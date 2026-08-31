"""Translate Needle tool calls into Home Assistant service calls.

The model returns grammar-constrained JSON like::

    {"name": "light_turn_on", "arguments": {"brightness_pct": 30}}

This module turns that into ``light.turn_on`` against the right entities.

**Targeting comes from the sentence, not from the model.** In order:

1. a device named outright - "turn on the reading lamp" - matched against the
   entities this installation actually has, by friendly name or Home Assistant
   alias. A named device wins over any room, because naming one is more
   specific than naming the room it sits in.
2. a whole-home marker: "בכל הבית". Kept distinct from naming no room at all,
   because conflating them turns "turn off the light" into "turn off every
   light in the house".
3. a room named in the sentence, matched against the area registry by
   :mod:`slot_match`. With more than one call and as many rooms as calls, the
   n-th room goes to the n-th call.
4. "all the <device>" with no place named at all - every area. Checked after
   rooms, so "turn on all the lights in the kitchen" still means the kitchen.
5. a room named in the sentence that *this house does not have* - refuse.
   "Turn on the light in the garage" in a house with no garage has to fail, not
   widen to every light in the building.
6. nothing named -> the area of the device the command came from. A satellite
   in the bedroom hearing "turn on the light" means *this* room.
7. nothing named and the device is in no area -> the room this conversation
   resolved a moment ago. Only ever reached when 1-6 all found nothing, where
   the alternative is an unconstrained match over the whole house.

The model's own ``area`` slug is not consulted. It is right 51.4% of the time
against the resolver's 99.0% on the same rows, it cannot express a room outside
the twelve it was trained on, and it invents one on commands that named no room
at all - "תכבה את האור" returns ``area="terrace"`` from a live engine. See
``slot_match`` for the measurement and the reasoning.

Entity matching uses Home Assistant's own ``intent.async_match_targets`` rather
than a hand-rolled registry walk, so exposure settings, duplicate names and
device-class handling behave exactly as they do for built-in intents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import monotonic
from typing import Any

from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from homeassistant.helpers import (
    intent,
)
from homeassistant.util import dt as dt_util

from . import direction, slot_match, tool_router
from .const import (
    ACTION_ARG,
    ACTIONS,
    ALL_WHEN_UNNAMED,
    BOOLEAN_SLOT,
    CALL_OF,
    CONF_MUSIC_PLAYER,
    DEVICE_CLASS_DOMAINS,
    DURATION_TOOLS,
    FALLBACK_DOMAINS,
    HEBREW_MONTHS,
    HELPER_IS_A_TIMER,
    LIST_INTEGRATIONS,
    LIST_ITEM_KEY,
    MEDIA_TOOLS,
    MUSIC_INTEGRATION,
    NAME_ADDRESSED,
    NON_SERVICE_ARGS,
    NUMBER_SLOT,
    QUERY_TOOLS,
    ROOM_MEMORY_SECONDS,
    ROUTINE_SIBLING,
    SERVICE_MAP,
    SETTING_SLOT,
    TOOL_ARGS,
    TOOL_DOMAIN,
    VIRTUAL_OF,
    WEATHER_STATES_HE,
)
from .hebrew_text import normalise

_LOGGER = logging.getLogger(__name__)


@dataclass
class CallOutcome:
    """What happened to one tool call."""
    tool: str
    ok: bool
    detail: str = ""
    entities: int = 0
    speech: str | None = None
    # The entities a *question* was answered from. Home Assistant's
    # conversation API distinguishes `action_done` from `query_answer`, and the
    # second carries the states it answered about so a caller can do something
    # with them rather than only hear a sentence. Separate from `entities`,
    # which is a count the spoken reply uses; this is the list itself, and it
    # stays empty for anything that actuates.
    answered_from: tuple[str, ...] = ()


class CallExecutor:
    """Executes Needle tool calls against Home Assistant."""

    def __init__(self, hass: HomeAssistant, entry: Any = None) -> None:
        self.hass = hass
        self.slots = slot_match.SlotIndex(hass)
        # The config entry rather than a snapshot of its options: Home
        # Assistant replaces the options mapping when the user saves the
        # dialog, so a copy taken at setup would go stale the first time
        # somebody changed the default speaker.
        self._entry = entry
        # conversation_id -> (when, area_ids). Bounded by
        # ROOM_MEMORY_SECONDS on read and by the fact that a household holds
        # one conversation at a time; see `_remembered_area`.
        self._recent_area: dict[str, tuple[float, list[str]]] = {}

    @property
    def options(self) -> dict[str, Any]:
        return dict(getattr(self._entry, "options", None) or {})

    # -- targeting ----------------------------------------------------------
    def _device_area_id(self, device_id: str | None) -> str | None:
        if not device_id:
            return None
        device = dr.async_get(self.hass).async_get(device_id)
        if device is None:
            return None
        if device.area_id:
            return device.area_id
        return None

    # -- what the last turn was about ---------------------------------------
    def _remember_area(self, conversation_id: str | None,
                       area_ids: list[str]) -> None:
        """Note the room this conversation just resolved.

        Expired entries are dropped on the way in. A conversation that never
        gets a second turn would otherwise leave its room here for the life of
        the process, and a household holds thousands of conversations.
        """
        if not (conversation_id and area_ids):
            return
        now = monotonic()
        for stale in [key for key, (when, _) in self._recent_area.items()
                      if now - when > ROOM_MEMORY_SECONDS]:
            del self._recent_area[stale]
        self._recent_area[conversation_id] = (now, list(area_ids))

    def _remembered_area(self, conversation_id: str | None) -> list[str]:
        """The room the last turn of this conversation named, if it is fresh.

        Deliberately not a conversation *history* - the model never sees this,
        and cannot: a Hebrew character costs 1.87 tokens against a 256-token
        window, so a previous turn would eat half the context to say what one
        registry lookup says exactly. This is one slot, remembered outside the
        model, which is the same division of labour as everything else here.

        ``monotonic`` rather than wall-clock time so a clock change cannot make
        a stale room look fresh.
        """
        if not conversation_id:
            return []
        remembered = self._recent_area.get(conversation_id)
        if remembered is None:
            return []
        when, area_ids = remembered
        if monotonic() - when > ROOM_MEMORY_SECONDS:
            del self._recent_area[conversation_id]
            return []
        return list(area_ids)

    def _device_class(self, tool: str, target: str, utterance: str,
                      args: dict[str, Any]) -> str | None:
        """Which kind of thing inside a domain that holds several."""
        if target not in DEVICE_CLASS_DOMAINS:
            return None
        if utterance and (said := slot_match.setting_from(
                utterance, "device_class")):
            return said
        guess = args.get("device_class")
        return guess if isinstance(guess, str) else None

    def _target_area(self, device_id: str | None, utterance: str = "",
                     index: int = 0, total: int = 1,
                     conversation_id: str | None = None,
                     floor: str | None = None
                     ) -> tuple[list[str], bool, bool]:
        """Resolve the target areas from what was said.

        Returns ``(area_ids, is_all, unresolvable)``. The list is usually one
        room; it is longer when one order named several - "turn off the light
        in the living room and in the kitchen" is one call and two rooms.

        ``unresolvable`` is the important one. If the speaker named a room and
        this house does not have it, the command must fail rather than widen.
        Without that, "turn on the light in the garage" in a house with no
        garage falls through to an unconstrained match and switches on every
        light in the building. Knowing a room was named is a separate question
        from knowing which one, and ``slot_match.names_a_room`` answers it from
        the project's whole Hebrew room lexicon rather than from this house's
        areas.
        """
        if utterance:
            if slot_match.mentions_whole_home(utterance):
                return [], True, False
            area_ids = self.slots.areas_for_call(utterance, index, total)
            if area_ids:
                self._remember_area(conversation_id, area_ids)
                return area_ids, False, False
            # A floor is a set of rooms. Consulted after rooms and before
            # "all the lights", so a sentence naming both a room and its floor
            # means the room - the more specific of the two - and one naming
            # only the floor reaches every area on it rather than the whole
            # house. See `slot_match.areas_on_floor`.
            floor_areas = self.slots.areas_on_floor(utterance)
            if floor_areas:
                _LOGGER.debug("a floor was named; targeting %s", floor_areas)
                return floor_areas, False, False
            # The model's own floor slug, for the phrasing the lexicon does
            # not carry. Same standing as its area slug and consulted for the
            # same reason the area slug is not: the sentence is right far more
            # often, so it goes first and this is what is left.
            if floor and (slug_areas := self.slots.areas_on_floor_slug(floor)):
                _LOGGER.debug("the model named floor %r; targeting %s",
                              floor, slug_areas)
                return slug_areas, False, False

            # "turn off all the lights" - every device, no place named.
            if slot_match.mentions_every_device(utterance):
                return [], True, False
            if slot_match.names_a_room(utterance):
                fallback = self._device_area_id(device_id)
                if fallback:
                    _LOGGER.debug(
                        "a room was named but this installation has no such "
                        "area; using the device's own area")
                    return [fallback], False, False
                _LOGGER.warning(
                    "%r names a room this installation does not have, and there "
                    "is no device area to fall back on; refusing rather than "
                    "targeting the whole house", utterance)
                return [], False, True

        # No room named: "wherever I am". The satellite's own area is the best
        # answer and needs no memory - a speaker in the bedroom saying "turn on
        # the light" means that room whatever was said a minute ago.
        own = self._device_area_id(device_id)
        if own:
            return [own], False, False

        # Nothing named and the satellite belongs to no area. This is the one
        # place a previous turn is worth consulting, and it is worth it because
        # of what it replaces: an empty list here means *unconstrained*, so
        # "תכבה את האור" typed into the web interface reaches every light in
        # the house. Inheriting the room the same conversation resolved seconds
        # ago is strictly better than that, and it can never override a room
        # the sentence named or a room the device is in - both return above.
        if (inherited := self._remembered_area(conversation_id)):
            _LOGGER.debug("no room named and no device area; this conversation "
                          "was about %s", inherited)
            return inherited, False, False
        return [], False, False

    def _match_entities(self, domain: str, area_ids: list[str],
                        all_areas: bool,
                        device_class: str | None = None) -> list[str]:
        """Entities of ``domain`` in every named area, in the order named."""
        if all_areas or not area_ids:
            return self._match_in_area(domain, None, device_class)
        registry = ar.async_get(self.hass)
        found: list[str] = []
        for area_id in area_ids:
            area = registry.async_get_area(area_id)
            found.extend(self._match_in_area(
                domain, area.name if area else None, device_class))
        return list(dict.fromkeys(found))

    def _match_in_area(self, domain: str, area_name: str | None,
                       device_class: str | None = None) -> list[str]:
        """Entities of one domain in one area, optionally of one kind.

        ``device_class`` narrows a domain that holds several kinds of thing.
        A living room with both blinds and curtains has two `cover` entities
        and "תפתח את הווילונות בסלון" names one of them; without this both
        move. The word comes out of the sentence, matched against the list
        Home Assistant's own Hebrew leaders wrote - see
        ``slot_match.SETTING_WORDS["device_class"]``.

        **Narrowing that finds nothing widens rather than refuses.**
        ``device_class`` is optional on a cover entity and a great many
        integrations leave it unset, so a house whose blinds are plain covers
        would otherwise stop answering a command it answers today. The
        narrowing can only ever make an over-broad match narrower, never turn
        a working command into "לא מצאתי מכשיר מתאים".
        """
        classes = [device_class] if device_class else None
        found = self._constrained(domain, area_name, classes)
        if not found and classes:
            _LOGGER.debug("no %s of class %s; matching the whole domain",
                          domain, device_class)
            found = self._constrained(domain, area_name, None)
        return found

    def _constrained(self, domain: str, area_name: str | None,
                     classes: list[str] | None) -> list[str]:
        constraints = intent.MatchTargetsConstraints(
            domains=[domain],
            area_name=area_name,
            device_classes=classes,
            assistant="conversation",
        )
        result = intent.async_match_targets(self.hass, constraints)
        if not result.is_match:
            # Retry without the assistant filter: entities that were never
            # explicitly exposed would otherwise be invisible, which reads to
            # the user as "the model is broken" rather than "this is unexposed".
            result = intent.async_match_targets(
                self.hass,
                intent.MatchTargetsConstraints(domains=[domain],
                                               area_name=area_name,
                                               device_classes=classes),
            )
        if not result.is_match:
            return []
        return [state.entity_id for state in result.states]

    def _match_named(self, domain: str, utterance: str,
                     fallback: str | None) -> list[str]:
        """Find the scene, script, automation, timer or helper that was asked for.

        The sentence first, matched against the entities this installation
        actually has. Only if that finds nothing does the model's ``name`` slug
        get a turn, and it is a weak second: the fine-tune maps Hebrew onto
        about forty invented English slugs, so a household whose scene is called
        ``מצב סרט`` was never addressable through it. Measured engine output for
        this slot includes ``"name": "בקיה"`` and ``"name": "ב\\ufffdבי\\ufffdתנמה"``.
        """
        if utterance:
            found = self.slots.entities(utterance, domain)
            if found:
                return found
        return self._match_by_slug(domain, fallback) if fallback else []

    def _match_by_slug(self, domain: str, name: str) -> list[str]:
        """Find scene/script/automation/timer/input_boolean by slug or name."""
        want = str(name).strip().casefold().replace(" ", "_")
        exact = f"{domain}.{want}"
        if self.hass.states.get(exact):
            return [exact]

        hits = []
        for state in self.hass.states.async_all(domain):
            object_id = state.entity_id.split(".", 1)[1].casefold()
            friendly = str(state.attributes.get("friendly_name", "")).casefold()
            if (want in (object_id, friendly)
                    or want in object_id.split("_")
                    or want.replace("_", " ") in friendly):
                hits.append(state.entity_id)
        return hits

    # -- argument translation ----------------------------------------------
    def _service_data(self, tool: str, args: dict[str, Any],
                      entity_ids: list[str],
                      utterance: str = "") -> dict[str, Any]:
        """Model arguments -> Home Assistant service data."""
        allowed = TOOL_ARGS.get(tool, frozenset())
        data: dict[str, Any] = {
            k: v for k, v in args.items()
            if k not in NON_SERVICE_ARGS and k in allowed
        }

        # Slots the sentence names outright. These come first and stand apart
        # from the arithmetic below, because they are not a translation of the
        # model's answer - they replace it. See `slot_match.SETTING_WORDS`.
        # The speaker said "שקט" or "גבוה", and which of the enum's values
        # that is does not need a model: 474 right and 0 wrong.
        # A tuple since v11: one behaviour can carry more than one such slot.
        # "תדליק אור חם בסלון" names a colour temperature and "תדליק אור אדום"
        # names a colour, and both belong to `light_turn_on`.
        for slot in SETTING_SLOT.get(tool, ()):
            if not utterance or slot not in allowed:
                continue
            if (said := slot_match.setting_from(utterance, slot)) is not None:
                data[slot] = said

        if tool == "media_set_volume":
            # HA takes volume_level as 0..1; the model speaks in percent.
            if "volume_pct" in data:
                data["volume_level"] = max(0.0, min(1.0, data.pop("volume_pct") / 100))
            if "volume_step_pct" in data:
                step = data.pop("volume_step_pct") / 100
                current = self._first_attr(entity_ids, "volume_level", 0.5)
                data["volume_level"] = max(0.0, min(1.0, current + step))

        elif tool == "climate_set_temperature":
            if "temperature_step" in data:
                step = data.pop("temperature_step")
                current = self._first_attr(entity_ids, "temperature", 23)
                data["temperature"] = max(7, min(35, current + step))

        elif tool == "timer_start":
            # Must carry into hours: the training data includes "שעתיים", and
            # "00:120:00" is not a duration Home Assistant accepts. Seconds
            # and hours are slots of their own since v11 - HassStartTimer has
            # all three, and "תעמיד טיימר לשעה וחצי" could only be said in
            # minutes before.
            if (total := self._seconds(data)) is not None:
                data["duration"] = (f"{total // 3600:02d}:"
                                    f"{total // 60 % 60:02d}:{total % 60:02d}")

        elif tool in ("timer_add", "timer_less"):
            # `timer.change` is one service with a signed duration, and its
            # own schema documents the integer-seconds form ("00:01:00, 60 or
            # -60"). Seconds rather than HH:MM:SS because a negative duration
            # string is not something the selector accepts.
            total = self._seconds(data) or 0
            data["duration"] = -total if tool == "timer_less" else total

        elif tool == "light_turn_on" and "color_temp_k" in data:
            # Home Assistant's own parameter name. The four values are the
            # ones `lists/he/lights.yaml` names, so this is the official
            # Hebrew list reaching the service it was written for.
            #
            # Coerced because the two sources disagree on type and both are
            # right: the schema declares an integer enum, and `SETTING_WORDS`
            # is a table of strings throughout. A Kelvin value arriving as
            # "2700" would be rejected by the service.
            try:
                data["color_temp_kelvin"] = int(data.pop("color_temp_k"))
            except (TypeError, ValueError):
                data.pop("color_temp_k", None)

        elif tool == "notify_send":
            # Take the message out of the sentence, not out of the model.
            # Hebrew reaches a tool argument as escape sequences - six exact
            # characters per letter - and the model gets them wrong: measured
            # output for this slot includes "끝4", a Hangul syllable
            # produced by dropping a hex digit. The sentence has the words.
            spoken = slot_match.extract_message(utterance) if utterance else None
            # Drop first, then set. The model's string must not survive even if
            # extraction fails: execute() refuses that case outright, and doing
            # it here too keeps the function honest on its own.
            data.pop("message", None)
            if spoken:
                data["message"] = spoken
            data.setdefault("title", "Home Assistant")

        elif tool == "broadcast":
            # Same rule, different service: `assist_satellite.announce` takes
            # the text under `message` and speaks it out loud.
            spoken = slot_match.extract_message(utterance) if utterance else None
            data.pop("message", None)
            if spoken:
                data["message"] = spoken

        # light.turn_on takes brightness_step_pct natively, so it passes through.
        return data

    @staticmethod
    def _seconds(data: dict[str, Any]) -> int | None:
        """Hours, minutes and seconds as one total, popped out of ``data``.

        ``None`` when the call carried no duration at all, which is what
        distinguishes "resume the timer" from "start a timer for nothing".
        """
        parts = [data.pop(k, None) for k in ("hours", "minutes", "seconds")]
        if all(v is None for v in parts):
            return None
        hours, minutes, seconds = (int(v or 0) for v in parts)
        return hours * 3600 + minutes * 60 + seconds

    def _first_attr(self, entity_ids: list[str], attr: str, default: float) -> float:
        for eid in entity_ids:
            state = self.hass.states.get(eid)
            if state and (val := state.attributes.get(attr)) is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return default

    # -- execution ----------------------------------------------------------
    async def execute(self, call: dict[str, Any], device_id: str | None,
                      context: Context, utterance: str = "",
                      index: int = 0, total: int = 1,
                      conversation_id: str | None = None) -> CallOutcome:
        """Run one tool call. ``index``/``total`` place it among its siblings,
        which is how a two-room sentence gets its two rooms in the right order.
        """
        name = call.get("name", "")
        args = dict(call.get("arguments") or {})

        # What the model emits is ``(tool, action)``; what everything below
        # this line speaks is the **virtual id** - the per-service name the
        # catalogue used before v11. Decoding here rather than threading the
        # pair through is what let the catalogue collapse from 42 tools to 20
        # without touching the direction guard, the service map, the routine
        # sibling rule, the timer correction or any of their measurements.
        #
        # A name that is already a virtual id passes through unchanged, so a
        # household still running v10 weights against this component keeps
        # working - and so does every test written before the rewrite.
        action = args.pop(ACTION_ARG, None)
        tool = VIRTUAL_OF.get(
            (name, action if isinstance(action, str) else None))
        if tool is None:
            tool = name if name in CALL_OF else ""
        if not tool:
            # Almost always one thing: weights trained against a different
            # catalogue. v11 replaced 42 per-service tools with 20 per-domain
            # ones, so a household running their own v10 fine-tune emits
            # `light_turn_off` where this expects `light_control{shut}` - and
            # `light_turn_off` still decodes, which is why the fallback above
            # exists. What reaches here is a name from neither catalogue.
            _LOGGER.warning(
                "%r is not a tool this catalogue has%s - if `weights_path` "
                "points at a fine-tune of your own, it was trained against a "
                "different tool set", name,
                f" (action {action!r})" if action else "")
            return CallOutcome(name, False, f"unknown tool {name}"
                               + (f" action {action!r}" if action else ""))

        # A helper toggle asked about a countdown is a timer command. Done
        # before the direction guard rather than after, so the guard settles
        # start against cancel on the tool this leaves behind: the toggle only
        # carries on or off, and "עצור את הטיימר" arrives as *turn_on*.
        # See const.HELPER_IS_A_TIMER for the corpus measurement.
        if (utterance and tool in HELPER_IS_A_TIMER
                and tool_router.names_a_timer(utterance)):
            _LOGGER.debug("%r is about a timer, not a helper toggle", utterance)
            tool = HELPER_IS_A_TIMER[tool]

        # Lock or unlock, open or close, on or off. The Hebrew verb settles
        # it and the model does not always agree with the verb - see
        # `direction`, where the signal is measured at 1337 right and 0 wrong
        # against gold. Corrected before anything else, because everything
        # below reads `tool`.
        if utterance and (settled := direction.settle(tool, utterance)) != tool:
            _LOGGER.debug("the sentence says %s, not %s", settled, tool)
            tool = settled

        # The noun says which family. The model disagrees with it far more
        # often than the noun is wrong: 19,410 agree and 7 disagree over the
        # corpus, and six of the seven are a word speech noise glued to its
        # neighbour. A starting point rather than a verdict - everything below
        # refines it. See `direction.FAMILY_ANCHOR`.
        if utterance:
            settled = direction.family_named(
                tool, tool_router.family_named(utterance))
            if settled != tool:
                _LOGGER.debug("%r names a %s, not a %s", utterance,
                              direction.family_of(settled), tool)
                tool = settled

        # A toggle has no opposite, so the pair above cannot reach it - and
        # v11 made that a live problem: all three of on, off and toggle are
        # values of one enum now, where the router used to rank the toggle
        # third and a tight shortlist usually cut it. 335 agree and 0 disagree
        # on the rows whose gold really is a toggle; see `direction.TOGGLES`.
        if utterance and (settled := direction.settle_toggle(tool, utterance)) != tool:
            _LOGGER.debug("%r says %s rather than a toggle", utterance, settled)
            tool = settled

        # A plug that is a speaker. A transport verb the model answered with a
        # switch is a media command - the television really is a plug in most
        # of this corpus, and a plug has no pause. `settle_media` below says
        # which of the eight. See `direction.TRANSPORT_IS_MEDIA`.
        if (utterance and tool in direction.TRANSPORT_IS_MEDIA
                and tool_router.names_a_transport(utterance)):
            _LOGGER.debug("%r is a transport verb, not a %s", utterance, tool)
            tool = "media_pause"

        # A door that is a blind. "תפתח את דלת החניה" is the garage, and דלת
        # alone is what a lock has, so the model answers `lock_control` and a
        # household that asked for the garage gets a bolt. All 147 corpus rows
        # naming one of the compound cover nouns are cover rows; see
        # `direction.NAMES_A_COVER`. After the toggle so a `switch_toggle` has
        # already become one side of its pair, and before the rest so `settle`
        # still says open or closed.
        if (utterance and tool in direction.NAMES_A_COVER
                and slot_match.names_a_compound_cover(utterance)):
            _LOGGER.debug("%r names a cover, not a %s", utterance, tool)
            tool = direction.NAMES_A_COVER[tool]
            tool = direction.settle(tool, utterance)

        # The number the model dropped, and the number it got wrong. "שנה את
        # הטמפרטורה ל20 מעלות" comes back as `climate_control{off}` with no
        # argument, and an air conditioner switches off when somebody asked
        # for twenty degrees.
        #
        # This used to fire only where the model left the slot empty. It is an
        # override now, because the sentence was measured to be right about
        # this whenever it speaks at all - 485 agree and 2 disagree over the
        # corpus, both of them "אשרים" - and a model that emits 23 for a
        # sentence saying 19 is the commoner failure of the two. Same rule as
        # the room, the floor, the name and the colour: the sentence decides.
        # See `slot_match.temperature_from`.
        if (utterance and tool in direction.CLIMATE_BEHAVIOURS
                and not args.get("temperature_step")
                and (degrees := slot_match.temperature_from(utterance))):
            _LOGGER.debug("the sentence says %s degrees", degrees)
            args["temperature"] = degrees

        # A climate call carrying a temperature is a call to set one. Not a
        # word in the sentence - the argument the model itself emitted, which
        # only one of the five behaviours has anywhere to put. Measured on
        # predictions rather than on gold: 160 agree, 1 disagrees. Unless the
        # clause asked for a fan speed or an hvac mode, in which case it asked
        # for that. See `direction.settle_climate`.
        if tool in direction.CLIMATE_BEHAVIOURS:
            settled = direction.settle_climate(
                tool, args, utterance,
                slot_match.mode_slot(utterance),
                None if (slot_match.setting_from(utterance, "fan_mode")
                         or slot_match.temperature_from(utterance))
                else slot_match.hvac_target(utterance))
            if settled != tool:
                _LOGGER.debug("%s carries a temperature, so it is %s",
                              tool, settled)
                tool = settled

        # And the two tools whose behaviours the sentence separates without
        # ever being wrong. An allow-list: the general form of this rule was
        # measured over the whole corpus at 647 disagreements and rejected.
        # See `direction.HINT_DECIDED`.
        if utterance and (name in direction.HINT_DECIDED
                          or CALL_OF.get(tool, ("",))[0] in direction.HINT_DECIDED):
            siblings = list(ACTIONS.get(CALL_OF[tool][0], {}).values())
            if (settled := direction.settle_action(tool, siblings, utterance)) != tool:
                _LOGGER.debug("the sentence says %s, not %s", settled, tool)
                tool = settled

        # Which of seven, for a countdown. The one tool the model does not do
        # at all - it collapsed onto two of the seven - and the one whose
        # behaviours the hint tables cannot separate either. 1262 agree and 6
        # disagree; see `direction.settle_timer`. It is also where a transport
        # verb aimed at a countdown lands back in the right family - see
        # `direction.TRANSPORT_IS_A_TIMER`.
        #
        # **After the allow-list, not before it.** `settle_action` reads the
        # router's hint lists, where עצור sits under cancel; "לעצור **רגע** את
        # הספירה" is a pause, and only `_T_PAUSE` carries that phrase. With
        # this block first, the allow-list then overruled it on eleven such
        # rows. Letting the specialist speak last costs nothing anywhere else
        # and gains 118 rows over the corpus - see `direction.HINT_DECIDED`.
        if utterance and tool in direction.SETTLES_A_TIMER:
            settled = direction.settle_timer(
                tool, args, utterance,
                tool_router.looks_like_question(utterance),
                tool_router.names_a_timer(utterance))
            if settled != tool:
                _LOGGER.debug("the sentence says %s, not %s", settled, tool)
                tool = settled

        # The clause that names its own behaviour outright: a fan asked to
        # turn, a cover asked to halt, a speaker asked to stop rather than
        # pause, a cover asked for a percentage. Four families
        # `settle_action`'s allow-list cannot take; see
        # `direction.settle_named` for the evidence.
        #
        # Before the numbers below, because a cover promoted to a placement
        # has to be one by the time `position` is filled in.
        if utterance:
            settled = direction.settle_named(
                tool, utterance,
                slot_match.percent_from(utterance) is not None,
                slot_match.setting_from(utterance, "color_name") is not None)
            if settled != tool:
                _LOGGER.debug("the clause names %s, not %s", settled, tool)
                tool = settled

        # Which of eight, for a speaker. 2020 agree, 1 disagree; see
        # `direction.settle_media`. **Before** the slot fills below, not
        # after: they are keyed on the behaviour, so a call promoted here to
        # `media_mute` had already been filled as whatever it was, and
        # `is_volume_muted` is not an argument of that. The same defect
        # `settle_climate` had, one family over - settle the behaviour, then
        # fill its slots. `evaluate.py` runs it in this place too.
        if utterance and tool in direction.MEDIA_BEHAVIOURS:
            settled = direction.settle_media(
                tool, utterance, slot_match.names_a_level(utterance))
            if settled != tool:
                _LOGGER.debug("the sentence says %s, not %s", settled, tool)
                tool = settled

        # And the other numbers Hebrew says out loud, read *after* every rule
        # that can still change which behaviour this is - unlike the
        # temperature above, which has to come first because
        # `settle_climate` promotes on it.
        #
        # Half of every number in the corpus is a word rather than a digit -
        # "שבעים וחמישה אחוז", "חצי", "רבע שעה" - and nothing here could read
        # one until `hebrew_numbers`. 1,175 agree and 3 disagree on the
        # percentage; see `const.NUMBER_SLOT`.
        if (utterance and (slot := NUMBER_SLOT.get(tool))
                and (said := slot_match.percent_from(utterance)) is not None):
            args[slot] = said

        # And the yes-or-no, read the same way and in the same place. A fan
        # asked to turn and a speaker asked to be silenced both carry one, and
        # the sentence always says which way; see `const.BOOLEAN_SLOT`.
        if utterance and (slot := BOOLEAN_SLOT.get(tool)):
            args[slot] = slot_match.switch_from(utterance, slot)

        # And the countdown. 553 exactly right, 7 wrong, and silent on every
        # timer row whose gold carries no duration - which is what keeps a
        # pause from acquiring one. The units are replaced together rather
        # than merged, because a model that said forty minutes for a sentence
        # saying two hours must not leave the forty behind.
        # See `const.DURATION_TOOLS`.
        if (utterance and tool in DURATION_TOOLS
                and (said_time := slot_match.duration_from(utterance))):
            for unit in ("hours", "minutes", "seconds"):
                args.pop(unit, None)
            args.update(said_time)

        # And the mirror of every rule above: a slot the sentence owns and
        # does not name is a slot the model invented. 459 invented arguments
        # in one release run; see `slot_match.SENTENCE_OWNS` for the nine
        # slots this is measured safe on and the four it is not.
        if utterance:
            for slot in [k for k in args if slot_match.unsupported(utterance, k)]:
                _LOGGER.debug("the sentence does not support %s=%r",
                              slot, args[slot])
                args.pop(slot, None)

        # The same verb also settles which way a relative argument points, and
        # `_service_data` below adds those to the device's current reading - so
        # a wrong sign moves the thermostat away from what was asked instead of
        # towards it. 1222 right and 1 wrong; see `direction`.
        if utterance:
            args = direction.settle_steps(
                args, utterance,
                "temperature_step" in TOOL_ARGS.get(tool, frozenset())
                and slot_match.unsupported(utterance, "temperature"))

        if tool in QUERY_TOOLS:
            return await self._answer_query(tool, args, device_id, utterance,
                                            index, total, conversation_id)

        # "Play" and "play *this*" are one verb apart in Hebrew, and which one
        # was meant is decided by whether a name follows - which the sentence
        # settles and the model has to guess. So when the model reaches for
        # some other media tool about a sentence that named something
        # specific, the sentence wins.
        #
        # This is the same rule the rest of this module runs on, and it is
        # safe here for the same reason: it never overrides which *domain* was
        # chosen. The model has already decided the utterance is about audio -
        # "תפעיל את השואב" gets vacuum_start and is never seen here - so all
        # that is being corrected is which media tool inside that decision. It
        # also makes the tool work before any model knows it exists, which is
        # what a household running the previous weights has.
        #
        # Two strengths of evidence, because they carry different risks. A
        # model that already said *play* has only the "what" left to get
        # wrong, so any title is enough. A model that said something else -
        # set the volume, pause - is being overruled on the verb too, so the
        # sentence has to have named the **kind** as well ("האלבום", "פלייליסט")
        # and no number: "שים את השיר על שישים" is a volume and says so.
        #
        # Measured over both splits, on the 663 media rows where the sentence
        # names a kind and a title and no number: 661 are music_play and the
        # two that are not are the same glued-ו artifact the narrow rule
        # already mishandles today, so the widening breaks nothing new. On the
        # 123 music rows of the held-out set it takes tool-set from 71.5% to
        # **88.6%** and argument F1 from 72.9% to 89.7% - twenty-one rows,
        # most of them "ערבב את האלבום", shuffle, answered with
        # media_set_volume.
        if tool in MEDIA_TOOLS and utterance:
            request = slot_match.extract_music(utterance)
            if request is not None and (
                    tool == "media_play"
                    or (request.media_type
                        and not slot_match.names_a_level(utterance))):
                _LOGGER.debug("the sentence names something to play; using "
                              "music_play instead of %s", tool)
                tool = "music_play"

        # And the mirror of that upgrade: a `music_play` about a sentence
        # that named nothing to play, and named one transport verb, is a
        # transport command. 383 agree, 0 disagree; see `direction`.
        if tool == "music_play" and utterance:
            settled = direction.settle_transport(
                tool, utterance, slot_match.extract_music(utterance) is not None)
            if settled != tool:
                _LOGGER.debug("nothing to play in %r; using %s", utterance, settled)
                tool = settled

        if tool == "music_play":
            return await self._play_music(args, device_id, context, utterance,
                                          index, total)

        if tool not in SERVICE_MAP:
            return CallOutcome(tool, False, f"unknown tool {tool}")

        domain, service = SERVICE_MAP[tool]
        # Where to look for entities, which is not always where the
        # service lives - see const.TOOL_DOMAIN.
        target = TOOL_DOMAIN[tool]

        if tool in NAME_ADDRESSED:
            named_domain = NAME_ADDRESSED[tool]
            entity_ids = self._match_named(named_domain, utterance,
                                           args.get("name"))
            if not entity_ids and (sibling := ROUTINE_SIBLING.get(tool)):
                # See ROUTINE_SIBLING: the registry knows whether this
                # household's "אווירת ערב" is a scene or a script.
                entity_ids = self._match_named(
                    NAME_ADDRESSED[sibling], utterance, args.get("name"))
                if entity_ids:
                    _LOGGER.debug("%s is a %s here, not a %s", args.get("name"),
                                  NAME_ADDRESSED[sibling], named_domain)
                    tool = sibling
                    domain, service = SERVICE_MAP[tool]
            if (not entity_ids and not args.get("name")
                    and tool in ALL_WHEN_UNNAMED):
                # "cancel the timer" with nothing to disambiguate: all of them.
                entity_ids = [s.entity_id
                              for s in self.hass.states.async_all(named_domain)]
        elif tool == "broadcast":
            # `HassBroadcast` -> `assist_satellite.announce`. Same rule as
            # notify_send: the words come out of the sentence, never out of
            # the model, because Hebrew reaches a tool argument as escape
            # sequences the model gets wrong. What differs is the audience -
            # this speaks out loud in the house, that one buzzes a phone.
            if not slot_match.extract_message(utterance):
                return CallOutcome(tool, False, "no message in the sentence")
            entity_ids = [st.entity_id
                          for st in self.hass.states.async_all("assist_satellite")]
        elif tool in LIST_INTEGRATIONS["todo"]:
            return await self._edit_list(tool, args, context, utterance)
        elif tool == "notify_send":
            # The words have to come out of the sentence - see _service_data for
            # why the model's own string is never safe to deliver. Its docstring
            # gives None the meaning "do not send"; honour that here rather than
            # notifying a phone with whatever survived decoding.
            if not slot_match.extract_message(utterance):
                return CallOutcome(tool, False, "no message in the sentence")
            entity_ids = [s.entity_id for s in self.hass.states.async_all("notify")]
        else:
            # A device named outright beats the room it stands in.
            entity_ids = (self.slots.entities(utterance, target)
                          if utterance and target in slot_match.NAMEABLE_DOMAINS
                          else [])
            if entity_ids:
                _LOGGER.debug("targeting named entities %s", entity_ids)
            else:
                area_ids, all_areas, unresolvable = self._target_area(
                    device_id, utterance, index, total, conversation_id,
                    args.get("floor"))
                if unresolvable:
                    return CallOutcome(tool, False, "no matching entities")
                # Which *kind*, for the domains that hold several. "תפתח את
                # הווילונות בסלון" and "תפתח את התריסים בסלון" are two
                # commands in a room that has both, and were one before this.
                #
                # The sentence first and the model second, which is the order
                # every slot in this module uses. The model's own guess is
                # consulted only where the sentence named no kind at all -
                # narrowing on a wrong class finds nothing, which is a refusal
                # rather than a wrong action, and it is still worth having
                # because a house can call a blind something this lexicon does
                # not carry.
                entity_ids = self._match_entities(
                    target, area_ids, all_areas,
                    self._device_class(tool, target, utterance, args))

                # Nothing of the primary domain in the room. Before refusing,
                # try the domains this tool stands in for: a boiler is switched
                # on with the same Hebrew verb as a plug, a valve opens with the
                # verb a blind opens with, and three of the eleven domains Home
                # Assistant exposes to Assist by default had no route here at
                # all. Reached only when the primary domain found nothing, so it
                # can turn a refusal into an action and never one action into a
                # different one.
                for fallback, its_domain, its_service in \
                        FALLBACK_DOMAINS.get(tool, ()):
                    if entity_ids:
                        break
                    entity_ids = self._match_entities(
                        fallback, area_ids, all_areas)
                    if entity_ids:
                        _LOGGER.debug("no %s here, but a %s - %r",
                                      target, fallback, utterance)
                        domain, service = its_domain, its_service

        if not entity_ids:
            return CallOutcome(tool, False, "no matching entities")

        data = self._service_data(tool, args, entity_ids, utterance)
        return await self._call(domain, service,
                                {"entity_id": entity_ids, **data},
                                context, tool, len(entity_ids))

    # -- lists ---------------------------------------------------------------
    async def _edit_list(self, tool: str, args: dict[str, Any],
                         context: Context, utterance: str) -> CallOutcome:
        """Add, complete or remove an item on a shopping or to-do list.

        `todo` is one of the eleven domains Home Assistant exposes to Assist by
        default, and a household could see its shopping list offered to the
        assistant and get "לא מצאתי מכשיר מתאים" - there was no route here at
        all before v11.

        Which integration serves the request is decided by the registry rather
        than by the model, for the same reason `ROUTINE_SIBLING` exists: a
        household has the modern `todo` domain, or the legacy `shopping_list`
        integration, or both, and the sentence cannot know which. The two do
        not even share a vocabulary - `todo` completes an item with
        `update_item` and a status, `shopping_list` with `complete_item` - so
        guessing would fail rather than merely mis-target.

        The item itself comes out of the sentence. Hebrew reaches a tool
        argument as escape sequences, six exact characters per letter, and
        model gets them wrong; see `_service_data` for the measurement that
        settled it for `notify_send`.
        """
        item = slot_match.extract_item(utterance) if utterance else None
        if not item:
            return CallOutcome(tool, False, "no item named in the sentence")

        wanted = (slot_match.setting_from(utterance, "list") if utterance
                  else None) or args.get("list")
        entity_ids = self.slots.list_entities(utterance, wanted)
        if entity_ids:
            kind = "todo"
        elif self.hass.services.has_service("shopping_list", "add_item"):
            kind, entity_ids = "shopping", []
        else:
            return CallOutcome(tool, False, "no list in this installation")

        domain, service, extra = LIST_INTEGRATIONS[kind][tool]
        data: dict[str, Any] = {LIST_ITEM_KEY[kind]: item, **extra}
        if kind == "todo":
            data["entity_id"] = entity_ids
        return await self._call(domain, service, data, context, tool,
                                max(len(entity_ids), 1))

    # -- music --------------------------------------------------------------
    def _music_players(self, area_ids: list[str], all_areas: bool) -> list[str]:
        """Music Assistant players, narrowed to an area when one was named.

        ``music_assistant.play_media`` targets media_player entities that the
        music_assistant integration provides - its own service definition says
        so - so an ordinary Sonos or Chromecast entity is not a legal target
        even though it plays audio. The registry's ``platform`` field is the
        same test the service applies.
        """
        registry = er.async_get(self.hass)
        players = [
            entry.entity_id
            for entry in registry.entities.values()
            if entry.domain == "media_player"
            and entry.platform == MUSIC_INTEGRATION
            and not entry.disabled_by
        ]
        if not players or all_areas or not area_ids:
            return players
        in_area = [
            eid for eid in players
            if (entry := registry.async_get(eid))
            and (entry.area_id or self._device_area_id(entry.device_id)) in area_ids
        ]
        # A named room with no Music Assistant player in it falls back to every
        # player rather than to silence: the caller has already decided the
        # room is real, and the configured default is checked next.
        return in_area or players

    async def _play_music(self, args: dict[str, Any], device_id: str | None,
                          context: Context, utterance: str,
                          index: int, total: int) -> CallOutcome:
        """Search Music Assistant for what the sentence named, and play it.

        Three things can happen, and all three are ordinary:

        * The sentence named something - play it.
        * The sentence asked for music without naming any - "תנגן מוזיקה
          בסלון". There is nothing to search for, so this resumes playback,
          which is what the words mean.
        * This house has no Music Assistant at all. The tool then degrades to
          ``media_player.media_play`` on whatever speaker the room has, so a
          household without it is no worse off than before the tool existed.
        """
        area_ids, all_areas, unresolvable = self._target_area(
            device_id, utterance, index, total)
        if unresolvable:
            return CallOutcome("music_play", False, "no matching entities")

        players = self._music_players(area_ids, all_areas)
        request = slot_match.extract_music(utterance) if utterance else None

        if not players:
            _LOGGER.debug("no Music Assistant player; resuming playback instead")
            speakers = self._match_entities("media_player", area_ids, all_areas)
            if not speakers:
                return CallOutcome("music_play", False, "no matching entities")
            return await self._call("media_player", "media_play",
                                    {"entity_id": speakers}, context, "music_play",
                                    len(speakers))

        if len(players) > 1 and (chosen := self._configured_player(players)):
            players = [chosen]

        if request is None:
            return await self._call("media_player", "media_play",
                                    {"entity_id": players}, context, "music_play",
                                    len(players))

        data: dict[str, Any] = {"entity_id": players, "media_id": request.media_id}
        # The sentence beats the model. It said "the album Shablul" in words;
        # the model guessed from five options.
        media_type = request.media_type or args.get("media_type")
        if media_type:
            data["media_type"] = media_type
        if request.artist:
            data["artist"] = request.artist
        return await self._call(*SERVICE_MAP["music_play"], data, context,
                                "music_play", len(players),
                                speech=f"מנגן {request.media_id}")

    def _configured_player(self, players: list[str]) -> str | None:
        """The speaker chosen in the options, if it is still a real player."""
        chosen = self.options.get(CONF_MUSIC_PLAYER)
        return chosen if chosen in players else None

    async def _call(self, domain: str, service: str, data: dict[str, Any],
                    context: Context, tool: str, entities: int,
                    speech: str | None = None) -> CallOutcome:
        """One service call, with the failure path every caller needs."""
        try:
            await self.hass.services.async_call(
                domain, service, data, blocking=True, context=context)
        except Exception as err:  # service validation, unavailable device, ...
            _LOGGER.error("%s.%s failed: %s", domain, service, err)
            return CallOutcome(tool, False, str(err))
        return CallOutcome(tool, True, entities=entities, speech=speech)

    # -- read-only ----------------------------------------------------------
    async def _answer_query(self, tool: str, args: dict[str, Any],
                            device_id: str | None, utterance: str = "",
                            index: int = 0, total: int = 1,
                            conversation_id: str | None = None) -> CallOutcome:
        # `HassGetCurrentTime` and `HassGetCurrentDate`. Two built-in intents
        # that needed no device, no registry and no model call - only Home
        # Assistant's own configured timezone, which `dt_util.now` reads.
        if tool == "get_time":
            now = dt_util.now()
            return CallOutcome(tool, True, speech=f"השעה {now.hour}:{now.minute:02d}")
        if tool == "get_date":
            now = dt_util.now()
            return CallOutcome(
                tool, True,
                speech=f"{now.day} ב{HEBREW_MONTHS[now.month - 1]} {now.year}")

        # `HassTimerStatus`. Answering it by calling `timer.start` - which is
        # what the shortlist used to make happen - restarts the very countdown
        # somebody wanted to know about.
        if tool == "timer_status":
            return self._answer_timer(tool, args, utterance)

        if tool == "get_weather":
            states = self.hass.states.async_all("weather")
            if not states:
                return CallOutcome(tool, False, "no weather entity")
            st = states[0]
            temp = st.attributes.get("temperature")
            # The raw state is an English slug; say it in Hebrew.
            parts = [f"מזג האוויר {WEATHER_STATES_HE.get(st.state, st.state)}"]
            if temp is not None:
                parts.append(f"{round(float(temp))} מעלות")
            return CallOutcome(tool, True, speech=", ".join(parts),
                               entities=1, answered_from=(st.entity_id,))

        # Which domain to look at comes out of the sentence, for the same
        # reason the area does: the router types a state question correctly on
        # 98.0% of the generated rows, against 0.9% argument F1 for the model
        # on this slot. "מה המצב של החלון" is a window contact, not a blind,
        # and the model has never once got that right. Its own guess is the
        # fallback for the sentences the nouns do not reach.
        domain = (tool_router.query_domain(utterance) if utterance else None) \
            or args.get("domain") or "light"
        entity_ids = (self.slots.entities(utterance, domain)
                      if utterance and domain in slot_match.NAMEABLE_DOMAINS
                      else [])
        if not entity_ids:
            area_ids, all_areas, unresolvable = self._target_area(
                device_id, utterance, index, total, conversation_id)
            if unresolvable:
                return CallOutcome(tool, False, "no matching entities")
            entity_ids = self._match_entities(domain, area_ids, all_areas)
        if not entity_ids:
            return CallOutcome(tool, False, "no matching entities")

        on_words = {"on", "open", "unlocked", "playing", "cleaning", "active", "home"}
        on, off, values = [], [], []
        for eid in entity_ids:
            # A separate name from the weather `st` above: that one is always a
            # state, this one is whatever the registry still has, which for an
            # entity removed mid-sentence is nothing.
            state = self.hass.states.get(eid)
            if state is None:
                continue
            name = state.attributes.get("friendly_name", eid)
            if domain in ("sensor",):
                unit = state.attributes.get("unit_of_measurement", "")
                values.append(f"{name}: {state.state} {unit}".strip())
            elif state.state in on_words:
                on.append(name)
            else:
                off.append(name)

        # "אילו אורות דולקים" asks for the lights that are *on*, not for all of
        # them. Without the filter the answer is "3 פעילים, 2 כבויים", which is
        # true and is not what was asked. The state word comes out of the
        # sentence against Home Assistant's own Hebrew list - see
        # `slot_match.SETTING_WORDS["state"]`.
        #
        # Only this branch is new: a question that names no state answers
        # exactly as it did before, so the phrasing every existing test asserts
        # is untouched. And it is `state_filter` rather than the state word on
        # its own, because "תבדוק אם האור בגן דולק" wants yes or no - reading
        # the word alone turned 127 such questions into lists.
        if (wanted := slot_match.state_filter(utterance) if utterance else None):
            named = on if wanted in ("on", "open") else off
            return CallOutcome(
                tool, True, entities=len(entity_ids),
                speech=self._say_which(utterance, named, len(entity_ids)),
                answered_from=tuple(entity_ids))

        if values:
            speech = ", ".join(values[:4])
        elif on and not off:
            speech = "כן, " + ", ".join(on[:4])
        elif off and not on:
            speech = "לא, " + ", ".join(off[:4])
        else:
            speech = f"{len(on)} פעילים, {len(off)} כבויים"
        return CallOutcome(tool, True, speech=speech, entities=len(entity_ids),
                           answered_from=tuple(entity_ids))

    def _answer_timer(self, tool: str, args: dict[str, Any],
                      utterance: str) -> CallOutcome:
        """How long is left on a countdown, and whether one is running."""
        entity_ids = self._match_named("timer", utterance, args.get("name"))
        if not entity_ids:
            entity_ids = [st.entity_id
                          for st in self.hass.states.async_all("timer")]
        running = []
        for eid in entity_ids:
            state = self.hass.states.get(eid)
            if state is None or state.state == "idle":
                continue
            left = state.attributes.get("remaining") or state.attributes.get("duration")
            name = state.attributes.get("friendly_name", eid)
            running.append(f"{name}: {left}" if left else name)
        if not running:
            return CallOutcome(tool, True, speech="אין טיימר פעיל",
                               entities=len(entity_ids),
                               answered_from=tuple(entity_ids))
        return CallOutcome(tool, True, speech=", ".join(running[:3]),
                           entities=len(entity_ids),
                           answered_from=tuple(entity_ids))

    @staticmethod
    def _say_which(utterance: str, named: list[str], total: int) -> str:
        """Answer a state-filtered question in the shape it was asked.

        Home Assistant's own Hebrew responses answer these three differently,
        and a household hears the difference: "האם האורות דולקים" wants yes or
        no, "אילו אורות דולקים" wants the list, "כמה אורות דולקים" wants the
        number. The tool and the entities are identical for all three - only
        the sentence says which answer was wanted.
        """
        text = normalise(utterance)
        if "כמה" in text:
            return str(len(named))
        if "אילו" in text or "איזה" in text:
            return ", ".join(named[:4]) if named else "אף אחד"
        # "האם כל האורות דולקים" is a different question from "האם האורות
        # דולקים": one asks about every device and the other about any, and
        # answering "כן" when two of five are on would be wrong.
        if "כל" in text:
            return "כן" if named and len(named) == total else (
                "לא, " + ", ".join(n for n in named[:3]) if named else "לא")
        if not named:
            return "לא"
        return "כן, " + ", ".join(named[:4])
