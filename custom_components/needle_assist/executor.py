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
from typing import Any

from homeassistant.core import HomeAssistant, Context
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
)

from . import direction, slot_match, tool_router
from .const import (
    ALL_WHEN_UNNAMED, CONF_MUSIC_PLAYER, MUSIC_INTEGRATION, NAME_ADDRESSED,
    NON_SERVICE_ARGS, QUERY_TOOLS, ROUTINE_SIBLING, SERVICE_MAP, TOOL_DOMAIN,
    WEATHER_STATES_HE,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class CallOutcome:
    """What happened to one tool call."""
    tool: str
    ok: bool
    detail: str = ""
    entities: int = 0
    speech: str | None = None


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

    @property
    def options(self) -> dict:
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

    def _target_area(self, device_id: str | None, utterance: str = "",
                     index: int = 0, total: int = 1
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
                return area_ids, False, False
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

        # No room named: "wherever I am". An unknown device area leaves this
        # unconstrained, which matches how Home Assistant's own intents behave
        # for an area-less command.
        own = self._device_area_id(device_id)
        return ([own] if own else []), False, False

    def _match_entities(self, domain: str, area_ids: list[str],
                        all_areas: bool) -> list[str]:
        """Entities of ``domain`` in every named area, in the order named."""
        if all_areas or not area_ids:
            return self._match_in_area(domain, None)
        registry = ar.async_get(self.hass)
        found: list[str] = []
        for area_id in area_ids:
            area = registry.async_get_area(area_id)
            found.extend(self._match_in_area(domain, area.name if area else None))
        return list(dict.fromkeys(found))

    def _match_in_area(self, domain: str, area_name: str | None) -> list[str]:
        constraints = intent.MatchTargetsConstraints(
            domains=[domain],
            area_name=area_name,
            assistant="conversation",
        )
        result = intent.async_match_targets(self.hass, constraints)
        if not result.is_match:
            # Retry without the assistant filter: entities that were never
            # explicitly exposed would otherwise be invisible, which reads to
            # the user as "the model is broken" rather than "this is unexposed".
            result = intent.async_match_targets(
                self.hass,
                intent.MatchTargetsConstraints(domains=[domain], area_name=area_name),
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
            if want == object_id or want in object_id.split("_") or want == friendly:
                hits.append(state.entity_id)
            elif want.replace("_", " ") in friendly:
                hits.append(state.entity_id)
        return hits

    # -- argument translation ----------------------------------------------
    def _service_data(self, tool: str, args: dict, entity_ids: list[str],
                      utterance: str = "") -> dict:
        """Model arguments -> Home Assistant service data."""
        data: dict[str, Any] = {
            k: v for k, v in args.items() if k not in NON_SERVICE_ARGS
        }

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
            if "minutes" in data:
                # Must carry into hours: the training data includes "שעתיים",
                # and "00:120:00" is not a duration Home Assistant accepts.
                total = int(data.pop("minutes"))
                data["duration"] = f"{total // 60:02d}:{total % 60:02d}:00"

        elif tool == "notify_send":
            # Take the message out of the sentence, not out of the model.
            # Hebrew reaches a tool argument as \uXXXX escapes - six exact
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

        # light.turn_on takes brightness_step_pct natively, so it passes through.
        return data

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
    async def execute(self, call: dict, device_id: str | None,
                      context: Context, utterance: str = "",
                      index: int = 0, total: int = 1) -> CallOutcome:
        """Run one tool call. ``index``/``total`` place it among its siblings,
        which is how a two-room sentence gets its two rooms in the right order.
        """
        tool = call.get("name", "")
        args = dict(call.get("arguments") or {})

        # Lock or unlock, open or close, on or off. The Hebrew verb settles
        # it and the model does not always agree with the verb - see
        # `direction`, where the signal is measured at 1091 right and 0 wrong
        # against gold. Corrected before anything else, because everything
        # below reads `tool`.
        if utterance and (settled := direction.settle(tool, utterance)) != tool:
            _LOGGER.debug("the sentence says %s, not %s", settled, tool)
            tool = settled

        # The same verb also settles which way a relative argument points, and
        # `_service_data` below adds those to the device's current reading - so
        # a wrong sign moves the thermostat away from what was asked instead of
        # towards it. 1222 right and 1 wrong; see `direction`.
        if utterance:
            args = direction.settle_steps(args, utterance)

        if tool in QUERY_TOOLS:
            return await self._answer_query(tool, args, device_id, utterance,
                                            index, total)

        # "Play" and "play *this*" are one verb apart in Hebrew, and which one
        # was meant is decided by whether a name follows - which the sentence
        # settles and the model has to guess. So when the model says "resume"
        # about a sentence that named something specific, the sentence wins.
        #
        # This is the same rule the rest of this module runs on, and it is
        # safe here for the same reason: it never overrides which *domain* was
        # chosen. The model has already decided the utterance is about audio -
        # "תפעיל את השואב" gets vacuum_start and is never seen here - so all
        # that is being corrected is which of two media tools inside that
        # decision. It also makes the tool work before any model knows it
        # exists, which is what a household running the previous weights has.
        if tool == "media_play" and utterance and slot_match.extract_music(utterance):
            _LOGGER.debug("the sentence names something to play; using music_play")
            tool = "music_play"

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
                    device_id, utterance, index, total)
                if unresolvable:
                    return CallOutcome(tool, False, "no matching entities")
                entity_ids = self._match_entities(target, area_ids, all_areas)

        if not entity_ids:
            return CallOutcome(tool, False, "no matching entities")

        data = self._service_data(tool, args, entity_ids, utterance)
        return await self._call(domain, service,
                                {"entity_id": entity_ids, **data},
                                context, tool, len(entity_ids))

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

    async def _play_music(self, args: dict, device_id: str | None,
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

    async def _call(self, domain: str, service: str, data: dict,
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
    async def _answer_query(self, tool: str, args: dict,
                            device_id: str | None, utterance: str = "",
                            index: int = 0, total: int = 1) -> CallOutcome:
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
            return CallOutcome(tool, True, speech=", ".join(parts))

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
                device_id, utterance, index, total)
            if unresolvable:
                return CallOutcome(tool, False, "no matching entities")
            entity_ids = self._match_entities(domain, area_ids, all_areas)
        if not entity_ids:
            return CallOutcome(tool, False, "no matching entities")

        on_words = {"on", "open", "unlocked", "playing", "cleaning", "active", "home"}
        on, off, values = [], [], []
        for eid in entity_ids:
            st = self.hass.states.get(eid)
            if st is None:
                continue
            name = st.attributes.get("friendly_name", eid)
            if domain in ("sensor",):
                unit = st.attributes.get("unit_of_measurement", "")
                values.append(f"{name}: {st.state} {unit}".strip())
            elif st.state in on_words:
                on.append(name)
            else:
                off.append(name)

        if values:
            speech = ", ".join(values[:4])
        elif on and not off:
            speech = "כן, " + ", ".join(on[:4])
        elif off and not on:
            speech = "לא, " + ", ".join(off[:4])
        else:
            speech = f"{len(on)} פעילים, {len(off)} כבויים"
        return CallOutcome(tool, True, speech=speech, entities=len(entity_ids))
