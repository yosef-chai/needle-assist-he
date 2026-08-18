# -*- coding: utf-8 -*-
"""Resolve the slots the model should never have been asked to fill.

The model's job is to choose a tool and fill the *typed* arguments - a
brightness, a temperature, an hvac mode. Those are small closed sets and it is
good at them. Three slots are not like that, and this module takes them over:

``area``     which room. Twelve English slugs were baked into the fine-tune, so
             a room outside those twelve was unreachable by voice in any house,
             and two rooms that folded onto one slug could not be told apart at
             all. Measured on the held-out corpus the model gets the area right
             **51.4%** of the time; matching the sentence against the areas the
             installation actually has gets **99.0%** on the same rows, and
             **100%** on the synonym probe held out of training entirely. See
             ``eval/slot_match_bench.py`` for both numbers and for the two house
             shapes the model cannot address at all.

``name``     which scene, script, automation, timer or plug. The fine-tune maps
             Hebrew onto about forty invented English slugs (``good_night``,
             ``vacation_mode``). A household that called its scene ``מצב סרט``
             is not in that list and never will be.

``message``  what to say in a notification. Free Hebrew text, and the engine
             emits non-ASCII as ``\\uXXXX`` escape sequences - six exact
             characters per Hebrew letter, which the model gets wrong: measured
             output includes ``\\ub05d4``, a Korean syllable produced by
             dropping one hex digit.

What every one of these has in common is that the answer is *already known* at
runtime. The areas are in the area registry. The scenes are in the state
machine. The message is in the sentence. None of it needs a 45M-parameter model,
and a model is strictly worse at it, because it was trained on one imagined
house and has to run in somebody else's.

This is the same move the project already made twice: tool selection went to
``tool_router`` because Needle's retrieval head cannot read Hebrew, and refusal
went to the same place because the model's refusal rate was 0.0%. The rule that
emerges is: *the model decides what cannot be looked up; everything that can be
looked up is looked up.*

Matching is delegated to :mod:`hebrew_text`, which knows about fused Hebrew
prefixes and about the ways speech-to-text mangles a word.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Final, Iterable

from .area_map import AREA_ALIASES, slug_for_name
from .hebrew_text import PhraseIndex, normalise

_LOGGER = logging.getLogger(__name__)

# Domains whose entities may be addressed by name in an ordinary command
# ("turn on the reading lamp"). Deliberately not every domain: a name match
# overrides the room, so it is only offered where a household actually names
# individual devices.
NAMEABLE_DOMAINS: Final = (
    "light", "switch", "fan", "cover", "climate", "media_player", "lock",
    "vacuum", "camera",
)

# Domains addressed by name and nothing else - there is no "the scene in the
# kitchen", only "the evening scene".
NAME_ONLY_DOMAINS: Final = (
    "scene", "script", "automation", "input_boolean", "timer",
)

# "the whole house". Kept apart from "no room was named", because conflating
# them turns "turn off the light" into "turn off every light in the building".
#
# Run through the same PhraseIndex as everything else rather than as plain
# regexes, so the whole-home markers get the same tolerance the room names get:
# the corpus contains "בכלהחדרים" (space lost) and "בכל הח דרים" (space gained)
# and both mean the whole house.
WHOLE_HOME_FORMS: Final = (
    "בכל הבית", "כל הבית", "בבית כולו", "בכל הדירה", "כל הדירה", "בכל דירה",
    "בכל מקום", "כל מקום", "בכל הקומה", "בכל הבניין",
    "בכל החדרים", "כל החדרים", "בכל חדר", "בכל הפינות",
    # "everywhere in the house" with no word for "all".
    "שיש בבית", "בבית",
)
_WHOLE_HOME = PhraseIndex()
for _form in WHOLE_HOME_FORMS:
    _WHOLE_HOME.add(_form, "all")

# Notification openers, and the marker that starts the message itself. Israelis
# say "tell the house THAT the food is ready" - the message begins at the ש.
_NOTIFY_TRIGGER: Final = re.compile(
    r"(הודעה|תודיע|תודיעי|להודיע|תעדכן|תעדכני|לעדכן|תכריז|להכריז|תגיד|תגידי)")
_NOTIFY_TAIL: Final = re.compile(r"[:\-]\s*(?P<msg>.+)$")
_POLITE_TAIL: Final = re.compile(
    r"\s*(בבקשה|תודה|אם אפשר|רבה|בבקשה תודה)+\s*$")


# Every room word this project knows, regardless of which house is running.
# Used for one question only: did the speaker name a room *at all*? A command
# that names a room this installation does not have must fail rather than widen
# - "turn on the light in the garage" in a house with no garage would otherwise
# fall through to an unconstrained match and light the whole building.
_ANY_ROOM = PhraseIndex()
for _slug, _names in AREA_ALIASES.items():
    _ANY_ROOM.extend(_names, _slug)
    _ANY_ROOM.add(_slug, _slug)


# "all the lights", with no room and no word for "house". The corpus generates
# these as whole-home commands and they read that way to a speaker too, but they
# carry no place word at all, so the fixed markers above cannot see them.
#
# Checked last, only when no room matched and no marker fired, because it is the
# broadest rule here: the difference between getting it wrong and getting it
# right is one room versus the whole building.
_EVERY_DEVICE: Final = re.compile(r"(^|\s)(את\s+)?כל\s+ה?[א-ת]{3,}")


def mentions_every_device(utterance: str) -> bool:
    """Did the speaker say "all the <device>" without naming a place?"""
    return bool(_EVERY_DEVICE.search(normalise(utterance)))


def names_a_room(utterance: str) -> bool:
    """Does this sentence name a room, whether or not this house has it?"""
    return _ANY_ROOM.find(utterance) is not None


def mentions_whole_home(utterance: str) -> bool:
    """Did the speaker mean the entire house rather than one room?

    No fuzzy pass here. "the whole house" is a fixed phrase, and a typo that
    turns one room into every room is the most expensive mistake this file can
    make - "בכניסה לבית" (at the entrance) is one edit away from "בבית" (in the
    house), and it means a specific room.
    """
    return _WHOLE_HOME.find(utterance, fuzzy=False) is not None


def extract_message(utterance: str) -> str | None:
    """The text of a notification, taken out of the sentence that asked for it.

    ``תודיע בבית שהאוכל מוכן`` -> ``האוכל מוכן``. Returns None when there is no
    recognisable message, which the caller should treat as "do not send", not
    as "send an empty notification".

    Unlike everything else here this works on the raw sentence, because the
    result is read out to a person rather than matched against a table.
    :func:`hebrew_text.normalise` folds final letters, which is right for
    comparison and wrong for a message: it would deliver ``האוכל מוכנ``.
    """
    text = " ".join(utterance.split())
    trigger = _NOTIFY_TRIGGER.search(text)
    if not trigger:
        return None
    rest = text[trigger.end():]

    if (explicit := _NOTIFY_TAIL.search(rest)):
        body = explicit.group("msg")
    else:
        # The message is whatever follows the subordinating ש. It is glued to
        # its word, so the token is found first and then the ש is peeled off.
        words = rest.split()
        for position, word in enumerate(words):
            if word.startswith("ש") and len(word) > 2:
                body = " ".join([word[1:]] + words[position + 1:])
                break
        else:
            return None

    body = _POLITE_TAIL.sub("", body).strip()
    return body or None


def _forms_for_area(name: str, aliases: Iterable[str],
                    area_id: str) -> tuple[set[str], set[str]]:
    """Every Hebrew surface form that should reach this area, in two tiers.

    Tier 0 is what the household itself calls the room: the area's name, its
    Home Assistant aliases, and the raw ``area_id`` for installations slugified
    in English. This is authoritative and it is the only part a user can change
    from the UI.

    Tier 1 is this integration's Hebrew synonym table, added only when the
    area's own name is recognisably one of the rooms in it. An area called
    ``סלון`` also answers to ``חדר אורחים`` and ``ליווינג``, because those are
    the same room. An area called ``חדר כביסה`` gets nothing extra, and needs
    nothing, because its own name is what people say - which is the entire
    reason a house with a laundry room works here without retraining anything.
    """
    own = {f for f in {name, area_id, *aliases} if f and f.strip()}
    synonyms: set[str] = set()
    for candidate in (name, *aliases):
        slug = slug_for_name(candidate)
        if slug:
            synonyms |= set(AREA_ALIASES[slug])
    return own, {f for f in synonyms if f and f.strip()} - own


def build_area_index(areas: Iterable[tuple[str, str, Iterable[str]]]) -> PhraseIndex:
    """Phrase index over ``(area_id, name, aliases)`` triples.

    Split out of :class:`SlotIndex` so the same code that runs on a device can
    be measured against the held-out corpus without a running Home Assistant -
    see ``eval/slot_match_bench.py``. A benchmark that exercised a reimplemented
    matcher would measure the reimplementation.

    A synonym that two areas both claim is dropped from both. It is the one
    situation where guessing is worse than nothing: a house with a ``מקלחת``
    and a ``שירותים`` has both rooms claiming ``אמבטיה`` through the table, and
    picking whichever came first in the registry is precisely the wrong-room
    actuation this module exists to end. Dropped, the sentence falls back to
    the room the speaker is standing in, and the household can settle it in one
    click by adding ``אמבטיה`` as a Home Assistant alias to the room it means -
    an alias is tier 0 and outranks everything here.
    """
    own: list[tuple[str, set[str]]] = []
    synonyms: list[tuple[str, set[str]]] = []
    for area_id, name, aliases in areas:
        mine, theirs = _forms_for_area(name, aliases, area_id)
        own.append((area_id, mine))
        synonyms.append((area_id, theirs))

    claimed_by_name = {normalise(f) for _, forms in own for f in forms}
    counts: dict[str, int] = {}
    for _, forms in synonyms:
        for form in forms:
            counts[normalise(form)] = counts.get(normalise(form), 0) + 1

    index = PhraseIndex()
    for area_id, forms in own:
        index.extend(forms, area_id, tier=0)
    for area_id, forms in synonyms:
        for form in forms:
            key = normalise(form)
            if counts[key] > 1 or key in claimed_by_name:
                _LOGGER.debug("dropping ambiguous area synonym %r", form)
                continue
            index.add(form, area_id, tier=1)
    return index


class SlotIndex:
    """Phrase indexes over one installation, rebuilt when its registries change.

    Building costs a few milliseconds per hundred entities, which is nothing
    once but far too much on every utterance, so the result is cached and
    :meth:`invalidate` is wired to the registry-updated events in
    ``__init__.py``. Nothing here is written to disk: the registries are the
    single source of truth and a stale copy would send commands to a room that
    was renamed an hour ago.
    """

    def __init__(self, hass: Any) -> None:
        self.hass = hass
        self._areas: PhraseIndex | None = None
        self._by_domain: dict[str, PhraseIndex] = {}
        self._counts: dict[str, int] = {}

    def invalidate(self, _event: Any = None) -> None:
        self._areas = None
        self._by_domain.clear()
        self._counts.clear()

    # -- areas --------------------------------------------------------------
    def _area_index(self) -> PhraseIndex:
        if self._areas is not None:
            return self._areas
        from homeassistant.helpers import area_registry as ar  # noqa: PLC0415

        index = build_area_index(
            (area.id, area.name, area.aliases or ())
            for area in ar.async_get(self.hass).async_list_areas()
        )
        _LOGGER.debug("area index: %d phrases", len(index))
        self._areas = index
        return index

    def areas(self, utterance: str) -> list[str]:
        """Every area named in the sentence, in the order it was said."""
        return [m.value for m in self._area_index().find_all(utterance)]

    def area_for_call(self, utterance: str, index: int = 0,
                      total: int = 1) -> str | None:
        """The room the ``index``-th of ``total`` tool calls should target.

        A sentence with exactly as many room mentions as calls pairs them off
        in order: "turn the light on in the living room and close the blind in
        the kitchen" is two calls and two rooms, and they are not
        interchangeable. Any other count gives every call the most specific
        room in the sentence, which is right for the ordinary "turn on the
        light and the air conditioning in the bedroom".
        """
        found = self._area_index().find_occurrences(utterance)
        if not found:
            return None
        if len(found) == total and 0 <= index < total:
            return found[index].value
        return max(found, key=lambda match: match.rank).value

    # -- entities -----------------------------------------------------------
    def _entity_index(self, domain: str) -> PhraseIndex:
        # Renames arrive as registry events, which clear the whole cache.
        # Appearances and disappearances do not always: a scene created by
        # `scene.create` never touches the entity registry. Counting is cheap
        # and catches those.
        live = len(self.hass.states.async_entity_ids(domain))
        cached = self._by_domain.get(domain)
        if cached is not None and self._counts.get(domain) == live:
            return cached
        from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

        registry = er.async_get(self.hass)
        blocked = _blocked_forms()
        index = PhraseIndex()
        for state in self.hass.states.async_all(domain):
            # Every one of these can be something other than a string.
            # ``entry.original_name`` in particular comes back as Home
            # Assistant's ``ComputedNameType`` sentinel for entities that
            # derive their name, which is truthy, is not a ``str``, and took
            # down a live installation's first state query with a TypeError
            # out of ``unicodedata.normalize``. Filter by type, not by truth.
            forms: set[str] = set()
            entry = registry.async_get(state.entity_id)
            candidates = [state.attributes.get("friendly_name")]
            if entry is not None:
                candidates.extend(entry.aliases or ())
                candidates.append(entry.original_name)
            forms |= {c for c in candidates if isinstance(c, str) and c.strip()}
            # ``scene.good_night`` is addressable as "good night" even with no
            # friendly name, which is what the model's slugs used to rely on.
            forms.add(state.entity_id.split(".", 1)[1].replace("_", " "))
            for form in forms:
                if _addressable(form, blocked):
                    index.add(form, state.entity_id)
        self._by_domain[domain] = index
        self._counts[domain] = live
        return index

    def entities(self, utterance: str, domain: str) -> list[str]:
        """Entities of ``domain`` named outright in the sentence.

        Empty for the ordinary case - "turn on the light in the kitchen" names
        no device - which is what lets the caller fall back to the room.
        """
        return [m.value for m in self._entity_index(domain).find_all(utterance)]


    # -- introspection ------------------------------------------------------
    def describe(self) -> dict:
        """What this installation looks like to the resolver.

        Surfaced through the diagnostics platform, because the two questions a
        new household actually asks - "why does it not find my room" and "why
        does it say it found no device" - are both answered by looking at the
        registries the way this module looks at them. Guessing at that from the
        outside is how the first live installation lost an hour.
        """
        from homeassistant.helpers import area_registry as ar  # noqa: PLC0415
        from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

        areas = list(ar.async_get(self.hass).async_list_areas())
        index = self._area_index()
        registry = er.async_get(self.hass)

        rooms = []
        for area in areas:
            own, synonyms = _forms_for_area(area.name, area.aliases or (), area.id)
            reachable = sorted(
                {phrase for phrase in own | synonyms
                 if index.find(phrase, fuzzy=False) == area.id})
            rooms.append({
                "area_id": area.id,
                "name": area.name,
                "aliases": list(area.aliases or ()),
                "recognised_as": slug_for_name(area.name),
                "reachable_by": reachable,
                "dropped_as_ambiguous": sorted(
                    (own | synonyms) - set(reachable)),
            })

        orphans = []
        for state in self.hass.states.async_all():
            domain = state.entity_id.split(".", 1)[0]
            if domain not in NAMEABLE_DOMAINS:
                continue
            entry = registry.async_get(state.entity_id)
            if entry is None or entry.area_id or (
                    entry.device_id and _device_area(self.hass, entry.device_id)):
                continue
            orphans.append(state.entity_id)

        return {
            "areas": rooms,
            "area_phrases": len(index),
            # An entity with no area is invisible to every room command. That is
            # Home Assistant's model, not this integration's, but it is the
            # single most common reason a working install looks broken.
            "entities_without_an_area": sorted(orphans),
        }


def _device_area(hass: Any, device_id: str) -> str | None:
    from homeassistant.helpers import device_registry as dr  # noqa: PLC0415

    device = dr.async_get(hass).async_get(device_id)
    return device.area_id if device else None


def _blocked_forms() -> set[str]:
    """Names too generic to be treated as naming one device.

    An entity called ``אור`` would otherwise swallow every "turn on the light",
    reducing a whole-room command to a single bulb. The device nouns come from
    the router's own tables, so the two modules cannot drift apart, and the
    room words come from the area table, so an entity named after its room does
    not shadow the room itself.
    """
    from . import tool_router  # noqa: PLC0415

    blocked = {normalise(n) for nouns in tool_router.FAMILY_NOUNS.values()
               for n in nouns}
    blocked |= {normalise(a) for names in AREA_ALIASES.values() for a in names}
    blocked |= {normalise(s) for s in AREA_ALIASES}
    return blocked


def _addressable(form: Any, blocked: set[str]) -> bool:
    """Is this name a string, and specific enough to target on its own?"""
    if not isinstance(form, str):
        return False
    text = normalise(form)
    return len(text) >= 3 and text not in blocked
