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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

from . import hebrew_numbers
from .area_map import (
    AREA_ALIASES,
    FLOOR_PHRASES,
    floor_slug_for_name,
    slug_for_name,
)
from .hebrew_text import PhraseIndex, normalise
from .tool_router import FAMILY_NOUNS, _fold, _hits, _tokens, _variants

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


# What goes on a list, and what comes off one. -------------------------------
#
# `HassListAddItem`, `HassListCompleteItem`, `HassListRemoveItem` and their
# three `shopping_list` counterparts all carry one slot, and it is free Hebrew
# text: "חלב", "לחם מלא", "לקחת את הכלב לווטרינר". The model cannot supply it -
# Hebrew reaches a tool argument as \uXXXX escapes, six exact characters per
# letter, and it gets them wrong - so the sentence does, exactly as it does for
# a notification's body.
#
# Written against the raw sentence rather than the normalised one for the same
# reason `extract_message` is: the result is read back to a person and stored
# on their list, and `normalise` folds final letters - it would store "לחמ".
_LIST_TRIGGER: Final = re.compile(
    r"(?:^|\s)(?:תוסיף|תוסיפי|הוסף|הוסיפי|להוסיף|תרשום|תרשמי|רשום|רשמי|"
    r"תכתוב|תכתבי|תמחק|תמחקי|מחק|מחקי|למחוק|תסיר|הסר|להסיר|תוציא|תורידי?|"
    r"תסמן|תסמני|סמן|סמני|לסמן|תוסף)\s+")

# The list itself, wherever the sentence names it. Everything from here on is
# the destination rather than the item.
_LIST_TAIL: Final = re.compile(
    r"\s*(?:[למב])?(?:ה)?(?:רשימת|רשימה|רשימ|הרשימה|קניות|הקניות|מטלות|"
    r"המטלות|משימות|המשימות|טודו|מצרכים|סופר)\b.*$")

# Politeness and pronouns that sit between the verb and the item.
_LIST_LEAD: Final = re.compile(r"^(?:לי|לנו|בבקשה|את|עוד|אם אפשר)\s+")

# "תסמן שקניתי חלב" - the item arrives inside a subordinate clause. One word
# after the ש and the item follows.
_LIST_CLAUSE: Final = re.compile(r"^ש\S+\s+")


def extract_item(utterance: str) -> str | None:
    """The list item named in the sentence, or ``None``.

    ``תוסיף חלב לרשימת קניות`` -> ``חלב``. ``None`` means "the sentence did not
    say what", which the caller must treat as a failure rather than as an empty
    item: a shopping list with a blank entry on it is worse than a command that
    said it did not understand.
    """
    text = " ".join(utterance.split())
    trigger = _LIST_TRIGGER.search(text)
    if not trigger:
        return None
    body = text[trigger.end():]
    body = _LIST_CLAUSE.sub("", body)
    while (trimmed := _LIST_LEAD.sub("", body)) != body:
        body = trimmed
    body = _LIST_TAIL.sub("", body)
    body = _POLITE_TAIL.sub("", body).strip(" ,.!?")
    return body or None


# The verbs that open a request for music, in the masculine and feminine
# imperative and future forms Israelis actually use. This is the same list the
# household's own Music Assistant automation triggers on, which is where the
# feature came from - the difference is that the automation hands the sentence
# to a cloud LLM to turn into JSON, and this does it here, offline.
_LISTEN_VERBS: Final = frozenset(
    ["האזין", "האזן", "האזני", "תאזין", "תאזיני"])

_MUSIC_VERB: Final = re.compile(
    # Anywhere in the sentence, not only at the head. Israelis wrap an order in
    # a frame far more often than they bark it: "אני צריך שתנגן לי", "תוכל
    # לשים". Anchoring at the start looked tidy and cost 72 of 97 requests on
    # the held-out music rows, which is the difference between the feature
    # working and the feature quietly resuming whatever was playing.
    r"(?:^|\s)"
    # The clitics that carry those frames. ל is the infinitive - לנגן, לשים,
    # להשמיע - and ש opens the subordinate clause the frame needs.
    r"(?:ש|ל|ו|כש|וש)?"
    r"(?P<verb>נגן|נגני|תנגן|תנגני|השמע|השמיעי|תשמיע|תשמיעי|"
    r"שים|שימי|תשים|תשימי|הפעיל|הפעל|הפעילי|תפעיל|תפעילי|"
    r"האזין|האזן|האזני|תאזין|תאזיני|ערבב|ערבבי|תערבב|תערבבי)"
    r"(?:\s+ל(?:י|נו))?\s+")

# A word that says which *kind* of thing to play, and what it maps to in
# ``music_assistant.play_media``. Order matters only in that each pattern is
# anchored, so the first one that matches consumes its own word.
_MEDIA_KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("track", r"ה?(?:שיר|רצועה|סינגל)"),
    ("album", r"ה?(?:אלבום|תקליט|דיסק)"),
    ("artist", r"ה?(?:אמן|אמנית|זמר|זמרת|להקה|הרכב)"),
    ("playlist", r"ה?(?:פלייליסט|רשימת\s+השמעה|רשימת\s+ההשמעה)"),
    ("radio", r"ה?(?:תחנת\s+ה?רדיו|רדיו|תחנה)"),
)
_MEDIA_KIND_RE: Final = tuple(
    (kind, re.compile(r"^(?:את\s+)?" + pattern + r"\b\s*"))
    for kind, pattern in _MEDIA_KINDS
)

# "play me some music by X" - the noun carries no type of its own, it just
# stands where one would be, and what follows ``של`` is an artist.
_MUSIC_FILLER: Final = re.compile(r"^(?:את\s+)?(?:קצת\s+)?(?:מוזיקה|מוסיקה|משהו)\s*")
_LEADING_ET: Final = re.compile(r"^\s*את\s+")
# Anchored on either a space or the start, because the filler noun in
# "play me some music by X" is consumed before this runs and leaves "של" first.
_OF: Final = re.compile(r"(?:^|\s)של\s+")

# Words that are grammar rather than a name. If nothing but these survives,
# the speaker did not actually say what to play.
# Every device noun the router knows, minus the media family - "מוזיקה" and
# "שיר" are music words, not devices. A residue made only of these is a
# machine in the house, not a record: "תפעיל את השואב" opens with a verb this
# module recognises and leaves "השואב" behind, and searching a music library
# for the vacuum cleaner is not what anybody meant.
# Folded, because `_variants` folds: Hebrew's five final letters are the
# same letters, and "הפן" reaches this set as "פנ" while the table spells
# it "פן". Unfolded, an air conditioner parsed as a record title.
_DEVICE_NOUNS: Final = frozenset(
    _fold(noun) for family, nouns in FAMILY_NOUNS.items() if family != "media"
    for noun in nouns
) | frozenset(_fold(w) for w in (
    # The media family is skipped above because its nouns are mostly the
    # thing being played. These few are not: they are the equipment it is
    # played on and the source it comes from, so a sentence that names one
    # is saying where, not what. "תנגן את הרמקול" is not a record called
    # "the speaker".
    #
    # Listed one by one rather than taken as "the media family minus the
    # content words", because that wider rule swept up שקט and לילה טוב -
    # a mute setting and a bedtime script, and also two playlists this
    # corpus actually contains, which cost eleven of the eighty-five titles.
    "רמקול", "רמקולים", "נגן", "עוצמה", "ווליום", "וליום",
    "סאונד", "שאונד", "קול", "מקור", "ערוץ",
    "ספוטיפיי", "יוטיוב", "אייראפליי", "בלוטות'",
    # Half of a device name the router spells with one word: its table
    # has "שואב", and a speaker who says "שואב האבק" leaves "האבק"
    # behind, which was neither a device word nor a grammar one.
    "אבק",
))

# Words that stand where a name would and are not one. Matched through a
# PhraseIndex rather than a set so that speech-to-text damage still lands -
# the corpus contains "מוזיכה" for "מוזיקה", and a request for something
# to listen to must not become a search for a misspelling.
_VAGUE = PhraseIndex()
for _word in ("משהו", "טוב", "טובה", "נחמד", "נעים", "כיף", "מוזיקה",
              "מוסיקה", "קצת", "עוד", "שיר", "שירים"):
    _VAGUE.add(_word, "vague")

_NOT_A_NAME: Final = frozenset((
    "הזה", "הזאת", "הזו", "זה", "זאת", "אותו", "אותה", "משהו", "מוזיקה",
    "מוסיקה", "שיר", "שירים", "את", "ה", "קצת", "עוד",
    # Transport control wearing a title's clothes: "play the next song" leaves
    # "הבא" behind once the kind word is consumed, and searching a library for
    # "the next" finds nothing.
    "הבא", "הקודם", "הבאה", "הקודמת", "אחרון", "אחרונה",
))


@dataclass(frozen=True)
class MusicRequest:
    """What to play, taken out of the sentence rather than out of the model.

    The fields are ``music_assistant.play_media``'s: ``media_id`` is what to
    search for and ``artist`` narrows it. Its ``album`` field is deliberately
    left alone - for "the album Shablul by Kaveret" the album name *is* the
    thing being searched for, and sending it as both narrows the search
    against itself.
    """

    media_id: str
    media_type: str | None = None
    artist: str | None = None


#: The enum values of ``vacuum_set_fan_speed`` and ``climate_set_fan_mode``, as
#: they are actually said. Both slots are a word the speaker chose out of a
#: fixed list, which makes them the same kind of argument as the room and the
#: music title: the sentence has it and the model guesses at it.
#:
#: Measured against gold over both splits, on every call carrying the slot:
#:
#:     fan_speed   180 agree, 0 disagree, 6 silent
#:     fan_mode    294 agree, 0 disagree, 4 silent
#:
#: ``חזק`` is deliberately in both tables and means different things in them -
#: the vacuum's strongest suction is ``turbo`` and the air conditioner's is
#: ``high`` - which is why they are separate tables keyed by slot rather than
#: one shared list.
#: ``color_name`` is here on a shorter leash than the other two, and the reason
#: is a morphological accident worth recording: ``להוריד`` - *to lower* - is
#: ``ל`` + ``ה`` + ``ורוד``, and ``ורוד`` is *pink*. The prefix chain that lets
#: "בסלון" find the living room turns "להוריד את התאורה" into a request for a
#: pink light, on 21 calls of the corpus that name no colour at all. So this
#: slot is only ever *corrected*, never *added*: see `executor._service_data`.
#: On the calls that do carry a colour the reading is 335 right and 0 wrong.
SETTING_WORDS: Final[dict[str, dict[str, str]]] = {
    "color_name": {
        "אדום": "red", "אדומה": "red",
        "כחול": "blue", "כחולה": "blue",
        "ירוק": "green", "ירוקה": "green",
        "צהוב": "yellow", "צהובה": "yellow",
        "כתום": "orange", "כתומה": "orange",
        "סגול": "purple", "סגולה": "purple",
        "ורוד": "pink", "ורודה": "pink",
        "לבן": "white", "לבנה": "white",
        "חמים": "warm_white", "חמימה": "warm_white",
        "קר": "cool_white", "קריר": "cool_white", "קרה": "cool_white",
    },
    "fan_speed": {
        "שקט": "silent", "שקטה": "silent",
        "רגיל": "standard", "רגילה": "standard",
        "בינוני": "medium", "בינונית": "medium",
        "חזק": "turbo", "חזקה": "turbo", "טורבו": "turbo", "מקסימום": "turbo",
    },
    "fan_mode": {
        "נמוך": "low", "נמוכה": "low", "חלש": "low",
        "בינוני": "medium", "בינונית": "medium",
        "גבוה": "high", "גבוהה": "high", "חזק": "high",
        "אוטומטי": "auto", "אוטו": "auto",
    },
    # Which *kind* of cover. "תפתח את התריסים בסלון" and "תפתח את הווילונות
    # בסלון" are two different commands in a room that has both, and without
    # this they are the same one: `executor` matches every cover in the area
    # and opens the lot.
    #
    # Copied verbatim from `lists/he/covers.yaml` in OHF-Voice/intents - these
    # are the words Home Assistant's own Hebrew leaders chose for the ten
    # `cover` device classes, and `test_the_official_cover_classes_are_carried`
    # fails if the vendored copy and this table ever disagree.
    #
    # Note what is *not* here: מוסך. The official list spells garage as
    # "דלת חניה", the door rather than the room, and that is the right call for
    # this project too - מוסך and חניה are *areas* in `area_map`, so a garage
    # device class carrying them would fight the room resolver for the same
    # word. Only consulted for cover and valve calls, which is what keeps דלת
    # and שער from colliding with the lock family that also claims them.
    "device_class": {
        "סוכך": "awning", "סככה": "awning",
        "סוככים": "awning", "סככות": "awning",
        "תריס": "blind", "תריסים": "blind",
        # Both spellings of the curtain. Hebrew writes a consonantal vav
        # doubled mid-word and speakers do it inconsistently; `FAMILY_NOUNS`
        # in `tool_router` already carries all four forms for the same reason.
        "וילון": "curtain", "וילונות": "curtain",
        "ווילון": "curtain", "ווילונות": "curtain",
        "דלת": "door", "דלתות": "door",
        # The official list writes the plural as `דלתות [ה]חניה` and the
        # singular as `דלת חניה` with no optional article. "תפתח את דלת החניה"
        # is the commoner way to say it, so the singular gets the same
        # definite form the plural already has.
        "דלת חניה": "garage", "דלת החניה": "garage",
        "דלתות חניה": "garage", "דלתות החניה": "garage",
        "שער": "gate", "שערים": "gate",
        "צילייה": "shade", "ציליה": "shade",
        # The same accident as the garage above, and the same repair.
        # Upstream writes the plural with the article and the singular
        # without, so "תפתח את תריס ההצללה" missed every two-word key and
        # fell through to תריס on its own - narrowing the room to a blind.
        # A wrong kind is the one outcome this table must not produce: it
        # finds the wrong entities rather than none.
        "תריס הצללה": "shutter", "תריס ההצללה": "shutter",
        "תריסי הצללה": "shutter", "תריסי ההצללה": "shutter",
        "חלון": "window", "חלונות": "window",
    },
    # The air-conditioner mode. Here for the same reason colour and suction
    # are: `hvac_mode` is one of the four Home Assistant enums whose values
    # collide at their first byte - `heat` against `heat_cool` - and those
    # values cannot be renamed, because HA rejects a call carrying anything
    # else. v4 measured what a colliding prefix does to a byte-level decode:
    # one value wins and the rest collapse. The sentence has no such problem.
    "hvac_mode": {
        "קירור": "cool", "קר": "cool", "מקרר": "cool",
        "חימום": "heat", "חם": "heat", "מחמם": "heat",
        # No "אוטו". It is the clipped form of אוטומטי and it is also the
        # Hebrew for *car*, so "הדליקי את המיזוג ליד האוטו" - beside the car,
        # a perfectly ordinary place for a thermostat - read as a request for
        # automatic mode. Measured: one wrong on the corpus, which is one more
        # than this table is allowed.
        "אוטומטי": "auto", "אוטומט": "auto", "אוטומטית": "auto",
        "יבש": "dry", "ייבוש": "dry", "לחות": "dry",
        "מאוורר": "fan_only", "אוורור": "fan_only", "פן": "fan_only",
        "כבוי": "off", "כיבוי": "off",
    },
    # What Music Assistant is being asked to look up. The fourth colliding
    # enum: `album` against `artist`.
    "media_type": {
        "שיר": "track", "רצועה": "track", "סינגל": "track",
        "אלבום": "album", "תקליט": "album", "דיסק": "album",
        "זמר": "artist", "זמרת": "artist", "אמן": "artist",
        "אמנית": "artist", "להקה": "artist", "הרכב": "artist",
        "פלייליסט": "playlist", "רשימת השמעה": "playlist",
        "רדיו": "radio", "תחנה": "radio", "תחנת רדיו": "radio",
    },
    # Which input a speaker or television is switched to.
    "source": {
        "ספוטיפיי": "spotify", "ספוטיפי": "spotify",
        "יוטיוב": "youtube", "יוטוב": "youtube",
        "רדיו": "radio",
        "טלוויזיה": "tv", "טיוי": "tv",
        "בלוטות": "bluetooth", "בלוטוס": "bluetooth",
        "אייראפליי": "airplay", "אירפליי": "airplay",
    },
    # Which list. From Home Assistant's two: the modern `todo` domain and the
    # legacy `shopping_list` integration.
    #
    # A bare "רשימה" is the shopping list. Israelis say "תוסיף לרשימה חלב"
    # far more often than they name which list, and the gold agrees: over
    # train and test, adding the bare forms takes this slot from 538 right to
    # **623 right, still 0 wrong**, and 174 rows stay silent. The construct
    # form רשימת is deliberately *not* here - it is the first half of "רשימת
    # מטלות" as readily as of "רשימת קניות", and adding it turned one
    # speech-noised "רשימת המשימוט" into a shopping list. Multi-word keys are
    # read first and win outright, so "רשימת מטלות" is a todo before any of
    # this is reached; see `setting_from`.
    "list": {
        "קניות": "shopping", "הקניות": "shopping", "לקניות": "shopping",
        "מצרכים": "shopping", "סופר": "shopping",
        "רשימת קניות": "shopping", "רשימת הקניות": "shopping",
        "רשימה": "shopping", "מהרשימה": "shopping",
        "מטלות": "todo", "המטלות": "todo", "משימות": "todo",
        "המשימות": "todo", "טודו": "todo",
        "רשימת מטלות": "todo", "רשימת משימות": "todo",
    },
    # The four colour temperatures `lists/he/lights.yaml` names, with the
    # Kelvin each maps to. Values are strings here because the table is one
    # type throughout; `setting_from` returns them as written and
    # `_service_data` hands `color_temp_kelvin` an int - see `_KELVIN`.
    "color_temp_k": {
        "אור נרות": "1900", "נר": "1900", "נרות": "1900",
        "לבן חם": "2700",
        "לבן קר": "4000",
        "אור יום": "6500", "אור טבעי": "6500",
    },
    # The state a question filters on: "אילו אורות דולקים" asks for the lights
    # that are on, not for all of them. From `lists/he/states.yaml` and
    # `lists/he/covers.yaml`, same provenance and same guarding test.
    "state": {
        "דולק": "on", "דולקים": "on", "דולקות": "on",
        "פועלים": "on", "פועלות": "on",
        "מופעלים": "on", "מופעלות": "on",
        "כבוי": "off", "כבוים": "off", "כבויים": "off",
        "כבויות": "off", "מכובים": "off", "מכובות": "off",
        "פתוח": "open", "פתוחה": "open",
        "פתוחים": "open", "פתוחות": "open",
        "סגור": "closed", "סגורה": "closed",
        "סגורים": "closed", "סגורות": "closed",
    },
}


#: The prefixes a *value* takes, which are not the prefixes a verb takes. This
#: is the whole reason these slots do not go through :class:`PhraseIndex` like
#: every other phrase in this module: the general chain strips ``ל`` and then
#: ``ה``, which turns ``להוריד`` - *to lower* - into ``ורוד``, *pink*, and asks
#: for a pink light on 21 corpus calls that name no colour at all. A colour or
#: a speed is a noun in a prepositional phrase - "בסגול", "לשקט", "וגבוה" - and
#: never carries a verb's prefixes. Narrowed to these, the three slots read
#: 335 / 180 / 293 right, **0 wrong**, and invent one on **0** of the calls
#: that name none.
#: ``וה`` completes the pair: "התריסים והווילונות" carries a conjunction over
#: an article, and without it only the first of the two was found - which
#: reads as one unambiguous value rather than as the two that settle nothing.
_VALUE_PREFIXES: Final = ("", "ב", "ל", "ו", "וב", "ול", "כ", "ה", "וה")

_SETTING_FOLDED: Final[dict[str, dict[str, str]]] = {
    slot: {normalise(word): value for word, value in table.items()}
    for slot, table in SETTING_WORDS.items()
}


#: Slots where "את ה<value>" names the device being changed rather than the
#: value it is being changed to. See the note inside :func:`setting_from`.
_OBJECT_IS_THE_DEVICE: Final = frozenset(
    ("hvac_mode", "fan_mode", "fan_speed", "source"))


def setting_from(utterance: str, slot: str) -> str | None:
    """The value the sentence names for ``slot``, or ``None``.

    ``None`` when the sentence names nothing, and also when it names two
    different values - "בין נמוך לגבוה" settles nothing and the model's answer
    is left alone.
    """
    table = _SETTING_FOLDED.get(slot)
    if not table or not utterance:
        return None
    found: set[str] = set()

    # Multi-word keys first, and they win outright. "דלת חניה" is a garage and
    # "דלת" on its own is an ordinary door, so scoring both would find two
    # values and settle nothing - which is how the more specific reading gets
    # lost. Same rule the area resolver uses for overlapping room names: the
    # most specific match wins rather than competing.
    #
    # Substring rather than token matching, because a multi-word key spans a
    # space that speech-to-text is free to move. The single-word tables have no
    # keys with spaces, so this branch cannot fire for colour or fan speed and
    # their measured "0 wrong" is untouched.
    phrase = normalise(utterance)
    for key, phrase_value in table.items():
        if " " in key and key in phrase:
            found.add(phrase_value)
    if len(found) == 1:
        return found.pop()
    if found:
        return None

    words = utterance.split()
    for position, raw in enumerate(words):
        word = normalise(raw)   # normalise already drops punctuation
        previous = normalise(words[position - 1]) if position else ""
        for prefix in _VALUE_PREFIXES:
            folded = normalise(prefix)
            if folded and not word.startswith(folded):
                continue
            value = table.get(word[len(folded):])
            if value is None:
                continue
            # The object of the accusative is what is being *changed*, not what
            # it is being changed to. "את הקירור במטבחון רק מאוורר" switches
            # the cooling to fan-only, and reading the first value gets it
            # exactly backwards - which is what it did: four of the corpus's
            # hvac rows, three of them with the target word corrupted by
            # injected speech noise so that only the object was legible.
            #
            # Only the bare and definite forms are skipped. A value carrying a
            # real preposition is a target wherever it stands - "תעביר את
            # המזגן לקירור" - and dropping those would cost every correct
            # reading this table has.
            #
            # And only for the slots where the accusative object is the
            # *device*. For `media_type` it is the content - "תנגן את האלבום
            # שבלול" plays the album - so the same rule applied there threw
            # away 487 correct readings. That is the whole reason it is a set
            # rather than a rule: the grammar is identical and the semantics
            # are opposite.
            #
            # Measured over all 31,519 rows, on every slot this function
            # serves: hvac_mode goes from 413 right and 4 wrong to **451 right
            # and 0 wrong**, and colour, colour temperature, fan mode, fan
            # speed, media type, source and list are all unchanged at 0 wrong.
            if (slot in _OBJECT_IS_THE_DEVICE and previous == "את"
                    and folded in ("", normalise("ה"))):
                break
            found.add(value)
    return found.pop() if len(found) == 1 else None


#: The words that turn a two-state slot *off*, per slot. Read only once the
#: behaviour is settled - "תוריד" is a volume almost everywhere and an unmute
#: only on a `media_mute` call - which is what keeps these lists short enough
#: to be right.
#:
#: Derived from gold rather than guessed: over all 31,519 corpus rows, the
#: words below appear on every `false` row of their slot and on no `true` one.
#: Measured as a rule, per clause: `oscillating` 209 agree and **0 disagree**,
#: `is_volume_muted` 221 agree and **0 disagree**.
_SWITCHED_OFF: Final[dict[str, tuple[str, ...]]] = {
    # "בלי סיבוב" is still a request to oscillate, so the negation lands here
    # rather than on the behaviour. See `direction.settle_named`.
    "oscillating": ("בלי", "ללא", "בלא", "בטל", "תבטל", "שתבטל", "לבטל",
                    "שיפסיק", "תפסיק", "הפסק", "להפסיק"),
    # An unmute is said as *undoing* one - cancel the muting, take the mute
    # off, bring the sound back - and never with a negative particle, which is
    # why the generic list above would miss two thirds of them.
    "is_volume_muted": ("בלי", "ללא", "בלא", "בטל", "תבטל", "שתבטל", "לבטל",
                        "השתקה", "הקול", "תוריד", "להוריד", "שתוריד",
                        "תחזיר", "שתחזיר", "להחזיר", "החזר"),
}


def switch_from(utterance: str, slot: str) -> bool:
    """Which way a two-state slot points, for a behaviour already settled.

    ``True`` unless the clause says otherwise, because that is what the corpus
    says: a request to oscillate that does not say "בלי" wants oscillation.
    The caller has already decided the behaviour, so this only answers which
    way - see :data:`_SWITCHED_OFF`.
    """
    words = _SWITCHED_OFF.get(slot)
    if not words or not utterance:
        return True
    return not _hits(list(words), _tokens(utterance), utterance)


#: The fan, as a *part* of the air conditioner rather than one of its modes.
#: "מהירות המאוורר", "הפן של המזגן" - a clause built around one of these is
#: asking about the fan, and its ``אוטומטי`` is the fan's automatic and not the
#: machine's. ``מהירות`` is matched as a substring because Hebrew glues the
#: preposition on ("למהירות") and the device nouns as whole words, because two
#: of them are short enough to appear inside unrelated ones.
_FAN_SPEED_NOUN: Final = normalise("מהירות")
_FAN_DEVICE_NOUNS: Final = frozenset(
    normalise(w) for w in ("פן", "הפן", "מאוורר", "המאוורר",
                           "וונטה", "הוונטה", "ונטה", "הונטה"))
#: ...except right after רק, where the bare noun is the *mode*: "לרק מאוורר"
#: is fan-only. The same exemption :func:`hvac_target` makes, and the
#: preposition is glued on here too.
_ONLY_FORMS: Final = tuple(normalise(w) for w in ("רק", "ורק", "לרק", "ברק"))


def _names_the_fan(utterance: str) -> bool:
    """True when the clause is built around the fan as a device."""
    folded = normalise(utterance)
    if _FAN_SPEED_NOUN in folded:
        return True
    words = folded.split()
    return any(
        word in _FAN_DEVICE_NOUNS
        and not (position and any(
            words[position - 1].endswith(only)
            and len(words[position - 1]) - len(only) <= 1
            for only in _ONLY_FORMS))
        for position, word in enumerate(words))


def mode_slot(utterance: str) -> str | None:
    """Which of the two climate mode slots the clause names, or ``None``.

    `direction.settle_climate` needs the slot and not just a yes: a
    `climate_set_temperature` with no temperature in it is a mode call, and
    which mode decides which behaviour it becomes. Three callers computed the
    yes as ``any(setting_from(text, s) for s in ...)`` and this replaces all
    three, so the reading lives in one place.

    The two tables overlap - ``אוטומטי`` is a machine mode and a fan speed
    both - so the device the clause is built around decides, and only then the
    value. Measured as the promotion rule it feeds, per clause over all 31,519
    corpus rows, against gold, with a temperature or a switching verb in the
    clause disqualifying it: **982 agree, 0 disagree**.
    """
    if not utterance:
        return None
    if _names_the_fan(utterance):
        return "fan_mode" if setting_from(utterance, "fan_mode") else None
    if setting_from(utterance, "hvac_mode"):
        return "hvac_mode"
    return "fan_mode" if setting_from(utterance, "fan_mode") else None


#: A number after the title is a level, not part of the name: "שים את השיר על
#: שישים" sets the volume. Used to keep :func:`names_a_level` from calling that
#: sentence a play - see `executor.execute`.
_LEVEL: Final = re.compile(
    r"(\d|אחוז|עשרים|שלושים|ארבעים|חמישים|שישים|שבעים|שמונים|תשעים|מאה)")


#: Questions that ask *which* things are in a state, rather than *whether* one
#: is. See :func:`state_filter`.
_WHICH_ARE: Final = re.compile(
    r"\b(אילו|איזה|איזו|כמה"
    r"|מה\s+פתוח|מה\s+סגור|מה\s+דולק|מה\s+כבוי|מה\s+פועל)\b"
    r"|\bהאם\s+כל\b")


def state_filter(utterance: str) -> str | None:
    """The state a question wants its answer *filtered* to, or ``None``.

    "אילו אורות דולקים" asks for the lights that are on, not for all of them,
    and answering "3 פעילים, 2 כבויים" is true and is not what was asked. But
    "תבדוק אם האור בגן דולק" names the same word and wants yes or no, and
    filtering that one to the lit lights answers a question nobody asked.

    Hebrew marks the difference and it is not the interrogative particle: both
    "אילו" and "האם כל" take the filter, while "האם" alone and "תבדוק אם" do
    not. Measured over all 28,233 corpus rows, on every row whose gold call can
    carry the slot:

        the sentence agrees with gold      420
        the sentence disagrees               0
        the sentence stays silent          132

    The 132 are rows the pattern does not reach, and they answer exactly as
    they did before this existed. Reading the state word without the
    interrogative test was measured too: 544 right and **127 wrong**, every one
    of them a yes/no question turned into a list.
    """
    if not utterance or not _WHICH_ARE.search(utterance):
        return None
    return setting_from(utterance, "state")


def names_a_level(utterance: str) -> bool:
    """True when the sentence names a number, which a title does not."""
    return bool(_LEVEL.search(utterance))


#: Units that belong to some other argument. A number wearing one of these is
#: a percentage or a duration and never a thermostat setting.
_NOT_DEGREES: Final = (
    r"(?![0-9])(?!\s*(?:אחוז|אחוזים|%|דקות|דקה|שניות|שניה|שעות|שעה))")

#: A target temperature: bound by `על` or `ל`, or wearing מעלות outright.
#: The conjunction is part of the preposition in Hebrew - "ועל 24" - and
#: it is matched so that a clause naming two of them is read as naming
#: none rather than as naming the first.
_TARGET_TEMP: Final = re.compile(
    r"(?:(?<=\s)|^)ו?(?:על|ל)[- ]?(\d{1,2})" + _NOT_DEGREES
    + r"|(?:(?<=\s)|^)(\d{1,2})\s*מעלות")

#: `ב` is the preposition Hebrew uses for a *step* - "תוריד ב2 מעלות" - which
#: is a different argument reaching a different service. It is not a target and
#: it must not be read as one.
_STEP_TEMP: Final = re.compile(r"(?:(?<=\s)|^)ו?ב[- ]?\d{1,2}\s*מעלות")


def temperature_from(utterance: str) -> int | None:
    """The target temperature the sentence names, or ``None``.

    Three of the four sentences Home Assistant's own Hebrew suite still gets
    wrong are one shape - "שנה את הטמפרטורה ל20 מעלות" - and what goes wrong is
    not the routing. The model answers `climate_control{off}` **with no
    argument at all**, so `direction.settle_climate`, which turns a climate
    call carrying a temperature into a call to set one, has nothing to fire on
    and an air conditioner switches off when somebody asked for twenty degrees.

    This is the sentence supplying what the model dropped, and it is read only
    for a climate behaviour, so a bare number bound by a preposition is a
    thermostat setting and nothing else. Measured per clause over all 30,613
    corpus rows, on every clause whose gold call can carry the slot:

        the sentence agrees with gold      442
        the sentence disagrees               0
        the sentence stays silent          970

    It fires on **none** of the 463 rows whose gold is a `temperature_step`,
    and on **none** of the 2,513 climate rows that want no number at all. In
    the whole corpus it matches exactly one clause outside a climate row.

    The 970 silent ones were mostly the half of the corpus that writes the
    number in words - "על עשרים ואחת" - which needed a Hebrew number parser
    with the gendered forms. :mod:`hebrew_numbers` is that parser, and it is
    the second half of this function now: over the training corpus the two
    together read **485 right against 2 wrong**, and both of the two are
    "אשרים" - speech noise for עשרים, in a sentence the model cannot read
    either.
    """
    if not utterance or _STEP_TEMP.search(utterance):
        return None
    found = {int(group) for match in _TARGET_TEMP.findall(utterance)
             for group in match if group}
    if len(found) != 1:
        # Not a digit anywhere the prepositions bind. Try the words.
        return hebrew_numbers.degrees_in(utterance)
    value = found.pop()
    # Home Assistant's own `climate` selector is wider than this, but a
    # thermostat asked for 3 or for 90 is a misread number rather than an
    # instruction, and the corpus has no row outside it.
    return value if 5 <= value <= 35 else None


#: Words that describe a room rather than name a mode. "בחדר **חם** מדי" is
#: too hot in here, not a request for the heater.
_MODE_IS_A_STATE: Final = frozenset(
    normalise(w) for w in ("חם", "קר", "יבש", "לחות"))

#: `מאוורר` is an hvac mode and also the noun in "מהירות **המאוורר**", which
#: is a fan mode. The following word decides and this is the one that must not.
_SPEED_NOUN: Final = normalise("מהירות")

#: A bare mode noun counts when this stands in front of it: "רק מאוורר".
_ONLY: Final = frozenset(normalise(w) for w in ("רק", "ורק"))

#: Prepositions that make the mode a target rather than a description.
_MODE_PREFIXES: Final = ("ל", "ב", "על", "ול", "לב")


#: Slots the sentence owns **in both directions**: it fills them, and its
#: silence empties them.
#:
#: Every rule in this module until now could only add. The model was free to
#: invent, and it does: 459 arguments in one release run that the utterance
#: never evidenced - `climate_control{run, hvac_mode: heat}` for "תדליק את
#: המזגן", `light_control{on, brightness_pct: 20}` for "תדליק את האור",
#: `media_control{pause, source: radio}` for a sentence about nothing of the
#: kind. Needle's own contract forbids these and the base model violates it
#: too, which is why `over_fill` has been a reported metric since v1. Nothing
#: acted on it.
#:
#: A slot is here only if the sentence was measured to be *complete* about it,
#: not merely right - counting, over train and test, the rows whose gold
#: carries the slot and whose sentence says nothing this module can read:
#:
#:     temperature 2 | brightness_pct 5 | volume_pct 2 | percentage 2
#:     color_name 4 | fan_speed 1 | fan_mode 5 | source 4 | device_class 9
#:
#: Thirty-four rows out of roughly four thousand that carry one of these, and
#: every single one is a value speech noise corrupted - אשרים for עשרים, כלש
#: for חלש, "אוטומטיבבקשה" glued, "בערך60" glued. The sentence is not merely
#: usually right about these; it is right whenever it is legible.
#:
#: **Four slots are deliberately absent**, and each says something:
#:
#: * `position` would break 14 - "תסגור את התריס **לגמרי**", all the way,
#:   which is a position without being a number and whose direction is the
#:   verb's business. See `hebrew_numbers._PCT_IDIOM`.
#: * `media_type` would break 51: the title carries the kind - "רשימת
#:   ההשמעה מוזיקה לעבודה" - and `extract_music` reads that, not this table.
#: * `hvac_mode` would break **240**, because the corpus gives a mode to a
#:   bare "תדליק את המזגן" and no sentence names one. That is a default, not a
#:   reading, and it belongs to the model.
#: * `temperature_step` and its siblings are idioms - "קצת", "שמץ" - with no
#:   number in them at all, so silence there means nothing.
SENTENCE_OWNS: Final[frozenset[str]] = frozenset((
    "temperature", "brightness_pct", "volume_pct", "percentage",
    "color_name", "fan_speed", "fan_mode", "source", "device_class"))


def unsupported(utterance: str, slot: str) -> bool:
    """True when ``slot`` is one the sentence owns and the sentence is silent.

    The caller drops the argument. Anything outside :data:`SENTENCE_OWNS` is
    never unsupported, however quiet the sentence - silence has to have been
    measured to mean something before it may be acted on.
    """
    if slot not in SENTENCE_OWNS or not utterance:
        return False
    if slot in SETTING_WORDS:
        return setting_from(utterance, slot) is None
    if slot == "temperature":
        return (temperature_from(utterance) is None
                and not hebrew_numbers.numbers_in(utterance))
    return (percent_from(utterance) is None
            and not hebrew_numbers.numbers_in(utterance))


#: The compound cover nouns, which are safe where their bare halves are not.
_COMPOUND_COVER: Final = tuple(
    normalise(key) for key in SETTING_WORDS["device_class"] if " " in key)


def names_a_compound_cover(utterance: str) -> bool:
    """True when the sentence names a cover by a phrase only a cover has.

    "דלת החניה" is the garage door and "דלת" alone is what a lock has; "תריס
    ההצללה" is an awning. All 147 corpus rows naming one of these eight
    phrases are cover rows. See `direction.NAMES_A_COVER`.
    """
    if not utterance:
        return False
    phrase = normalise(utterance)
    return any(key in phrase for key in _COMPOUND_COVER)


def hvac_target(utterance: str) -> str | None:
    """The hvac mode the sentence asks the machine to be *put into*, or None.

    Distinct from ``setting_from(utterance, "hvac_mode")``, which reads the
    slot for a call that is already known to set a mode. This answers the
    harder question - *is this sentence a request to change the mode at all* -
    and it has to be stricter, because the mode words double as adjectives and
    as device nouns: "בחדר חם מדי" wants a colder number and "מהירות המאוורר"
    wants a fan speed, and both read as a mode to the loose test.

    So: bound by a preposition, or bare only after רק; never an adjective;
    never the noun in "מהירות המאוורר"; and exactly one, since two settle
    nothing.

    It exists because of a hole in the pipeline rather than in a lexicon. The
    setting slots are read for the behaviour the model chose - `SETTING_SLOT`
    is keyed on the virtual id - so when the model answers "תעביר את המזגן
    לאוורור" with `climate_control{temp}`, nothing ever reads the mode: the
    behaviour that carries the slot was never selected. Thirty of the four
    hundred failures in one dump were exactly that, all of them fan_only.

    Measured over the training corpus on single-clause single-call climate
    rows, with a fan mode or a target temperature in the sentence
    disqualifying it: **119 agree, 0 disagree**.
    """
    if not utterance:
        return None
    table = _SETTING_FOLDED.get("hvac_mode") or {}
    words = normalise(utterance).split()
    found: set[str] = set()
    for position, word in enumerate(words):
        previous = words[position - 1] if position else ""
        if previous.startswith(_SPEED_NOUN):
            continue
        for prefix in (*_MODE_PREFIXES, ""):
            if prefix and (not word.startswith(prefix)
                           or len(word) <= len(prefix)):
                continue
            stem = word[len(prefix):]
            value = table.get(stem)
            if value is None:
                continue
            if stem in _MODE_IS_A_STATE:
                break
            if not prefix and previous not in _ONLY:
                break
            found.add(value)
            break
    return found.pop() if len(found) == 1 else None


def percent_from(utterance: str) -> int | None:
    """The percentage the sentence names, or ``None``.

    One function for four slots - `brightness_pct`, `volume_pct`, `position`
    and `percentage` - because Hebrew says all four the same way and only the
    behaviour decides which of them it is. `const.NUMBER_SLOT` is that
    decision; this is the reading.

    Digits and words alike, and the fraction idioms an Israeli reaches for
    instead of either: "חצי", "רבע", "מקסימום". Measured over the training
    corpus on single-clause single-call rows whose gold carries one of the
    four: **1,175 right, 3 wrong**, all three a value speech noise corrupted.

    It fires on **none** of the 779 rows whose gold is a relative step - the
    corpus says those as "קצת" and "הרבה יותר" and never as a percentage - so
    unlike :func:`temperature_from` this one needs no step guard.
    """
    return hebrew_numbers.percent_in(utterance)


def duration_from(utterance: str) -> dict[str, int]:
    """The countdown the sentence names, as ``{hours, minutes, seconds}``.

    Empty when it names none. Measured over the training corpus on timer
    rows: **553 exactly right, 66 the same countdown spelled with different
    units, 7 wrong, 5 silent** - and it speaks on none of the timer rows whose
    gold carries no duration at all, which is what keeps a pause or a cancel
    from acquiring one.

    The 66 are the corpus disagreeing with itself: "טיימר של שעתיים" is
    `{hours: 2}` in one row and `{minutes: 120}` in the next. Both are the
    same two hours to `executor._seconds`, and `data/generate.py` now spells
    them one way. See :func:`hebrew_numbers.duration_in`.
    """
    return hebrew_numbers.duration_in(utterance)


def extract_music(utterance: str) -> MusicRequest | None:
    """Parse "play me X by Y" into a Music Assistant search.

    ``תנגן לי את אם ננעלו של עומר אדם`` ->
    ``MusicRequest("אם ננעלו", "track", artist="עומר אדם")``.

    Returns None when the sentence asked for music but never said which - the
    caller should resume playback rather than search for the word "music".

    Deterministic on purpose. The model is never asked for the name: Hebrew
    reaches a tool argument as ``\\uXXXX`` escapes, six exact characters per
    letter, and a 45M-parameter model gets them wrong - the same reason
    :func:`extract_message` exists. It also cannot be asked to *know* anything:
    the household automation this replaces sends the sentence to a cloud LLM to
    have song titles inferred, and nothing here does that. Music Assistant's
    own search is what turns these words into a track, which is the component
    that has the library.

    Works on the raw sentence, like :func:`extract_message` and for the same
    reason: the result is a search string, and folding final letters would ask
    the library for ``עומר אדמ``.
    """
    text = " ".join(utterance.split())
    # The last verb, not the first: "תוכל לשים לי" has the frame's verb in
    # front of the real one, and what follows the real one is the title.
    openers = list(_MUSIC_VERB.finditer(text))
    if not openers:
        return None
    rest = _POLITE_TAIL.sub("", text[openers[-1].end():]).strip()
    if openers[-1].group("verb") in _LISTEN_VERBS and rest.startswith("ל"):
        rest = rest[1:]

    media_type: str | None = None
    for kind, pattern in _MEDIA_KIND_RE:
        if (hit := pattern.match(rest)):
            media_type, rest = kind, rest[hit.end():]
            break
    else:
        if (hit := _MUSIC_FILLER.match(rest)):
            # "some music by X" names an artist and nothing narrower.
            rest = rest[hit.end():]
            media_type = "artist" if _OF.match(rest) or rest.startswith("של ") else None

    rest = _LEADING_ET.sub("", rest).strip()
    # ``על`` opens where to play or what to set it to, never a name.
    rest = _strip_on_phrase(rest)
    # The room is targeting, not part of the search. It has to go before the
    # ``של`` split, because a room can contain one: "בחדר של הילדים".
    rest = _strip_trailing_room(rest)

    artist = None
    if (split := _OF.search(rest)):
        artist = rest[split.end():].strip()
        rest = rest[:split.start()].strip()
    rest = _LEADING_ET.sub("", rest).strip()

    if not rest and artist:
        rest, artist = artist, None
    if _names_a_device(rest) or (artist and _names_a_device(artist)):
        return None
    if not rest or _is_vague(rest, media_type):
        # "play something by X" - the artist is the only name in the sentence.
        if artist:
            return MusicRequest(artist, media_type or "artist")
        return None

    return MusicRequest(rest, media_type, artist=artist)


def _is_vague(text: str, media_type: str | None) -> bool:
    """Does this name nothing in particular?

    A kind word switches the test off entirely: "פלייליסט רגוע" is a playlist
    that is actually called רגוע, while a bare "משהו טוב" is a mood. Every word
    has to be vague for the phrase to be, so "לילה טוב" survives.
    """
    words = text.split()
    if not words:
        return True
    # Grammar words are never a name, whatever kind was said: "the next song"
    # states a kind and still names nothing.
    if all(word in _NOT_A_NAME for word in words):
        return True
    if media_type:
        return False
    return all(word in _NOT_A_NAME or _VAGUE.find(word) is not None
               for word in words)


def _names_a_device(text: str) -> bool:
    """Is every word here the name of something in the house rather than music?"""
    words = text.split()
    # "שים את הרובוט על שקט" is "set the vacuum to quiet", not a record called
    # "the vacuum on quiet". A trailing על-phrase is the value of a setting -
    # any על that named a room was already removed by _strip_trailing_room - so
    # the device test looks at what comes before it. Only when something does:
    # "שיר על אהבה" has nothing to the left and stays a title.
    if "על" in words and words.index("על") > 0:
        words = words[:words.index("על")]
    # At least one real device noun, with grammar words tolerated around it.
    # Without the first condition a residue of nothing but stop words - which
    # "שירים של היהודים" leaves behind after the split - counted as a device
    # and threw the artist away.
    if not any(_variants(word) & _DEVICE_NOUNS for word in words):
        return False
    return all(
        word in _NOT_A_NAME or bool(_variants(word) & _DEVICE_NOUNS)
        for word in words
    )


def _strip_on_phrase(text: str) -> str:
    """Drop a trailing ``על`` phrase: it is a target or a value, not a name.

    "תנגן את פינק פלויד על המרפסת" plays Pink Floyd on the balcony, and "שים
    את העוצמה על שישים אחוז" is a volume. Hebrew's ``על`` opens neither a
    title nor an artist here - except when nothing comes before it, which is
    where "שיר על אהבה" lives: the kind word has already been taken off the
    front by then, so its ``על`` is first and the phrase survives whole.

    Runs before the room strip rather than instead of it, because what is
    left can still end in a room: "העוצמה בחדר שינה על שישים אחוז".

    :func:`_names_a_device` does the same cut for its own purposes and keeps
    doing it: it is also reached from callers that never came through here.
    """
    words = text.split()
    if "על" in words and words.index("על") > 0:
        return " ".join(words[:words.index("על")])
    return text


def _strip_trailing_room(text: str) -> str:
    """Drop a room named at the end of a music request.

    "play Kaveret in the living room" is a search for Kaveret, not for
    "Kaveret in the living room". The room is matched against every room word
    the project knows rather than this house's own, because getting it out of
    the search string is right either way, and the house's own areas are what
    :class:`SlotIndex` will use a moment later to decide where to play it.
    """
    words = text.split()
    # Shortest tail first. Longest-first looked more careful and was wrong:
    # "playlist לילה טוב בסלון" matches a room across all three words, because
    # the room is inside them, and stripping all three leaves nothing to play.
    # The shortest tail that is itself a room phrase is the room.
    for take in range(1, min(3, len(words)) + 1):
        tail = " ".join(words[-take:])
        if tail.startswith(("ב", "ל", "על")) and _ANY_ROOM.find(tail, fuzzy=False):
            return " ".join(words[:-take])
    return text


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


def _outside(matches: list[Any], wider: list[Any]) -> list[Any]:
    """``matches`` with anything strictly inside a ``wider`` span removed.

    One rule, used twice, because the same ambiguity turns up at two levels: a
    device whose name sits inside a room's ("המחשב" in "חדר המחשב") is the room,
    and a room whose name sits inside a floor's ("כניסה" in "קומת הכניסה") is
    the floor. In both, the longer phrase is the one the speaker actually said
    and the shorter is a substring of it, so containment settles it with
    nothing to tune. Equal spans are left alone - that is the same phrase, not
    a narrower reading of it.
    """
    if not matches or not wider:
        return matches
    return [m for m in matches
            if not any(w.start <= m.start and m.end <= w.end
                       and (w.end - w.start) > (m.end - m.start)
                       for w in wider)]


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
        self._floors: PhraseIndex | None = None
        self._by_domain: dict[str, PhraseIndex] = {}
        self._counts: dict[str, int] = {}

    def invalidate(self, _event: Any = None) -> None:
        self._areas = None
        self._floors = None
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

    # -- floors -------------------------------------------------------------
    def _floor_index(self) -> PhraseIndex:
        """Phrases for the floors this installation has.

        The floor registry first, at tier 0: a household that named a floor
        "קומה עליונה" is matched on "בקומה העליונה" by the same prefix chain
        that finds a room, and Home Assistant's per-floor aliases extend it
        with no code change - the same answer `area_map` reaches for rooms
        nobody anticipated.

        Then `area_map.FLOOR_PHRASES`, at tier 1, for the floors this house
        actually has. Until v11 there was no such table, because the corpus had
        no floor rows to measure one against; the family exists now and every
        phrase in it fires on zero rows that name no floor. It is tier 1 so the
        household's own name always wins, and it is keyed on the *slug* the
        registry name resolves to, so it can only ever reach a floor this
        installation already has. Measured on the held-out set, it takes the
        floor rows the sentence reaches from 46 of 76 to **63 of 76**, with
        zero contradicted. The remaining 13 all say "למטה", which is excluded
        on purpose: see `area_map.FLOOR_PHRASES`.
        """
        if self._floors is not None:
            return self._floors
        from homeassistant.helpers import floor_registry as fr  # noqa: PLC0415

        index = PhraseIndex()
        for floor in fr.async_get(self.hass).async_list_floors():
            for form in {floor.name, *(floor.aliases or ())}:
                if form:
                    index.add(form, floor.floor_id)
            # And the measured Hebrew, for the floor this one answers to.
            slug = floor_slug_for_name(floor.name) or floor.floor_id
            for phrase in FLOOR_PHRASES.get(slug, ()):
                index.add(phrase, floor.floor_id, tier=1)
        _LOGGER.debug("floor index: %d phrases", len(index))
        self._floors = index
        return index

    def areas_on_floor(self, utterance: str) -> list[str]:
        """Every area on a floor the sentence names, or an empty list.

        A floor is a set of rooms, so "תכבה הכל בקומה העליונה" resolves to the
        areas on it and the rest of the executor proceeds exactly as it would
        for a sentence that had named them all. Nothing downstream has to learn
        what a floor is.

        Only consulted when no room matched: a sentence naming both a room and
        its floor means the room, which is the more specific of the two.
        """
        found = self._floor_index().find_all(utterance)
        if not found:
            return []
        # The most specific floor, on the same rule overlapping rooms use.
        floor_id = max(found, key=lambda m: m.rank).value
        from homeassistant.helpers import area_registry as ar  # noqa: PLC0415

        return [area.id for area in ar.async_get(self.hass).async_list_areas()
                if area.floor_id == floor_id]

    def areas_on_floor_slug(self, slug: str | None) -> list[str]:
        """Every area on the floor the *model* named, or an empty list.

        The same standing the model's ``area`` slug has, and consulted for the
        same reason: the sentence is right far more often, so
        :meth:`areas_on_floor` runs first and this is what is left - the
        household whose floor is called something the phrase index cannot
        reach from the words that were said.

        Matched against the floor registry by id, by name and by the same
        slug rule `area_map` uses for rooms, so "upper" reaches a floor named
        "Upper" or "קומה עליונה" with an alias.
        """
        if not slug or not isinstance(slug, str):
            return []
        from homeassistant.helpers import area_registry as ar  # noqa: PLC0415
        from homeassistant.helpers import floor_registry as fr  # noqa: PLC0415

        want = slug.strip().lower().replace(" ", "_")
        floor_id = None
        for floor in fr.async_get(self.hass).async_list_floors():
            names = {floor.floor_id, floor_slug_for_name(floor.name)}
            names |= {floor_slug_for_name(a) for a in (floor.aliases or ())}
            if want in names:
                floor_id = floor.floor_id
                break
        if floor_id is None:
            return []
        return [area.id for area in ar.async_get(self.hass).async_list_areas()
                if area.floor_id == floor_id]

    def list_entities(self, utterance: str, kind: str | None = None) -> list[str]:
        """The `todo` entities a list command should act on.

        A household can have several lists, and adding milk to all of them is
        not a reading of "תוסיף חלב לרשימת קניות". So: a list named outright
        in the sentence wins; failing that, the single list this house has;
        failing that, the one whose name matches the *kind* the sentence
        implied - a shopping list rather than a chore list.

        Empty when nothing matched, which sends the caller to the legacy
        `shopping_list` integration if this installation has one.
        """
        named = [m.value for m in self._entity_index("todo").find_all(utterance)]
        if named:
            return named
        states = self.hass.states.async_all("todo")
        if len(states) == 1:
            return [states[0].entity_id]
        if kind:
            words = SETTING_WORDS["list"]
            wanted = [state.entity_id for state in states
                      if any(kind == value and normalise(word) in
                             normalise(state.attributes.get("friendly_name", ""))
                             for word, value in words.items())]
            if wanted:
                return wanted
        return []

    def area_for_call(self, utterance: str, index: int = 0,
                      total: int = 1) -> str | None:
        """The single room the ``index``-th of ``total`` calls should target."""
        found = self.areas_for_call(utterance, index, total)
        return found[0] if found else None

    def areas_for_call(self, utterance: str, index: int = 0,
                       total: int = 1) -> list[str]:
        """The rooms the ``index``-th of ``total`` tool calls should target.

        A sentence with exactly as many room mentions as calls pairs them off
        in order: "turn the light on in the living room and close the blind in
        the kitchen" is two calls and two rooms, and they are not
        interchangeable.

        One call and several rooms is the other shape - "turn off the light in
        the living room and in the kitchen" - and it means all of them. That is
        a list, not a choice, and answering it with one room leaves a light on.

        The one thing that has to be got right here is telling those apart from
        two *readings of the same words*. A house with a room called ``חדר``
        and another called ``חדר ילדים`` matches both on the phrase "חדר
        ילדים", and lighting the parent room because the child's room was named
        is exactly the kind of over-reach the rest of this module exists to
        prevent. Overlapping matches are therefore competing answers to one
        question and the most specific wins; only rooms named in spans that do
        not touch are two rooms.

        The same rule reaches one level up. "תכבה את האורות בקומת הכניסה" names
        a *floor*, and `כניסה` is one of this project's words for the hallway,
        so the room index claimed it and `_target_area` never got as far as
        asking about floors - the entrance floor lit one corridor. The floor
        phrase contains the room phrase outright, so the spans settle that too.
        16 rows over the corpus, every one of them a floor read as a room, and
        none the other way.
        """
        found = _outside(self._area_index().find_occurrences(utterance),
                         self._floor_index().find_occurrences(utterance))
        if not found:
            return []
        if len(found) == total and 0 <= index < total:
            return [found[index].value]

        if total == 1:
            chosen: list[Any] = []
            for match in sorted(found, key=lambda m: (-m.rank[0], m.start)):
                if any(match.start < other.end and other.start < match.end
                       for other in chosen):
                    continue
                chosen.append(match)
            names = list(dict.fromkeys(
                m.value for m in sorted(chosen, key=lambda m: m.start)))
            if len(names) > 1:
                return names

        return [max(found, key=lambda match: match.rank).value]

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

        A device whose name sits *inside* a room's is the room. "תכבה את המפסק
        בחדר המחשב" is the switch in the computer room, and a house with a
        switch called "המחשב" was turning off the computer instead, because the
        executor consults this before it consults the areas at all. The room
        phrase is the longer of the two and contains the device phrase outright,
        so the spans settle it with nothing to tune: 8 rows over the corpus, and
        every one of them a wrong device rather than a wrong room - which is the
        same failure class, one register down.

        "תכבה את המחשב בסלון" is untouched: the device is at 8-13 and the room
        at 14-19, so neither contains the other and both are read.
        """
        return [m.value for m in _outside(
            self._entity_index(domain).find_all(utterance),
            self._area_index().find_occurrences(utterance))]


    # -- introspection ------------------------------------------------------
    def describe(self) -> dict[str, Any]:
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
