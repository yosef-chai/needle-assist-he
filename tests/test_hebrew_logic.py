# -*- coding: utf-8 -*-
"""The deterministic half of the integration, tested without Home Assistant.

Everything here runs before or after the model, and every case is one that was
wrong at some point. The houses are invented; nothing in this file describes a
real installation.

    python -m pytest tests -q
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conftest import load  # noqa: E402

TEXT = load("hebrew_text")
ROUTER = load("tool_router")
SLOT = load("slot_match")
CLAUSE = load("clause_split")


# --- invented installations -------------------------------------------------
#
# Three shapes that a fixed twelve-slug design cannot serve: rooms outside any
# lexicon, two rooms that transliterate to the same slug, and area ids that
# drifted from the names after a rename.

WIDE = [
    ("hall", "הול", []),
    ("laundry", "חדר כביסה", []),
    ("store", "מחסן", []),
    ("gym", "חדר כושר", []),
    ("coffee", "פינת קפה", []),
    ("living", "סלון", []),
]

SPLIT = [
    ("shower", "מקלחת", []),
    ("toilet", "שירותים", []),
    ("kitchen", "מטבח", []),
]

DRIFTED = [
    # An area renamed after creation keeps its original id. Both of these
    # transliterate to something that contradicts the room they now name.
    ("mtbkh", "מסדרון", []),          # id says kitchen, room is a corridor
    ("khdr_shynh", "מקלחת", []),      # id says bedroom, room is a shower
    ("mtbkh_2", "מטבח", []),
    ("bedroom", "חדר הורים", ["חדר שינה"]),
]


def room(house, utterance):
    index = SLOT.build_area_index(house)
    hits = index.find_all(utterance)
    return hits[0].value if hits else None


# --- rooms ------------------------------------------------------------------

def test_a_room_no_model_has_a_word_for_still_resolves():
    """The point of matching the registry instead of twelve baked-in slugs."""
    for utterance, expected in (
        ("תדליק את האור בהול", "hall"),
        ("תדליק את האור בחדר הכביסה", "laundry"),
        ("תדליק את האור במחסן", "store"),
        ("תדליק את האור בחדר הכושר", "gym"),
        ("תדליק את האור בפינת הקפה", "coffee"),
    ):
        assert room(WIDE, utterance) == expected, utterance


def test_two_rooms_that_share_one_slug_stay_apart():
    """Both transliterate to `washroom`; only the sentence separates them."""
    assert room(SPLIT, "תדליק את האור במקלחת") == "shower"
    assert room(SPLIT, "תדליק את האור בשירותים") == "toilet"


def test_an_area_id_is_not_a_name():
    """A drifted id must never win over the name the household says."""
    assert room(DRIFTED, "תדליק את האור במטבח") == "mtbkh_2"
    assert room(DRIFTED, "תדליק את האור במסדרון") == "mtbkh"
    assert room(DRIFTED, "תדליק את האור במקלחת") == "khdr_shynh"


def test_an_alias_reaches_the_area_that_carries_it():
    """Aliases set in Home Assistant are read; a household says what it says."""
    assert room(DRIFTED, "תדליק את האור בחדר השינה") == "bedroom"


# --- Hebrew surface forms ---------------------------------------------------

def test_the_prefix_chain_is_ordered_not_a_bag_of_letters():
    """"המרחב המוגן" must not be read as a garden.

    Stripping "up to three prefix letters" lets המו parse as prefixes and finds
    גן inside the word. The real chain is ordered - ו, then ש/כש, then ב/ל/כ/מ,
    then ה - and המו is not a legal chain.
    """
    house = [("garden", "גינה", ["גן"]), ("safe", "ממד", ["מרחב מוגן"])]
    assert room(house, "תדליק את האור במרחב המוגן") == "safe"
    assert room(house, "תדליק את האור בגן") == "garden"


def test_a_room_name_inside_a_longer_word_is_not_a_room():
    """גן lives inside מזגן, and a climate command is not a garden command."""
    house = [("garden", "גן", [])]
    assert room(house, "תכבה את המזגן") is None


def test_speech_to_text_damage_still_resolves():
    """Glued words and misplaced final letters both reach the right room."""
    assert room(SPLIT, "תדליק אתהאור במטבח") == "kitchen"
    assert room(SPLIT, "תדליק את האור במטבח") == "kitchen"
    assert room(SPLIT, "תדליק את האור במתבח") == "kitchen"   # typo, one edit


def test_the_whole_home_is_not_a_room():
    assert SLOT.mentions_whole_home("תכבה את הכל בבית")
    assert SLOT.mentions_whole_home("תכבה את האורות בכל הבית")
    assert not SLOT.mentions_whole_home("תכבה את האור בכניסה לבית")


# --- questions --------------------------------------------------------------

def test_a_question_is_never_given_a_tool_that_can_actuate():
    """Asked for a light's state, an earlier build switched the light on.

    A tool the router does not declare cannot be emitted: the decoding grammar
    is compiled from the declared set.
    """
    for question in ("מה המצב של האור במטבח",
                     "מה קורה עם המזגן בסלון",
                     "האם החלון בממד סגור",
                     "כמה מעלות בחדר השינה",
                     "תבדוק את התאורה בסלון",
                     "תגיד לי מה קורה עם הווילונות"):
        assert ROUTER.looks_like_question(question), question
        assert not [t for t in ROUTER.select_tool_names(question)
                    if t not in ("get_state", "get_weather")], question


def test_an_order_is_never_mistaken_for_a_question():
    """The expensive direction. These are the measured near misses."""
    for order in ("תשאיר את האור דולק",
                  "אפשר קצת אור בסלון",
                  "תוכל לפתוח את התריס",
                  "תדליק את האור במטבח"):
        assert not ROUTER.looks_like_question(order), order
        assert [t for t in ROUTER.select_tool_names(order)
                if t not in ("get_state", "get_weather")], order


def test_a_device_question_is_not_answered_with_the_weather():
    assert ROUTER.select_tool_names("מה המצב של האור בשירותים") == ["get_state"]
    assert ROUTER.select_tool_names("מה מזג האוויר") == ["get_weather"]
    assert ROUTER.select_tool_names("תגיד לי אם ירד גשם מחר") == ["get_weather"]
    # Nothing named: the model still chooses between the two.
    assert set(ROUTER.select_tool_names("מה המצב")) == {"get_state", "get_weather"}


def test_a_state_question_is_typed_from_the_sentence():
    assert ROUTER.query_domain("מה המצב של האור במטבח") == "light"
    assert ROUTER.query_domain("מה קורה עם המזגן בסלון") == "climate"
    assert ROUTER.query_domain("מה רמת הלחות במרפסת") == "sensor"
    # חלון is a window contact; וילון and תריס are a blind. One letter apart.
    assert ROUTER.query_domain("האם החלון בממד סגור") == "binary_sensor"
    assert ROUTER.query_domain("מה המצב של הווילונות בסלון") == "cover"
    assert ROUTER.query_domain("מה המצב") is None


# --- refusal ----------------------------------------------------------------

def test_negation_is_blocked_but_correction_is_not():
    """Bare לא corrects mid-sentence far more often than it negates."""
    assert ROUTER.looks_negated("אל תדליק את האור")
    assert ROUTER.looks_negated("בלי להדליק את האור")
    assert ROUTER.looks_negated("אין צורך להדליק")
    assert not ROUTER.looks_negated("תדליק את המנורה, לא לא, תכבה")


def test_off_topic_is_refused_and_a_command_is_not():
    assert ROUTER.looks_off_topic("מה בירת צרפת")
    assert ROUTER.looks_off_topic("ספר לי בדיחה")
    assert not ROUTER.looks_off_topic("תדליק את האור בסלון")
    assert not ROUTER.looks_off_topic("סגור את התריס במטבח")


# --- notifications ----------------------------------------------------------

def test_notification_text_comes_out_of_the_sentence():
    """Final letters are folded for matching and must not be for a message.

    Folding would deliver "האוכל מוכנ", which is a different word to read out.
    """
    assert SLOT.extract_message("תודיע בבית שהאוכל מוכן") == "האוכל מוכן"
    # Nothing extractable means do not send, not send an empty notification.
    assert SLOT.extract_message("תשלח הודעה") is None


# --- catalogue integrity ----------------------------------------------------

def test_every_routed_tool_exists_in_the_catalogue():
    """A tool the router can declare but the model was never given is dead."""
    import json
    catalogue = {t["name"] for t in json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "custom_components"
         / "needle_assist" / "tools.json").read_text(encoding="utf-8"))}
    routed = {name for tools in ROUTER.FAMILY_TOOLS.values() for name in tools}
    assert routed <= catalogue, routed - catalogue
    assert set(ROUTER.FALLBACK) <= catalogue


def test_the_model_is_bundled():
    """A release without the weights installs and then does nothing useful."""
    weights = (pathlib.Path(__file__).resolve().parents[1] / "custom_components"
               / "needle_assist" / "needle_he.cact")
    assert weights.is_file(), "needle_he.cact is missing from the release"
    assert weights.stat().st_size > 10_000_000, "weights look truncated"


# --- several orders in one sentence -----------------------------------------

def test_a_sentence_with_two_orders_becomes_two_clauses():
    """Asked for both at once the model answers with one call.

    Measured on the held-out set: 94 of 97 such sentences come back as a single
    call, usually the verb of one clause with the room of another, for 0.0%
    tool-set accuracy. Cutting first, on the same weights, scores 75.3%.
    """
    assert CLAUSE.split_clauses("תדליק את האור בסלון וסגור את התריסים בחדר שינה") == [
        "תדליק את האור בסלון", "סגור את התריסים בחדר שינה"]
    assert CLAUSE.split_clauses("כבה את האור במטבח וגם תנעל את הדלת") == [
        "כבה את האור במטבח", "תנעל את הדלת"]
    assert CLAUSE.split_clauses(
        "תפעיל את הרובוט, נעל את הדלת, תסגור את התריסים בסלון") == [
        "תפעיל את הרובוט", "נעל את הדלת", "תסגור את התריסים בסלון"]


def test_coordination_that_is_not_a_second_order_is_left_alone():
    """A cut survives only if both sides carry an action verb.

    Each of these contains a joining word and one order. Cutting any of them
    would invent a second command with nothing to act on.
    """
    for sentence in (
        "תדליק את האור בסלון ובמטבח",      # two rooms
        "תדליק את האור והמזגן בסלון",      # two devices
        "אה, סגור את האור בסלון",           # a filler is not an order
        "תכבה את האור בסלון, תודה",         # nor is a polite tail
        "מה המצב של האור בסלון ובמטבח",     # a question is answered as one
    ):
        assert CLAUSE.split_clauses(sentence) == [sentence], sentence


def test_each_clause_is_still_a_command_the_router_can_read():
    """A clause that lost its verb would route to nothing."""
    clauses = CLAUSE.split_clauses(
        "תדליק את האור בסלון וסגור את התריסים בחדר שינה וגם תנעל את הדלת")
    assert [ROUTER.select_tool_names(c)[0] for c in clauses] == [
        "light_turn_on", "cover_close", "lock_lock"]


def test_the_splitter_and_the_router_share_one_verb_list():
    """A verb added for routing has to become a cut point too."""
    for family, verbs in ROUTER.FAMILY_VERBS.items():
        if family == "query":
            continue
        for verb in verbs:
            assert ROUTER._fold(verb) in CLAUSE.ACTION_VERBS, verb
    assert "מה" not in CLAUSE.ACTION_VERBS


# --- one order, several rooms -----------------------------------------------

def rooms(house, utterance, index=0, total=1):
    """`SlotIndex.areas_for_call` over an invented registry."""
    import types
    built = SLOT.build_area_index(house)
    return SLOT.SlotIndex.areas_for_call(
        types.SimpleNamespace(_area_index=lambda: built), utterance, index, total)


def test_one_order_naming_two_rooms_targets_both():
    """"Turn the light off in the living room and the kitchen" is two rooms.

    The splitter leaves this as one clause on purpose - "ובמטבח" is a place,
    not a second order - so the room list is what makes both lights go out.
    """
    assert rooms(DRIFTED, "תכבה את האור בסלון ובמטבח") == ["mtbkh_2"]
    assert rooms(WIDE, "תכבה את האור בסלון ובמחסן") == ["living", "store"]


def test_two_readings_of_one_room_phrase_are_not_two_rooms():
    """The case that makes overlap matter.

    "חדר הורים" also matches the alias "חדר שינה" logic and any shorter room
    word inside it. One room, named once, read two ways - and lighting a second
    room because of it is exactly the over-reach the resolver exists to stop.
    """
    assert rooms(DRIFTED, "תדליק את האור בחדר הורים") == ["bedroom"]
    assert rooms(SPLIT, "תדליק את האור במקלחת") == ["shower"]


def test_as_many_rooms_as_calls_still_pairs_them_off():
    """n rooms and n calls go in order; that rule is untouched."""
    sentence = "תדליק את האור בסלון וסגור את התריסים במחסן"
    assert rooms(WIDE, sentence, index=0, total=2) == ["living"]
    assert rooms(WIDE, sentence, index=1, total=2) == ["store"]


# --- music ------------------------------------------------------------------

def test_music_is_read_out_of_the_sentence():
    """The title never comes from the model, for the same reason a message does not.

    Hebrew reaches a tool argument as escape sequences, six exact characters
    per letter, and the model gets them wrong. So the tool has no slot for a
    title at all: the model is asked only for the kind of thing.
    """
    found = SLOT.extract_music("תנגן לי את אם ננעלו של עומר אדם")
    assert (found.media_id, found.artist) == ("אם ננעלו", "עומר אדם")

    found = SLOT.extract_music("שים לי את האלבום שבלול של כוורת")
    assert (found.media_id, found.media_type, found.artist) == (
        "שבלול", "album", "כוורת")

    found = SLOT.extract_music("תנגן רדיו גלגלצ במטבח")
    assert (found.media_id, found.media_type) == ("גלגלצ", "radio")

    found = SLOT.extract_music("תנגן לי מוזיקה של שלמה ארצי")
    assert (found.media_id, found.media_type) == ("שלמה ארצי", "artist")


def test_a_request_that_names_nothing_is_not_a_search():
    """None means resume, not search the library for the word "music"."""
    for sentence in ("תנגן מוזיקה", "תנגן קצת מוזיקה בסלון",
                     "תנגן את השיר הבא",          # transport control
                     "תפעיל את השואב",            # a device, not a record
                     "תדליק את האור בסלון"):
        assert SLOT.extract_music(sentence) is None, sentence


def test_the_room_is_not_part_of_the_search():
    """Otherwise the library is asked for "Kaveret in the living room"."""
    assert SLOT.extract_music("תנגן לי כוורת בסלון").media_id == "כוורת"


def test_the_router_offers_the_music_tool_when_a_kind_is_named():
    """A word for the kind of thing separates playing from resuming."""
    for sentence in ("שים לי את האלבום שבלול", "תנגן רדיו גלגלצ",
                     "תשמיע לי פלייליסט רגוע"):
        assert ROUTER.select_tool_names(sentence)[0] == "music_play", sentence
    assert "music_play" in ROUTER.select_tool_names("תנגן לי כוורת")
    # "Stop the song" is transport control; the noun must not hijack it.
    assert ROUTER.select_tool_names("תעצור את השיר")[0] == "media_pause"
    assert "music_play" not in ROUTER.select_tool_names("תדליק את האור בסלון")
