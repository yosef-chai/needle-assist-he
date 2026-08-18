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
