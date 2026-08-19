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
import pytest  # noqa: E402
from conftest import load  # noqa: E402

TEXT = load("hebrew_text")
ROUTER = load("tool_router")
SLOT = load("slot_match")
CLAUSE = load("clause_split")
CONST = load("const")
REPLY = load("reply")
DIRECTION = load("direction")


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


def test_a_framed_request_still_names_its_music():
    """Israelis wrap an order in a frame far more often than they bark it.

    Anchoring the verb at the head of the sentence looked tidy and lost 72 of
    97 held-out music requests. Hebrew also builds the infinitive from a stem
    that is not the imperative - להפעיל is ל + הפעיל - and puts the definite
    article on the second noun of a construct chain.
    """
    assert SLOT.extract_music(
        "אני רוצה שתנגן לי את אם ננעלו של עומר אדם").media_id == "אם ננעלו"
    assert SLOT.extract_music(
        "אפשר להפעיל לנו את הפלייליסט שירים ישראלים").media_id == "שירים ישראלים"
    assert SLOT.extract_music(
        "אני צריך שתפעילי לנו את תחנת הרדיו אקו 99").media_id == "אקו 99"
    assert SLOT.extract_music("תוכל להאזין לפינק פלויד").media_id == "פינק פלויד"


def test_a_device_setting_is_not_a_record():
    """"שים את הרובוט על שקט" sets the vacuum, and שקט is also a media noun.

    That combination is what lets the router offer a play tool on a sentence
    about a vacuum cleaner, so the extractor has to be the thing that says no.
    """
    for sentence in ("תפעיל לנו משהו טוב בחדר ההורים",
                     "אני רוצה שתנגן לי מוזיכה",
                     "שים את הרובוט על שקט",
                     "שים את הפן של המיזוג במטבחון על חזק"):
        assert SLOT.extract_music(sentence) is None, sentence


def test_a_kind_word_protects_a_title_that_looks_vague():
    """"רגוע" is a mood and also the name of a playlist."""
    assert SLOT.extract_music("תשמיע פלייליסט רגוע").media_id == "רגוע"
    assert SLOT.extract_music("תשמיע לי משהו טוב") is None


# --- what the assistant says back -------------------------------------------

class Outcome:
    """Stand-in for executor.CallOutcome, which needs Home Assistant."""

    def __init__(self, ok, speech=None, detail=""):
        self.ok, self.speech, self.detail = ok, speech, detail


def test_nothing_to_do_is_an_answer_not_a_failure():
    """An empty call list is Needle refusing, which is a valid outcome."""
    assert REPLY.compose([]).error is None
    assert REPLY.compose([]).speech == CONST.SPEECH_NOTHING


def test_a_plain_success_confirms_and_a_speaking_one_speaks():
    assert REPLY.compose([Outcome(True)]).speech == CONST.SPEECH_OK
    assert REPLY.compose([Outcome(True, "מנגן שבלול")]).speech == "מנגן שבלול"
    # Several orders, several confirmations, one sentence.
    both = REPLY.compose([Outcome(True, "מנגן שבלול"), Outcome(True, "22 מעלות")])
    assert both.speech == "מנגן שבלול. 22 מעלות"


def test_a_partly_successful_sentence_says_so():
    """Speaking only the success leaves a dropped order sounding like it ran."""
    said = REPLY.compose([Outcome(True, "מנגן שבלול"), Outcome(False, detail="boom")])
    assert said.error is None                      # something did happen
    assert said.speech.startswith("מנגן שבלול. ")
    assert "נכשלה" in said.speech
    # With nothing to announce, the partial report stands on its own.
    quiet = REPLY.compose([Outcome(True), Outcome(False, detail="boom")])
    assert quiet.speech == CONST.SPEECH_OK + " חלקית, פעולה אחת נכשלה"


def test_hebrew_counts_the_failures_the_way_hebrew_counts():
    """"1 פעולות נכשלו" is a machine talking; the noun and verb agree here."""
    assert REPLY.failures(1) == "פעולה אחת נכשלה"
    assert REPLY.failures(2) == "שתי פעולות נכשלו"
    assert REPLY.failures(4) == "ארבע פעולות נכשלו"
    # Past the clause limit a digit reads naturally again.
    assert REPLY.failures(7) == "7 פעולות נכשלו"


def test_finding_nothing_is_a_different_error_from_going_wrong():
    """Home Assistant has two codes and they mean different things.

    "no lamp in the study" is the house not having one; anything else is the
    integration failing, and only the second is worth putting internals into.
    """
    empty = REPLY.compose([Outcome(False, detail=REPLY.NO_TARGET)])
    assert empty.error == REPLY.NO_TARGETS
    assert empty.speech == CONST.SPEECH_NO_TARGET

    broke = REPLY.compose([Outcome(False, detail="engine failure")], "bad export")
    assert broke.error == REPLY.FAILED
    assert broke.speech.endswith("bad export")

    # One of each is not "nothing was there", so it reports the failure.
    mixed = REPLY.compose([Outcome(False, detail=REPLY.NO_TARGET),
                           Outcome(False, detail="engine failure")])
    assert mixed.error == REPLY.FAILED


# --- what the miss list taught the router -----------------------------------
#
# Each of these was a family of misses in `eval/router_recall.py` before the
# word behind it went into a table. Together they took recall on the held-out
# set from 97.6% to 99.1%, and they are here so a table edit cannot quietly
# undo one.

@pytest.mark.parametrize("query,expected", [
    # A scene is asked for by mood as often as by name.
    ("תעשה לי אווירת ערב בבקשה", "scene_activate"),
    ("אפשר לעשות לי אווירת שינה", "scene_activate"),
    # ...and an automation named outright keeps its slot even though its own
    # name ("מצב חופשה") is also a helper.
    ("תשבית את האוטומציה מצב חופשה", "automation_turn_off"),
    ("תדליק את האוטומציה מצב חופשה", "automation_turn_on"),
    # "next in line" is how the next track gets asked for.
    ("הבא בתור בחדר השינה בבקשה", "media_next_track"),
    # A playlist has a Hebrew name as well as a borrowed one.
    ("הפעל לנו את רשימת ההשמעה לילה טוב", "music_play"),
    # Speech-to-text writes a borrowed word the way it sounds.
    ("שים לי תיימר של עשר דקות", "timer_start"),
])
def test_router_reaches_the_tool_the_sentence_named(query, expected):
    assert expected in ROUTER.select_tool_names(query), query


def test_a_weather_word_inside_a_room_name_is_not_a_forecast():
    """"מרפסת שמש" is the balcony and "בחוץ" is the garden.

    Both were weather words on equal footing with "מזג אוויר", so a question
    about a device standing in either room was answered with the forecast. Of
    the 34 sentences in the held-out set containing שמש, none ask about the
    sky.
    """
    assert ROUTER.select_tool_names("מה קורה עם המאוורר במרפסת שמש") == ["get_state"]
    assert ROUTER.select_tool_names("מה קורה עם התאורה בחוץ") == ["get_state"]
    # The sky still answers for itself, including when what is asked for is a
    # number rather than a thing - a thermostat is a מזגן, not a טמפרטורה.
    assert ROUTER.select_tool_names("מה מזג האוויר היום") == ["get_weather"]
    assert ROUTER.select_tool_names("מה הטמפרטורה בחוץ") == ["get_weather"]
    # A statement with no interrogative in it never reaches the weather
    # test at all - it falls through to the family scorer, which keeps a
    # query slot for exactly this case. get_weather still has to be in
    # the shortlist; it just does not have it to itself.
    assert "get_weather" in ROUTER.select_tool_names("חם בחוץ")


def test_a_vague_request_reaches_audio_without_unlocking_the_gate():
    """"תפעיל לי משהו" is a request to play; "תספר לי משהו" is chatter.

    The word carries a weak weight for exactly this reason. Entered as a device
    noun it also scored three points on the four off-topic rows that say
    "תספר לי משהו על הפירמידות", which then stopped being refused.
    """
    assert "media_play" in ROUTER.select_tool_names("תפעיל לי משהו בסלון")
    assert ROUTER.looks_off_topic("תספר לי משהו על הפירמידות")
    assert ROUTER.looks_off_topic("ספר לי משהו מעניין")


@pytest.mark.parametrize("sentence,media_id", [
    # על says where to play it...
    ("תנגן את פינק פלויד על המרפסת", "פינק פלויד"),
    # ...or what to set it to, and neither is part of the name.
    ("שים את העוצמה בחדר שינה על שישים אחוז", None),
    ("שים את שואב האבק על מצב שקט", None),
    # A speaker, a volume and a source are where the music comes out, not
    # what comes out of them.
    ("תנגן את הרמקול על הדשא", None),
    ("שים ספוטיפיי בחדר האורחים", None),
    ("תשים את הקול במוסך על 15 אחוז", None),
    # ...but a playlist really called שקט survives all of it, which is why the
    # equipment words are listed one by one instead of taken as a family.
    ("נגן את הפלייליסט שקט", "שקט"),
    ("הפעל את רשימת ההשמעה לילה טוב", "לילה טוב"),
])
def test_where_to_play_it_is_not_what_to_play(sentence, media_id):
    found = SLOT.extract_music(sentence)
    assert (found.media_id if found else None) == media_id, sentence


def test_naming_a_room_says_the_sentence_is_about_the_house():
    """A room is worth two of the three the gate asks for, and never three.

    "תפעיל לי משהו בסלון" scores one for the play verb and was being thrown
    away before the model ever saw it - 34 real commands were, on the held-out
    set. Nobody says "בסלון" about the pyramids.

    Two rather than three because a room names a place and not a thing to act
    on: at three a room passes the gate on its own, and "הגינה של השכנים
    מוזנחת" is not a question, so it would be handed a shortlist that can
    actuate.
    """
    assert not ROUTER.looks_off_topic("תפעיל לי משהו בסלון")
    assert not ROUTER.looks_off_topic("תנגן לנו משהו טוב בחדר השינה")
    # Chatter stays refused, with a room in it or without one.
    assert ROUTER.looks_off_topic("תספר לי משהו על הפירמידות")
    assert ROUTER.looks_off_topic("הגינה של השכנים מוזנחת")
    assert ROUTER.looks_off_topic("לאיזה מוסך כדאי לקחת את האוטו")


# --- which way round --------------------------------------------------------

@pytest.mark.parametrize("said,sentence,settled", [
    # The worst thing a home assistant can do, and the reason this exists: the
    # shipped model answered 45 of these with lock_unlock.
    ("lock_unlock", "נעל את הדלת בחדר ההורים", "lock_lock"),
    ("lock_unlock", "תנעל לי את המנעול בכניסה", "lock_lock"),
    ("lock_lock", "תפתח את המנעול במטבח", "lock_unlock"),
    ("lock_lock", "תשחרר את הנעילה בגינה", "lock_unlock"),
    # ...and the rest of the pairs.
    ("cover_close", "תפתח את התריס בסלון", "cover_open"),
    ("cover_open", "תוריד את הווילון בחדר שינה", "cover_close"),
    ("light_turn_off", "תדליק את האור במטבח", "light_turn_on"),
    ("light_turn_on", "תכבה את המנורה בסלון", "light_turn_off"),
    ("switch_turn_off", "תדליק את הדוד במקלחת", "switch_turn_on"),
    ("camera_turn_on", "תכבי את המצלמה בחניון", "camera_turn_off"),
    # A tool with no opposite is returned untouched.
    ("vacuum_start", "תפעיל את השואב", "vacuum_start"),
    ("get_state", "מה המצב של האור בסלון", "get_state"),
])
def test_the_verb_decides_which_way_round(said, sentence, settled):
    assert DIRECTION.settle(said, sentence) == settled


def test_a_speaker_taking_a_verb_back_is_left_alone():
    """"תכבה את המנורה, לא לא, תעשה את המנורה" says the wrong verb first.

    Sixteen of the eighteen disagreements with gold were this shape, and every
    one of them is a light the speaker wanted *on*. The correction markers are
    what the corpus's own `correction` family is built from.
    """
    sentence = "תכבה את המנורה בחדר האוכל, לא לא, תעשה את המנורה בחדר האוכל"
    assert DIRECTION.settle("light_turn_on", sentence) == "light_turn_on"
    assert DIRECTION.settle("light_turn_off", sentence) == "light_turn_off"


def test_silence_leaves_the_model_alone():
    """No direction verb, no correction. The common case and the safe one."""
    assert DIRECTION.settle("cover_open", "את התריס בסלון בבקשה") == "cover_open"
    assert DIRECTION.settle("cover_close", "את התריס בסלון בבקשה") == "cover_close"


def test_dimming_is_a_brightness_and_not_an_off():
    """עמעם sits in light_turn_off's routing hints, which is right there.

    "תעמעם קצת פחות" is light_turn_on carrying a brightness argument, so the
    guard does not read that word as a direction.
    """
    assert DIRECTION.settle(
        "light_turn_on", "תעמעם קצת פחות את הנורה בחדר הביטחון") == "light_turn_on"


def test_play_and_pause_are_deliberately_not_guarded():
    """נגן is both "play!" and "the player", so no counting separates them.

    "תעצור את הנגן" contains a pause verb and a play hint. Music that keeps
    playing is not a door that opens, so the pair is left out entirely.
    """
    assert "media_play" not in DIRECTION.OPPOSITE
    assert DIRECTION.settle("media_play", "תעצור את הנגן בסלון") == "media_play"


def test_a_routine_falls_back_to_its_sibling_domain_and_no_further():
    """A scene and a script are the same act; an automation is not.

    Home Assistant keeps a household's routines in three domains and a Hebrew
    sentence cannot say which one "אווירת ערב" landed in. Both models trained
    here confuse them. The registry knows, so a name absent from the domain the
    model chose is looked for in the sibling one.

    Automations are excluded on purpose: `automation.turn_on` *enables* an
    automation rather than running it, so guessing wrong there would leave a
    household with one quietly switched on.
    """
    assert CONST.ROUTINE_SIBLING == {"scene_activate": "script_run",
                                     "script_run": "scene_activate"}
    for tool, sibling in CONST.ROUTINE_SIBLING.items():
        assert CONST.ROUTINE_SIBLING[sibling] == tool
        assert tool in CONST.NAME_ADDRESSED and sibling in CONST.NAME_ADDRESSED
        assert CONST.NAME_ADDRESSED[tool] != CONST.NAME_ADDRESSED[sibling]
    assert not any(t.startswith("automation") for t in CONST.ROUTINE_SIBLING)


def test_naming_nothing_means_all_of_them_only_for_timers():
    """"בטל את הטיימר" means every timer. "תפעיל סצנה" does not mean every scene."""
    assert CONST.ALL_WHEN_UNNAMED == {"timer_start", "timer_cancel"}
    assert CONST.ALL_WHEN_UNNAMED <= set(CONST.NAME_ADDRESSED)
    assert all(CONST.NAME_ADDRESSED[t] == "timer" for t in CONST.ALL_WHEN_UNNAMED)


def test_every_name_addressed_tool_targets_a_real_domain():
    for tool, domain in CONST.NAME_ADDRESSED.items():
        assert tool in CONST.SERVICE_MAP, tool
        assert CONST.SERVICE_MAP[tool][0] == domain, tool


@pytest.mark.parametrize("sentence,clauses", [
    # Taking an order back is not giving two orders. Cutting these in two
    # turned the light off and then on again - 33 of the 74 correction rows of
    # the held-out set were being split that way.
    ("תכבה את המנורה בחדר האוכל, לא לא, תעשה את המנורה בחדר האוכל",
     ["תעשה את המנורה בחדר האוכל"]),
    ("אה, תסגור אור בחדר העבודה, לא, בעצם תעשי אור בחדר העבודה",
     ["תעשי אור בחדר העבודה"]),
    ("נו תכבה לי את המנורה על הבלקון, טעות, פתחי את המנורה על הבלקון",
     ["פתחי את המנורה על הבלקון"]),
    ("תכבה לי את האור בחצר, אה לא פתחי את האור בחצר",
     ["פתחי את האור בחצר"]),
    # An afterthought is not a retraction: dropping the first half here would
    # drop the only verb in the sentence, so the rule stands down.
    ("תדליק את האור בסלון ובעצם גם במטבח",
     ["תדליק את האור בסלון ובעצם גם במטבח"]),
    # And two real orders are still two.
    ("תכבה את האור בסלון וסגור את התריס במטבח",
     ["תכבה את האור בסלון", "סגור את התריס במטבח"]),
])
def test_an_order_taken_back_is_not_a_second_order(sentence, clauses):
    assert CLAUSE.split_clauses(sentence) == clauses


def test_the_correction_pattern_is_shared_with_the_direction_guard():
    """One definition, because both modules need the same sentences.

    `clause_split` must not cut them and `direction` must not read a verb off
    them, and for the same reason: the verb before the retraction is the one
    the speaker withdrew.
    """
    assert DIRECTION.CORRECTION is CLAUSE.CORRECTION


@pytest.mark.parametrize("said,sentence,settled", [
    # Not a direction, the same question: which of two tools in one domain.
    ("vacuum_start", "שים את הרובוט על שקט", "vacuum_set_fan_speed"),
    ("climate_set_temperature", "תעביר את המזגן למהירות נמוכה",
     "climate_set_fan_mode"),
    # ...and the other way round, so the rule is not a one-way ratchet.
    ("vacuum_set_fan_speed", "תפעיל את שואב האבק", "vacuum_start"),
    ("climate_set_fan_mode", "תוריד את המזגן ל-22 מעלות",
     "climate_set_temperature"),
])
def test_a_setting_is_not_a_start(said, sentence, settled):
    assert DIRECTION.settle(said, sentence) == settled


def test_volume_against_play_is_left_out_on_purpose():
    """65 agree and one disagrees, and one is not zero.

    The bar for overruling a model is that the rule is never wrong on the
    held-out set, not that it is usually right.
    """
    assert "media_set_volume" not in DIRECTION.OPPOSITE


@pytest.mark.parametrize("sentence,way", [
    # The verb decides.
    ("בסלון הגדול חם מדי, תנמיך משמעותית", -1),
    ("במרפסת קר מדי, תגביר קצת", 1),
    ("תעמעם את האור בסלון", -1),
    ("תרים את העוצמה במטבח", 1),
    # No verb, so the adjective decides - and "יותר חלש" is more *quiet*,
    # not more. Reading the comparative first gets all 32 of these backwards.
    ("יותר חלש בחוץ", -1),
    ("משמעותית יותר חלש בפינת המטבח", -1),
    ("קצת יותר חזק בסלון", 1),
    # Neither, so the bare comparative decides. 170 brightness rows of the
    # corpus say only this much.
    ("קצת יותר בסלון", 1),
    ("טיפה פחות במטבח", -1),
    # And silence is allowed.
    ("תעשה משהו בסלון", None),
])
def test_the_sentence_says_which_way(sentence, way):
    assert DIRECTION.which_way(sentence) == way


def test_a_retraction_is_not_a_direction():
    """The verb before "לא לא" is the one the speaker withdrew."""
    assert DIRECTION.which_way("תנמיך, לא לא, תגביר") is None


@pytest.mark.parametrize("sentence,before,after", [
    # An eight-degree error from a one-character one.
    ("בסלון הגדול חם מדי, תנמיך משמעותית",
     {"area": "living_room", "temperature_step": 4},
     {"area": "living_room", "temperature_step": -4}),
    ("יותר חלש בחוץ",
     {"area": "garden", "volume_step_pct": 20},
     {"area": "garden", "volume_step_pct": -20}),
    ("תגביר קצת יותר את האור",
     {"area": "salon", "brightness_step_pct": -10},
     {"area": "salon", "brightness_step_pct": 10}),
    # Already right: left alone, magnitude and all.
    ("תנמיך קצת", {"area": "salon", "temperature_step": -1},
     {"area": "salon", "temperature_step": -1}),
    # Nothing is invented: an absolute temperature is not a step, an argument
    # the model did not emit stays absent, and a silent sentence changes
    # nothing.
    ("תנמיך", {"area": "salon", "temperature": 24},
     {"area": "salon", "temperature": 24}),
    ("תכבה את האור", {"area": "salon"}, {"area": "salon"}),
    ("תעשה משהו", {"area": "salon", "temperature_step": 3},
     {"area": "salon", "temperature_step": 3}),
])
def test_the_sentence_settles_the_sign_and_nothing_else(sentence, before, after):
    assert DIRECTION.settle_steps(before, sentence) == after


def test_settling_a_sign_does_not_touch_the_caller_s_arguments():
    args = {"area": "salon", "temperature_step": 4}
    DIRECTION.settle_steps(args, "חם מדי, תנמיך")
    assert args == {"area": "salon", "temperature_step": 4}


def test_every_relative_argument_is_guarded():
    """The set must match what `executor._service_data` resolves against a
    current reading; a fourth one added there without being added here would
    ship with an unguarded sign."""
    assert DIRECTION.RELATIVE == {
        "temperature_step", "brightness_step_pct", "volume_step_pct"}
