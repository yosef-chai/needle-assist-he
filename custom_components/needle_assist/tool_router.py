# -*- coding: utf-8 -*-
"""Pick at most five tools to declare for a given Hebrew utterance.

Why this exists
---------------
Needle runs a built-in retrieval head when more than five tools are declared:
it renders only its top five per turn and constrains the grammar to that subset.
A tool outside that five cannot be emitted at all - the grammar forbids it.

That head is trained contrastively upstream against English tool descriptions,
and nothing in ``cactus-needle``'s fine-tuning path touches it: ``finetune.py``
and ``cli.py`` contain no reference to retrieval, contrastive training or the
tool index. So it does not learn Hebrew from this fine-tune, and measurement
says it does not handle Hebrew at all. On 250 held-out utterances:

    all 41 tools declared      6.8% tool-set correct
    the row's own <=5 declared 54.4% tool-set correct

An eightfold difference, and the failures are not random - the head returns a
near-constant top five, so ``light_turn_on`` came back as ``cover_open``,
``climate_turn_off`` as ``cover_close`` and ``media_set_volume`` as
``cover_open``. It is embedding Hebrew it cannot read.

Declaring five or fewer tools bypasses the head entirely. That is also exactly
the shape the model was fine-tuned on: every training row carries a candidate
set of about five tools (gold plus same-family adversarial distractors), so
declaring the whole catalogue at inference is a distribution shift away from
training as well as a retrieval problem.

Hence this module. It is deliberately a plain keyword scorer rather than
anything learned:

* it is deterministic and inspectable - a wrong routing decision is one lookup
  to find and one list entry to fix;
* it runs in microseconds with no model call, so it costs nothing on device;
* the keywords are the same Hebrew surface forms the training data was
  generated from, so the router and the model agree on vocabulary by
  construction;
* Home Assistant's own conversation pipeline resolves intents by deterministic
  sentence matching too, so this is idiomatic for the platform rather than a
  workaround.

The model still chooses the tool and fills every argument. The router only
decides which shortlist it chooses from, which is the part the measurement says
the model cannot do in Hebrew.
"""

from __future__ import annotations

import re
from typing import Final

from .area_map import AREA_ALIASES
from .hebrew_text import PhraseIndex, normalise

MAX_TOOLS: Final = 5
_NOUN_WEIGHT: Final = 3

# Tool names grouped the way the training data groups them. Within a family the
# order is the order tools are offered when the shortlist has room to spare.
FAMILY_TOOLS: Final[dict[str, list[str]]] = {
    "light": ["light_turn_on", "light_turn_off", "light_toggle"],
    "climate": ["climate_set_temperature", "climate_set_hvac_mode",
                "climate_set_fan_mode", "climate_turn_off"],
    "fan": ["fan_turn_on", "fan_turn_off", "fan_oscillate"],
    "cover": ["cover_open", "cover_close", "cover_stop", "cover_set_position"],
    "lock": ["lock_lock", "lock_unlock"],
    "camera": ["camera_turn_on", "camera_turn_off"],
    "vacuum": ["vacuum_start", "vacuum_return_to_base", "vacuum_pause",
               "vacuum_set_fan_speed"],
    "media": ["music_play", "media_play", "media_pause", "media_set_volume",
              "media_mute", "media_next_track", "media_select_source"],
    "switch": ["switch_turn_on", "switch_turn_off"],
    "routine": ["scene_activate", "script_run", "automation_turn_on",
                "automation_turn_off"],
    "helper": ["input_boolean_turn_on", "input_boolean_turn_off",
               "timer_start", "timer_cancel", "notify_send"],
    "query": ["get_state", "get_weather"],
}

# Device nouns. These are the strong signal: Hebrew imperatives are heavily
# overloaded (תפתח opens a blind, unlocks a door, and colloquially turns on a
# light) so the noun decides the domain and the verb only breaks ties.
FAMILY_NOUNS: Final[dict[str, list[str]]] = {
    # Each list also carries the transliterated English the code-switching
    # recipe generates ("את הלייט", "את הא.יי.סי"), because those utterances
    # are Hebrew-framed and never reach an English tool description.
    "light": ["אור", "אורות", "תאורה", "מנורה", "מנורות", "נורה", "נורות",
              "ספוט", "ספוטים", "לד", "דימר", "אורה", "לייט", "לייטס"],
    # "קר מדי" / "חם מדי" name the sensation instead of the device ("בסלון קר
    # מדי, תעלה"). Without them the verb decides, and תעלה/תוריד raise and lower
    # a blind, so these routed to cover - 41 of the test misses. Two-word forms
    # again: bare "חם" and "קר" are also hvac modes ("על חם").
    "climate": ["מזגן", "מזגנים", "מיזוג", "אינוורטר", "עינוורטר", "קירור",
                "חימום", "מאייד", "טמפרטורה", "מעלות", "מחמם",
                "א.יי.סי", "אייר", "קונדישן", "קר מדי", "חם מדי"],
    "fan": ["מאוורר", "מאווררים", "פן", "ונטה", "וונטה", "מפוח", "וונטילטור"],
    "cover": ["תריס", "תריסים", "וילון", "וילונות", "ווילון", "ווילונות",
              "שאטרס", "רפפות", "תריסול", "גגון", "סטורים", "בליינדס"],
    "lock": ["דלת", "מנעול", "נעילה", "שער", "בריח"],
    "camera": ["מצלמה", "מצלמות", "מצלמת"],
    # "שיעשה סיבוב" is the elliptical "send the robot round". The whole phrase
    # is the key because bare "סיבוב" is how a fan's oscillation is asked for
    # ("תפעיל סיבוב"), and that belongs to fan_oscillate.
    "vacuum": ["שואב", "רובוט", "רומבה", "שואבת", "ווקום", "שיעשה סיבוב"],
    # "יותר חזק" / "יותר חלש" are the elliptical volume commands ("טיפה יותר
    # חזק בסלון") - they name no device at all, so without them the utterance
    # scores nothing and falls through to the generic shortlist. The two-word
    # form is deliberate: bare "חזק" is also a vacuum suction level and a
    # climate fan speed.
    "media": ["מוזיקה", "מוסיקה", "שיר", "רמקול", "רמקולים", "נגן", "פודקאסט",
              "שירים", "אלבום", "תקליט", "רצועה", "סינגל", "זמר", "זמרת",
              "אמן", "אמנית", "להקה", "תחנה", "דיסק",
              "רדיו", "ווליום", "וליום", "עוצמה", "נגינה", "פלייליסט",
              "מיוזיק", "סאונד", "שאונד", "קול", "השתקה", "שקט", "מיוט",
              "יותר חזק", "יותר חלש",
              "בלוטות'", "מקור", "ספוטיפיי", "יוטיוב", "אייראפליי", "ערוץ",
              # A tool hint only orders a family something else has already
              # reached, so these have to name the family themselves.
              "הבא בתור", "רשימת השמעה", "השמעה"],
    # דוד / מיחם / בוילר / מחשב / טלוויזיה / מטען are the switch entities the
    # training data names, so "turn on the TV" is a switch, not a media command.
    "switch": ["שקע", "תקע", "מפסק", "שקעים", "בוילר", "דוד", "מיחם", "מחשב",
               "טלוויזיה", "טיוי", "מטען", "פלאג", "סוקט"],
    # Scenes and scripts are referred to by name ("מצב סרט", "סצנת ערב",
    # "תעשה השקיה"), so the names themselves have to be in the table. They are
    # listed in their two-word form wherever the bare word is ambiguous -
    # "אורחים" alone is part of "חדר האורחים", the living room.
    "routine": ["סצנה", "סצינה", "סצנת", "תרחיש", "סקריפט", "אוטומציה",
                "אוטומציות", "רוטינה", "שגרה",
                "מצב ערב", "מצב בוקר", "מצב סרט", "מצב רומנטי", "מצב מסיבה",
                "מצב שינה", "מצב קריאה", "מצב שבת", "מצב אורחים",
                # A mood, asked for as one: "תעשה לי אווירת ערב". Unlike
                # bare "מצב", which also means "state", nothing else in a
                # house is an אווירה - so the word carries on its own and
                # a household scene called "אווירת קפה" is reachable too.
                "אווירה", "אווירת",
                "ניקיון", "השקיה", "יציאה מהבית", "חזרה הביתה", "לילה טוב",
                "תריסים בבוקר", "אורות בלילה"],
    # input_boolean helpers are all named "מצב <something>", so the two-word
    # form is the key - bare "מצב" also means "state" and would collide with
    # state queries and with scenes.
    # "ספירה" (the countdown) names a timer and nothing else, but it lived only
    # in timer_start's hints, so "תעצור את הספירה" scored no noun at all and the
    # verb handed the shortlist to media - six of the recall misses, every one
    # of them timer_cancel.
    "helper": ["טיימר", "תיימר", "שעון עצר", "תזכורת", "הודעה", "התראה",
               "דגל",
               "ספירה", "הספירה", "ספירה לאחור",
               "מצב אורחים", "מצב חופשה", "מצב לילה", "מצב שקט", "מצב חיסכון",
               "נעדר"],
    # No state adjectives here (דולק, סגור, פתוח, נעול): they are far more
    # common as commands than as questions, and scoring them as query nouns
    # pulled get_state into the shortlist ahead of the real domain's tools.
    # "מחר" carries weather on its own ("יהיה חם מחר"). "בחוץ" deliberately does
    # not: it is also a colloquial name for the garden, so "תדליק בחוץ" is a
    # light command, and the interrogative verbs below separate the two.
    #
    # The temperature-outside phrasings are listed as two-word keys for exactly
    # that reason. "חם בחוץ" and "קר היום בחוץ" are the commonest way to ask about
    # the weather without naming it, and they were reaching no family at all:
    # measured on the v8 test set, every one of them was answered
    # `climate_set_temperature`, which is a plausible guess for an unrouted
    # temperature word and is marked wrong. Multi-word keys match by substring
    # (see `_hits`), so these cannot fire on a bare "בחוץ".
    "query": ["מזג", "תחזית", "גשם", "לחות", "מחר",
              "חם בחוץ", "קר בחוץ", "חם היום", "קר היום", "חם למעלה"],
}

# Verbs. Weak signal, used only to rank families that the nouns already matched,
# or to guess when no noun matched at all.
FAMILY_VERBS: Final[dict[str, list[str]]] = {
    "light": ["הדלק", "תדליק", "תדליקי", "הדליקי", "כבה", "תכבה", "תכבי", "כבי",
              "עמעם", "תעמעם", "האר", "עמעמי", "תעמעמי"],
    "climate": ["קרר", "תקרר", "חמם", "תחמם", "תכוון", "כוון", "תכווני", "כווני"],
    "fan": ["תסובב", "יסתובב", "לסובב", "תסובבי"],
    "cover": ["תרים", "הרם", "תוריד", "הורד", "תעלה", "תפתח", "פתח", "תסגור",
              "סגור", "תרימי", "הרימי", "תורידי", "הורידי", "תעלי", "תפתחי", "פתחי", "תסגרי", "סגרי"],
    "lock": ["נעל", "תנעל", "תנעלי", "נעלי", "תשחרר", "שחרר", "תשחררי", "שחררי"],
    "camera": ["תצלם", "צלם", "תצלמי", "צלמי"],
    "vacuum": ["תשאב", "שאב", "לשאוב", "לנקות", "תנקה", "נקה", "שיחזור",
               "לבסיס", "לתחנה", "לעגינה", "תשאבי", "שאבי", "תנקי", "נקי"],
    "media": ["נגן", "תנגן", "השמע", "תשמיע", "השתק", "תשתיק", "תדלג",
              "האזן", "תאזין", "האזני", "תאזיני", "ערבב", "תערבב", "ערבבי",
              "תערבבי",
              "דלג", "עצור", "תעצור", "תפסיק", "השהה", "תשהה", "נגני", "תנגני", "השמיעי", "תשמיעי", "השתיקי", "תשתיקי", "תדלגי", "דלגי", "עצרי", "תעצרי", "תפסיקי", "תשהי", "תשהי"],
    "switch": ["טרן", "און", "אוף", "תסוויץ'"],
    "routine": ["הפעל", "תפעיל", "הרץ", "תריץ", "הפעילי", "תפעילי", "הריצי", "תריצי"],
    "helper": ["תזכיר", "הזכר", "תשלח", "שלח", "הודע", "תודיע", "תעמיד", "תזכירי", "תשלחי", "שלחי", "הודיעי", "תודיעי", "תעמידי", "תעדכן", "תעדכני"],
    # Interrogatives, plus the state adjectives that were demoted out of the
    # noun table - as verbs they still nudge a question toward get_state
    # without outweighing the device noun that names the domain.
    "query": ["מה", "האם", "כמה", "תבדוק", "בדוק", "תגיד", "מהי", "מהו",
              "איך", "יהיה", "צפוי",
              "דולק", "דולקת", "כבוי", "כבויה", "פתוח", "פתוחה", "נעול",
              "נעולה", "סגורה", "תבדקי", "בדקי", "תגידי"],
}

# Words that point at a family without naming anything in it. They score
# like a verb - one, not three - and that is exactly the strength they
# should have.
#
# "תפעיל לי משהו בסלון" is a request to play something, and the corpus
# labels it media_play; nine rows of the held-out set were lost because
# "תפעיל" is a routine verb and nothing else in the sentence said audio.
# One point is enough to reach a spare slot in the shortlist and not enough
# to make the family strong - and, measured, not enough to pass the refusal
# gate either: the four off-topic rows that say "תספר לי משהו על הפירמידות"
# stay refused, which they do not if the word is entered as a noun.
FAMILY_WEAK: Final[dict[str, list[str]]] = {
    "media": ["משהו"],
}

# Sub-ranking inside a family. This is what decides *which* of a family's tools
# survives when the shortlist is tight, so the on/off pairs need it as much as
# the crowded families do: without hints, ``light_turn_off`` sat second in its
# family and got cut from commands that plainly said "turn off".
#
# Verbs are deliberately allowed to appear under several tools - "תסגור" closes
# a blind, locks a door and colloquially turns a light off. The family score has
# already chosen the domain by then, so the overlap costs nothing.
TOOL_HINTS: Final[dict[str, list[str]]] = {
    "light_turn_on": ["הדלק", "תדליק", "תדליקי", "הדליקי", "האר", "פתח",
                      "תפתח", "און", "פתחי", "תפתחי"],
    "light_turn_off": ["כבה", "תכבה", "תכבי", "כבי", "סגור", "תסגור", "אוף",
                       "תעמעם", "עמעם", "סגרי", "תסגרי", "תעמעמי", "עמעמי"],
    "light_toggle": ["תחליף", "החלף", "הפוך", "תהפוך", "תחליפי", "החליפי"],
    "climate_turn_off": ["כבה", "תכבה", "תכבי", "כבי", "סגור", "תסגור",
                         "תפסיק", "כיבוי", "אוף", "סגרי", "תסגרי", "תפסיקי"],
    "climate_set_temperature": ["מעלות", "טמפרטורה", "תעלה", "תוריד", "חם",
                                "שים", "תשים", "תשימי", "שימי", "הגדר",
                                "תגדיר", "תסדר", "תסדרי",
                                "קר", "מעלה", "תעלי", "תורידי",
                                "קר מדי", "חם מדי", "תחמם", "תקרר"],
    "climate_set_hvac_mode": ["קירור", "חימום", "אוטומטי", "יבש", "מאוורר"],
    "climate_set_fan_mode": ["מהירות", "נמוך", "גבוה", "בינוני", "פן"],
    "fan_turn_on": ["הדלק", "תדליק", "הפעל", "תפעיל", "און", "הדליקי", "תדליקי", "הפעילי", "תפעילי"],
    "fan_turn_off": ["כבה", "תכבה", "תכבי", "סגור", "תסגור", "אוף", "כבי", "סגרי", "תסגרי"],
    "fan_oscillate": ["תסובב", "יסתובב", "לסובב", "סיבוב", "תסובבי"],
    "cover_open": ["פתח", "תפתח", "תרים", "הרם", "תעלה", "העלה", "פתחי", "תפתחי", "תרימי", "הרימי", "תעלי", "העלי"],
    "cover_close": ["סגור", "תסגור", "תוריד", "הורד", "סגרי", "תסגרי", "תורידי", "הורידי"],
    "cover_stop": ["עצור", "תעצור", "תפסיק", "די", "עצרי", "תעצרי", "תפסיקי"],
    "cover_set_position": ["אחוז", "אחוזים", "חצי", "מחצית", "רבע"],
    "lock_lock": ["נעל", "תנעל", "תנעלי", "נעלי", "סגור", "תסגור", "סגרי", "תסגרי"],
    "lock_unlock": ["תפתח", "פתח", "תשחרר", "שחרר", "תפתחי", "פתחי", "תשחררי", "שחררי"],
    "camera_turn_on": ["הדלק", "תדליק", "הפעל", "תפעיל", "הדליקי", "תדליקי", "הפעילי", "תפעילי"],
    "camera_turn_off": ["כבה", "תכבה", "תכבי", "כבה", "כבי", "כבי"],
    "vacuum_start": ["תשאב", "שאב", "תנקה", "נקה", "תתחיל", "הפעל", "תפעיל", "תשאבי", "שאבי", "תנקי", "נקי", "הפעילי", "תפעילי"],
    "vacuum_return_to_base": ["תחזיר", "חזור", "לעגינה", "לבסיס", "לתחנה",
                              "הביתה", "תחזור", "שיחזור", "תחזירי", "חזרי", "תחזרי"],
    "vacuum_pause": ["השהה", "תשהה", "עצור", "תעצור", "תפסיק", "תשהי", "תשהי", "עצרי", "תעצרי", "תפסיקי"],
    "vacuum_set_fan_speed": ["מהירות", "חזק", "שקט", "עוצמה"],
    "switch_turn_on": ["הדלק", "תדליק", "הפעל", "תפעיל", "און", "טרן", "הדליקי", "תדליקי", "הפעילי", "תפעילי"],
    "switch_turn_off": ["כבה", "תכבה", "תכבי", "סגור", "תסגור", "אוף", "כבי", "סגרי", "תסגרי"],
    # No "שיר" here, deliberately. It is a media noun - it belongs in the
    # family table above, where it says "this sentence is about audio" - but as
    # a tool hint it ties with "תעצור" on "תעצור את השיר" and wins the tie on
    # position alone, which turns "stop the song" into a request to play one.
    # What actually separates playing a named thing from resuming is the word
    # for the *kind* of thing.
    "music_play": ["אלבום", "תקליט", "דיסק", "רצועה",
                   "סינגל", "פלייליסט", "רשימת השמעה", "רדיו", "תחנה",
                   "זמר", "זמרת",
                   "אמן", "אמנית", "להקה", "הרכב",
                   "האזן", "תאזין", "האזני", "תאזיני",
                   "ערבב", "תערבב", "ערבבי", "תערבבי"],
    "media_play": ["נגן", "תנגן", "השמע", "תשמיע", "שים", "תשים", "נגני", "תנגני", "השמיעי", "תשמיעי"],
    "media_set_volume": ["ווליום", "וליום", "קול", "עוצמה", "סאונד", "שאונד",
                         "תגביר", "תנמיך", "הגבר", "נמיך", "אחוז", "חצי",
                         "תכוון", "כוון", "תגבירי", "תנמיכי", "הגבירי", "הנמיכי", "תכווני", "כווני",
                         "יותר חזק", "יותר חלש", "תרים", "תעלה", "תוריד", "תחליש"],
    "media_mute": ["השתק", "תשתיק", "מיוט", "בשקט", "שקט", "השתקה", "השתיקי", "תשתיקי"],
    "media_next_track": ["הבא", "דלג", "תדלג", "הבאה", "דלגי", "תדלגי"],
    "media_select_source": ["ערוץ", "מקור", "ספוטיפיי", "יוטיוב", "תעביר",
                            "בלוטות'", "אייראפליי", "טלוויזיה", "תעבירי",
                            "שים", "תשים", "תחליף", "רדיו"],
    "media_pause": ["השהה", "תשהה", "עצור", "תעצור", "תפסיק", "תשהי", "תשהי", "עצרי", "תעצרי", "תפסיקי"],
    "scene_activate": ["סצנה", "סצינה", "סצנת", "תרחיש", "מצב",
                       "אווירה", "אווירת"],
    "script_run": ["סקריפט", "הרץ", "תריץ", "ניקיון", "השקיה", "תעשה",
                   "יציאה", "חזרה", "הריצי", "תריצי", "תעשי"],
    "automation_turn_on": ["אוטומציה", "אוטומציות", "הפעל", "תפעיל",
                           "הפעילי", "תפעילי", "תדליק", "תדליקי",
                           "תאפשר", "תאפשרי"],
    "automation_turn_off": ["אוטומציה", "אוטומציות", "תבטל", "בטל", "תכבה",
                            "תבטלי", "בטלי", "תכבי", "תשבית", "תשביתי"],
    "input_boolean_turn_on": ["תפעיל", "הפעל", "תדליק", "תפעילי", "הפעילי", "תדליקי"],
    "input_boolean_turn_off": ["תכבה", "תבטל", "תכבי", "בטל", "תבטלי", "בטלי"],
    "get_state": ["מה", "האם", "כמה", "דולק", "כבוי", "פתוח", "נעול", "סגור", "סגרי"],
    "get_weather": ["מזג", "אוויר", "תחזית", "גשם", "יורד", "מעונן"],
    "timer_start": ["טיימר", "תיימר", "שעון עצר", "דקות", "תזכיר", "תעמיד",
                    "ספירה", "תזכירי", "תעמידי"],
    "timer_cancel": ["בטל", "תבטל", "בטלי", "תבטלי"],
    "notify_send": ["הודעה", "תשלח", "שלח", "התראה", "הודע", "תודיע", "תשלחי", "שלחי", "הודיעי", "תודיעי", "תעדכן", "תעדכני", "כולם"],
}

# The infinitive, which is how a Hebrew speaker asks politely: "אתה יכול
# לנעול את הדלת" is the same order as "נעל את הדלת" and was the last shape the
# direction guard could not read - every lock inversion left in the shipped
# evaluation was one of these. Appended rather than written inline so that the
# lists above stay readable as what they are, one imperative per line.
for _tool, _forms in {
    "light_turn_on": ["להדליק", "להאיר", "לפתוח"],
    "light_turn_off": ["לכבות", "לסגור", "לעמעם"],
    "climate_turn_off": ["לכבות", "לסגור", "להפסיק"],
    "climate_set_temperature": ["לחמם", "לקרר", "לשים", "להגדיר"],
    "lock_lock": ["לנעול", "לסגור"],
    "lock_unlock": ["לפתוח", "לשחרר"],
    "cover_open": ["לפתוח", "להרים", "להעלות"],
    "cover_close": ["לסגור", "להוריד"],
    "switch_turn_on": ["להדליק", "להפעיל"],
    "switch_turn_off": ["לכבות", "לסגור"],
    "fan_turn_on": ["להדליק", "להפעיל"],
    "fan_turn_off": ["לכבות", "לסגור"],
    "camera_turn_on": ["להדליק", "להפעיל"],
    "camera_turn_off": ["לכבות"],
    "automation_turn_on": ["להפעיל", "להדליק", "לאפשר"],
    "automation_turn_off": ["לבטל", "לכבות", "להשבית"],
    "input_boolean_turn_on": ["להפעיל", "להדליק"],
    "input_boolean_turn_off": ["לכבות", "לבטל"],
    "vacuum_start": ["לשאוב", "לנקות", "להתחיל", "להפעיל"],
    "vacuum_pause": ["לעצור", "להפסיק", "להשהות"],
    "vacuum_return_to_base": ["לחזור", "להחזיר"],
    "media_play": ["לנגן", "להשמיע", "להמשיך"],
    "media_pause": ["לעצור", "להפסיק", "להשהות"],
    "music_play": ["לנגן", "להשמיע", "לשמוע"],
    "scene_activate": ["להפעיל", "להדליק"],
    "script_run": ["להריץ", "להפעיל"],
    "timer_start": ["לכוון", "להתחיל"],
    "timer_cancel": ["לבטל", "לעצור"],
}.items():
    TOOL_HINTS.setdefault(_tool, []).extend(_forms)

# When nothing matches - an unrelated utterance, or one whose vocabulary this
# table does not cover - offer a spread across the most common domains rather
# than nothing. The model can still return an empty call list, which is the
# refusal, so a wrong shortlist here costs a refusal rather than a wrong action.
FALLBACK: Final[list[str]] = [
    "light_turn_on", "light_turn_off", "climate_set_temperature",
    "cover_open", "get_state",
]

# Hebrew glues single-letter particles onto the front of words, so "במזגן",
# "והמזגן" and "שהמזגן" all contain the noun "מזגן". Longest first so that
# "וה" is tried before "ו".
_CLITICS: Final[tuple[str, ...]] = (
    "וכש", "ולכ", "ומה", "כשה", "לכש", "מה", "וה", "ול", "וב", "ומ", "וש",
    "כש", "שה", "ב", "ל", "כ", "מ", "ש", "ו", "ה",
)

_PUNCT: Final[str] = ".,!?;:'\"״׳()[]-–—"

# Hebrew writes five letters differently at the end of a word. Speech-to-text
# and fast typing both put the wrong variant mid-word ("םזגן" for "מזגן",
# "הםיזוג" for "המיזוג"), which breaks an exact match on the one keyword that
# decides the domain. Folding them is safe because the two forms are the same
# letter - no two distinct Hebrew words differ only in a final form.
#
# Wider phonetic folding was measured and rejected: merging א/ע, כ/ק/ח, ט/ת and
# ב/ו moved router recall by -0.4 to +0.2pp against this alone, because the
# false matches it created cost about what the repairs gained.
_FINALS: Final = str.maketrans("םןץףך", "מנצפכ")


def _fold(text: str) -> str:
    return text.translate(_FINALS)


def _variants(word: str) -> set[str]:
    """A token and the forms left after stripping Hebrew prefix particles.

    Matching whole tokens rather than substrings matters: "אור" (light) is a
    substring of "אורחים" (guests), and "חדר האורחים" is the living room, so a
    naive substring match would route every command mentioning the living room
    to the light domain.
    """
    word = _fold(word.strip(_PUNCT))
    out = {word}
    for clitic in _CLITICS:
        if word.startswith(clitic) and len(word) > len(clitic) + 1:
            out.add(word[len(clitic):])
    return out


def _tokens(query: str) -> set[str]:
    out: set[str] = set()
    for word in query.split():
        out |= _variants(word)
    return out


# One phrase index per family, over the device nouns, built once at import.
#
# `_hits` matches whole tokens, which is exactly right until speech-to-text
# drops a space: "תסגור אתהתאורה" is one token and matches nothing, so the
# utterance scored no noun and the verb sent it to the wrong domain. Five of the
# recall misses were that. The index carries the glued and de-spaced passes from
# `hebrew_text`, with the same boundary rules - which is what keeps "אור" from
# being found inside "מאוורר" the moment the boundary is relaxed.
#
# Used only as a fallback, when whole-token matching found nothing, so ordinary
# scoring is untouched and the off-topic gate keeps its calibration.
#
# The fuzzy pass stays off here, and that was measured rather than assumed.
# Letting misspelled device nouns match raised nothing (recall 98.6% -> 98.5%)
# and cost the refusal gate nearly ten points - off-topic refusal 73.9% -> 64.4%
# - because a sentence about anything at all lands within one edit of some
# device noun. Halving false refusals (3.35% -> 1.47%) does not pay for that:
# a false refusal asks the user to repeat themselves, a false actuation opens
# a blind because somebody mentioned the weather.
_NOUN_INDEX: Final[dict[str, PhraseIndex]] = {}


def _noun_index(family: str) -> PhraseIndex:
    index = _NOUN_INDEX.get(family)
    if index is None:
        index = PhraseIndex()
        index.extend(FAMILY_NOUNS.get(family, []), family)
        _NOUN_INDEX[family] = index
    return index


def _hits(keywords: list[str], toks: set[str], query: str) -> int:
    """Count matching keywords. Multi-word keys fall back to substring."""
    folded = _fold(query)
    n = 0
    for key in keywords:
        if " " in key:
            if _fold(key) in folded:
                n += 1
        # Keys are stripped the same way query tokens are, or a keyword that
        # carries punctuation can never match: "בלוטות'" and "תסוויץ'" are
        # written with a geresh, which _variants removes from the token.
        elif _fold(key).strip(_PUNCT) in toks:
            n += 1
    return n


def score_families(query: str) -> list[tuple[str, int]]:
    """Families with a non-zero score, most likely first."""
    toks = _tokens(query)
    scored = []
    for fam in FAMILY_TOOLS:
        # Nouns weigh triple: they identify the device, which is what decides
        # the domain. Verbs only separate families the nouns already reached.
        nouns = _hits(FAMILY_NOUNS.get(fam, []), toks, query)
        if not nouns and _noun_index(fam).find(query, fuzzy=False):
            nouns = 1
        score = (_NOUN_WEIGHT * nouns
                 + _hits(FAMILY_VERBS.get(fam, []), toks, query)
                 + _hits(FAMILY_WEAK.get(fam, []), toks, query))
        if score:
            scored.append((fam, score))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


def _rank_within(family: str, query: str, toks: set[str]) -> list[str]:
    """Order a family's tools, floating ones whose specific hints matched."""
    tools = FAMILY_TOOLS[family]
    return sorted(
        tools,
        key=lambda t: (-_hits(TOOL_HINTS.get(t, []), toks, query), tools.index(t)),
    )




# A question must not be able to move anything. ------------------------------
#
# Only 930 of the 18,806 generated rows are state questions - 5.0%. The model
# sees "turn on the light" seventeen times for every "is the light on", and
# with four actuation tools in a five-tool shortlist that prior wins: on the
# live installation "מה המצב של האור במטבח" ("what is the state of the kitchen
# light") turned three kitchen lights on. Retraining might shift that prior;
# not declaring the tools removes the possibility, because a tool outside the
# declared set cannot be emitted at all - the grammar forbids it.
#
# Every pattern here was measured against the whole corpus, and only patterns
# that fire on ZERO of the 15,709 actuation rows survived. Together they cover
# 82.0% of the query rows at 0.00% false positives, and no row that mixes a
# question with a command is affected: the corpus contains none.
#
# Measured and rejected:
#
#   state adjectives (דולק, פתוח, נעול)  159 query hits, but 227 actuation
#       ones - "תשאיר את האור דולק" is an order, not a question.
#   "אני רוצה"                          95 query hits, 740 actuation.
#   a PhraseIndex instead of regexes    84.2% recall at 12.12% false
#       positives: its glued pass finds האם inside the command "אם אפשר".
#
# The 18% of questions this misses are the ones with no interrogative at all
# ("התריס בחדר אוכל פתוח"), which are ambiguous in writing too. They fall
# through to the scorer below, which still keeps a query slot for them.
#
# Spaces inside the multi-word patterns are optional. Speech-to-text glues
# words together - the corpus has מההמצב for מה המצב - and the looser form
# recovers those at no cost: +0.3pp query recall, still 0.00% on actuation.
#
# Patterns are written as ordinary Hebrew and folded at import, because the
# text they run against has its final letters folded: עם becomes עמ.
_QUESTION: Final = tuple(re.compile(_fold(p)) for p in (
    r"\bמה\s*ה?מצב\b",                       # מה המצב של / מה מצב
    r"\bמה\s*קורה\s*עם\b",                   # מה קורה עם האור
    r"\bהאם\b",                              # האם האור דולק
    r"\b[תי]?בדו?ק[יו]?\b",                  # תבדוק / בדוק / תבדקי
    r"\bכמה\b",                              # כמה מעלות בסלון
    r"\bמה\s*ה?(טמפרטורה|לחות|רמת|חום)\b",
    r"מזג\s*ה?או+יר",                        # מזג האוויר, and the one-vav spelling
    r"\bתגיד[יי]?\s*לי\b",                   # תגיד לי מה קורה עם...
))


def looks_like_question(query: str) -> bool:
    """True when the sentence asks about state rather than changing it."""
    return any(rx.search(normalise(query)) for rx in _QUESTION)


# Which of the two read-only tools, and what it should look at. ---------------
#
# The model cannot tell them apart. Asked "מה המצב של האור בשירותים" with both
# declared it answered the weather, and on a 200-row probe of query utterances
# it chose get_weather for a device question again and again. The distinction
# is lexical in Hebrew, so it does not need a model.
#
# Weather words, measured over every generated row: they appear in 90.0% of the
# get_weather rows and 2.0% of the get_state ones. They are only ever consulted
# after :func:`looks_like_question` has fired, so a command cannot reach them.
# Two tiers, because two of these words name a place rather than the sky.
# "מרפסת שמש" is the balcony and "בחוץ" is the garden - both are areas in
# this project's own room lexicon. Measured over the held-out set, `שמש`
# fires on 34 sentences and not one of them asks about the weather, and
# `בחוץ` fires on 37 of which 28 do not. Left in the same tier they answered
# "מה קורה עם המאוורר במרפסת שמש" with the forecast.
_WEATHER_SKY: Final = tuple(re.compile(_fold(p)) for p in (
    r"מזג\s*ה?או+יר",
    r"\bגש[מו]",                        # גשם, גשום, יגשם
    r"\bמעונן",
    r"\bתחזית\b",
    r"\bשלג",
    r"\bרוח\b",
    r"\b(חם|קר)\s+היום\b",
    r"\bיהיה\s*(חם|קר)\b",
    r"\bאיך\s*יהיה\b",
))

# These decide only when the question names no device at all. "מה יש בחוץ"
# is the sky; "מה קורה עם התאורה בחוץ" is a lamp standing in the garden.
# שמש additionally refuses the balcony outright, because that is the only
# thing it ever names here: 34 occurrences in the held-out set, every one
# of them "מרפסת שמש", and none of them a question about the weather.
_WEATHER_PLACE: Final = tuple(re.compile(_fold(p)) for p in (
    r"(?<!מרפסת )\bשמש",
    r"\bבחוץ\b",
))

# What is being asked *about*, when it is a number rather than a thing.
# "מה הטמפרטורה בחוץ" names no device - a thermostat is a מזגן - so the
# place word is left to decide, and it says the sky. Without this the
# climate family scored on the quantity and answered from the thermostat.
_QUANTITY: Final = re.compile(_fold(r"ה?(?:טמפרטורה|טמפ|מעלות|חום|לחות)"))

# Home Assistant domains that answer a state question but own no service, so
# they are not families in the table above and never appear in a shortlist.
# Both are single-word signals in the corpus: a humidity or generic sensor is
# חיישן/לחות, and a window contact is חלון - distinct from the וילון/תריס that
# make a cover. Temperature is deliberately absent: a climate entity reports it
# too, and "מה הטמפרטורה בסלון" is answered from the thermostat.
SENSOR_NOUNS: Final[dict[str, list[str]]] = {
    "sensor": ["חיישן", "חיישנים", "לחות", "רמת לחות", "מדחום"],
    "binary_sensor": ["חלון", "חלונות"],
}

# Every other family answers about its own domain. media is the only one whose
# Home Assistant domain is not the family name.
_FAMILY_DOMAIN: Final[dict[str, str]] = {
    "light": "light", "climate": "climate", "fan": "fan", "cover": "cover",
    "lock": "lock", "camera": "camera", "vacuum": "vacuum", "switch": "switch",
    "media": "media_player", "helper": "timer",
}


def asks_about_weather(query: str) -> bool:
    """True when a question is about the sky rather than about a device."""
    folded = normalise(query)
    if any(rx.search(folded) for rx in _WEATHER_SKY):
        return True
    if not any(rx.search(folded) for rx in _WEATHER_PLACE):
        return False
    return query_domain(_QUANTITY.sub(" ", folded)) is None


def query_domain(query: str) -> str | None:
    """The Home Assistant domain a state question is asking about, or None.

    Derived from the same device nouns that route a command, so the router and
    the executor agree by construction. Right on 83.5% of the get_state rows
    from the family table alone and 97.4% with the two service-less domains
    added, against a measured 0.9% argument F1 for the model on the same slot.
    """
    toks = _tokens(query)
    for domain, nouns in SENSOR_NOUNS.items():
        if _hits(nouns, toks, query):
            return domain
    for family, _score in score_families(query):
        if family in _FAMILY_DOMAIN:
            return _FAMILY_DOMAIN[family]
    return None



def select_tool_names(query: str, limit: int = MAX_TOOLS) -> list[str]:
    """Up to ``limit`` tool names to declare for this utterance."""
    if looks_like_question(query):
        # Read-only tools and nothing else. The asymmetry is the whole reason:
        # a command misread as a question costs an unanswered sentence, while a
        # question misread as a command moves a device in someone's house.
        #
        # Which of the two read-only tools is lexical, and the model cannot do
        # it: with both declared it answered "מה המצב של האור בשירותים" with
        # the weather. Declaring one leaves the grammar no room to.
        #
        # The sky is asked about first. Weather words appear in 2.0% of the
        # device questions, but a weather question names a device noun far more
        # often than that - "יהיה חם מחר" scores the climate family - so asking
        # about the device first sent 64% of the weather rows to get_state.
        # Neither signal present - "מה המצב" and nothing else - keeps both.
        if asks_about_weather(query):
            return ["get_weather"]
        if query_domain(query):
            return ["get_state"]
        return FAMILY_TOOLS["query"][:limit]

    toks = _tokens(query)
    families = score_families(query)
    if not families:
        return FALLBACK[:limit]

    # A family counts as a genuine domain only once a device *noun* has named
    # it. Verb-only matches are near-worthless on their own, because Hebrew
    # imperatives are shared across domains - scoring "סגור מזגן" by verb alone
    # put cover_open and get_state in the shortlist and pushed climate_turn_off
    # out of it, which was 144 of the 385 first-pass misses.
    strong = [f for f, s in families if s >= _NOUN_WEIGHT] or [families[0][0]]

    # A question names a device too, so it scores that device's family on the
    # noun and loses the tool that could actually answer it: "האם האור דולק
    # בסלון" filled the shortlist with light_* and cut get_state, which the
    # grammar then forbids outright. 159 of the misses were exactly this. When
    # an interrogative fires, the query family keeps a slot whatever the noun
    # scored - it is the second-largest family in the data and costs one slot.
    if "query" not in strong and any(f == "query" for f, _ in families):
        strong.append("query")

    ranked = {fam: _rank_within(fam, query, toks) for fam in strong}
    chosen: list[str] = []

    # One slot per strong family first, so a two-domain command ("turn off the
    # light and lock the door") keeps both reachable...
    for fam in strong[:limit]:
        chosen.append(ranked[fam][0])
    # ...then families expand in score order into whatever room is left. With a
    # single domain matched this hands the whole shortlist to that domain, which
    # is what the training distribution looks like.
    for fam in strong:
        for tool in ranked[fam][1:]:
            if len(chosen) >= limit:
                break
            if tool not in chosen:
                chosen.append(tool)
    # Verb-only matches are last, and only if nothing better claimed the slot.
    if len(chosen) < limit:
        for fam, _ in families:
            if fam in strong:
                continue
            for tool in _rank_within(fam, query, toks):
                if len(chosen) >= limit:
                    break
                if tool not in chosen:
                    chosen.append(tool)
    return chosen[:limit]


REFUSE_BELOW: Final = 3

# A room is worth two: strong evidence that a sentence is about the house,
# and deliberately less than the three a device noun is worth, because a
# room names a place and not a thing to act on. See `looks_off_topic`.
ROOM_WEIGHT: Final = 2

_ROOMS: Final = PhraseIndex()
for _slug, _forms in AREA_ALIASES.items():
    _ROOMS.extend(_forms, _slug)


def looks_off_topic(query: str, threshold: int = REFUSE_BELOW) -> bool:
    """True when nothing in the utterance names a device this house controls.

    The model will not refuse. Measured across v5 through v8 on the held-out
    test set, correct refusal is 0.0% and false actuation is ~100%: an off-topic
    Hebrew sentence gets a tool call, because the grammar always permits one and
    fourteen epochs of "emit a call" overwhelm the 11.5% of the corpus that says
    otherwise. Raising the refusal share to 25% was tried and made everything
    worse (the `exp` run) - it is not a ratio problem.

    The router already holds the information the model is failing to use. Scored
    over the whole test set, the top family's score separates the two classes
    cleanly, because a real command almost always names its device and a noun is
    worth 3:

        actionable  67.2% score exactly 3, 21.8% score 4, only 3.3% below 3
        off-topic   73.9% below 3

    So the gate sits at 3, which caught 73.9% of off-topic utterances while
    refusing 3.35% of genuine ones. Threshold 2 cost almost the same (3.27%)
    and caught only 60.6%, and 4 is off a cliff - it would block 70.6% of real
    commands, since scoring exactly 3 is the *normal* case for one device noun.
    (Those four are the figures that chose the threshold, on the corpus of the
    time. The current ones are two paragraphs down.)

    The asymmetry is what justifies it. A false refusal says "I didn't
    understand" and the user repeats themselves; a false actuation opens a blind
    or unlocks a door because someone mentioned the weather. This trades the
    benign error for the dangerous one, and it is the same reasoning that put
    tool selection in a deterministic router rather than in the retrieval head.

    A named room is worth two of the three on its own. Nobody says "בסלון"
    about the pyramids, and one signal short of the bar is where the genuine
    commands were piling up - "תפעיל לי משהו בסלון" scores one for the play
    verb and was thrown away before the model ever saw it. Measured over the
    held-out set, the room bonus rescues 34 real commands (3.42% wrongly
    refused down to 1.87%) and lets three off-topic ones through, from 76.6%
    caught to 75.0%.

    All three of those are questions - "כמה עולה לשכור חניה בתל אביב" - so
    :func:`looks_like_question` has already restricted them to the two
    read-only tools and none of them can move anything. That is also why the
    bonus stops at two: at three a room passes the gate by itself, which lets
    through "הגינה של השכנים מוזנחת" and "לאיזה מוסך כדאי לקחת את האוטו" -
    neither of them a question, both handed a shortlist that can actuate.
    """
    scored = score_families(query)
    score = scored[0][1] if scored else 0
    if _ROOMS.find(query, fuzzy=False):
        score += ROOM_WEIGHT
    return score < threshold


# Hebrew negation. Deliberately a short list of *unambiguous* forms.
#
# The engine has a negation detector of its own - it sets `validation.negation`
# on the response - but its vocabulary is "don't", "dont", "do not", "never",
# "no longer": English, all of it, and it never fires on a Hebrew sentence. The
# corpus does not fill the gap either. Across all 22,000 generated rows there is
# not one example of "אל תדליק", so nothing has ever taught the model to refuse
# a negated command, and the grammar always permits a call.
#
# What the corpus does contain, 327 times, is bare לא used to *correct*: "תדליק
# את המנורה, לא לא, תכבה". Those must actuate - they are the correction family -
# so bare לא cannot be a negation marker. Every pattern below requires לא or אל
# to govern a following verb, which the correction rows never do.
_NEGATION: Final = tuple(re.compile(p) for p in (
    r"\bאל\s+[תל]",          # אל תדליק, אל להדליק
    r"\bבלי\s+ל",            # בלי להדליק
    r"\bלא\s+צריך",          # לא צריך להדליק
    r"\bלא\s+רוצה\s+ש",      # לא רוצה שתדליק
    r"\bאין\s+צורך",
    r"\bאסור\s+ל",
    r"\bעדיף\s+ש?לא\b",
))


def looks_negated(query: str) -> bool:
    """True when the speaker asked for a command *not* to happen."""
    text = " ".join(query.replace(",", " ").split())
    return any(pattern.search(text) for pattern in _NEGATION)


def select_tools(query: str, catalogue: list[dict],
                 limit: int = MAX_TOOLS) -> list[dict]:
    """Filter a full tool catalogue down to the shortlist for this utterance."""
    by_name = {t["name"]: t for t in catalogue}
    names = [n for n in select_tool_names(query, limit) if n in by_name]
    if not names:  # catalogue does not contain our names at all
        return catalogue[:limit]
    return [by_name[n] for n in names]
