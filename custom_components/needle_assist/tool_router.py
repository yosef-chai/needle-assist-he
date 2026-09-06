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

import json
import re
from typing import Any, Final

from . import hebrew_numbers
from .area_map import AREA_ALIASES
from .const import ACTIONS, CALL_OF
from .hebrew_text import PhraseIndex, _distance, normalise

MAX_TOOLS: Final = 5
_NOUN_WEIGHT: Final = 3

# What the model is offered, and what this module speaks internally.
#
# v11 collapsed the catalogue: a domain is one tool now and the behaviour is an
# ``action`` enum inside it (see ``tools/ha_tools.py``). Everything in this
# module still ranks the old per-service names - what that file calls a
# **virtual id** - because every measurement in it was taken at that
# granularity: TOOL_HINTS separates ``light_turn_on`` from ``light_turn_off``,
# ``direction.settle`` swaps between them at 1337 agreements and 0
# disagreements, and the corpus generator names them at every call site.
#
# Only :func:`select_tool_names` converts, at the last step, to the tool names
# the model actually sees. The conversion is many-to-one, which is the whole
# point: a light command used to spend three of its five slots on
# light_turn_on/off/toggle and now spends one, so a sentence naming three
# domains gets all three instead of one and a half.
TOOL_OF: Final[dict[str, str]] = {}

# Tool names grouped the way the training data groups them. Within a family the
# order is the order tools are offered when the shortlist has room to spare.
FAMILY_TOOLS: Final[dict[str, list[str]]] = {
    "light": ["light_turn_on", "light_turn_off", "light_toggle"],
    "climate": ["climate_set_temperature", "climate_set_hvac_mode",
                "climate_set_fan_mode", "climate_turn_off",
                "climate_turn_on"],
    "fan": ["fan_turn_on", "fan_turn_off", "fan_oscillate", "fan_toggle"],
    "cover": ["cover_open", "cover_close", "cover_stop", "cover_set_position"],
    # A valve is its own Home Assistant domain and its own tool. It used to be
    # reached through `cover` and widened by the executor, which worked and
    # left the model unable to say so; the noun separates the two cleanly in
    # Hebrew - ברז against תריס - so the family does too. The executor's
    # widening stays as a second chance, never as the only one.
    "valve": ["valve_open", "valve_close", "valve_stop", "valve_set_position"],
    "lock": ["lock_lock", "lock_unlock"],
    "camera": ["camera_turn_on", "camera_turn_off"],
    "vacuum": ["vacuum_start", "vacuum_return_to_base", "vacuum_pause",
               "vacuum_set_fan_speed"],
    "media": ["music_play", "media_play", "media_pause", "media_set_volume",
              "media_mute", "media_next_track", "media_previous_track",
              "media_stop", "media_select_source"],
    "switch": ["switch_turn_on", "switch_turn_off", "switch_toggle"],
    "routine": ["scene_activate", "script_run", "automation_turn_on",
                "automation_turn_off", "button_press"],
    # Timers left the helper family in v11. They were together because both
    # answered to `input_boolean`-shaped verbs and the executor had to undo the
    # confusion afterwards (see const.HELPER_IS_A_TIMER); now they are two
    # different tools, so the shortlist can simply not offer the wrong one.
    "helper": ["input_boolean_turn_on", "input_boolean_turn_off"],
    "timer": ["timer_start", "timer_cancel", "timer_pause", "timer_resume",
              "timer_add", "timer_less", "timer_status"],
    "notify": ["notify_send", "broadcast"],
    # `todo` is one of the eleven domains Home Assistant exposes to Assist by
    # default and had no route here at all.
    "list": ["list_add_item", "list_complete_item", "list_remove_item"],
    "query": ["get_state", "get_weather"],
    "datetime": ["get_date", "get_time"],
}

TOOL_OF.update({virtual: tool for virtual, (tool, _) in CALL_OF.items()})

assert not {v for names in FAMILY_TOOLS.values() for v in names} - set(TOOL_OF), (
    "a family names a virtual id the catalogue does not have")

#: How many tools a family can contribute to the shortlist. Three families
#: hold more than one: media (transport control against playing something
#: named), notify (a phone against every speaker in the house) and query
#: (devices against the sky).
TOOLS_PER_FAMILY: Final[dict[str, int]] = {
    fam: len({TOOL_OF[v] for v in names}) for fam, names in FAMILY_TOOLS.items()
}

# Device nouns. These are the strong signal: Hebrew imperatives are heavily
# overloaded (תפתח opens a blind, unlocks a door, and colloquially turns on a
# light) so the noun decides the domain and the verb only breaks ties.
FAMILY_NOUNS: Final[dict[str, list[str]]] = {
    # Each list also carries the transliterated English the code-switching
    # recipe generates ("את הלייט", "את הא.יי.סי"), because those utterances
    # are Hebrew-framed and never reach an English tool description.
    # "מנורת" is the construct form and a separate token - "מנורת שינה" never
    # matches "מנורה". "בהירות" and "עוצמת האור" name the slot rather than the
    # device, which is how the official Hebrew suite asks for it ("קבע את
    # הבהירות של מנורת שינה ל50%"); both fire on zero off-topic rows. "צבע" was
    # measured with them and rejected at 29 - "איזה צבע לצבוע את הקיר" is not a
    # light command, and the sentence that needs it names מנורת anyway.
    "light": ["אור", "אורות", "תאורה", "מנורה", "מנורות", "נורה", "נורות",
              "ספוט", "ספוטים", "לד", "דימר", "אורה", "לייט", "לייטס",
              "מנורת", "בהירות", "עוצמת האור",
              # The two colour temperatures that name no light on their own.
              # `lists/he/lights.yaml` spells four - אור נרות, לבן חם, לבן קר,
              # אור יום - and two of them already carry the word אור, which is
              # a light noun. The other two do not, so "אני רוצה לבן קר בחדר
              # רחצה" scored a room and nothing else and was refused: 11 of
              # the 156 `extras` rows in the held-out set. Zero off-topic hits.
              "לבן חם", "לבן קר"],
    # "קר מדי" / "חם מדי" name the sensation instead of the device ("בסלון קר
    # מדי, תעלה"). Without them the verb decides, and תעלה/תוריד raise and lower
    # a blind, so these routed to cover - 41 of the test misses. Two-word forms
    # again: bare "חם" and "קר" are also hvac modes ("על חם").
    "climate": ["מזגן", "מזגנים", "מיזוג", "אינוורטר", "עינוורטר", "קירור",
                "חימום", "מאייד", "טמפרטורה", "מעלות", "מחמם",
                "א.יי.סי", "אייר", "קונדישן", "קר מדי", "חם מדי",
                # `rules/he/climate.yaml` writes the rule as (טמפ|טמפרטורה),
                # so the clipped form is how Assist itself expects to be asked;
                # "מה טמפ" is in the official suite. תרמוסטט was missing
                # outright. Both fire on zero off-topic rows.
                "טמפ", "תרמוסטט"],
    "fan": ["מאוורר", "מאווררים", "פן", "ונטה", "וונטה", "מפוח", "וונטילטור"],
    # A valve opens and closes with exactly the verbs a blind does, so it is
    # reached through this family rather than through one of its own; the
    # executor widens `cover_*` to `valve.*` when the room holds no cover.
    # See const.FALLBACK_DOMAINS. All four fire on zero off-topic rows.
    # The last six are the `cover` device classes from `lists/he/covers.yaml`
    # in OHF-Voice/intents, and they were missing: "תרים את דלת החניה" scored
    # the lock family on דלת and "תפתח את הסוכך" scored nothing at all - 12
    # router-recall misses, every one of them a command Home Assistant's own
    # Hebrew list can express. Measured over all 29,623 rows, each fires on
    # zero off-topic ones. `דלת` on its own stays out: it is a lock in most
    # houses, and `slot_match` is what tells the two apart once the family is
    # settled.
    "cover": ["תריס", "תריסים", "וילון", "וילונות", "ווילון", "ווילונות",
              "שאטרס", "רפפות", "תריסול", "גגון", "סטורים", "בליינדס",
              "דלת חניה", "דלת החניה", "דלתות חניה", "דלתות החניה",
              "סוכך", "סככה", "סוככים", "סככות",
              "צילייה", "ציליה", "תריס הצללה", "תריסי הצללה",
              # A gate is a `cover` device class in Home Assistant and a lock
              # in this house's lexicon, and both readings are right - so it
              # names both families and the shortlist carries a tool from
              # each, exactly as טלוויזיה does for switch and media. The
              # plural was missing from both.
              "שער", "שערים",
              # And the same for the other two words the official Hebrew list
              # maps to a cover class while something else here already
              # claims them. A door that *opens* is a cover and a door that is
              # *locked* is a lock; a window that opens is a cover and a
              # window that is *asked about* is a binary_sensor contact.
              # Naming both families is the honest reading - the shortlist
              # carries one tool from each and the registry settles which
              # exists. Measured over all 31,519 rows, each fires on zero
              # off-topic ones.
              "דלת", "דלתות", "חלון", "חלונות"],
    # The four words that used to live in `cover` above, now naming a family
    # and a tool of their own. Deliberately NOT given cover's verbs: a valve
    # noun scores three and cover's verb scores one, so the noun already
    # settles it, and duplicating תרים/תעלה/הורד into a second family would
    # cost them their place in DECISIVE_VERBS - which the refusal gate reads.
    "valve": ["ברז", "ברזים", "שסתום", "ואלב", "ברז ראשי", "ברז המים"],
    # The plurals were missing outright, which a question is far more likely
    # to use than a command: "איזה הדלתות פתוחים" scored nothing at all and
    # the refusal gate threw it away. Measured over all 29,623 rows, both fire
    # on zero off-topic ones.
    "lock": ["דלת", "דלתות", "מנעול", "מנעולים", "נעילה", "שער", "שערים",
             "בריח"],
    "camera": ["מצלמה", "מצלמות", "מצלמת"],
    # "שיעשה סיבוב" is the elliptical "send the robot round". The whole phrase
    # is the key because bare "סיבוב" is how a fan's oscillation is asked for
    # ("תפעיל סיבוב"), and that belongs to fan_oscillate.
    # "לתחנה" and "לעמדה" are where the robot goes home. `תחנה` on its own is
    # a radio station and sits under `media` for that reason, so the docking
    # phrases are here as phrases - without them "שיחזור לתחנה" names a media
    # noun and nothing else, which is 33 of the 40 rows where the sentence and
    # the gold disagree about which family was named.
    "vacuum": ["שואב", "רובוט", "רומבה", "שואבת", "ווקום", "שיעשה סיבוב",
               # A robot mower rides this family to reach
               # `FALLBACK_DOMAINS["vacuum_start"]`, and without its own
               # noun it could not: "תפעיל את המכסחת" scored nothing for
               # vacuum and went to the automations. `דשא` is
               # deliberately absent - it is the lawn, an area in this
               # project's own room lexicon, on 398 corpus rows.
               #
               # Neither form appears anywhere in the corpus or the
               # frozen benchmark, so this cannot move a measured row;
               # it is coverage for a device the generator does not
               # simulate, exactly like `מאדה` on the switch family.
               "מכסחת", "מכסחה", "מכסחת דשא", "רובוט דשא",
               "לתחנה", "לעמדה", "לבסיס", "לעגינה"],
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
              # `HassMediaStop`. "סטופ" names nothing else in a house; zero
              # off-topic hits.
              "סטופ",
              "יותר חזק", "יותר חלש",
              "בלוטות'", "מקור", "ספוטיפיי", "יוטיוב", "אייראפליי", "ערוץ",
              # A tool hint only orders a family something else has already
              # reached, so these have to name the family themselves.
              "הבא בתור", "רשימת השמעה", "השמעה",
              # `HassMediaPrevious` had no route at all: "תחזור אחורה" and
              # "השיר הקודם" name the previous track and nothing else in a
              # house is gone back to.
              "הקודם", "הקודמת", "אחורה", "השיר הקודם",
              # A television is a `switch` in the house this corpus was written
              # for - it hangs off a smart plug - and a `media_player` in Home
              # Assistant's own tests and in most houses. Both are true, so the
              # word names both families and the shortlist carries tools from
              # each; `executor` resolves whichever entity actually exists.
              # Listing it twice does not move the refusal gate, which reads the
              # top family's score and already saw three from `switch`.
              "טלוויזיה", "טיוי"],
    # דוד / מיחם / בוילר / מחשב / טלוויזיה / מטען are the switch entities the
    # training data names, so "turn on the TV" is a switch, not a media command.
    "switch": ["שקע", "תקע", "מפסק", "שקעים", "בוילר", "דוד", "מיחם", "מחשב",
               "טלוויזיה", "טיוי", "מטען", "פלאג", "סוקט",
               # The official suite's word for a switch entity, and absent
               # here: "האם המתגים דולקים" reached no family at all.
               "מתג", "מתגים",
               # `humidifier` is one of the eleven domains Home Assistant
               # exposes to Assist by default and had no Hebrew word here at
               # all. A humidifier and a dehumidifier are the same domain and
               # the same on/off verb as a plug, so they ride the switch
               # family and the executor widens the search - see
               # const.FALLBACK_DOMAINS. "דוד שמש" was measured with them and
               # rejected at 17 off-topic hits; bare "דוד" is already here.
               "מאדה", "אדים", "מייבש"],
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
                "תריסים בבוקר", "אורות בלילה",
                # `button` and `input_button` press from this family: a
                # doorbell or a "run once" button is a routine by another
                # name, and HassTurnOn targets both domains.
                "כפתור", "הכפתור", "לחצן", "הלחצן"],
    # input_boolean helpers are all named "מצב <something>", so the two-word
    # form is the key - bare "מצב" also means "state" and would collide with
    # state queries and with scenes.
    # "ספירה" (the countdown) names a timer and nothing else, but it lived only
    # in timer_start's hints, so "תעצור את הספירה" scored no noun at all and the
    # verb handed the shortlist to media - six of the recall misses, every one
    # of them timer_cancel.
    "helper": ["דגל",
               "מצב אורחים", "מצב חופשה", "מצב לילה", "מצב שקט", "מצב חיסכון",
               "נעדר"],
    # Countdowns left `helper` in v11. They were together because both
    # answered to the same on/off verbs and the executor had to undo the
    # confusion afterwards; now they are two tools, and a family each is what
    # lets the shortlist decline to offer the wrong one.
    "timer": ["טיימר", "טיימרים", "תיימר", "שעון עצר", "תזכורת",
              "ספירה", "הספירה", "ספירה לאחור", "סטופר",
              "כמה זמן נשאר", "זמן נשאר", "כמה נשאר"],
    "notify": ["הודעה", "התראה", "נוטיפיקציה", "פוש",
               # `HassBroadcast` speaks out loud in the house rather than
               # buzzing a phone, and Hebrew names it: one announces, one
               # sends. Both tools sit in this family and the tool hints
               # below separate them.
               "הכרזה", "כריזה", "רמקולים"],
    # `todo` is one of Home Assistant's eleven default-exposed domains and had
    # no Hebrew word anywhere in this file. Two-word keys wherever the bare
    # word is ambiguous: "קניות" alone is also "כמה קניות עשית".
    "list": ["רשימה", "רשימת", "הרשימה",
             "רשימת קניות", "רשימת הקניות", "לרשימת", "מהרשימה",
             "רשימת מטלות", "רשימת משימות", "מטלות", "משימות", "טודו",
             # "תוסיף חלב לסופר" is how an Israeli says it, and it was the
             # commonest list phrasing this table could not reach: 10 of the
             # held-out list rows scored no family at all. Only the prefixed
             # form - bare "סופר" fires on 25 off-topic rows ("איזה סופר הכי
             # זול"), and "לסופר" on none.
             "לקניות", "מצרכים", "לסופר",
             # And the mirror of "לקניות", which was missing: taking something
             # *off* the list is "תוציא חלב מהקניות", and the router sent four
             # of those to the blinds. Prefixed only, on the same rule "לסופר"
             # follows - measured over all 28,233 rows it fires on 54 genuine
             # list rows, zero off-topic ones and zero rows of any other family.
             "מהקניות"],
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
    # The sensor nouns below name devices this house genuinely has and that no
    # other family claims, but they lived only in SENSOR_NOUNS - which types a
    # question after the fact and is never scored. So "האם יש חלונות פתוחים"
    # scored one for the interrogative and was refused before inference, on a
    # sentence the official Hebrew suite requires. Measured over all 20,814
    # rows, each fires on zero off-topic ones, and חלון and חיישן between them
    # rescue 15 genuine commands the gate is currently throwing away.
    # `HassGetCurrentDate` and `HassGetCurrentTime` deliberately have no nouns
    # here. They name no device, so a family score cannot separate them from
    # the world clock: measured over the corpus, "מה השעה" fires on 22
    # off-topic rows and every one of them is "מה השעה בניו יורק". The whole
    # utterance decides instead - see :func:`names_a_clock` - and
    # `looks_off_topic` consults it directly.
    "datetime": [],
    "query": ["מזג", "תחזית", "גשם", "לחות", "מחר",
              "חם בחוץ", "קר בחוץ", "חם היום", "קר היום", "חם למעלה",
              "חלון", "חלונות", "חיישן", "חיישנים", "מדחום", "רמת לחות"],
}

# Verbs. Weak signal, used only to rank families that the nouns already matched,
# or to guess when no noun matched at all.
FAMILY_VERBS: Final[dict[str, list[str]]] = {
    "light": ["הדלק", "תדליק", "תדליקי", "הדליקי", "כבה", "תכבה", "תכבי", "כבי",
              "עמעם", "תעמעם", "האר", "עמעמי", "תעמעמי"],
    "climate": ["קרר", "תקרר", "חמם", "תחמם", "תכוון", "כוון", "תכווני", "כווני"],
    "fan": ["תסובב", "יסתובב", "לסובב", "תסובבי"],
    "cover": ["תרים", "הרם", "תוריד", "הורד", "תעלה", "תפתח", "פתח", "תסגור",
              "סגור", "תרימי", "הרימי", "תורידי", "הורידי", "תעלי", "תפתחי", "פתחי",
              "תסגרי", "סגרי"],
    "lock": ["נעל", "תנעל", "תנעלי", "נעלי", "תשחרר", "שחרר", "תשחררי", "שחררי"],
    "camera": ["תצלם", "צלם", "תצלמי", "צלמי"],
    "vacuum": ["תשאב", "שאב", "לשאוב", "לנקות", "תנקה", "נקה", "שיחזור",
               "לבסיס", "לתחנה", "לעגינה", "תשאבי", "שאבי", "תנקי", "נקי"],
    "media": ["נגן", "תנגן", "השמע", "תשמיע", "השתק", "תשתיק", "תדלג",
              "האזן", "תאזין", "האזני", "תאזיני", "ערבב", "תערבב", "ערבבי",
              "תערבבי",
              "דלג", "עצור", "תעצור", "תפסיק", "השהה", "תשהה", "נגני", "תנגני",
              "השמיעי", "תשמיעי", "השתיקי", "תשתיקי", "תדלגי", "דלגי", "עצרי", "תעצרי",
              "תפסיקי", "תשהי", "תשהי"],
    "switch": ["טרן", "און", "אוף", "תסוויץ'"],
    "routine": ["הפעל", "תפעיל", "הרץ", "תריץ", "הפעילי", "תפעילי", "הריצי", "תריצי",
                "תלחץ", "לחץ", "תלחצי", "לחצי"],
    "helper": [],
    "timer": ["תזכיר", "הזכר", "תזכירי", "תעמיד", "תעמידי"],
    "notify": ["תשלח", "שלח", "הודע", "תודיע",
               "תשלחי", "שלחי", "הודיעי", "תודיעי", "תעדכן", "תעדכני",
               "תכריז", "הכרז", "תכריזי", "הכריזי", "תשדר", "לשדר"],
    # No "תוסיף" here, and it is the commonest form of the verb. Measured over
    # the corpus it fires on 21 off-topic rows - all of them "תוסיף פגישה
    # ליומן", a calendar this assistant does not have - and on zero genuine
    # ones, because every genuine list sentence names the list. Scoring it
    # would have handed those 21 an actuating tool in a shortlist the refusal
    # gate had already let through on מחר. It stays as a hint under
    # `list_add_item`, where it orders a family the noun has already chosen
    # and cannot reach one on its own.
    "list": ["הוסף", "תוסיפי", "הוסיפי", "תמחק", "מחק", "תמחקי",
             "מחקי", "תסמן", "סמן", "תסמני", "סמני", "תרשום", "רשום",
             "תרשמי", "רשמי"],
    "valve": [],
    "datetime": [],
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
#
# ``שקט`` is here for the mirror-image reason. It is a media noun, because
# "תעשה שקט" is a mute, and it is also the vacuum's quietest suction setting -
# so "שים את הרובוט על שקט" scored media 3 against vacuum 3 and lost the tie on
# table order, handing the sentence four media tools and one vacuum tool. As a
# weak term it breaks the tie the right way (vacuum 4, media 3) without ever
# carrying a sentence on its own: "תעשה שקט" is still media 3 against vacuum 1.
FAMILY_WEAK: Final[dict[str, list[str]]] = {
    "media": ["משהו"],
    "vacuum": ["שקט"],
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
    # The last six are how Home Assistant's own Hebrew suite asks for a
    # brightness or a colour - "שנה את הבהירות ... ל50 אחוז", "קבע את
    # טמפרטורת הצבע ... ל2700" - and setting either is a `turn_on`. Without
    # them the sentence named no light behaviour at all and the model's
    # `flip` stood: 8 of the 11 HassLightSet sentences ran `light.toggle`.
    # Measured over all 31,519 rows, each fires on zero off-topic ones; bare
    # `שנה` was measured with them and rejected at 91.
    # `הדלקה` and `כיבוי` are the nouns of two verbs already in these lists,
    # and the suite asks with one of them: "הדלקה של אור ראשי" is a gerund and
    # not an imperative, so nothing named a light behaviour and the model's
    # `flip` stood. They are added as a pair rather than one of them, for the
    # reason `למעלה` is kept out of FLOOR_PHRASES - keeping half of a symmetric
    # pair because only half of it was exercised is fitting the suite rather
    # than the language. Neither occurs anywhere in the 30,613 corpus rows, in
    # any clause, which is also what says they cannot move a shortlist and so
    # may land between a generation and its run. See `_QUESTION` for the same
    # argument spelled out.
    "light_turn_on": ["הדלק", "תדליק", "תדליקי", "הדליקי", "האר", "פתח",
                      "תפתח", "און", "פתחי", "תפתחי",
                      "תשנה", "תשני", "קבע", "תקבע", "קבעי",
                      "בהירות", "הבהירות", "טמפרטורת הצבע", "הדלקה",
                      # The four colour temperatures `lists/he/lights.yaml`
                      # names. "שנה את מנורת חדר השינה ללבן חם" asks for one
                      # without ever saying "טמפרטורת הצבע", so nothing named
                      # a light behaviour and `flip` stood. Measured over the
                      # corpus: 9, 15 and 22 rows say the first three and all
                      # 46 want `on`; none is off-topic, and "אור נרות"
                      # appears nowhere but is its pair.
                      "לבן חם", "לבן קר", "אור יום", "אור נרות"],
    "light_turn_off": ["כבה", "תכבה", "תכבי", "כבי", "סגור", "תסגור", "אוף",
                       "תעמעם", "עמעם", "סגרי", "תסגרי", "תעמעמי", "עמעמי",
                       "כיבוי"],
    # `טוגל` was in `switch_toggle` and `fan_toggle` and not here, which is a
    # plain omission and an expensive one: "תעשה טוגל לאור בסלון" scored no
    # toggle word, so `direction.settle_toggle` saw a sentence naming neither
    # side of the pair and left the model's answer alone - 50 rows of one
    # template, and the reason the do-verb tier below it could not ship.
    "light_toggle": ["תחליף", "החלף", "הפוך", "תהפוך", "טוגל", "תחליפי",
                     "החליפי"],
    "climate_turn_off": ["כבה", "תכבה", "תכבי", "כבי", "סגור", "תסגור",
                         "תפסיק", "כיבוי", "אוף", "סגרי", "תסגרי", "תפסיקי"],
    "climate_set_temperature": ["מעלות", "טמפרטורה", "תעלה", "תוריד", "חם",
                                "שים", "תשים", "תשימי", "שימי", "הגדר",
                                "תגדיר", "תסדר", "תסדרי",
                                "קר", "מעלה", "תעלי", "תורידי",
                                "קר מדי", "חם מדי", "תחמם", "תקרר"],
    "climate_set_hvac_mode": ["קירור", "חימום", "אוטומטי", "יבש", "מאוורר"],
    "climate_set_fan_mode": ["מהירות", "נמוך", "גבוה", "בינוני", "פן",
                             "אוטומטי", "נמוכה", "גבוהה"],
    "fan_turn_on": [ "הדלק", "תדליק", "הפעל", "תפעיל", "און", "הדליקי", "תדליקי",
                    "הפעילי", "תפעילי"],
    "fan_turn_off": [ "כבה", "תכבה", "תכבי", "סגור", "תסגור", "אוף", "כבי", "סגרי",
                     "תסגרי"],
    "fan_oscillate": ["תסובב", "יסתובב", "לסובב", "סיבוב", "תסובבי"],
    "cover_open": [ "פתח", "תפתח", "תרים", "הרם", "תעלה", "העלה", "פתחי", "תפתחי",
                   "תרימי", "הרימי", "תעלי", "העלי"],
    "cover_close": [ "סגור", "תסגור", "תוריד", "הורד", "סגרי", "תסגרי", "תורידי",
                    "הורידי"],
    "cover_stop": ["עצור", "תעצור", "תפסיק", "די", "עצרי", "תעצרי", "תפסיקי"],
    "cover_set_position": ["אחוז", "אחוזים", "חצי", "מחצית", "רבע"],
    "lock_lock": ["נעל", "תנעל", "תנעלי", "נעלי", "סגור", "תסגור", "סגרי", "תסגרי"],
    "lock_unlock": ["תפתח", "פתח", "תשחרר", "שחרר", "תפתחי", "פתחי", "תשחררי", "שחררי"],
    "camera_turn_on": [ "הדלק", "תדליק", "הפעל", "תפעיל", "הדליקי", "תדליקי", "הפעילי",
                       "תפעילי"],
    "camera_turn_off": ["כבה", "תכבה", "תכבי", "כבה", "כבי", "כבי"],
    "vacuum_start": [ "תשאב", "שאב", "תנקה", "נקה", "תתחיל", "הפעל", "תפעיל", "תשאבי",
                     "שאבי", "תנקי", "נקי", "הפעילי", "תפעילי"],
    "vacuum_return_to_base": ["תחזיר", "חזור", "לעגינה", "לבסיס", "לתחנה",
                              "הביתה", "תחזור", "שיחזור", "תחזירי", "חזרי", "תחזרי"],
    "vacuum_pause": [ "השהה", "תשהה", "עצור", "תעצור", "תפסיק", "תשהי", "תשהי", "עצרי",
                     "תעצרי", "תפסיקי"],
    # Every value of the fan_speed enum as it is actually said, because
    # this is the half of a guarded pair that carries the evidence:
    # "שים את השואב על רגיל" is a setting and nothing else in it says so.
    "vacuum_set_fan_speed": ["מהירות", "חזק", "שקט", "עוצמה", "רגיל",
                             "בינוני", "טורבו", "שקטה"],
    "switch_turn_on": [ "הדלק", "תדליק", "הפעל", "תפעיל", "און", "טרן", "הדליקי",
                       "תדליקי", "הפעילי", "תפעילי"],
    "switch_turn_off": [ "כבה", "תכבה", "תכבי", "סגור", "תסגור", "אוף", "כבי", "סגרי",
                        "תסגרי"],
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
    # "המשך"/"חדש" are how a *paused* player is asked to carry on, which is
    # `media_player.media_play` and which nothing here spelled: Home Assistant's
    # Hebrew suite asks it three ways and all three ran the wrong service.
    "media_play": [ "נגן", "תנגן", "השמע", "תשמיע", "שים", "תשים", "נגני", "תנגני",
                   "השמיעי", "תשמיעי",
                   "המשך", "תמשיך", "המשיכי", "תמשיכי", "חדש", "חדשי"],
    "media_set_volume": ["ווליום", "וליום", "קול", "עוצמה", "סאונד", "שאונד",
                         "תגביר", "תנמיך", "הגבר", "נמיך", "אחוז", "חצי",
                         "תכוון", "כוון", "תגבירי", "תנמיכי", "הגבירי", "הנמיכי",
                         "תכווני", "כווני",
                         "יותר חזק", "יותר חלש", "תרים", "תעלה", "תוריד", "תחליש"],
    "media_mute": ["השתק", "תשתיק", "מיוט", "בשקט", "שקט", "השתקה", "השתיקי", "תשתיקי"],
    "media_next_track": ["הבא", "דלג", "תדלג", "הבאה", "דלגי", "תדלגי"],
    "media_select_source": ["ערוץ", "מקור", "ספוטיפיי", "יוטיוב", "תעביר",
                            "בלוטות'", "אייראפליי", "טלוויזיה", "תעבירי",
                            "שים", "תשים", "תחליף", "רדיו"],
    # The bare imperatives beside the future-as-imperative forms already here.
    # "הפסק את המוזיקה" carried no pause hint at all and settled as a play.
    "media_pause": [ "השהה", "תשהה", "עצור", "תעצור", "תפסיק", "תשהי", "תשהי", "עצרי",
                    "תעצרי", "תפסיקי", "הפסק", "הפסיקי", "השהי"],
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
    # "בטל את הטיימר" scored one for `בטל` here, one for `בטל` under
    # input_boolean_turn_off and one for `טיימר` under timer_start - a
    # three-way tie broken by position in FAMILY_TOOLS, where input_boolean
    # sits second and this sits fourth. So the shortlist led with
    # `input_boolean_turn_off` and the model took it: 0 of the 13 HassCancelTimer
    # sentences in Home Assistant's own Hebrew suite reached `timer.cancel`,
    # and the ceiling measurement could not see it because the right tool *was*
    # declared - just not first.
    #
    # The noun belongs here for the same reason it belongs under timer_start:
    # a sentence that says both "cancel" and "timer" matches this tool twice
    # and the helper toggles once, which is the whole job of a tool hint. The
    # extra verbs are `rules/he/timers.yaml`'s own `timer_cancel` rule -
    # (בטל|בטלי|עצור|עצרי|הפסק|הפסיקי) - which is how Assist itself expects a
    # timer to be stopped. עצור and הפסק are media verbs too, and that costs
    # nothing: a hint only orders tools inside a family the score already
    # chose.
    "timer_cancel": ["בטל", "תבטל", "בטלי", "תבטלי",
                     "טיימר", "טיימרים", "תיימר", "ספירה",
                     "עצור", "עצרי", "הפסק", "הפסיקי"],
    "notify_send": [ "הודעה", "תשלח", "שלח", "התראה", "הודע", "תודיע", "תשלחי", "שלחי",
                    "הודיעי", "תודיעי", "תעדכן", "תעדכני"],
    # `HassBroadcast` -> `assist_satellite.announce`. What separates it from a
    # phone notification is the audience, and Hebrew says the audience:
    # "תכריז בכל הבית" against "תשלח לי הודעה". "כולם" moved here from
    # notify_send for that reason - "תודיע לכולם שהאוכל מוכן" is an
    # announcement, not a push.
    "broadcast": ["תכריז", "הכרז", "תכריזי", "הכריזי", "הכרזה", "כריזה",
                  "כולם", "לכולם", "בכל הבית", "תשדר", "לשדר", "ברמקולים"],
    # Which end of a cover, on the valve tool. Same words, because a tap opens
    # and closes with the verbs a blind does; the noun already chose the
    # family by the time these are read.
    "valve_open": ["פתח", "תפתח", "תרים", "הרם", "פתחי", "תפתחי", "לפתוח"],
    "valve_close": ["סגור", "תסגור", "תוריד", "הורד", "סגרי", "תסגרי", "לסגור"],
    "valve_stop": ["עצור", "תעצור", "תפסיק", "די", "עצרי", "תעצרי"],
    "valve_set_position": ["אחוז", "אחוזים", "חצי", "מחצית", "רבע"],
    # The five timer operations Home Assistant has intents for and this
    # catalogue could not express. `timer.change` is what adds or removes
    # time; `timer.start` is what resumes a paused one.
    "timer_pause": ["השהה", "תשהה", "השהי", "תשהי", "פאוזה", "להשהות"],
    "timer_resume": ["המשך", "תמשיך", "המשיכי", "תמשיכי", "חדש", "תחדש",
                     "להמשיך", "תחזיר"],
    "timer_add": ["עוד", "תוסיף", "הוסף", "להוסיף", "תאריך את", "תוסיפי"],
    "timer_less": ["תקצר", "לקצר", "תוריד", "הורד", "פחות", "תקצרי"],
    "timer_status": ["כמה", "נשאר", "מצב", "זמן", "מתי", "נותר"],
    # Shopping list and to-do list.
    "list_add_item": ["תוסיף", "הוסף", "תוסיפי", "הוסיפי", "להוסיף",
                      "תרשום", "רשום", "תרשמי", "רשמי", "תכתוב", "צריך"],
    "list_complete_item": ["תסמן", "סמן", "תסמני", "סמני", "קניתי", "לקחתי",
                           "בוצע", "השלמתי", "עשיתי", "לסמן"],
    "list_remove_item": ["תמחק", "מחק", "תמחקי", "מחקי", "למחוק", "תוריד",
                         "הסר", "תסיר", "תוציא", "להוריד"],
    # A button is pressed, not turned on. `input_button` is the same verb one
    # domain over; see const.FALLBACK_DOMAINS.
    "button_press": ["תלחץ", "לחץ", "תלחצי", "לחצי", "ללחוץ", "לחיצה",
                     "כפתור", "הכפתור"],
    # "תדליק את המזגן" - the commonest way in Hebrew to start an air
    # conditioner, and `climate.turn_on` was simply not in the map.
    "climate_turn_on": ["הדלק", "תדליק", "הפעל", "תפעיל", "און", "הדליקי",
                        "תדליקי", "הפעילי", "תפעילי", "להדליק", "להפעיל"],
    "media_stop": ["עצור", "תעצור", "תפסיק", "הפסק", "סטופ", "לעצור",
                   "עצרי", "תעצרי", "תפסיקי"],
    "media_previous_track": ["הקודם", "הקודמת", "אחורה", "חזור", "תחזור",
                             "קודם", "לאחור", "תחזרי", "חזרי"],
    "switch_toggle": ["תחליף", "החלף", "הפוך", "תהפוך", "טוגל", "תחליפי"],
    "fan_toggle": ["תחליף", "החלף", "הפוך", "תהפוך", "טוגל", "תחליפי"],
    "get_date": ["תאריך", "התאריך", "יום", "היום", "בשבוע"],
    "get_time": ["שעה", "השעה", "זמן", "עכשיו", "כרגע"],
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
    "valve_open": ["לפתוח", "להרים"],
    "valve_close": ["לסגור", "להוריד"],
    "button_press": ["ללחוץ"],
    "climate_turn_on": ["להדליק", "להפעיל"],
    "broadcast": ["להכריז", "להודיע"],
    "list_add_item": ["להוסיף", "לרשום"],
    "list_remove_item": ["למחוק", "להסיר"],
    "list_complete_item": ["לסמן"],
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

#: The verbs that open a request to play something, in every form Israelis use.
#:
#: Only consulted when the scorer found **nothing at all**, which is what makes
#: them safe here despite being shared: "שים את המזגן על 22" scores the climate
#: family and never reaches this, and "תפעיל את השואב" scores vacuum. What is
#: left is the shape the family table cannot reach by construction - a play
#: verb and a proper noun: "שימי ברי סחרוף בחדר המוגן", "תשים עידן רייכל".
#: An artist is not a device and no keyword list can hold every name.
#:
#: Measured over all 30,706 rows, on the 42 where nothing scored a family and
#: one of these appears: **33 are a media tool** and one is what the generic
#: fallback below would have got right. Thirty-three against one is not a
#: close call, and it cannot make anything worse - this branch only runs where
#: the alternative was a guess across four unrelated domains.
PLAY_VERBS: Final[tuple[str, ...]] = (
    "שים", "שימי", "תשים", "תשימי", "לשים",
    "נגן", "נגני", "תנגן", "תנגני", "לנגן",
    "השמע", "השמיעי", "תשמיע", "תשמיעי", "להשמיע",
    "האזן", "האזני", "תאזין", "תאזיני", "לשמוע",
    "ערבב", "ערבבי", "תערבב", "תערבבי",
)

#: What to offer when nothing scored but the sentence asks for something to be
#: played. `music_play` leads because the shape that reaches here is "play
#: <name>" rather than "play"; `media_play` follows for the sentences that
#: named nothing after all.
FALLBACK_MUSIC: Final[list[str]] = ["music_play", "media_play"]

# Hebrew glues single-letter particles onto the front of words, so "במזגן",
# "והמזגן" and "שהמזגן" all contain the noun "מזגן". Longest first so that
# "וה" is tried before "ו".
_CLITICS: Final[tuple[str, ...]] = (
    "וכש", "ולכ", "ומה", "כשה", "לכש", "מה", "וה", "ול", "וב", "ומ", "וש",
    "כש", "שה", "ב", "ל", "כ", "מ", "ש", "ו", "ה",
)

# Written as escapes: these two are invisible in a diff, and a pass that
# rewrites dashes in prose silently drops one of them from the class.
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


# ק and כ are one sound in Israeli Hebrew and speech-to-text swaps them freely:
# the corpus carries קבי for כבי, תדליכי for תדליקי, תקבי for תכבי.
_KAF_FOLD: Final = str.maketrans("ק", "כ")

# The particles Hebrew glues to the front of a word, as
# `hebrew_text._PREFIX_LETTERS` spells them. Only the anchored
# reading below needs them, and only to let a match begin just after one.
_PREFIX_CLITICS: Final = "ובהלכמש"


def _hits_noisy(keywords: list[str], query: str) -> int:
    """:func:`_hits` again, deaf to a lost space and to the ק/כ homophone.

    **Not a replacement for `_hits` and not for the router.** Wider phonetic
    folding was measured at the routing level and rejected - see the note above
    `_FINALS` - because a false match there opens a blind on a sentence about
    the weather. This is for the *settlers* in :mod:`direction`, which run after
    the router has chosen the family and the model has answered: the only thing
    left to decide is which of two directions, so a false match costs a
    direction and cannot cost a device. That weaker consequence is what buys
    the weaker test.

    Two readings, and a keyword counts if either finds it:

    ``anchored``  the query with every space removed, but the match must still
                  begin where a word began, give or take the prefix cluster -
                  `hebrew_text._find_despaced`'s left edge, without its right
                  edge. Giving up the right edge is the whole point: "כבהאת"
                  is the verb with the next word glued to it and has no
                  boundary at its end. Three characters is enough here.
    ``plain``     no boundary at all, four characters and up. Recovers the
                  split "לסג ור", which no word-start rule can see because the
                  match begins in the middle of the *following* token.

    The floors are where the damage is. Unanchored at three, ``כבה`` is found
    across the seam of "במוסך בהרבה" - the final ך folds to כ, the ב of the
    next word follows, and seven brightenings became switch-offs. Anchored,
    that seam is not a word start and the same three-letter key is safe.

    Measured over train, dev and test on every clause where the strict reader is
    silent on **both** sides of a toggle: anchored alone **33 agree, 0
    disagree**, plain alone **47 and 0**, together **49 and 0**. They overlap
    but neither contains the other, which is why both are here.
    """
    folded = _fold(query).translate(_KAF_FOLD)
    flat, starts, at = [], set(), 0
    for word in folded.split():
        starts.add(at)
        flat.append(word)
        at += len(word)
    joined = "".join(flat)

    n = 0
    for key in keywords:
        needle = _fold(key).strip(_PUNCT).translate(_KAF_FOLD)
        if " " in needle or len(needle) < 3:
            continue
        if len(needle) >= 4 and needle in joined:
            n += 1
            continue
        found = joined.find(needle)
        while found >= 0:
            if any(found - back in starts
                   and all(c in _PREFIX_CLITICS for c in joined[found - back:found])
                   for back in range(4)):
                n += 1
                break
            found = joined.find(needle, found + 1)
    return n


# Verbs exactly one family claims. ------------------------------------------
#
# "נקה כאן" and "דלג" name no device at all, so they scored one point for the
# verb and the refusal gate threw them away before the model ever saw them -
# both are sentences Home Assistant's own Hebrew test suite requires. But
# neither is ambiguous: `נקה` is the vacuum and nothing else in this house is
# cleaned, `דלג` is the next track and nothing else is skipped. A verb no other
# family claims names a domain exactly as decisively as a noun does, so in the
# gate it is worth what a noun is worth.
#
# Derived from FAMILY_VERBS rather than listed out, so a verb added there
# becomes decisive on its own and there is no second table to fall out of step
# with; `_AMBIGUOUS` carries the exceptions and nothing else has to.
#
# Measured over all 20,814 generated rows, per verb, the way every other
# addition to this module was measured - 119 of the 129 family-unique verbs
# fire on **zero** off-topic rows, and between them they rescue 318 genuine
# commands that the gate is currently refusing. The ten that do not are listed
# below with what rejected them; three of the ten are the prefix-stripping trap
# this module already documents for מזגן/גן.
_AMBIGUOUS: Final = frozenset((
    "סגור",    # 31 off-topic: "כמה עולה לסגור מרפסת", "הוא סגור בעניין הזה"
    "תזכיר",   # 27: "תזכיר לי איך קוראים לשחקן ההוא"
    "תשלח",    # 27: "תשלח מייל לבוס שאני חולה"
    "תעמיד",   # 19: "תעמיד פנים שאתה פיראט"
    "תחמם",    # 18: "הדוד בבית של אמא לא מתחמם" - מתחמם strips to תחמם
    "תפתח",    # 18: "תפתח לי את הראש קצת", "תפתח לי את הדפדפן"
    "תדליק",   # 15: "תדליק לי סיגריה"
    "קרר",     #  14: "איזה מקרר הכי חסכוני" - מקרר strips to קרר
    "תצלם",    #  8: "תצלם אותי"
    "שלח",     #  7: "שלח הודעה לדני בוואטסאפ"
))


def _build_decisive() -> dict[str, str]:
    """Folded verb -> the one family that claims it."""
    owners: dict[str, set[str]] = {}
    for family, verbs in FAMILY_VERBS.items():
        for verb in verbs:
            owners.setdefault(_fold(verb).strip(_PUNCT), set()).add(family)
    ambiguous = {_fold(v).strip(_PUNCT) for v in _AMBIGUOUS}
    return {
        key: next(iter(families))
        for key, families in owners.items()
        # The query family is interrogatives, not imperatives. "מה" naming only
        # that family must never carry a sentence past the gate on its own -
        # that is what `looks_like_question` is for, and it declares read-only
        # tools rather than letting anything act.
        if len(families) == 1 and "query" not in families and key not in ambiguous
    }


DECISIVE_VERBS: Final[dict[str, str]] = _build_decisive()

# What such a verb adds in the gate. A verb already scores one, so two more
# brings it to the three a device noun is worth - deliberately equal, because
# the claim is that the two signals are equally decisive, not that one wins.
DECISIVE_VERB_BONUS: Final = _NOUN_WEIGHT - 1


def names_a_domain(query: str) -> str | None:
    """The family a verb only one family claims names, if the sentence has one.

    Word order, not `_tokens` order. `_tokens` unions every word's variants
    into one set, so a sentence carrying two decisive verbs from different
    families would answer differently between runs - which the refusal gate
    would not notice, since it only asks whether there is one at all, and
    anything reading the family later would.
    """
    for word in query.split():
        for variant in _variants(word):
            family = DECISIVE_VERBS.get(variant)
            if family is not None:
                return family
    return None


#: The words that name a countdown. Measured over all 20,814 generated rows:
#: 473 sentences carry one of these, and their gold calls are `timer_start`
#: (328), `timer_cancel` (116) and `get_state` (29) - **not one** is an
#: `input_boolean`. That is what licenses `executor` to read a helper toggle
#: over a timer sentence as a timer command; see HELPER_IS_A_TIMER.
TIMER_NOUNS: Final[tuple[str, ...]] = (
    "טיימר", "טיימרים", "תיימר", "שעון עצר", "ספירה", "ספירה לאחור",
    # `HassTimerStatus` is asked without naming the thing: "כמה זמן נשאר" is
    # the one sentence in Home Assistant's Hebrew suite that this whole stack
    # could not reach, because it scores no family, names no room and carries
    # no decisive verb - so the refusal gate threw it away before inference.
    # Multi-word keys, so none of them can fire on a bare נשאר. Measured over
    # all 29,623 rows: zero off-topic hits each.
    "כמה זמן נשאר", "זמן נשאר", "כמה נשאר",
)


def names_a_timer(query: str) -> bool:
    """True when the sentence is about a countdown."""
    return bool(_hits(list(TIMER_NOUNS), _tokens(query), query))


def family_named(query: str) -> str | None:
    """The one family this sentence's nouns name, or ``None``.

    ``None`` when they name none and when they name two - a sentence
    supporting two readings settles nothing, which is the rule every table in
    this project follows and the reason "תדליק את האוטומציה תריסים בבוקר" stays
    an automation instead of becoming a blind.

    `query` is excluded from the vote deliberately: its "nouns" are
    interrogatives rather than devices, and every sentence carrying one also
    names the device it is asking about. See `direction.family_named`, which
    is where the answer is used and where the measurement lives.
    """
    tokens = _tokens(query)
    named = [family for family, nouns in FAMILY_NOUNS.items()
             if family != "query" and _hits(list(nouns), tokens, query)]
    return named[0] if len(named) == 1 else None


#: The verbs that only a speaker can answer. Pause, resume, skip, go back.
#:
#: Deliberately *not* the on/off verbs. "כבה את הטלוויזיה" is a smart plug in
#: 43 of this corpus's 59 television rows and a media player in the rest, and
#: nothing separates them; "השהה את הטלוויזיה" is a media player and cannot be
#: anything else, because a plug has no pause.
_TRANSPORT_VERBS: Final[tuple[str, ...]] = (
    "השהה", "תשהה", "השהי", "תשהי", "פאוזה", "להשהות",
    "המשך", "תמשיך", "המשיכי", "תמשיכי", "חדש", "חדשי", "תחדש", "להמשיך",
    "נגן", "תנגן", "נגני", "תנגני", "השמע", "תשמיע", "השמיעי", "תשמיעי",
    "הבא", "הבאה", "דלג", "תדלג", "דלגי", "תדלגי",
    "הקודם", "הקודמת", "אחורה", "לאחור", "חזור", "תחזור",
    # Halting is a transport verb too, and it had to be measured apart because
    # הפסק and עצור *can* mean switching something off. Paired with a media
    # noun they do not: 342 rows over train and test, **every one of them a
    # media behaviour**. Home Assistant's own "הפסיקי את הטלוויזיה" is one.
    "הפסק", "הפסיקי", "תפסיק", "תפסיקי", "להפסיק",
    "עצור", "תעצור", "עצרי", "תעצרי", "לעצור", "סטופ",
)


def names_a_transport(query: str) -> bool:
    """True when the sentence asks a *speaker* to do something only it can do.

    A transport verb and a media noun together, and neither on its own: "נגן"
    is the player as well as the imperative, and "טלוויזיה" is a smart plug as
    often as a media player in this house.

    It exists because the pair is decisive where each half is not. Home
    Assistant's own Hebrew suite asks "השהה את הטלוויזיה" and "המשך את
    הטלוויזיה", the model answers `switch_control{flip}` - the television is a
    plug in most of the corpus - and five of the suite's nine remaining
    failures were that. Measured over train and test on single-clause
    single-call rows, excluding questions and countdowns: **1,204 rows have a
    transport verb and a media noun, and every one of them is a media or music
    behaviour.**
    """
    tokens = _tokens(query)
    return bool(_hits(list(_TRANSPORT_VERBS), tokens, query)
                and _hits(list(FAMILY_NOUNS.get("media", ())), tokens, query)
                and not looks_like_question(query)
                and not names_a_timer(query))


#: The phrases that ask for the wall clock or the calendar.
#:
#: `HassGetCurrentTime` and `HassGetCurrentDate` are built-in intents that
#: Home Assistant's own Hebrew suite tests and this project could not answer at
#: all. They need their own test rather than a family score because what they
#: name is not a device: "מה השעה" scores nothing, and "כמה זמן נשאר" - which
#: is a timer, not a clock - would otherwise land here on the word זמן.
#:
#: Every key is a phrase for that reason. Bare שעה sets a countdown ("עוד
#: שעה"), bare יום is weather vocabulary ("חם היום"), and neither may reach
#: this on its own.
CLOCK_PHRASES: Final[tuple[str, ...]] = (
    "מה השעה", "השעה עכשיו", "השעה כרגע", "מה הזמן", "מה שעה",
    "מה התאריך", "איזה תאריך", "התאריך היום", "תאריך היום",
    "איזה יום", "מה היום", "יום בשבוע", "איזה יום היום",
)


#: The words a local clock question may contain, beyond the phrase itself.
#: From the five sentences Home Assistant's own Hebrew suite tests - "מה
#: השעה", "מה השעה עכשיו", "מה השעה כרגע", "מה התאריך", "מה התאריך היום" -
#: plus the ordinary Israeli variants of the same question.
_CLOCK_WORDS: Final = frozenset(
    _fold(w) for w in (
        "מה", "מהי", "שעה", "השעה", "תאריך", "התאריך", "זמן", "הזמן",
        "איזה", "יום", "היום", "שבוע", "בשבוע", "עכשיו", "כרגע", "הזה",
        "לי", "עכשו",
        # The politeness this corpus wraps every utterance in. "מה השעה אם
        # אפשר" is the same question as "מה השעה", and without these it was
        # not one: 13 of the 35 datetime rows in the held-out set failed the
        # residue test on a word like אם or יאללה.
        "אם", "אפשר", "יאללה", "קדימה", "בבקשה", "תודה", "נא", "אנא",
        "תוכל", "תוכלי", "תגיד", "תגידי", "תשמע", "תשמעי", "תקשיב",
        "תקשיבי", "רגע", "אוקיי", "אה", "אממ", "עממ", "כאילו", "נו",
    ))


def _near_filler(word: str) -> bool:
    """Is this word a one-edit misspelling of a politeness or clock word?"""
    if len(word) < 4:
        return False
    return any(_distance(word, known, 1) <= 1
               for known in _CANCEL_NOISE_FOLDED | _CLOCK_WORDS
               if abs(len(known) - len(word)) <= 1)


def names_a_clock(query: str) -> bool:
    """True when the whole utterance asks the local time or date.

    Whole-utterance rather than phrase-matching, and the reason is measured:
    "מה השעה" appears on 22 off-topic rows of this project's corpus and every
    one of them is "מה השעה בניו יורק". A world clock is not something this
    assistant knows, so the phrase alone cannot carry the intent - what
    separates the two readings is whether anything is left over once the
    politeness comes off.

    That is the same shape as :func:`looks_like_cancel`, and it costs nothing:
    all five sentences the official Hebrew suite tests are the bare question,
    and the 22 off-topic rows all name a city.

    Guarded against the countdown reading too. "כמה זמן נשאר בטיימר" is a
    timer status, not a clock, and a sentence that names a countdown is never
    this.
    """
    if not query or names_a_timer(query):
        return False
    folded = _fold(query)
    matched = [p for p in CLOCK_PHRASES if _fold(p) in folded]
    if not matched:
        return False
    # Take the phrase out first, then look at what is left. Removing it as a
    # substring rather than word by word is what handles the space
    # speech-to-text drops: "מה התאריךהיום" leaves "היום", which is a clock
    # word, where a token test leaves "התאריךהיום", which is nothing.
    rest = folded
    for phrase in sorted(matched, key=len, reverse=True):
        rest = rest.replace(_fold(phrase), " ")
    for word in rest.split():
        variants = _variants(word)
        if variants & _CANCEL_NOISE_FOLDED or variants & _CLOCK_WORDS:
            continue
        # A leftover word one edit from a filler is that filler. "מה השעה
        # כרגא", "עוקיי מה השעה כרגע", "מה השעה קרגע" - five of the benchmark's
        # datetime rows were refused outright because one letter of the
        # politeness was wrong, and the sentence is otherwise the bare
        # question this function exists to recognise.
        #
        # Safe *here* and not in the noun matching, for the reason the whole
        # function exists: what this is guarding against is "מה השעה בניו
        # יורק", and a city is not one edit from a word meaning "please".
        # The set is closed and small, the floor of four keeps two- and
        # three-letter words out, and it is measured: over the whole corpus
        # this reads a clock on **182** genuine datetime rows against 173
        # before, and on **zero** rows that are not one, unchanged.
        if any(_near_filler(v) for v in variants):
            continue
        return False
    return True


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
    # "מתי הטיימר נגמר" - when does the timer end. Without it `settle_timer`
    # never sees a question and answers by *starting* a countdown, which is
    # four benchmark rows and the worst reading available: the household asked
    # how long was left and got a new timer.
    #
    # Measured over the whole corpus: 19 clauses say it and every one is gold
    # `timer_control{query}` - **not one row that would actuate anything**. It
    # is also on 122 off-topic rows, "מתי נולד רמברנדט" and its siblings, and
    # those name no device, so they score nothing, take no room bonus and are
    # refused exactly as they were.
    r"\bמתי\b",                              # מתי הטיימר נגמר
    # `טמפ` is the abbreviation an Israeli actually says, and Home Assistant's
    # own Hebrew suite uses it - "מה טמפ" was read as an order and answered by
    # *setting* a temperature, which is the failure this whole block exists to
    # prevent, arriving through a word this corpus happens never to use.
    # `_QUANTITY` below already knew it; this table did not.
    #
    # Landing it between a corpus generation and the run that corpus feeds is
    # normally forbidden, because `Corpus.distractors` aligned three quarters
    # of the training rows to `select_tool_names` and a moved shortlist trains
    # them against candidates inference will not present. The rule exists for a
    # reason rather than as a ritual, so the reason was checked: over all
    # 30,613 rows and every clause of every one of them, the added alternative
    # newly fires **zero** times. A change that moves no row cannot break an
    # alignment.
    r"\bמה\s*ה?(טמפרטורה|טמפ|לחות|רמת|חום)\b",
    r"מזג\s*ה?או+יר",                        # מזג האוויר, and the one-vav spelling
    r"\bתגיד[יי]?\s*לי\b",                   # תגיד לי מה קורה עם...
    # "אילו אורות דולקים" is how Home Assistant's own Hebrew suite asks which
    # of a domain is on, and it was the one shape here that reached an
    # actuation tool: the sentence names a light, so it scores three and passes
    # the refusal gate, and with no interrogative matching, the shortlist
    # filled with light_*. That is precisely the failure this block exists to
    # prevent - "מה המצב של האור במטבח" turning three kitchen lights on -
    # arriving through a word this corpus happens never to use. Measured: zero
    # hits on all 18,459 actuation rows.
    r"\bאילו\b",
    # And its singular. "איזה מאווררים כבויים" is the same question asked with
    # the commoner word, and it was reaching the fan family's actuation tools:
    # the sentence names a device, so it scores three, and with no
    # interrogative matching the shortlist filled with fan_control. Measured
    # over all 29,623 rows: **zero** hits on the 24,046 actuation ones, 188 on
    # the query ones, and 47 on off-topic ones - which this makes safer rather
    # than worse, since a question can only reach the read-only tools.
    r"\bאיזה\b",
    # Three more interrogatives Home Assistant's own Hebrew suite uses and
    # this corpus never does. Both were reaching `climate.turn_off`: a
    # question about the heat outside switching the air conditioner off is
    # the exact failure this block exists to prevent, arriving through
    # words the corpus happens not to contain. Measured over all 31,519
    # rows: zero hits on every class - actuation, query and off-topic.
    r"\bמהו\b",
    r"\bמהי\b",
    r"\bה?סטטוס\b",
    # A timer is asked after by name rather than with an interrogative: "מצב
    # הטיימר", "כמה זמן נשאר". Both were reaching timer_start, which restarts
    # the very countdown somebody just asked about. Zero actuation hits.
    r"\bמצב\s*ה?טיימר",
    r"\bכמה\s*זמן\s*נשאר\b",
    # The clock and the calendar. `HassGetCurrentTime` and
    # `HassGetCurrentDate` name no device, so nothing else here fires on them
    # and they reached the actuation shortlist: "מה השעה" scored the timer
    # family on nothing at all and "איזה יום היום" scored the weather family
    # on היום. Both are unambiguous questions in Hebrew - there is no
    # imperative reading of "מה השעה" - and both measure zero hits on the
    # actuation rows.
    r"\bמה\s*ה?שעה\b",
    r"\bמה\s*ה?תאריך\b",
    r"\bאיזה\s*(יום|תאריך)\b",
))


#: The passive participles a device is described *by*. Not verbs: nothing is
#: done to a window by saying it is open.
#:
#: The other half of this list is `slot_match.SETTING_WORDS["state"]`, which
#: maps each of them to the state it names, plus the singular forms already in
#: `FAMILY_VERBS["query"]`. It cannot be read from here - `slot_match` imports
#: this module - so the three are kept in step by
#: `test_a_state_adjective_is_not_a_verb`, which fails if any of them moves.
_STATE_ADJECTIVES: Final[frozenset[str]] = frozenset(
    _fold(word) for word in (
        "דולק", "דולקת", "דולקות", "דולקים",
        "כבוי", "כבויה", "כבויות", "כבויים", "כבוים",
        "מופעלות", "מופעלים", "מכובות", "מכובים",
        "פועלות", "פועלים",
        "נעול", "נעולה",
        "סגור", "סגורה", "סגורות", "סגורים",
        "פתוח", "פתוחה", "פתוחות", "פתוחים",
    ))

#: Asking for a state to be *kept*. No family lists these, because they name
#: no family - "תשאיר את האור דולק" is an order and the only thing in it that
#: says so is the verb. They belong with the imperatives below and nowhere
#: else: a state adjective is exactly what they take as their object, so
#: without them every one of them reads as a question about that adjective.
_KEEP_AS_IS: Final[tuple[str, ...]] = (
    "תשאיר", "תשאירי", "השאר", "השאירי", "להשאיר",
    "תשמור", "תשמרי", "שמור", "שמרי", "לשמור",
)

#: Every family's imperatives, flattened - and **without the query family**,
#: whose "verbs" are the interrogatives and the adjectives above rather than
#: anything anybody does to a device.
#:
#: ``סגור`` is in both this and `_STATE_ADJECTIVES`, and that is not a mistake
#: to be resolved: in Hebrew it is both "close!" and "closed", spelled
#: identically. Being in both means a clause carrying it never reaches
#: :func:`_asks_about_a_state`, which is the safe reading - the model's own
#: answer stands rather than the gate guessing.
_IMPERATIVES: Final[tuple[str, ...]] = tuple(sorted(
    {verb for family, verbs in FAMILY_VERBS.items() if family != "query"
     for verb in verbs} | set(_KEEP_AS_IS)))


def _asks_about_a_state(query: str) -> bool:
    """A device described by its state, with nothing asked of it.

    Hebrew forms a yes/no question with intonation and no interrogative at all,
    and written down "החלון פתוח" is exactly a statement. `_QUESTION` looks for
    האם, איזה, מה - and there is none - so fifteen benchmark rows reached the
    model with the whole catalogue in front of them and came back as
    `cover_control`, `lock_control`, `light_control`: a question about a window
    answered by opening it.

    What separates the two readings is not the interrogative, it is the
    **verb** - or rather the absence of one. ``פתוח`` is a passive participle
    and describes a window; ``תפתח`` is an imperative and opens it. A sentence
    carrying the first and none of the second asks for nothing.

    **The adjectives are matched on bare tokens**, and that is the rule rather
    than an optimisation. `_tokens` strips the Hebrew clitics, and stripping the
    ל of the infinitive ``לפתוח`` leaves ``פתוח`` - 1,051 plain orders read as
    questions, "אתה יכול לפתוח את האורות" among them. An adjective is not
    introduced by a preposition, so it has no business being clitic-stripped.

    Measured per clause over the whole corpus: **981 agree, 1 disagree**, the
    one being "כצי פתוח" where speech noise hid the half from the level guard.
    80 of the 981 are clauses `_QUESTION` does not reach today.

    It also fires on 58 of the 4,462 off-topic rows, and that cost is real but
    bounded: :func:`looks_like_question` hands them the two read-only tools and
    nothing else, so the worst of them is a state report where a refusal was
    wanted. Nothing can be switched by one.
    """
    bare = {_fold(word.strip(_PUNCT)) for word in query.split()}
    if not bare & _STATE_ADJECTIVES:
        return False
    if _hits(list(_IMPERATIVES), _tokens(query), query):
        return False
    # "חצי פתוח" is not a question about a blind, it is an instruction to put
    # it at fifty, and nobody asks whether something is half open. Ten corpus
    # clauses say it and all ten are `cover_control`.
    return hebrew_numbers.percent_in(query) is None


def looks_like_question(query: str) -> bool:
    """True when the sentence asks about state rather than changing it."""
    if any(rx.search(normalise(query)) for rx in _QUESTION):
        return True
    return _asks_about_a_state(query)


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
#: A door with nothing qualifying it. See :func:`query_domain`.
_BARE_DOOR: Final = re.compile(r"(?<![א-ת])ה?דלת(?:ות)?(?![א-ת])")
_GARAGE_DOOR: Final = re.compile(r"דלת(?:ות)?\s+ה?חני")

SENSOR_NOUNS: Final[dict[str, list[str]]] = {
    "sensor": ["חיישן", "חיישנים", "לחות", "רמת לחות", "מדחום"],
    "binary_sensor": ["חלון", "חלונות"],
}

# Every other family answers about its own domain. media is the only one whose
# Home Assistant domain is not the family name.
_FAMILY_DOMAIN: Final[dict[str, str]] = {
    "light": "light", "climate": "climate", "fan": "fan", "cover": "cover",
    "lock": "lock", "camera": "camera", "vacuum": "vacuum", "switch": "switch",
    "media": "media_player", "timer": "timer", "valve": "valve",
    "helper": "input_boolean",
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
    # A bare door, *asked about*, is a lock. `דלת` deliberately names both
    # families - see FAMILY_NOUNS["cover"], where the reasoning is that a door
    # which opens is a cover and a door which is locked is a lock - and for a
    # command that is right, because the registry settles which the house has.
    # A question has no registry to fall through to: it types the domain and
    # answers about it, so the tie has to break somewhere, and every one of the
    # 98 disagreements this slot had was this word. Measured over all 28,233
    # corpus rows it takes the domain from 1,371 right / 98 wrong to 1,467
    # right / 2 wrong. The garage door keeps its cover reading, because
    # "דלת החניה" names the class outright.
    if _BARE_DOOR.search(query) and not _GARAGE_DOOR.search(query):
        return "lock"
    for family, _score in score_families(query):
        if family in _FAMILY_DOMAIN:
            return _FAMILY_DOMAIN[family]
    return None



def select_virtual_names(query: str, limit: int = MAX_TOOLS) -> list[str]:
    """Up to ``limit`` **virtual ids**, best first.

    The old ``select_tool_names``, unchanged apart from its name. Every
    keyword table in this module discriminates at this granularity - the hints
    that separate ``light_turn_on`` from ``light_turn_off`` are the whole
    content of :data:`TOOL_HINTS` - so the ranking is done here and the
    conversion to what the model is shown happens one function down.
    """
    # The clock leads, ahead of the interrogative test. "מה הזמן עכשיו" and
    # "מה היום" are questions that `_QUESTION` does not match and cannot
    # safely be made to - `\bמה\s*היום\b` collides with the weather rows this
    # corpus is full of - so a clock sentence reached them only when some
    # *other* pattern happened to fire. Seven of the nineteen datetime rows in
    # the held-out set failed that way.
    #
    # It is allowed to lead because it is exact rather than heuristic: the
    # whole utterance has to be the question plus politeness, which is what
    # keeps "מה השעה בניו יורק" out. See :func:`names_a_clock`.
    if names_a_clock(query):
        return FAMILY_TOOLS["datetime"][:limit]

    if looks_like_question(query):
        # Read-only tools and nothing else. The asymmetry is the whole reason:
        # a command misread as a question costs an unanswered sentence, while a
        # question misread as a command moves a device in someone's house.
        #
        # Which of the read-only tools is lexical, and the model cannot do it:
        # with two declared it answered "מה המצב של האור בשירותים" with the
        # weather. Declaring one leaves the grammar no room to.
        #
        # The clock is asked first, because "מה השעה" and "איזה יום היום" name
        # no device and no sky and would otherwise fall through to a state
        # query with no domain to look at.
        # The sky is asked about first here - the clock was settled above.
        # Weather words appear in 2.0% of the device questions, but a weather
        # question names a device noun far more often than that - "יהיה חם מחר"
        # scores the climate family - so asking about the device first sent 64%
        # of the weather rows to get_state.
        # Neither signal present - "מה המצב" and nothing else - keeps both.
        if asks_about_weather(query):
            return ["get_weather"]
        # A question about a countdown is a timer status, not a device state:
        # "כמה זמן נשאר" has no domain for get_state to look at, and answering
        # it with `timer.start` restarts the countdown that was asked about.
        if names_a_timer(query):
            return ["timer_status"]
        if query_domain(query):
            return ["get_state"]
        return FAMILY_TOOLS["query"][:limit]

    toks = _tokens(query)
    families = score_families(query)
    if not families:
        # A play verb and a proper noun. See PLAY_VERBS: 33 of the 42 such
        # sentences in the corpus are a media tool, against one for the
        # generic fallback.
        if _hits(list(PLAY_VERBS), toks, query):
            return FALLBACK_MUSIC[:limit]
        return FALLBACK[:limit]

    # A family counts as a genuine domain only once a device *noun* has named
    # it. Verb-only matches are near-worthless on their own, because Hebrew
    # imperatives are shared across domains - scoring "סגור מזגן" by verb alone
    # put cover_open and get_state in the shortlist and pushed climate_turn_off
    # out of it, which was 144 of the 385 first-pass misses.
    strong = [f for f, sc in families if sc >= _NOUN_WEIGHT] or [families[0][0]]

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
    # ...then families expand in score order into whatever room is left.
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


def _to_tools(virtuals: list[str], limit: int) -> list[str]:
    """Virtual ids -> the tool names that carry them, in order, deduplicated."""
    out: list[str] = []
    for virtual in virtuals:
        tool = TOOL_OF.get(virtual, virtual)
        if tool not in out:
            out.append(tool)
        if len(out) >= limit:
            break
    return out


def select_tool_names(query: str, limit: int = MAX_TOOLS) -> list[str]:
    """Up to ``limit`` **tool names** to declare for this utterance.

    What the model is actually shown. Ranking happens over virtual ids in
    :func:`select_virtual_names`; this maps them through :data:`TOOL_OF` and
    drops the duplicates that mapping creates.

    Deduplication is the point rather than a side effect. In v10 a plain light
    command spent three of its five slots on light_turn_on, light_turn_off and
    light_toggle, so a sentence naming three domains could not fit them: the
    shortlist held the first domain's whole family and one tool of the second.
    One tool per domain means the same five slots hold five domains, and the
    choice those three slots used to express is now the ``action`` enum inside
    one of them - constrained by the same grammar, decided by the same
    :mod:`direction` guard.

    Filling stays greedy in family-score order, and a family's *ranked* virtual
    ids are walked rather than its declared order, so that where a family maps
    to more than one tool - media, notify, query - the tool whose hints matched
    leads. That is what keeps "תנגן את האלבום" declaring music_play first and
    "תפסיק" declaring media_control first.
    """
    if (names_a_clock(query) or looks_like_question(query)
            or not score_families(query)):
        return _to_tools(select_virtual_names(query, limit), limit)

    toks = _tokens(query)
    families = score_families(query)
    strong = [f for f, sc in families if sc >= _NOUN_WEIGHT] or [families[0][0]]
    if "query" not in strong and any(f == "query" for f, _ in families):
        strong.append("query")
    ranked = {fam: _rank_within(fam, query, toks) for fam in strong}

    chosen: list[str] = []

    def take(virtual: str) -> bool:
        tool = TOOL_OF.get(virtual, virtual)
        if tool in chosen or len(chosen) >= limit:
            return False
        chosen.append(tool)
        return True

    # One tool per strong family first...
    for fam in strong:
        if len(chosen) >= limit:
            break
        for virtual in ranked[fam]:
            if take(virtual):
                break
    # ...then the families that map to more than one tool expand into whatever
    # room is left.
    for fam in strong:
        for virtual in ranked[fam]:
            if len(chosen) >= limit:
                break
            take(virtual)
    # Verb-only matches are last, and only if nothing better claimed the slot.
    for fam, _ in families:
        if fam in strong or len(chosen) >= limit:
            continue
        for virtual in _rank_within(fam, query, toks):
            if take(virtual):
                break
    return chosen[:limit]


REFUSE_BELOW: Final = 3

# A room is worth two: strong evidence that a sentence is about the house,
# and deliberately less than the three a device noun is worth, because a
# room names a place and not a thing to act on. See `looks_off_topic`.
ROOM_WEIGHT: Final = 2

_ROOMS: Final = PhraseIndex()
for _slug, _forms in AREA_ALIASES.items():
    _ROOMS.extend(_forms, _slug)

# Things that contain devices and are not this house. A car has an air
# conditioner, a bag has a lock, a phone has a camera, a bicycle has a lock,
# a computer has a fan - so the noun scores three and the utterance reaches
# the gate looking exactly like a command. The preposition is what carries
# the meaning, which is why the forms here are prefixed and specific: "ברכב"
# is *in the car*, while a bare "רכב" would also match "לרכב" (to ride), and
# "לאוטו" measures 35 genuine orders against zero off-topic ones.
#
# The list is a closed category rather than a set of phrases lifted off the
# corpus, so siblings of the ones that fire are in it too ("בנייד", "בלפטופ",
# "של המכונית") even though nothing in this corpus says them. Leaving those
# out is what would be fitting the corpus - the same argument that keeps
# "למעלה" out of FLOOR_PHRASES, run in the other direction.
#
# It is consulted by the sentence gate only. Running it per clause as well was
# the obvious next step and was measured instead: of the 1,610 rows that split
# into more than one clause, **zero** have a clause naming a foreign container,
# so the rule would correct nothing and does not ship. See
# `clause_names_nothing`, which stays the narrower test it was written as.
NOT_THIS_HOUSE: Final[tuple[str, ...]] = (
    "ברכב", "באוטו", "במכונית", "של הרכב", "של האוטו", "של המכונית",
    "בתיק", "של התיק",
    "בטלפון", "של הטלפון", "בנייד", "בסלולרי",
    "לאופניים", "של האופניים",
    "במחשב", "של המחשב", "בלפטופ",
    "בבית של", "אצל אמא", "אצל השכן", "אצל השכנים",
    "של השכן", "של השכנים",
)

# Asking *about* a song is not asking for one, and the gate cannot see the
# difference by scoring words: "מי שר את השיר על האור בקצה המנהרה" names a
# song and a light and scores well past the bar, so all twelve of the corpus's
# phrasings of it were handed `media_control` and came back as `next` - the
# assistant skipping a track at somebody asking a trivia question.
#
# A closed category rather than the phrases the corpus happens to say, the same
# way NOT_THIS_HOUSE is written: the siblings are in it too. Measured over all
# 31,519 rows, the list is matched by 208 refusals and by **zero** rows that
# want a call, so it answers the gate outright rather than adjusting a score -
# the same licence :func:`names_a_clock` has, and for the same reason.
ASKS_ABOUT_CONTENT: Final[tuple[str, ...]] = (
    "מי שר", "מי שרה", "מי כתב", "מי הלחין", "מי מנגן",
    "מי הזמר", "מי הזמרת", "מי המבצע", "מי הלהקה",
    "של מי השיר", "איך קוראים לשיר", "מה שם השיר",
)

# The same argument, widened past songs, and it is the piece `looks_off_topic`
# names as the remaining work: "twenty-three of the 285 can still reach a tool
# that moves something, and every one of them is a sentence whose device noun
# is real and whose meaning is not". Scoring words cannot separate them,
# because the nouns really are this house's - a lamp, a blind, a computer, a
# television - and the sentence is a question about the world that happens to
# mention one.
#
# What they share is a *shape*, and Hebrew marks each one plainly:
#
#   a price          כמה עולה מזגן חדש לסלון
#   a purchase       כדאי לקנות שואב אבק רובוטי
#   an errand        כמה זמן לוקח להתקין תריסים חשמליים
#   the impersonal   איך מכבים מחשב שנתקע      <- plural, never an imperative
#   a comparison     מה ההבדל בין מזגן אינוורטר לרגיל
#   a request to     תמליץ לי על מוזיקה לריצה
#   compose or pick  כתוב לי שיר על החתול שלי
#   the assistant    אתה אדם או מחשב
#   the world's      מה מזג האוויר הטיפוסי באלסקה בחורף
#   listings         מה יש בטלוויזיה הערב
#   a name, a body   אור זה שם יפה לילדה / יש לי חום שלושים ושמונה מעלות
#   another app      שלח הודעה לדני בוואטסאפ / תוסיף פגישה ליומן מחר
#
# `תכתוב לי` is deliberately absent and `כתוב לי` is not: the prefixed form is
# how this corpus adds to a list - "תכתוב לי בננות לרשימת המצרכים" - and it is
# said by 37 rows that want `list_edit`. `לילה טוב` went the same way, on 66
# rows naming the playlist. Closed categories are written with their siblings
# in, so the forms the corpus never says are here too.
#
# Measured over all 40,631 rows: matched by 700 of the 4,462 refusals and by
# **zero** of the 36,169 rows that want a call, which is the licence to answer
# the gate outright rather than adjust a score.
ASKS_ABOUT_THE_WORLD: Final[tuple[str, ...]] = (
    "כמה עולה", "כמה יעלה", "כדאי לקנות", "שווה את הכסף", "יש הנחה",
    "איפה יש", "איפה קונים",
    "כמה זמן לוקח",
    "איך מכבים", "איך מדליקים", "איך מתקינים", "איך עושים",
    "איך פותחים", "איך סוגרים", "איך מחליפים",
    "מה ההבדל בין", "מה עדיף",
    "תמליץ", "המלץ", "כתוב לי", "כתבי לי",
    "אתה אדם",
    "מה יש בטלוויזיה", "הטיפוסי",
    "זה שם", "יש לי חום",
    "בוואטסאפ", "בווטסאפ", "ליומן",
)


_ABOUT_CONTENT: Final = re.compile(
    "(?<![א-ת])(?:"
    + "|".join(re.escape(normalise(_p))
               for _p in ASKS_ABOUT_CONTENT + ASKS_ABOUT_THE_WORLD)
    + ")(?![א-ת])")


_NOT_HERE: Final = re.compile(
    "(?<![א-ת])(?:"
    + "|".join(re.escape(normalise(_p)) for _p in NOT_THIS_HOUSE)
    + ")(?![א-ת])")


#: The mirror of the two categories above, and it answers the gate the other
#: way. Israelis ask for music by asking for *something*: "שים משהו",
#: "תפעילי משהו טוב", "ערבבי לנו משהו". `משהו` is a pronoun, so the sentence
#: names no device, scores nothing for any family and was refused - 22 genuine
#: media requests on this corpus, and it is the one non-typo class left in the
#: gate's false refusals.
#:
#: The verb is what makes it safe, and the corpus draws the line sharply:
#: **196 utterances pair `משהו` with a play-or-put verb and every one of them
#: wants a call**, while the 84 that say `משהו` and want none are all "תספר לי
#: משהו על הדינוזאורים" - tell me *about* something - which names no such verb
#: and stays refused.
_SOMETHING: Final = re.compile(r"(?<![א-ת])משהו(?![א-ת])")

_PUT_SOMETHING_ON: Final[tuple[str, ...]] = (
    *TOOL_HINTS.get("media_play", ()), *TOOL_HINTS.get("music_play", ()),
    "הפעל", "תפעיל", "הפעילי", "תפעילי", "להפעיל",
    # The feminine and infinitive of `שים`, which the router's hint list does
    # not carry because it never needed them to *route*.
    "שימי", "תשימי", "לשים",
)


def asks_for_something(query: str) -> bool:
    """True when the sentence asks for *something* to be put on."""
    if not query or not _SOMETHING.search(normalise(query)):
        return False
    return bool(_hits(list(_PUT_SOMETHING_ON), _tokens(query), query))


def _sounds_like_an_order(query: str) -> bool:
    """Something to do, rather than something being reported.

    Not which domain the sentence is about - whether it asks for an action at
    all. `names_a_domain` was the obvious test and it is the wrong one:
    "תפתח" is claimed by covers and by locks, so it is not decisive, and
    "תפתח את השער לאופניים" is an ordinary thing to say to a house that has a
    gate. Nothing in the corpus says it, which is precisely why the guard has
    to be written from the language rather than from the rows.

    What decides is the *shape* of the verb, not its presence. Israeli
    Hebrew gives an order with the second-person future - תפתח, תדליק,
    תסגור - or with the ה-imperative - הדלק, הפעל - and both are visible in
    the first letter. Reporting verbs are not: "קונים", "מתחמם", "מטפטף".
    Testing for any family verb instead was measured and spared eight
    off-topic rows for nothing; this spares none of the 173 and still keeps
    every imperative.
    """
    for word in normalise(query).split():
        if not word.startswith(("ת", "ה")):
            continue
        if any(_hits(FAMILY_VERBS.get(fam, []), _variants(word), word)
               for fam in FAMILY_TOOLS):
            return True
    return False


def asks_for_a_reminder(query: str) -> bool:
    """"תזכיר לי בעוד שלוש דקות" - an order that names no device.

    A countdown is the one thing this house does that has no device noun, so
    no family score can reach it and the gate refuses the sentence before the
    model ever sees it. Both halves of the test are needed and both are exact:
    the router must offer **only** the timer, so there is nothing else the
    sentence could be asking for, and the sentence must say **how long**,
    which is what separates a countdown from "תזכיר לי לקנות חלב" - a to-do
    this house has no tool for.

    Measured over the 27,210 rows of the corpus the shipping adapter was
    trained on. Of the 3,384 rows the gate refuses, this **rescues 53 that
    gold gives a call to and lets through none that gold refuses.** The looser
    version - any sentence whose shortlist holds exactly one tool - rescues 57
    and leaks 450, which is the gate's whole purpose undone.
    """
    from . import slot_match  # noqa: PLC0415 - slot_match imports this module

    return (select_tool_names(query, MAX_TOOLS) == ["timer_control"]
            and bool(slot_match.duration_from(query)))


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

    Exempting every question from the gate outright was measured and rejected,
    and the numbers are worth keeping because the idea is a tempting one. It is
    safe - of the 283 off-topic rows it lets through, **zero** are handed a tool
    that can call a service, because :func:`looks_like_question` has already
    restricted them to the two read-only ones - but it takes correct refusal
    from 75.4% to 63.4%, and answering the weather at somebody discussing house
    prices is still a wrong answer. The sentences it was meant to rescue are
    reached instead by naming their devices: see the sensor nouns in
    FAMILY_NOUNS["query"], which cost nothing because a window contact really is
    a device this house has.

    Re-measured on the v12 corpus, because the plan requires the threshold be
    chosen again whenever the corpus moves and v11 added seven families. It
    did not move - 3 is still the only habitable value - and the curve is worth
    keeping because it shows why so plainly (`eval/refuse_curve.py`, 3,975
    held-out rows, 285 of them off-topic):

        threshold   off-topic refused   orders refused   let through and
                                                         able to actuate
            1             42%               0.2%               55
            2             69%               0.4%               37
            3             85%               0.9%               23
            4             94%              16.0%                0
            5             99%              18.9%                0

    Four buys the last eleven points of refusal by throwing away one order in
    six, which is not a trade, it is a broken assistant: scoring exactly three
    is the normal case for a sentence with one device noun, so the threshold
    lands on top of the whole actionable distribution rather than beside it.

    The fourth column is where the remaining work is, and NOT_THIS_HOUSE is
    the first instalment: it took the row of threshold 3 from 75% refused and
    51 actuating to 85% and 23, at exactly zero cost to the orders column.
    Twenty-three of the 285 can still reach a tool that moves something, and
    every one of them is a sentence whose device noun is real and whose
    meaning is not - "מה יש בטלוויזיה הערב", "איך מכבים מחשב שנתקע". Those are
    not reachable by scoring words, which is what the hard negatives were
    written to demonstrate.
    """
    # The clock names no device and never will, so no family score can reach
    # it. The test is the whole utterance and it is exact - see
    # :func:`names_a_clock` - which is why it is allowed to answer the gate
    # outright rather than adding to a score.
    if names_a_clock(query):
        return False
    # And the other sentence that names no device and is still an order.
    # See :func:`asks_for_something`.
    if asks_for_something(query):
        return False
    # And its mirror: a question about who performed something, what a thing
    # costs or how one is installed is about the world, not about this house.
    # See :data:`ASKS_ABOUT_CONTENT` and :data:`ASKS_ABOUT_THE_WORLD`.
    if _ABOUT_CONTENT.search(normalise(query)):
        return True
    scored = score_families(query)
    score = scored[0][1] if scored else 0
    if _ROOMS.find(query, fuzzy=False):
        score += ROOM_WEIGHT
    # A verb only one family claims is as decisive as a device noun; see
    # DECISIVE_VERBS for the per-verb measurement that decided which qualify.
    if names_a_domain(query):
        score += DECISIVE_VERB_BONUS
    elif (_NOT_HERE.search(normalise(query))
          and not _sounds_like_an_order(query)):
        # The noun is real but the thing it names is not in this house, so it
        # is worth what a noun is worth and no more. The scope cancels the
        # noun rather than vetoing the sentence, so a second and domestic
        # device noun still carries it over the bar.
        #
        # Measured over all 30,613 rows: the phrases are matched by 173 of
        # the 3,574 refusals and by **zero** of the 27,039 rows that want a
        # call, and not one of the 173 is phrased as an order - so the guard
        # costs nothing here and still covers the case the corpus does not
        # contain. See :func:`_sounds_like_an_order`.
        score -= _NOUN_WEIGHT
    if score >= threshold:
        return False
    # About to refuse, and one order still has no device noun to score: a
    # countdown. Asked here rather than beside the other two exceptions
    # because it runs the router, and on the path that passes this gate the
    # router runs again a moment later anyway - so it is asked only where the
    # answer would otherwise be "no". See :func:`asks_for_a_reminder`.
    return not asks_for_a_reminder(query)


def clause_names_nothing(clause: str) -> bool:
    """True when a *clause* of a longer sentence names nothing this house has.

    The sentence gate above runs once, on the whole utterance, and then
    `clause_split` cuts the sentence up and every piece goes to the model. So a
    sentence that is half context and half order had its context half answered
    with a tool call, and the corpus is full of that shape:

        "צריך מים מינרלים, תוסיפי לרשימה"     -> climate_control{temp} + list_edit
        "כבר לקחתי שוקולד, תסמן לרשימת הקניות" -> cover_control{close}  + list_edit

    The first clause of each is a *statement* - what is needed, what was already
    done - and Hebrew marks it plainly: it names no device, no room and no
    family this house controls. The second is the order. Answering the first
    with a call is a device moving because somebody said what they needed.

    The test is deliberately narrower than :func:`looks_off_topic`, which
    refuses anything scoring under three. A clause scoring one or two still
    named *something* - "תזכיר לי בעוד שעה" scores one for the countdown - and
    those must reach the model. This fires only on a clause whose family score
    is empty, which is the machine-readable form of "nothing here is ours".

    Measured over all 28,233 corpus rows, on the 1,368 that split into more
    than one clause:

        clauses dropped, the row still able to meet its gold   303
        clauses dropped, a gold call lost                        2

    Both of the two are word-merge speech noise - "תעשי את האורבגראז'" for
    "תעשי את האור בגראז'", "האינוורטראצל" for "האינוורטר אצל" - where the merge
    swallows the device noun, so the clause really does name nothing readable
    and the model could not have read it either. The failure direction is the
    benign one this whole layer is built on: a dropped clause does nothing,
    where the bug it replaces moved a blind.

    The caller never drops every clause. A sentence that passed the gate above
    holds an order somewhere, and if each piece looks empty on its own the
    split is what is wrong, not the sentence - so the whole thing goes to the
    model as it did before.
    """
    return not score_families(clause)


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


# "Never mind". ---------------------------------------------------------------
#
# `HassNevermind` is one of Home Assistant's built-in intents and its whole job
# is to do nothing: the speaker started a request and withdrew it. The official
# Hebrew suite spells it with six words - לא משנה, ביטול, עצור, עצרי, בטל, בטלי -
# and four of the six were already answered correctly here, by accident: they
# name no device, so the refusal gate threw them away and the household heard
# "לא הבנתי מה לעשות". That is the right *action* and the wrong *reason*, and
# the reason started to matter the moment `עצור` became a decisive media verb,
# because then a bare "עצור" paused the music instead.
#
# What separates the two readings is not the verb but whether it governs
# anything. "עצור את המוזיקה" names what to stop; bare "עצור" does not, and in
# Hebrew a transitive imperative standing alone is a withdrawal, not an order.
# So the rule is deliberately about the *whole utterance*: strip politeness and
# fillers, and if nothing is left but cancel words, nothing happens.
#
# Measured over all 20,814 generated rows: fires on **zero** actuation rows -
# the corpus has no bare-imperative row, since every generated command names its
# device - and on 4 off-topic ones, which it refuses anyway.
_CANCEL: Final = frozenset((
    "משנה", "ביטול", "עצור", "עצרי", "בטל", "בטלי",
    # Ordinary Israeli withdrawals the official list does not carry. Each is
    # only ever reached as a whole utterance, so none can swallow a command.
    "עזוב", "עזבי", "שכח", "שכחי", "תשכח", "תשכחי", "מזה", "די", "מספיק",
    "כלום", "התחרטתי",
    # "לא צריך" is deliberately absent: `looks_negated` already owns it, and
    # bare "לא" is the correction word this corpus uses 327 times mid-sentence.
))

# Politeness and hesitation that can surround a bare cancel without changing it.
# The first seven are `skip_words` from the official `sentences/he/_common.yaml`;
# the rest are the fillers this corpus generates around every utterance.
_CANCEL_NOISE: Final = frozenset((
    "אנא", "נא", "בבקשה", "תוכל", "תוכלי", "אפשר", "יכול", "יכולה", "אתה", "את",
    "אה", "אממ", "כאילו", "נו", "רגע", "אוקיי", "תודה", "תשמע", "תשמעי",
    "תקשיב", "תקשיבי", "לי", "לא", "כבר",
))


_CANCEL_FOLDED: Final = frozenset(_fold(w).strip(_PUNCT) for w in _CANCEL)
_CANCEL_NOISE_FOLDED: Final = frozenset(
    _fold(w).strip(_PUNCT) for w in _CANCEL_NOISE)


def looks_like_cancel(query: str) -> bool:
    """True when the whole utterance is a withdrawal and nothing else.

    `HassNevermind`. Distinguished from a real command by governing nothing:
    "עצור" is a withdrawal, "עצור את המוזיקה" is a media stop.

    Tested per word rather than against :func:`_tokens`, which unions the
    prefix-stripped variants of every word into one set. A subset test over that
    set fails on exactly the words this needs: ב and מ are clitics, so `ביטול`
    contributes `יטול` and `לא משנה` contributes `שנה`, and neither stripped form
    is a cancel word. Each word only has to match on *one* of its variants.
    """
    meaningful = 0
    for word in query.split():
        variants = _variants(word)
        if variants & _CANCEL_NOISE_FOLDED:
            continue
        if not variants & _CANCEL_FOLDED:
            return False
        meaningful += 1
    # Politeness alone is not a cancel - it is not anything, and the refusal
    # gate is the honest answer for it.
    return meaningful > 0


#: How much rendered schema the shortlist may spend, in characters of JSON.
#:
#: Needle decodes through a 256-token sliding KV window and this project
#: measured ``_doc_prefix_len`` at 0, so nothing is pinned: a long tools block
#: pushes its own head out of the window before generation starts. v10's
#: shortlist rendered to about 288 tokens and worked; v11's domain tools are
#: larger individually and fewer per sentence, so the median improved (208
#: tokens) and the tail got worse (462 at p95, on the sentences that reach
#: four or five domains).
#:
#: This caps the tail rather than the count. 1500 characters of ASCII JSON is
#: about 400 tokens at the 0.27 tokens/char this tokenizer gives English,
#: which is the budget the plan set. Trimming happens from the back, so what
#: is dropped is always the lowest-scoring speculative family - never the one
#: the sentence actually named.
#:
#: Counting characters rather than tokens is deliberate: the component has no
#: tokenizer on the event loop, the ratio is stable for ASCII, and the cap is
#: a budget rather than a limit that must be exact.
MAX_TOOL_CHARS: Final = 1500


# The declared ``action`` enum, ordered - and sometimes cut - by the sentence.
#
# This is the five-tool shortlist one level down, and it exists for the same
# measured reason. v11 collapsed 42 tools into 20, which moved "which
# behaviour" out of the router - deterministic, and right 87.6% of the time on
# its top pick - and into the model, which is right 37% of the time on the same
# question. On Home Assistant's own Hebrew suite the model answered
# `timer.change` to "בטל את הטיימר" and `timer.pause` to "מצב הטיימר": seven
# values, and it had collapsed onto two.
#
# Two things are done with that, and they carry very different risk.
#
# **Reordering costs nothing.** Every value stays reachable, so no gold answer
# can be made unemittable, and v10 measured that the model takes the first
# plausible entry it is offered - that is how `timer_cancel` ranked fourth and
# scored 0 of 13. Putting the sentence's own ranking first turns that prior
# into an asset.
#
# **Cutting is the shortlist's risk again.** A value the grammar does not
# declare cannot be emitted at all, so a wrong cut is unrecoverable. Measured
# over all 25,542 action decisions in the corpus, cutting to the supported
# values alone would force **67.2%** of them to the right answer outright - and
# drop the right answer on **1.87%**. Per tool it splits cleanly, and the ones
# that fail are the ones this project already documents as lexically
# overloaded:
#
#     lock_control      997 forced right,  0 dropped
#     vacuum_control   1237 forced right,  0 dropped
#     list_edit        1222 forced right,  0 dropped
#     switch_control    751 forced right,  0 dropped
#     helper_toggle     398 forced right,  0 dropped
#     camera_control    207 forced right,  0 dropped
#     get_datetime      162 forced right,  0 dropped
#     ------------------------------------------------ the rest keep the enum
#     fan_control       724 forced right,   1 dropped
#     valve_control     439 forced right,  12 dropped
#     routine_run       519 forced right,  17 dropped
#     cover_control    3792 forced right,  29 dropped
#     light_control    3245 forced right,  51 dropped
#     timer_control       8 forced right,  69 dropped
#     media_control    1265 forced right,  77 dropped
#     climate_control  2199 forced right, 222 dropped
#
# `climate_control`'s 222 are one shape - "תנמיך את הקירור" lowers the
# temperature of the cooling, and קירור is the machine as well as the mode, the
# same collision `slot_match` documents. `light_control`'s 51 are the dimming
# case `direction._NOT_A_DIRECTION` already names: עמעם routes to
# `light_turn_off` and *means* a brightness.
#
# So the cut ships for the seven that measure zero and the order ships for
# everything.
CUT_ACTIONS: Final[frozenset[str]] = frozenset((
    "lock_control", "vacuum_control", "list_edit", "switch_control",
    "helper_toggle", "camera_control", "get_datetime",
))


def ranked_actions(query: str, tool: str) -> tuple[list[str], bool]:
    """``(actions best-first, safe to cut)``.

    An empty list means the sentence supports nothing and the enum should be
    left exactly as the catalogue declares it.
    """
    mapping = ACTIONS.get(tool)
    if not mapping:
        return [], False
    toks = _tokens(query)
    scored = [
        (_hits(TOOL_HINTS.get(virtual, []), toks, query), index, action)
        for index, (action, virtual) in enumerate(mapping.items())
    ]
    if not any(hits for hits, _, _ in scored):
        return [], False
    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [action for _, _, action in scored]
    supported = [action for hits, _, action in scored if hits]
    if tool in CUT_ACTIONS and supported:
        return supported, True
    return ordered, False


def select_tools(query: str, catalogue: list[dict[str, Any]],
                 limit: int = MAX_TOOLS) -> list[dict[str, Any]]:
    """Filter a full tool catalogue down to the shortlist for this utterance."""
    by_name = {t["name"]: t for t in catalogue}
    names = [n for n in select_tool_names(query, limit) if n in by_name]
    if not names:  # catalogue does not contain our names at all
        return catalogue[:limit]
    chosen = [by_name[n] for n in names]

    # The enum the model is shown, ordered by the sentence and cut where
    # cutting has been measured never to lose the right answer. A copy, because
    # the catalogue is shared and read on every turn.
    ordered = []
    for tool in chosen:
        actions, _cut = ranked_actions(query, tool["name"])
        spec = tool["parameters"]["properties"].get("action")
        if not actions or spec is None or actions == spec["enum"]:
            ordered.append(tool)
            continue
        properties = dict(tool["parameters"]["properties"])
        properties["action"] = {**spec, "enum": actions}
        ordered.append({
            **tool,
            "parameters": {**tool["parameters"], "properties": properties},
        })
    chosen = ordered

    # Spend no more than the budget, and always keep the first tool whatever
    # it costs - a sentence with no declared tool at all can only be refused.
    spent = 0
    for i, tool in enumerate(chosen):
        spent += len(json.dumps(tool, separators=(",", ":")))
        if i and spent > MAX_TOOL_CHARS:
            return chosen[:i]
    return chosen
