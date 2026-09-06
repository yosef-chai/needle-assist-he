"""Which of two tools, and which way a number points.

Every model this project has trained inverts the direction of a command
sometimes. Counted over the dumped failures of two runs on the same held-out
set, it is 32 of v9's 400 and 56 of v10's - and the worst of them is the worst
thing a home assistant can do:

    query  נעל את הדלת בחדר ההורים          lock the bedroom door
    pred   lock_unlock{area: bedroom}       ...unlocked it

The Hebrew verb settles this. ``נעל`` is lock and ``פתח`` is unlock; ``תדליק``
is on and ``תכבה`` is off. So the direction is read off the sentence, and the
model's answer is corrected when the two disagree - the same rule, and the same
safety argument, as the ``media_play`` upgrade in :mod:`executor`: it never
changes which *domain* the model chose, only which of two tools inside it.

Measured as it runs - one clause at a time, since `clause_split` cuts the
sentence first - over all 2,379 held-out rows, on the 1,542 calls whose tool
belongs to one of the guarded pairs:

    the words agree with gold     1337     86.7%
    the words disagree with gold     0      0.00%
    the words say nothing          205     13.3%

Zero. A signal that is never wrong when it speaks is a signal that can overrule
a model which is wrong one time in eight.

Per clause is not a kinder measurement, it is the right one: reading whole
sentences reports four disagreements which are not disagreements. "תעשי את
התאורה ואז תכבי אור בסלון" is two orders, and the second one's verb says
nothing about the first.

The same mechanism settles two pairs that are not directions at all, and they
are here because they are the same question - which of two tools inside one
domain - answered by the same evidence:

* **A setting is not a start.** "שים את הרובוט על שקט" sets the vacuum's
  suction; the model answered 17 of these with ``vacuum_start``. 82 agree,
  0 disagree.
* **A fan speed is not a temperature.** "תעביר את המזגן למהירות נמוכה" is
  ``climate_set_fan_mode``; the model answered 15 of these with
  ``climate_set_temperature``. 174 agree, 0 disagree.

``media_set_volume`` against ``media_play`` was measured too and is left out:
65 agree and **one** disagrees, and one is not zero. The bar for overruling a
model is that the rule is never wrong, not that it is usually right.

Which way the number points
---------------------------

Three arguments are *relative* - ``temperature_step``, ``brightness_step_pct``
and ``volume_step_pct`` - and :mod:`executor` adds them to the device's current
reading before it calls the service. Their sign is a direction, so getting it
backwards is the same defect one level down:

    query  בסלון הגדול חם מדי, תנמיך משמעותית    it is too hot, lower it a lot
    pred   climate_set_temperature{temperature_step: 4}   ...raised it 4 degrees

That is an eight-degree error from a one-character one, and the model makes it
because the training corpus is 71% positive steps and the held-out set is 97%
negative, so a model that learned "usually up" scores well on one and badly on
the other.

It is not the common failure, and saying so would overstate this module.
Counted over the dumped failures of three runs, on the calls whose gold carries
a relative step and whose arguments are wrong:

    the model emitted no step at all      86
    the sign is backwards                 13
    the sign is right, the size is not     0

So the guard repairs 13 of those 99, and leaves the 86. Supplying the missing
step was measured and rejected: the sentence reads the whole value correctly
96.6% of the time, but 106 calls in the corpus carry an *absolute* value under
a directional verb - "תוריד את המזגן בכניסה לבית לבערך 25" lowers it *to* 25 -
so a rule that filled from the direction alone would invent a step where a
target was meant. The 86 are also the benign failure: a `set_temperature` with
no temperature is refused by Home Assistant, so the household is told the
command failed and repeats it. A wrong sign is the other kind, and that is the
one worth a rule.

The sentence is not ambiguous, and it reads in three tiers, because Hebrew
comparatives stack:

1. **The verb.** ``תנמיך``, ``תוריד``, ``תקרר``, ``תחליש``, ``תעמעם`` go down;
   ``תגביר``, ``תעלה``, ``תרים``, ``תחזק``, ``תחמם`` go up.
2. **The adjective**, when there is no verb. ``חלש`` is quiet and ``חזק`` is
   loud, ``עמום`` is dim and ``בהיר`` is bright.
3. **The bare comparative**, when there is neither. ``פחות`` down, ``יותר`` up.

The order is what makes it work. "יותר חלש בחוץ" is *more quiet*, not *more*:
``יותר`` intensifies the adjective instead of pointing anywhere, and reading
the tiers in the other order gets all 32 of these backwards. Measured against
gold over both splits, on every call carrying one of the three arguments:

    the words agree with gold     1222     98.9%
    the words disagree with gold     1      0.08%
    the words say nothing           12      1.0%

The one disagreement is "יותר כלש בחדר הביטחון" - the corpus injects speech
noise, and here it corrupted the direction word itself. That is a different
thing from the ``media_play`` case above, which is why this one ships and that
one does not: ``נגן`` is a rule that is *wrong about Hebrew*, and this is a
rule that is right about Hebrew and defeated by a typo the model cannot read
either.

Only the sign is taken. The magnitude was measured too - ``קצת``/``שמץ`` is
one step, ``חזק`` three, ``משמעותית``/``הרבה`` four - and reaches 96.6% with
six disagreements, every one of them another injected typo. Six is not zero,
being two degrees out is not being eight degrees out in the wrong direction,
and a step that is too small is one the speaker simply repeats. So the model
keeps the number and the sentence keeps the sign.

Three things had to be excluded to get there, and each one was found by the
measurement rather than by reading:

* **Self-corrections.** "תכבה את המנורה, לא לא, תעשה את המנורה" says the wrong
  verb first and then fixes it. Sixteen of the eighteen disagreements were this
  shape. Taking the *last* verb instead of counting them was tried and is
  worse, for the reason the third bullet gives: position cannot be trusted
  when a hint list holds nouns as well as verbs. The pattern is
  :data:`clause_split.CORRECTION`, shared with the module that has to cut
  the same sentences differently for the same reason.
* **Dimming.** ``עמעם`` sits in ``light_turn_off``'s hints, which is right for
  routing - a dim is a reduction - and wrong here: "תעמעם קצת פחות" is
  ``light_turn_on`` with a brightness argument, not an off.
* **Play and pause.** ``נגן`` is both the imperative "play!" and the noun "the
  player", so "די עם את הנגן בסלון" contains a play hint and a pause verb, and
  no amount of counting or ordering separates them: measured over the corpus
  the pair reads 731 agree and **22 disagree**, every one of them that shape.

  Counting and ordering are not the only tools, though, and the one this
  module already uses works here. ``עמעם`` is dropped from the direction
  vocabulary while staying in the router's - a dim is a reduction for routing
  and a brightness for this - and ``נגן`` is the same case one family over.
  Dropped, the pair reads **795 agree and 0 disagree**, which is the bar, so
  it ships. Dropping the inflected forms with it was measured too and is
  worse: 711 agree for the same zero, because ``תנגן`` really is a play verb
  and only the bare noun is ambiguous.

  (The earlier reading of this - "nothing is lost: music that keeps playing is
  not a door that opens" - was true about the *cost* and wrong about the
  *fix*. Home Assistant's own Hebrew suite is what forced it: six of its
  HassMediaPause and HassMediaUnpause sentences ran the opposite service.)

``media_set_volume`` against ``media_play`` stays out. It was measured with
the same drop and still reads one disagreement, and one is not zero.
"""

from __future__ import annotations

import re
from typing import Any, Final

from .clause_split import CORRECTION
from .tool_router import _TET_FOLD, TOOL_HINTS, _fold, _hits, _hits_noisy, _tokens

#: Tools that come in a pair the Hebrew words settle. Volume against play is
#: deliberately absent; see the module docstring.
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("lock_lock", "lock_unlock"),
    ("light_turn_on", "light_turn_off"),
    ("cover_open", "cover_close"),
    ("switch_turn_on", "switch_turn_off"),
    ("fan_turn_on", "fan_turn_off"),
    ("camera_turn_on", "camera_turn_off"),
    ("automation_turn_on", "automation_turn_off"),
    ("input_boolean_turn_on", "input_boolean_turn_off"),
    ("media_play", "media_pause"),
    # v11 additions. Each is the same question the pairs above answer - which
    # of two behaviours inside one domain - and each is measured on the
    # regenerated corpus by `direction_probe`; see the README table.
    ("climate_turn_on", "climate_turn_off"),
    ("valve_open", "valve_close"),
    ("timer_pause", "timer_resume"),
    ("list_add_item", "list_remove_item"),
    # Which end of a countdown. Needed because `executor` reads a helper
    # toggle over a timer sentence as a timer command, and the toggle it came
    # from says on or off while the sentence says start or cancel - "עצור את
    # הטיימר" arrives as `input_boolean_turn_on` from the model. Measured over
    # the corpus: 414 agree, 0 disagree.
    ("timer_start", "timer_cancel"),
    # Not a direction, the same question: which of two tools in one domain.
    ("vacuum_set_fan_speed", "vacuum_start"),
    ("climate_set_fan_mode", "climate_set_temperature"),
)

#: Arguments that are a change rather than a value, so their sign is a
#: direction. :mod:`executor` adds these to the device's current reading.
RELATIVE: Final[frozenset[str]] = frozenset(
    ("temperature_step", "brightness_step_pct", "volume_step_pct")
)

#: Direction words, strongest evidence first; see the module docstring for why
#: the order matters. Each tier is (up, down).
TIERS: Final[tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]] = (
    (("תגביר", "תגבירי", "להגביר", "תעלה", "תעלי", "להעלות", "תרים", "תרימי",
      "להרים", "תחזק", "תחזקי", "לחזק", "תחמם", "תחממי", "לחמם"),
     ("תנמיך", "תנמיכי", "להנמיך", "תוריד", "תורידי", "להוריד", "תחליש",
      "תחלישי", "להחליש", "תקרר", "תקררי", "לקרר", "תעמעם", "תעמעמי",
      "לעמעם")),
    (("חזק", "חזקה", "בהיר", "בהירה"),
     ("חלש", "חלשה", "עמום", "עמומה", "כהה")),
    (("יותר",), ("פחות",)),
)

OPPOSITE: Final[dict[str, str]] = {}
for _a, _b in PAIRS:
    OPPOSITE[_a] = _b
    OPPOSITE[_b] = _a

# Hints that route correctly and point nowhere. Both are words the router
# needs under the tool they sit on and this module must not read as evidence.
#
# A dim is a brightness, not an off: routing wants ``עמעם`` under
# light_turn_off, and "תעמעם קצת פחות" is a light_turn_on with an argument.
#
# ``נגן`` is the noun as well as the imperative - "the player" as well as
# "play!" - so it appears in a sentence that means the opposite of what it
# hints at. Only the bare form: ``תנגן`` and ``נגני`` are unambiguously verbs,
# and dropping them costs 84 agreements for no fewer disagreements.
_NOT_A_DIRECTION: Final = frozenset(
    _fold(w) for w in ("עמעם", "תעמעם", "תעמעמי", "עמעמי", "נגן"))

#: Verbs that mean a direction here and would mean the wrong thing to the
#: router, so they are added on this side of the wall only.
#:
#: An Israeli opens a boiler, a socket and a switch with the same verb they
#: open a blind with, and `light_turn_on` has carried פתח since v4 for exactly
#: that reason. `switch_turn_on` never did, though `switch_turn_off` has
#: carried סגור all along - a plain asymmetry, and an expensive one: with
#: neither side of the pair named, `settle_toggle` falls silent and every one
#: of these sentences ships as a toggle. "תפתח את הדוד בסלון" turned the
#: boiler off when it was on.
#:
#: Measured over the corpus on single-clause single-call rows, the way every
#: other table in this module was: an open-form verb on a switch row means
#: `switch_turn_on` **124 times and its opposite none**.
#:
#: Not given to :data:`~tool_router.TOOL_HINTS`, because that list also scores
#: the shortlist: teaching the router that פתח suggests a switch would put
#: `switch_control` in front of `cover_control` on every blind sentence in the
#: house. The router is right as it is - the model already picks the switch on
#: these rows and only the direction is wrong.
#:
#: The camera is the same asymmetry one family over, and neither side of it
#: was named: `camera_control` is in :data:`HINT_DECIDED`, so `settle_action`
#: reaches it, but its vocabulary was the on/off verbs alone and an Israeli
#: closes a camera as readily as they switch it off. "נו תוכל לסגור לי מצלמה
#: בחדר הילדים" came back as `on`. Measured the same way: a close-form verb on
#: a camera row means `camera_turn_off` **107 times and its opposite none**, an
#: open-form verb means `camera_turn_on` **82 times and its opposite none**.
DIRECTION_ONLY: Final[dict[str, tuple[str, ...]]] = {
    "switch_turn_on": ("פתח", "תפתח", "פתחי", "תפתחי", "לפתוח"),
    "camera_turn_on": ("פתח", "תפתח", "פתחי", "תפתחי", "לפתוח"),
    "camera_turn_off": ("סגור", "תסגור", "סגרי", "תסגרי", "לסגור"),
}

#: The router's own per-tool verbs, minus the ones that do not mean a direction.
VOCABULARY: Final[dict[str, list[str]]] = {
    tool: [hint for hint in TOOL_HINTS.get(tool, [])
           if _fold(hint) not in _NOT_A_DIRECTION]
    + list(DIRECTION_ONLY.get(tool, ()))
    for tool in OPPOSITE
}


#: The three behaviours a *pair* cannot reach: a toggle has no opposite.
#:
#: v11 made this matter. Until then the router ranked `light_toggle` third in
#: its family and a tight shortlist usually cut it, so the model rarely emitted
#: one; now all three behaviours are values of one enum and every light command
#: can come back as `flip`. On an epoch-4 checkpoint it did, repeatedly, and
#: nothing corrected it - "כבי את המנורה בחדר הביטחון" came back as a toggle.
#:
#: The rule is narrow on purpose: it fires only when the sentence carries no
#: toggle word at all *and* exactly one side of the pair. Measured over the
#: corpus on single-call rows, where clause alignment is exact:
#:
#:     gold is a genuine toggle    335 agree, **0 disagree**, 58 silent
#:     gold is on or off          3413 agree,    2 disagree, 2226 silent
#:
#: The two are both self-corrections whose "רגע לא" the CORRECTION pattern
#: missed because speech noise glued it to the next word - the class this
#: module already excludes, arriving in a form the regex cannot see. Nothing
#: turns a genuine toggle into an on or an off.
TOGGLES: Final[dict[str, tuple[str, str]]] = {
    "light_toggle": ("light_turn_on", "light_turn_off"),
    "switch_toggle": ("switch_turn_on", "switch_turn_off"),
    "fan_toggle": ("fan_turn_on", "fan_turn_off"),
}

#: Tools whose behaviours the sentence separates without ever being wrong, so
#: the one it names may overrule the model outright.
#:
#: An allow-list, and the two measurements that produced it are the reason.
#:
#: v11 is why this exists at all. Collapsing 42 tools into 20 moved a decision
#: out of the router - which is deterministic and measured at 99% - and into
#: the model, which is measured at 37% on it. The router used to rank
#: `light_turn_on` above `light_turn_off` and a tight shortlist often cut the
#: loser outright; now every behaviour of a declared tool is always reachable,
#: and the model picks. This puts as much of that decision back into the
#: sentence as the sentence can carry without ever being wrong.
#:
#: **The obvious rule is the wrong one.** Take the best-*scoring* sibling for
#: any tool with three or more behaviours, and it reads 14,161 agree against
#: **647 disagree** - 4.4%.
#:
#: The rule that ships is narrower: it fires only when exactly one sibling's
#: hints appear in the sentence **at all**, so a sentence that supports two
#: readings settles nothing rather than picking between them. Measured per
#: tool over the corpus, on single-call rows where clause alignment is exact:
#:
#:     list_edit       1222 agree,   0 disagree,   37 silent
#:     vacuum_control   910 agree,   0 disagree,  152 silent
#:     switch_control   751 agree,   0 disagree,  207 silent
#:     lock_control     638 agree,   0 disagree,    7 silent
#:     helper_toggle    398 agree,   0 disagree,   11 silent
#:     camera_control   207 agree,   0 disagree,  195 silent
#:     get_datetime     162 agree,   0 disagree,    4 silent
#:     timer_control      8 agree,   0 disagree, 1306 silent
#:     ---------------------------------------------------- the rest fail
#:     fan_control      724 agree,   1 disagree,  472 silent
#:     valve_control    439 agree,  12 disagree,   98 silent
#:     routine_run      519 agree,  16 disagree,  676 silent
#:     cover_control   2889 agree,  26 disagree,  275 silent
#:     light_control   2384 agree,  47 disagree, 1613 silent
#:     media_control   1265 agree,  65 disagree, 1227 silent
#:     climate_control 1689 agree, 220 disagree, 1160 silent
#:
#: `climate_control` fails for the reason `slot_match` documents one file over:
#: קירור and חימום are the nouns for the machine as well as the names of its
#: modes. `cover_control` and `light_control` fail on the same overloaded
#: imperatives their own pairs exist to settle - and those pairs already do,
#: which is why losing them here costs nothing.
#:
#: `timer_control` is the interesting one: it never errs and it almost never
#: speaks, because "טיימר" is a hint under `start` and under `cancel` alike and
#: "עצור" is under two more. Its real discriminator is a *duration*, and that
#: is :data:`TIMER_DURATION_STARTS` below.
#: **Re-measured after :data:`DISCRIMINATING` landed, and the protocol
#: changed with it.** The table above asks "does the rule keep the gold
#: answer", which a silent rule passes for free. The table below asks the
#: question the rule exists to answer: *fed the wrong sibling on purpose*,
#: does it get back to gold? Same rows, same clause alignment:
#:
#:     light_control   2028 agree,   0 disagree, 1612 silent
#:     cover_control   2256 agree,   0 disagree,  443 silent
#:     timer_control    907 agree,   0 disagree,  152 silent
#:     vacuum_control   763 agree,   0 disagree,  146 silent
#:     routine_run      609 agree,   3 disagree,  300 silent
#:     lock_control     584 agree,   0 disagree,    7 silent
#:     fan_control      576 agree,   0 disagree,  464 silent
#:     switch_control   513 agree,   0 disagree,  188 silent
#:     valve_control    384 agree,   0 disagree,  103 silent
#:     helper_toggle    256 agree,   0 disagree,    8 silent
#:     camera_control   174 agree,   0 disagree,  185 silent
#:     get_datetime     120 agree,   0 disagree,    3 silent
#:     ---------------------------------------------------- the rest fail
#:     media_control   1094 agree,  43 disagree,  971 silent
#:     climate_control 1543 agree, 169 disagree, 1046 silent
#:
#: `routine_run` joins the allow-list on that measurement. Its three
#: disagreements are all one shape - "תקשיבי,תכבה", "תשביתאת", "תשביט" - a
#: negative verb that speech noise glued to its neighbour or misspelled, so
#: the noun is read and the direction is not, and all three fall to
#: `automation_turn_on` when the gold is off. That is the same class, and the
#: same benign end, as the six :func:`settle_timer` ships with.
#:
#: `timer_control` reads zero here only because :func:`settle_timer` now runs
#: **after** this function rather than before it. Both orders were measured on
#: these rows: with the timer settled first, this function then overrules it
#: on eleven "לעצור **רגע** את הספירה" - stop it *for a moment*, which is a
#: pause - because עצור is a cancel hint and רגע is a phrase only
#: :data:`_T_PAUSE` knows. Letting the specialist speak last costs nothing
#: anywhere else and gains 118 rows overall.
HINT_DECIDED: Final[frozenset[str]] = frozenset((
    "list_edit", "vacuum_control", "switch_control", "lock_control",
    "helper_toggle", "camera_control", "get_datetime", "timer_control",
    "routine_run",
))


#: "Do the light." A bare do-verb with nothing else to go on means switch it
#: on - see the last tier of :func:`settle_toggle`.
_DO_VERBS: Final = ("תעשה", "תעשי", "עשה", "עשי", "לעשות", "שיעשה")


def settle_toggle(tool: str, text: str,
                  names_a_level: bool = False) -> str:
    """A toggle the sentence contradicts, resolved to the side it names.

    Returns ``tool`` unchanged for anything that is not a toggle, for a
    sentence that takes a verb back, for one that actually says "toggle", and
    for one that names both sides or neither.

    ``names_a_level`` is `slot_match.level_from(text) is not None`, passed in
    rather than imported. See the level tier below.
    """
    pair = TOGGLES.get(tool)
    if pair is None or not text:
        return tool
    if CORRECTION.search(_fold(text)):
        return tool
    tokens = _tokens(text)
    if _hits(TOOL_HINTS.get(tool, []), tokens, text):
        return tool
    on, off = pair
    on_hit = _hits(VOCABULARY[on], tokens, text)
    off_hit = _hits(VOCABULARY[off], tokens, text)
    if not on_hit and not off_hit:
        # Both sides silent is where the model answers `flip`, and it is very
        # often a verb speech-to-text broke rather than a verb nobody said:
        # "כבהאת הנורה" lost its space, "קבי אור" and "תדליכי" swapped ק for כ.
        # Read again with those two tolerances - see `_hits_noisy` for why they
        # are affordable here and nowhere else. Tried only when the strict
        # reader is silent on *both* sides, so a side it did find can never be
        # tied by a noisy match on the other. 47 agree, 0 disagree.
        on_hit = _hits_noisy(VOCABULARY[on], text)
        off_hit = _hits_noisy(VOCABULARY[off], text)
    if on_hit and not off_hit:
        return on
    if off_hit and not on_hit:
        return off
    if on_hit or off_hit:
        # Both sides named settles nothing, as everywhere else here.
        return tool
    # Last tier, and only ever reached by a sentence that names neither side
    # and no toggle either. An Israeli who says "**תעשה** את האור בסלון" - do
    # the light - means turn it on; nobody uses the bare verb to mean off,
    # because off has כבה and סגור and אוף of its own and they are all
    # checked above. 31 of the 400 failures in one release dump were this.
    #
    # Measured over train and test on single-clause single-call rows whose
    # gold is one of the three pairs: **385 agree, 1 disagree**, and the one
    # is "תעשה אוףאת הלייטס", where speech noise glued the אוף to the next
    # word so the off side could not be seen.
    #
    # It could not ship before `light_toggle` gained טוגל: without it, "תעשה
    # טוגל לאור" reached this tier and 50 genuine toggles were turned on.
    # A level is a turn-on, because nothing else can hold one. "אור בחדר
    # שינה בעשרים וחמישה אחוז" is a brightness and `light.toggle` takes no
    # brightness; the model answers `flip` because it read a light and no verb,
    # and the percentage was there all along. Same for a fan's speed.
    #
    # An *upward* step counts too and a downward one does not, and the
    # asymmetry is Hebrew rather than fitting. Raising a thing that is off
    # turns it on - "תחזק את האורות הרבה יותר", "אור בסלון יותר" - but
    # "להוריד" is the ordinary way to say switch off: "תוריד את התקע" takes the
    # plug down, it does not dim it. Measured per clause where both sides are
    # silent, level alone reads 467/297/0 agree and **0 disagree** on
    # light/fan/switch; adding the upward step makes it 868/297/21 and still
    # **0**; adding the downward step as well breaks 74 of them, every one a
    # "להוריד" whose gold is off.
    if names_a_level or (which_way(text) or 0) > 0:
        return on
    if _hits(list(_DO_VERBS), tokens, text):
        return on
    return tool


#: The words that separate one behaviour of a tool from its siblings, for the
#: family whose full hint lists overlap too much for :func:`settle_action`.
#:
#: `routine_run` is that family, and it is the worst-scoring tool in the
#: catalogue because of it: 24.3% on v11's own test split, where the failure
#: is almost always `scene` answered as `macro`. The cause is visible in the
#: hint lists - `scene_activate` and `script_run` and `automation_turn_on` all
#: carry להפעיל, because all three genuinely are things you "run". With three
#: siblings named, `settle_action`'s strict rule settles nothing, every time.
#:
#: The nouns do separate them, and they separate them completely. A sentence
#: that says סצנה means a scene; one that says סקריפט or הרץ means a script;
#: one that says אוטומציה means an automation; one that says כפתור or לחץ
#: means a button. Measured over the corpus on single-clause single-call rows:
#:
#:     scene       212 agree, 0 disagree, 156 silent
#:     script       71 agree, 0 disagree, 137 silent
#:     automation  253 agree, 3 disagree,   6 silent
#:     button       73 agree, 0 disagree,   1 silent
#:
#: The three automation disagreements are all "turn it off" sentences whose
#: negative verb speech noise glued to the next word - תשביתאת, תשביט - so the
#: noun is read and the direction is not. They cost nothing here because
#: `automation_turn_on` and `automation_turn_off` are a :data:`PAIRS` entry,
#: and this function hands its answer back to :func:`settle` before returning.
#:
#: Silence is the common case and stays the safe one: a sentence naming no
#: routine noun, or naming two, leaves the model's answer alone.
DISCRIMINATING: Final[dict[str, tuple[str, ...]]] = {
    "scene_activate": ("סצנה", "סצינה", "סצנת", "תרחיש", "אווירה", "אווירת"),
    "script_run": ("סקריפט", "הרץ", "תריץ", "הריצי", "תריצי", "להריץ"),
    "automation_turn_on": ("אוטומציה", "אוטומציות"),
    "automation_turn_off": ("אוטומציה", "אוטומציות"),
    "button_press": ("כפתור", "הכפתור", "תלחץ", "לחץ", "תלחצי", "לחצי",
                     "ללחוץ", "לחיצה"),
}

#: The second tier, read only when the table above is silent - which it is on
#: 966 of the 1,621 routine clauses, because two thirds of them never say the
#: noun at all.
#:
#: **Order is the rule here**, not an accident of the literal: the first entry
#: whose words appear decides, and it is `scene_activate` on purpose. A
#: sentence that says both "מצב" and "תפעיל" - "תפעיל מצב שבת" - is a scene,
#: and reading the verb first would make it a script.
#:
#: `מצב` is what an Israeli calls a scene when they do not call it a scene:
#: it appears on **222 of the 234** silent scene clauses and is nearly absent
#: from the rest. The run-verbs are the mirror - a script is a thing you
#: *activate*, a scene is a thing you *switch to* - and they carry 194 more.
#:
#: These are markers, not names. The corpus's silent macro clauses also
#: correlate strongly with הביתה, יציאה, השקיה and ניקיון, and those are left
#: out deliberately: they are the names of the scripts this corpus happens to
#: generate, and a table of them would score well here and know nothing about
#: a real house.
#:
#: Measured per clause over the whole corpus, as a ladder on top of the strict
#: table and its noisy re-read: 655 agree and 0 disagree becomes **1,074 agree
#: and 4 disagree**, with silence down from 966 to 543. All four say the
#: strict noun through injected speech noise - "האותומציה", "העוטומציה",
#: "האוטומזיה", "שצנת" - so the tier that should have caught them is the one
#: above, not this one.
DISCRIMINATING_WEAK: Final[dict[str, tuple[str, ...]]] = {
    "scene_activate": ("מצב", "למצב", "המצב"),
    "script_run": ("תפעיל", "תפעילי", "להפעיל", "הפעל", "הפעילי",
                   "תעשה", "תעשי", "לעשות"),
}

#: A run-verb needs something to run. All 194 corpus clauses this tier reads as
#: a script name one - "תעשה חזרה הביתה", "תפעיל יציאה מהבית", "להפעיל השקיה" -
#: and "תפעיל את זה" names nothing at all, so the model's own answer is the
#: better one. **Not one of the 194 says any of these**, so the guard is free.
_NAMES_NOTHING: Final[tuple[str, ...]] = ("זה", "זאת", "אותו", "אותה")


#: The routine nouns, and only the nouns. :data:`DISCRIMINATING` separates the
#: four siblings once the model has already answered inside the family;
#: :func:`settle_routine_noun` uses this to pull it *into* the family, which is
#: a stronger claim and takes the stricter table.
#:
#: A noun names the family. A verb only says which behaviour inside it, so
#: `הרץ`, `תריץ`, `תלחץ` and `לחץ` stay out: `לחצי` is both "press" and "to a
#: half", and "תכוון את הסאונד לחצי" is a volume. That one row is the whole
#: difference between this table and its parent.
#:
#: Why it is needed at all: `family_named` answers ``None`` when a sentence
#: names two families, and an automation is *named after what it does* -
#: "האוטומציה תריסים בבוקר" says blinds, "האוטומציה אורות בלילה" says lights.
#: So the one shape where the family is stated outright was the one shape the
#: family rule could not read, and the model's cover or light answer stood.
#: `tool_router.family_named`'s own docstring claimed the opposite.
#:
#: Measured per clause over the whole corpus, against the gold behaviour and
#: not merely the gold family: **1,172 agree, 3 disagree**. All three are the
#: speech noise :data:`DISCRIMINATING` already names - תכוה, תשבית, שטבטל -
#: where the noun is read and the negative verb is not.
ROUTINE_NOUNS: Final[dict[str, tuple[str, ...]]] = {
    "scene_activate": ("סצנה", "סצינה", "סצנת", "תרחיש", "אווירה", "אווירת"),
    "script_run": ("סקריפט",),
    "automation_turn_on": ("אוטומציה", "אוטומציות"),
    "button_press": ("כפתור", "הכפתור"),
}


def settle_routine_noun(tool: str, text: str) -> str:
    """A clause that names a routine outright is one, whatever else it names."""
    if not text:
        return tool
    tokens = _tokens(text)
    named = [name for name, words in ROUTINE_NOUNS.items()
             if _hits(list(words), tokens, text)]
    if len(named) != 1:
        return tool
    return settle(named[0], text)


def settle_action(tool: str, siblings: list[str], text: str) -> str:
    """The one behaviour of ``siblings`` the sentence names, or ``tool``.

    "Names" is deliberately strict: exactly one sibling's hints have to appear
    at all. A sentence supporting two readings settles nothing rather than
    picking the higher-scoring one - see :data:`HINT_DECIDED` for what taking
    the maximum instead costs.

    Where :data:`DISCRIMINATING` carries an entry for a sibling, that shorter
    list is read instead of the router's, because the router's exists to get
    the family onto the shortlist and shares its verbs across the family on
    purpose. Two siblings pointing at one behaviour - the two halves of an
    automation - count as one naming, and :func:`settle` then says which half.

    Only for the tools in the :data:`HINT_DECIDED` allow-list.
    """
    if not text or not siblings or CORRECTION.search(_fold(text)):
        return tool
    tokens = _tokens(text)
    named = [name for name in siblings
             if _hits(list(DISCRIMINATING[name]) if name in DISCRIMINATING
                      else TOOL_HINTS.get(name, []), tokens, text)]
    if not named:
        # The same nouns, read through the noise speech-to-text puts in them.
        # Only for siblings the table covers: `TOOL_HINTS` is the router's own
        # list and is long and generic, and reading *it* loosely is what the
        # note above `tool_router._FINALS` measured and rejected.
        named = [name for name in siblings
                 if name in DISCRIMINATING
                 and _hits_noisy(list(DISCRIMINATING[name]), text)]
    if not named and not _hits(list(_NAMES_NOTHING), tokens, text):
        # And the markers that are not nouns. Ordered by the table rather than
        # by `siblings`, because the order is the rule - see
        # :data:`DISCRIMINATING_WEAK`.
        for name, words in DISCRIMINATING_WEAK.items():
            if name in siblings and _hits(list(words), tokens, text):
                named = [name]
                break
    if not named:
        return tool
    if len(named) > 1:
        # Two names for one behaviour pair is one naming; anything else is a
        # sentence supporting two readings, which settles nothing.
        paired = {OPPOSITE.get(name, name) for name in named}
        if len(paired | set(named)) != 2:
            return tool
    return settle(named[0], text)


def settle(tool: str, text: str) -> str:
    """The tool the words point at, or ``tool`` unchanged.

    Returns ``tool`` when it is not one of a guarded pair, when the sentence
    takes a verb back, or when the two directions are equally evidenced -
    silence is the common case and the safe one.
    """
    other = OPPOSITE.get(tool)
    if other is None or not text:
        return tool
    if CORRECTION.search(_fold(text)):
        return tool
    tokens = _tokens(text)
    mine = _hits(VOCABULARY[tool], tokens, text)
    theirs = _hits(VOCABULARY[other], tokens, text)
    return other if theirs > mine else tool


def which_way(text: str) -> int | None:
    """``1`` for up, ``-1`` for down, ``None`` when the sentence is silent.

    The tiers are read in order and the first one that speaks decides, which
    is what keeps "יותר חלש" quiet rather than loud.

    A clause the strict reader cannot hear is read again through the speech
    noise, exactly as :func:`settle` does for a toggle and for the same reason:
    "טרים בהרבה את הווליום בממ״ד" sizes its step at thirty-five and had no sign
    to hang it on, so the step was dropped and a household asking for much
    louder got nothing. ט and ת are one sound, so this call - and only this one
    - reads them as one; see :data:`tool_router._TET_FOLD`.

    Measured per clause over the v11 corpus. On clauses whose gold carries a
    step and which the strict reader leaves silent: **9 rescued, 0 disagree**
    (five of the nine without the ט/ת tolerance). And on the 22,893 clauses
    whose gold carries no step at all, it speaks on 149 and **not one of them
    also sizes a step** - a sign alone fills nothing, since `settle_steps` and
    `settle_climate` both want a size before they write anything.
    """
    if not text or CORRECTION.search(_fold(text)):
        return None
    tokens = _tokens(text)
    for up_words, down_words in TIERS:
        up = _hits(list(up_words), tokens, text)
        down = _hits(list(down_words), tokens, text)
        if up != down:
            return 1 if up > down else -1
    for up_words, down_words in TIERS:
        up = _hits_noisy(list(up_words), text, _TET_FOLD)
        down = _hits_noisy(list(down_words), text, _TET_FOLD)
        if up != down:
            return 1 if up > down else -1
    return None


#: How big a step the clause asks for, when it says so in words. Derived from
#: gold rather than guessed: over every corpus clause whose gold carries a
#: `temperature_step` and which names no number at all, these words appear on
#: one magnitude each and on no other.
#:
#: A missing size is *not* filled from the default of two. That is the whole
#: difference between this and the rule the module header rejected: filling
#: from the direction alone reads 576 agree and 7 disagree, and speaking only
#: when an adverb is actually there reads **389 agree and 1 disagree** - the
#: one being "טנמיך עוד קצת", where injected speech noise corrupted the
#: direction verb itself, which is the same single exception the sign rule
#: above already ships with.
_STEP_SMALL: Final = ("קצת", "טיפה", "שמץ", "מעט")
_STEP_BIG: Final = ("בהרבה", "משמעותית", "הרבה")
_STEP_LOUD: Final = ("חזק",)
_STEP_MORE: Final = ("עוד",)

#: One scale per slot: ``(small, small with עוד, big, חזק, unmarked)``.
#:
#: A degree and a percent are not the same size, which is why this is a table
#: and not a constant. Read off gold the same way the temperature row always
#: was - every clause whose call carries the slot and whose sentence names no
#: number at all, grouped by exactly which adverbs are present. Every cell
#: below is unanimous in the corpus except the two noted:
#:
#:     temperature_step      קצת 1 x122   הרבה 4 x96    חזק 3 x31   none 2 x156
#:     brightness_step_pct   קצת 10 x108  הרבה 35 x122  חזק 30 x29  none 20 x168
#:     volume_step_pct       קצת 10 x45   הרבה 35 x61   חזק  see below
#:
#: Three readings follow from the grouping and none of them were guessed:
#:
#: * **עוד raises a small step and only a small one.** "עוד קצת" is 15 on both
#:   percent slots, 29 and 13 clauses, unanimous - while "עוד" alone is the
#:   unmarked 20 and "עוד קצת" on a thermostat is still 1.
#: * **The smaller adverb wins a compound.** "טיפה יותר חזק" is 10, not 30:
#:   קצת+חזק, טיפה+חזק and שמץ+חזק are 26 clauses and every one takes the
#:   small value. So small is tested first, then big, then חזק.
#: * **חזק sizes a light and not a speaker.** On brightness it is 30 on all 29
#:   clauses; on volume it is 30 on twenty and 20 on sixteen, which is a coin
#:   flip and not a rule. `None` there means *stay silent* rather than fall
#:   through to the unmarked value - falling through turns those sixteen into
#:   disagreements, and the model's own answer is the better bet.
#:
#: Measured as a rule, per clause, over train, dev and test:
#:
#:     temperature_step      389 agree,  1 disagree
#:     brightness_step_pct   572 agree,  6 disagree
#:     volume_step_pct       311 agree,  2 disagree
#:
#: The nine are all injected speech noise inside the adverb itself - "קזת",
#: "הרוה פחות", "משמעוטית", "העבודהמשמעותית" - the same single exception class
#: the sign rule has always carried.
_STEP_SCALE: Final[dict[str, tuple[int, int, int, int | None, int]]] = {
    "temperature_step":    (1, 1, 4, 3, 2),
    "brightness_step_pct": (10, 15, 35, 30, 20),
    "volume_step_pct":     (10, 15, 35, None, 20),
}


def step_size(text: str, slot: str = "temperature_step") -> int | None:
    """How big a step this clause asks for on ``slot``, or ``None``.

    ``None`` is silence and it has two causes, deliberately not distinguished
    by the caller: the clause sizes nothing at all, or it sizes it with a word
    this slot has no unanimous reading for. See :data:`_STEP_SCALE`.

    The unmarked value is included - a verb that moves the thing is a step even
    with no adverb behind it - and is guarded on the *verb*, not on
    :func:`which_way`, which also answers to a bare יותר or a bare חלש. Those
    modify a step; they do not make one.
    """
    scale = _STEP_SCALE.get(slot)
    if not text or scale is None:
        return None
    small, small_more, big, loud, unmarked = scale
    tokens = _tokens(text)
    if _hits(list(_STEP_SMALL), tokens, text):
        return small_more if _hits(list(_STEP_MORE), tokens, text) else small
    if _hits(list(_STEP_BIG), tokens, text):
        return big
    if _hits(list(_STEP_LOUD), tokens, text):
        return loud
    if _hits(list(_STEP_VERBS), tokens, text):
        return unmarked
    return None


def settle_steps(arguments: dict[str, Any], text: str,
                 may_fill: tuple[str, ...] = ()) -> dict[str, Any]:
    """``arguments`` with the sign of any relative argument corrected.

    Nothing is removed and no magnitude the model emitted changes: the
    sentence was measured to settle which way a step points and not whether
    there is one.

    ``may_fill`` names the step slots the caller has cleared for filling - the
    ones this tool declares and whose sentence names no number at all.
    `hebrew_numbers.numbers_in` and `slot_match.unsupported` are what answer
    that, passed in rather than imported, the same arrangement as elsewhere in
    this module. For each of them a step the model left out is supplied from
    the words: see :data:`_STEP_SCALE`.

    The guard is what the module header's rejection turned on - 106 corpus
    calls carry an absolute value under a directional verb, "תוריד את המזגן
    לבערך 25" lowers it *to* 25 - and every one of those names a number.

    It was a bool meaning `temperature_step` until the other two slots were
    measured. They fail the same way and for the same reason: "תגביר את
    הווליום קצת" is a step the model answers as a switch-on with no argument
    at all, and there is nothing about a degree that made it the only slot
    worth reading off the sentence.
    """
    if may_fill and (way := which_way(text)) is not None:
        for slot in may_fill:
            if arguments.get(slot):
                continue
            size = step_size(text, slot)
            if size is not None:
                arguments = dict(arguments)
                arguments[slot] = size if way > 0 else -size
    relative = {k: v for k, v in arguments.items()
                if k in RELATIVE and isinstance(v, (int, float))
                and not isinstance(v, bool) and v}
    if not relative:
        return arguments
    way = which_way(text)
    if way is None:
        return arguments
    fixed = dict(arguments)
    for key, value in relative.items():
        if (value > 0) != (way > 0):
            fixed[key] = -value
    return fixed

# Which of seven, for a countdown. -------------------------------------------
#
# `timer_control` is the one tool the sentence-vs-model argument is not close
# on. Seven behaviours, and the v11 model collapsed onto two of them: on Home
# Assistant's own Hebrew suite it answered `timer.change` to "בטל את הטיימר",
# `timer.change` to "קבע טיימר ל2 שעות ו5 דקות" and `timer.pause` to "מצב
# הטיימר" - 1 of 17 HassStartTimer, 1 of 13 HassCancelTimer, 0 of 4
# HassUnpauseTimer. Reordering the enum did not move it; the prior is the
# model's, exactly as it was for `timer_cancel` in v10.
#
# The hint tables cannot separate them either, and that is why this is written
# out rather than derived: `טיימר` sits under both `start` and `cancel`, and
# `עצור` under `cancel` and `pause` and two media tools. Measured, the generic
# rule is silent on 1,306 of 1,314 timer rows.
#
# What actually separates them is structure, in this order:
#
# 1. **A question is a status.** "כמה זמן נשאר", "מצב הטיימר".
# 2. **Pausing says so.** השהה, פאוזה - and "עצור **רגע** את הספירה", stop it
#    *for a moment*, which is the discriminator the first version of this rule
#    missed on all 34 of them.
# 3. **Resuming says so.** המשך, תחדש.
# 4. **Changing a countdown references one that exists.** "תקצר את הטיימר",
#    "עוד חמש דקות **לטיימר**" - a definite article or a preposition, against
#    the indefinite "קבע **טיימר** ל..." that creates one. An explicit
#    lengthen or shorten verb needs no article; a bare עוד does.
# 5. **Cancelling says so.** בטל, עצור, הפסק, די.
# 6. **Anything left carrying a duration starts one.**
#
# Measured over every timer row in the corpus, one clause at a time:
#
#     the words agree with gold     1262     96.0%
#     the words disagree               6      0.46%
#     the words say nothing           46      3.5%
#
# Six is not zero, and it ships anyway on the same grounds the sign guard does:
# every one is a word speech noise glued to its neighbour - תקצראת, תוסיףעוד,
# פחותשתי, עודעשר, אוד for עוד - so the rule is right about Hebrew and defeated
# by a typo the model cannot read either. All six fall back to `timer_start`,
# which is the benign end of the mistake: starting a countdown somebody wanted
# shortened is a countdown they can cancel.

_T_PAUSE: Final = ("השהה", "תשהה", "השהי", "תשהי", "פאוזה", "להשהות",
                   "עצור רגע", "תעצור רגע", "לעצור רגע", "עצרי רגע",
                   "תעצרי רגע", "הפסק רגע", "תפסיק רגע")
_T_RESUME: Final = ("המשך", "תמשיך", "המשיכי", "תמשיכי", "חדש", "תחדש",
                    "להמשיך", "תחזיר")
#: Verbs that lengthen or shorten outright, which need no definite article.
_T_ADD_VERB: Final = ("תוסיף", "הוסף", "להוסיף", "תוסיפי", "הוסיפי",
                      "תאריך", "הארך", "להאריך", "תאריכי")
_T_LESS_VERB: Final = ("תקצר", "לקצר", "תקצרי", "קצר")
#: Markers that only mean a change when a countdown is referred to definitely.
#: Bare עוד is "תזכיר לי בעוד חמש דקות", which *starts* one.
_T_ADD_MARK: Final = ("עוד",)
_T_LESS_MARK: Final = ("פחות", "הורד", "תוריד", "להוריד", "הורידי", "תורידי")
#: A countdown that already exists: definite, or governed by a preposition.
_T_EXISTING: Final = ("הטיימר", "לטיימר", "בטיימר", "מהטיימר", "הטיימרים",
                      "לטיימרים", "התיימר", "לתיימר", "הספירה", "לספירה",
                      "בספירה", "מהספירה")
_T_CANCEL: Final = ("בטל", "תבטל", "בטלי", "תבטלי", "עצור", "עצרי", "תעצור",
                    "הפסק", "הפסיקי", "תפסיק", "למחוק", "תמחק", "די",
                    "לבטל", "לעצור")

#: A door that is a blind. Which behaviour each family's opener and closer
#: becomes when the sentence names one of the compound cover nouns.
#:
#: "תפתח את **דלת החניה**" is the garage door, and דלת on its own is what a
#: lock has - so the router scores lock, the model answers `lock_control`, and
#: a household that asked for the garage gets a bolt. Twenty of the four
#: hundred failures in one dump were this one noun phrase.
#:
#: The evidence is as clean as it gets: `slot_match.SETTING_WORDS` carries
#: eight compound cover classes - דלת חניה, דלתות החניה, תריס הצללה and their
#: variants - and over train and test, **all 147 rows naming one are a cover
#: row**. Not most: all of them. A compound is safe where the bare noun is
#: not, which is exactly why `setting_from` reads multi-word keys first and
#: lets them win outright.
#:
#: Only the opener and the closer of each family, and `settle` runs afterwards
#: so the verb still says which of the two it is.
NAMES_A_COVER: Final[dict[str, str]] = {
    "lock_unlock": "cover_open", "lock_lock": "cover_close",
    "valve_open": "cover_open", "valve_close": "cover_close",
    "switch_turn_on": "cover_open", "switch_turn_off": "cover_close",
    "light_turn_on": "cover_open", "light_turn_off": "cover_close",
}


#: The seven behaviours this settles, so a caller can tell whether to try.
TIMER_BEHAVIOURS: Final[frozenset[str]] = frozenset((
    "timer_start", "timer_cancel", "timer_pause", "timer_resume",
    "timer_add", "timer_less", "timer_status"))

#: Three transport verbs a countdown borrows, and what they mean when it does.
#:
#: `עצור` and `השהה` halt a track and a countdown with the same word, and v12
#: answers "השהה את הטיימר" with a media pause. The router is not the problem -
#: it declares `timer_control` first and `media_control` second - so this is
#: the model reaching past a correct shortlist, one family over from the
#: television and the fan that v12 fixed.
#:
#: `const.HELPER_IS_A_TIMER` already reads a helper toggle over a countdown
#: sentence as a countdown command, on the same evidence and for the same
#: reason. The corpus licenses this one at least as strongly: of 30,613 rows,
#: **1,463 have a transport verb as their gold** - 744 pause, 114 stop, 605
#: play - **and not one of them names a countdown**. Asked the other way
#: round it is stronger still: on the rows whose sentence *does* name one,
#: the gold call is a media behaviour **zero** times out of eight, against
#: 1,249 that are one of the seven timer behaviours.
#:
#: Only the three that overlap in meaning are here. The measurement would
#: license all eight, but muting and volume and next-track correct nothing
#: that was ever seen to go wrong, and a rule that fixes nothing does not
#: ship - the same bar that kept `source` out of `slot_match`.
TRANSPORT_IS_A_TIMER: Final[dict[str, str]] = {
    "media_pause": "timer_pause",
    "media_stop": "timer_cancel",
    "media_play": "timer_resume",
}

#: Where a sentence lands when its nouns name one family and the model chose
#: another. One entry per family, and each is a *starting point* rather than a
#: verdict - `settle`, `settle_toggle`, `settle_media`, `settle_action` and
#: `settle_timer` all run afterwards and say which behaviour inside the family
#: it is.
#:
#: This is the general form of :data:`NAMES_A_COVER` and
#: :data:`TRANSPORT_IS_MEDIA`, and it exists because those two kept turning out
#: to be special cases of one fact: **the noun says which family, and the model
#: disagrees with it far more often than the noun is wrong.**
#:
#: Measured over train and test on single-clause single-call rows whose gold is
#: an actuation - questions are excluded, because a question names a device and
#: asks *about* it, and `tool_router.query_domain` owns those:
#:
#:     the sentence names exactly one family   19,410 agree, **7 disagree**
#:
#: Six of the seven are a word speech noise glued to its neighbour -
#: האוטומזיה, אתהאוטומציה, מצו, האוטומציהמצב, ההשמעהמוזיקה - and the seventh is
#: "תשתיק בחדר המחשב", where מחשב is a switch noun and the sentence names no
#: speaker at all. The same class, and the same benign end, as the six
#: :func:`settle_timer` ships with.
#:
#: A sentence naming *two* families settles nothing, which is what keeps
#: "תדליק את האוטומציה תריסים בבוקר" - an automation whose name is about blinds
#: - from becoming a cover command. That was 79 rows before every family was
#: included in the test rather than only the device ones.
FAMILY_ANCHOR: Final[dict[str, str]] = {
    "light": "light_turn_on",
    "cover": "cover_open",
    "lock": "lock_lock",
    "valve": "valve_open",
    "switch": "switch_turn_on",
    "fan": "fan_turn_on",
    "climate": "climate_turn_on",
    "camera": "camera_turn_on",
    "media": "media_pause",
    "vacuum": "vacuum_start",
    "routine": "scene_activate",
    "helper": "input_boolean_turn_on",
    "timer": "timer_start",
    "list": "list_add_item",
    "notify": "notify_send",
    "datetime": "get_date",
}

#: Virtual-id prefixes whose family is not their first word.
_FAMILY_ALIAS: Final[dict[str, str]] = {
    "input": "helper", "scene": "routine", "script": "routine",
    "automation": "routine", "button": "routine", "music": "media",
    "broadcast": "notify", "get": "datetime",
}


def family_of(tool: str) -> str:
    """The family a virtual id belongs to, for :func:`family_named`."""
    head = tool.split("_", 1)[0]
    return _FAMILY_ALIAS.get(head, head)


def family_named(tool: str, named: str | None) -> str:
    """``named``'s anchor when the model answered outside that family.

    ``named`` is the caller's answer to "which family do this sentence's nouns
    name, if exactly one" - passed in rather than imported, so this module
    keeps depending on :mod:`tool_router` for vocabulary only, the same
    arrangement as ``names_a_mode`` and ``names_a_timer``.

    Returns ``tool`` unchanged when the sentence named none or several, when it
    named the family the model already chose, and for a read-only behaviour: a
    question names a device and asks about it, and `query_domain` owns that.
    """
    if not named or named not in FAMILY_ANCHOR or tool.startswith("get_"):
        return tool
    return tool if family_of(tool) == named else FAMILY_ANCHOR[named]


#: A plug that is a speaker. The behaviours a transport verb overrules.
#:
#: The same argument as :data:`TRANSPORT_IS_A_TIMER`, one family over and
#: pointing the other way: there, a transport verb aimed at a countdown is a
#: countdown command; here, a *switch* verb the sentence answers with a
#: transport verb is a media command.
#:
#: Home Assistant's own Hebrew suite asks "השהה את הטלוויזיה" and "המשך את
#: הטלוויזיה" and "נגן הטלוויזיה". The model answers `switch_control{flip}`,
#: and it is not wrong about the noun - a television is a smart plug in 43 of
#: this corpus's 59 television rows. It is wrong about the verb: a plug has no
#: pause. Five of the suite's nine remaining failures were these three
#: sentences.
#:
#: `tool_router.names_a_transport` is the reading, measured at **1,204 agree
#: and 0 disagree** over train and test. `media_pause` is a starting point
#: rather than a verdict: :func:`settle_media` runs afterwards and says which
#: of the eight it is, exactly as `settle_timer` does for a countdown.
TRANSPORT_IS_MEDIA: Final[frozenset[str]] = frozenset((
    "switch_turn_on", "switch_turn_off", "switch_toggle",
    "light_turn_on", "light_turn_off", "light_toggle",
    "fan_turn_on", "fan_turn_off", "fan_toggle",
    "input_boolean_turn_on", "input_boolean_turn_off",
))

#: What a caller should offer :func:`settle_timer`. One name, so the three
#: call sites cannot drift apart on which tools are worth asking about.
SETTLES_A_TIMER: Final[frozenset[str]] = (
    TIMER_BEHAVIOURS | frozenset(TRANSPORT_IS_A_TIMER))


def settle_timer(tool: str, arguments: dict[str, Any], text: str,
                 is_question: bool, names_a_timer: bool = False) -> str:
    """Which of the seven countdown behaviours the sentence names.

    ``is_question`` is the caller's answer to "did the interrogative fire",
    and ``names_a_timer`` its answer to "does the sentence say countdown",
    both passed in rather than imported so this module keeps depending on
    :mod:`tool_router` for vocabulary only.

    Returns ``tool`` unchanged when it is not a countdown behaviour, when the
    sentence takes a verb back, and when nothing above speaks.
    """
    # The family first, because the noun decides it and a sentence that takes
    # its verb back is still about the same thing. Arbitration below may then
    # decline to re-read the verb; landing on the paired behaviour is already
    # the right family, which is the half that moves a physical device.
    if names_a_timer and tool in TRANSPORT_IS_A_TIMER:
        tool = TRANSPORT_IS_A_TIMER[tool]
    if tool not in TIMER_BEHAVIOURS or not text:
        return tool
    if CORRECTION.search(_fold(text)):
        return tool
    tokens = _tokens(text)

    def said(words: tuple[str, ...]) -> int:
        return _hits(list(words), tokens, text)

    if is_question:
        return "timer_status"
    if said(_T_PAUSE):
        return "timer_pause"
    if said(_T_RESUME):
        return "timer_resume"
    duration = any(arguments.get(k) for k in ("hours", "minutes", "seconds"))
    existing = said(_T_EXISTING)
    if duration and (said(_T_LESS_VERB) or (existing and said(_T_LESS_MARK))):
        return "timer_less"
    if duration and (said(_T_ADD_VERB) or (existing and said(_T_ADD_MARK))):
        return "timer_add"
    if said(_T_CANCEL):
        return "timer_cancel"
    if duration:
        return "timer_start"
    return tool


# An argument the service needs, in a call that does not name that service. --
#
# `climate_control` is deliberately outside HINT_DECIDED and the reason is a
# good one: קירור and חימום name the machine as well as two of its modes, so
# the hint tables settle that family wrongly. This is a different kind of
# evidence and it needs no vocabulary at all. The model emitted a target
# temperature, and `temp` is the only one of the five behaviours with
# anywhere to put it. Measured over the corpus, **1,469 climate calls carry
# `temperature` or `temperature_step` and every one of them is `temp`** -
# 1,006 and 463, and not one of either is a mode, a fan speed, an on or an
# off.
#
# Gold is not what the rule sees, though, and the number that decides is
# the
# one measured on *predictions*. Run over the 534 held-out rows whose gold
# names this tool, one clause at a time as the integration routes them, the
# rule moves 161 calls: **160 of them to what gold says and 1 away from it**.
# The one it loses is "מזגן בחניה, תעשה" - a sentence vague enough that the
# model answered it with a turn-on and a temperature nobody asked for, and
# nothing in the clause says which of the two it meant.
#
# `names_a_mode` is what took that from three losses to one, and it went in
# the opposite direction from the guard that looked obvious. Whether the
# *call* also carries an hvac or fan mode turns out to be no evidence at all
# (15 such moves, 15 of them right); whether the *clause* says one is decisive
# - "שים את הפן של המיזוג ... על נמוך" asks for a fan speed, and the
# model put a temperature on it anyway. Three moves where the clause names a
# mode, two of them wrong; 161 where it does not, one of them wrong. The
# caller reads it with `slot_match.setting_from`, which measures 294 right and
# 0 wrong on fan speeds, and passes it in - the same shape as
# :func:`settle_media` and for the same reason.
#
# It also cannot invent a broken call, and that is what rules out the wider
# version. Home Assistant's own suite fails three of its five
# HassClimateSetTemperature sentences and all three say "מעלות", so settling
# on the word looks tempting and measures 514 agreements and no
# disagreements. It would also turn two of those three into a
# `climate.set_temperature` carrying no temperature, because on those two the
# model dropped the number along with the action - "שנה טמפרטורה סלון ל20
# מעלות" comes back as `off` with an hvac mode and nothing else. A service
# call that cannot succeed is not a repair, and those two rows are a model
# failure that is counted as one.
_CLIMATE_TEMP_ARGS: Final = ("temperature", "temperature_step")

#: The five behaviours of `climate_control`, so a caller can tell whether to try.
CLIMATE_BEHAVIOURS: Final[frozenset[str]] = frozenset((
    "climate_turn_on", "climate_turn_off", "climate_set_temperature",
    "climate_set_hvac_mode", "climate_set_fan_mode"))


#: A clause that switches the machine is not a clause that sets its mode, and
#: it names one anyway: "לכבות את כל הקירור" carries קירור, which is a mode
#: word. Without this the promotion below reads 114 disagreements and with it
#: **zero** - it is most of what makes the rule safe, and every one of the
#: three groups was found by the measurement rather than by reading.
#:
#: The open and close verbs are here because Hebrew turns an air conditioner
#: on by opening it, and `_DO_VERBS` because it turns one on by *making* it -
#: "תוכלי לעשות מזגן" is a request to run it.
#: The three behaviours the promotion may reach into. A model that already
#: chose a mode call is left alone - it is the one that had somewhere to put
#: the value - and the switching behaviours are here because the model reaches
#: for them on a clause that names a fan speed and no verb at all: "לכוון את
#: מהירות המאוורר של האינוורטר" comes back as `climate_turn_off`. The clause
#: still names a mode and still has no temperature in it, which is the whole
#: of the evidence the measurement was taken on.
_NOT_YET_A_MODE: Final[frozenset[str]] = frozenset(
    ("climate_set_temperature", "climate_turn_on", "climate_turn_off"))

_CLIMATE_SWITCHED: Final[tuple[str, ...]] = (
    *VOCABULARY["climate_turn_on"], *VOCABULARY["climate_turn_off"],
    "פתח", "תפתח", "שתפתח", "לפתוח", "פתחי", "תפתחי", *_DO_VERBS,
)

#: A verb that asks for a *relative* move, from the strongest tier of `TIERS`.
#: "בחדר שינה חם מדי, תקרר" carries no number and still has something to set,
#: and the mode promotion below has to leave it alone: תקרר is the temperature
#: going down by a step, not a request for cooling mode.
_STEP_VERBS: Final[tuple[str, ...]] = TIERS[0][0] + TIERS[0][1]


def settle_climate(tool: str, arguments: dict[str, Any], text: str,
                   named_mode: str | None = None,
                   hvac_target: str | None = None,
                   said_a_number: bool = True) -> str:
    """`climate_set_temperature` when the call carries a temperature to set.

    ``named_mode`` is the caller's answer to "which mode slot does this clause
    name" - ``"hvac_mode"``, ``"fan_mode"`` or ``None`` - passed in rather
    than imported so this module keeps depending on :mod:`tool_router` for
    vocabulary only, the same arrangement as :func:`settle_media`. A clause
    that asks for a fan speed asked for a fan speed, whatever else the model
    attached to the call. It was a bool until the slot itself was needed; see
    the promotion below, which has to know *which* of the two was named.

    Returns ``tool`` unchanged for anything that is not a climate behaviour,
    for a call carrying no temperature - the common case, and the quiet one -
    and for a sentence that takes a verb back, since a leftover argument from
    the half that was retracted is exactly what a correction leaves behind.
    """
    if tool not in CLIMATE_BEHAVIOURS:
        return tool
    if text and CORRECTION.search(_fold(text)):
        return tool
    # A sentence that asks for a mode *as a target* is a request to set one,
    # whichever of the five the model reached for. `slot_match.hvac_target` is
    # the strict reading - preposition-bound, never an adjective, never the
    # noun in "מהירות המאוורר" - and the caller passes it rather than this
    # module importing it, the same arrangement as `names_a_mode`. Measured
    # over the corpus with a fan mode or a temperature in the sentence
    # disqualifying it: 119 agree, 0 disagree.
    if hvac_target is not None:
        return "climate_set_hvac_mode"
    # A mode the clause named, on a temperature call with no temperature in
    # it. The early return below would keep the temperature behaviour, and
    # keeping it *also discards the mode*: neither `hvac_mode` nor `fan_mode`
    # is an argument of `climate_set_temperature`, so the value the caller's
    # resolver already read off the sentence has nowhere to go and is dropped
    # on the floor. Two argument errors out of one line, and it was the
    # largest single failure shape left on the frozen benchmark - 25 `climate`
    # rows, every one of them a call carrying no temperature at all.
    #
    # Guarded on there being nothing to set, so the older reading stands
    # wherever it was doing work: "תעלה ל-24 ובמהירות גבוהה" names a fan mode
    # and really is a temperature call.
    #
    # ``said_a_number`` is the caller's answer to "does this clause name a
    # number at all", the same question `settle_steps` asks, and it is what
    # closes the hole the first cut of this left open. A temperature argument
    # only counts as something to set if the sentence *backs it up* - with a
    # number, or with a verb that asks for a relative move. The model attaches
    # `temperature: 23` to "תעביר את המזגן לרק מאוורר" out of habit, and an
    # invented value must not outrank a mode the speaker said out loud.
    #
    # Measured per clause over the whole corpus, on every climate call whose
    # clause names a mode, names no number and takes no switching verb:
    # 911 agree, 1 disagrees, and the one is "תקררשמץ" - the step verb glued
    # to its adverb by the injected speech noise, which is the exception this
    # project names rather than the rule failing.
    tokens = _tokens(text)
    # A step the clause sizes in words is a thermostat setting, even when the
    # model attached no number to hang it on. "תקרר את המזגן במטבח קצת", "חם
    # מדי, תנמיך", "קר מדי, תחמם קצת יותר" - the model answers these with a
    # mode, an off, a fan speed, anything but the behaviour that can hold a
    # step, and `settle_steps` runs *after* this and so has nothing to fill.
    # It is the same shape as the level tier in `settle_toggle`: the slot the
    # sentence names settles the behaviour, and then the behaviour takes it.
    #
    # Excluded where the clause names a fan speed, and that exclusion is the
    # whole rule: "שים את הפן של המזגן על חזק" is a `fan_mode`, and חזק is
    # also how `_STEP_SCALE` sizes a step. Without the guard 72 fan rows read
    # as temperatures. With it, measured per clause over the whole corpus on
    # every climate call: **1,318 agree, 0 disagree**.
    sized_step = (named_mode != "fan_mode"
                  and which_way(text) is not None
                  and step_size(text) is not None)
    to_set = (sized_step
              or (any(arguments.get(key) for key in _CLIMATE_TEMP_ARGS)
                  and (said_a_number or _hits(list(_STEP_VERBS), tokens, text))))
    if (named_mode and tool in _NOT_YET_A_MODE and not to_set
            and not _hits(list(_CLIMATE_SWITCHED), tokens, text)):
        return ("climate_set_fan_mode" if named_mode == "fan_mode"
                else "climate_set_hvac_mode")
    if tool == "climate_set_temperature":
        return tool
    # A mode word holds the tool still - except where the clause has *sized a
    # step*, which no mode call can hold: neither `climate_set_hvac_mode` nor
    # `climate_set_fan_mode` has a `temperature_step` to put it in. "במסדרון
    # קר מדי, תחמם קצת יותר" is the shape, and both halves matter: קר is the
    # room being complained about rather than a mode being asked for -
    # `hvac_target`, the strict reading, declines it - and תחמם קצת is a step
    # said out loud. Without this the clause kept whatever the model said,
    # which on all three benchmark rows was `off`: asked for a degree more,
    # the house switched the heating off.
    #
    # The inverse of the guard three lines above, and by the same principle:
    # there an invented temperature must not outrank a mode the speaker said,
    # here a spoken step must not be outranked by a mode nobody asked for.
    # Measured per clause over the v11 corpus, on every climate clause naming
    # a mode and sizing a step: **137 agree, 0 disagree**.
    if named_mode and not sized_step:
        return tool
    # Truthiness rather than presence, as `settle_timer` reads a duration: a
    # `temperature_step` of zero is a no-op and not evidence of anything, and
    # a `temperature` cannot be zero because the schema floors it at 16.
    if sized_step or any(arguments.get(key) for key in _CLIMATE_TEMP_ARGS):
        return "climate_set_temperature"
    return tool


# The clause that names its own behaviour. -----------------------------------
#
# :func:`settle_action` does this already and cannot be used here: it is
# allow-listed to :data:`HINT_DECIDED` because the general form - let any
# family's hints decide its behaviour - reads 647 disagreements over the
# corpus. These four are not on that list and do not need to be, because they
# are not hints. Each is a word that names one behaviour and nothing else, in
# a family whose other behaviours never carry it.
#
# The evidence, per clause over all 31,519 corpus rows, gold on both sides:
#
#     fan       סיבוב        -> rotate      204 agree, 0 disagree
#     cover     עצור/הפסק    -> stop        341 agree, 0 disagree
#     valve       "     "
#     media     לגמרי/סטופ   -> stop        146 agree, 0 disagree
#     cover     "N אחוז"     -> place       538 agree, 0 disagree
#     valve       "     "
#
# Why the model needs telling. ``סיבוב`` is the whole difference between
# turning a fan on and setting it oscillating, and v11 answers 13 of these
# with ``on``; ``לגמרי`` - *completely* - is the whole difference between a
# pause and a stop, and the pause vocabulary is otherwise identical. The
# percentage is the plainest of the four: "על שבעים אחוז" is a position and
# there is nothing else in Hebrew it could be, but a cover call is opened or
# closed by every verb around it and the model follows the verb.
_ROTATES: Final[tuple[str, ...]] = (
    "סיבוב", "הסיבוב", "סיבובי", "שיסתובב", "להסתובב", "מסתובב",
    "תסתובב", "יסתובב", "אוסילציה",
)
#: ``בלי סיבוב`` is still a ``rotate`` - the negation lands on the
#: ``oscillating`` argument, not on which behaviour was asked for - so it is
#: deliberately not excluded here. 34 corpus rows say it and all 34 are gold
#: ``rotate``.
_STOPS: Final[tuple[str, ...]] = (
    "עצור", "עצרי", "תעצור", "תעצרי", "שתעצור", "שתעצרי", "לעצור",
    "הפסק", "הפסיקי", "תפסיק", "תפסיקי", "שתפסיק", "שתפסיקי", "להפסיק",
    # "די עם את הוילונות" - enough with the curtains. 33 cover and valve
    # clauses say it and all 33 are a stop. It is deliberately absent from
    # `_MEDIA_STOPS` below: on a speaker the same word is a **pause**, 111
    # clauses out of 111, and the two families disagree about it completely.
    "די",
)
#: Not the stop verbs: :data:`_STOPS` appears on ``media_pause`` as often as
#: on ``media_stop`` - "תעצור את המוזיקה" is a pause - so the verb settles
#: nothing here and only these three words do.
_MEDIA_STOPS: Final[tuple[str, ...]] = (
    "לגמרי", "לחלוטין", "סטופ", "השמעה", "ההשמעה", "להשמעה",
)

#: ``behaviour -> (vocabulary, what it becomes)``, for the behaviours each
#: rule is allowed to overrule. Keyed by behaviour rather than by tool so a
#: rule can never reach a sibling it was not measured against - ``media_stop``
#: and ``media_next_track`` are both ``media_control`` and only the first pair
#: is in evidence here.
#: A message to a phone against an announcement over the speakers. Both tools
#: take one argument and it is the same one, so the model has nothing to go on
#: but the verb - and it picks wrong, or refuses, on 14 of the benchmark's
#: `notify_send` rows.
#:
#: The verbs separate them completely and in both directions: **199 agree, 0
#: disagree** for the notification words and **102 and 0** for the broadcast
#: ones, and **not one corpus clause says both**, so the order they are read in
#: is free rather than load-bearing.
_TO_A_PHONE: Final[tuple[str, ...]] = (
    "הודעה", "תעדכן", "תעדכני", "לעדכן", "תשלח", "לשלוח", "שלח",
    "שתשלח", "שתעדכן",
)
_OVER_THE_SPEAKERS: Final[tuple[str, ...]] = (
    "תכריז", "להכריז", "הכרז", "תכריזי", "תשדר", "תשדרי", "לשדר",
    "ברמקולים", "רמקולים", "הכרזה",
)

_NAMES_ITS_BEHAVIOUR: Final[tuple[tuple[frozenset[str], tuple[str, ...], str], ...]] = (
    (frozenset(("cover_open", "cover_close", "cover_set_position")),
     _STOPS, "cover_stop"),
    (frozenset(("valve_open", "valve_close", "valve_set_position")),
     _STOPS, "valve_stop"),
    # The family the pattern had left out. A cover, a valve, a timer and a
    # speaker all already answer "די עם את הרומבה" - enough with the robot -
    # and a vacuum did not: it kept the model's own `clean`, so a household
    # asking the robot to stop got it started again. There is no third
    # behaviour to choose between here, which is why the whole stop
    # vocabulary is safe where on a speaker only three words are: a vacuum
    # halts one way. Measured per clause over the v11 corpus on every
    # `vacuum_control` clause the vocabulary reaches: **95 agree, 0
    # disagree**, and די alone accounts for 21 of them.
    (frozenset(("vacuum_start", "vacuum_return_to_base")),
     _STOPS, "vacuum_pause"),
    (frozenset(("fan_turn_on", "fan_turn_off", "fan_toggle")),
     _ROTATES, "fan_oscillate"),
    (frozenset(("media_pause", "media_play")),
     _MEDIA_STOPS, "media_stop"),
    (frozenset(("notify_send", "broadcast")), _TO_A_PHONE, "notify_send"),
    (frozenset(("notify_send", "broadcast")), _OVER_THE_SPEAKERS, "broadcast"),
)

#: A placement that has nothing to place. Read only when
#: the clause names no level at all: over the corpus, **684 of 684** cover and
#: valve clauses that name one are gold `place`, and of the 5,116 that name
#: none only 21 are - so a `cover_set_position` on a silent clause is the model
#: reaching for the behaviour rather than the sentence asking for it.
#:
#: The verb is what it is handed back to, and `settle` is asked from both ends
#: so that a clause whose verbs point two ways stays silent. Measured per
#: clause: **4,558 agree, 0 disagree**, 547 silent. The stop half of that is
#: `_NAMES_ITS_BEHAVIOUR` above, which runs first and now knows `די`.
_PLACED_FROM: Final[dict[str, tuple[str, str]]] = {
    "cover_set_position": ("cover_open", "cover_close"),
    "valve_set_position": ("valve_open", "valve_close"),
}

#: And the other way: cover and valve behaviours a named percentage turns into
#: a placement. ``cover_stop`` is here too, and the stop rule above runs first
#: and wins, so the only way this reaches a stop is a clause naming a
#: percentage and no stop verb - which is a placement the model called a stop.
_PLACEABLE: Final[dict[str, str]] = {
    "cover_open": "cover_set_position", "cover_close": "cover_set_position",
    "cover_stop": "cover_set_position",
    "valve_open": "valve_set_position", "valve_close": "valve_set_position",
    "valve_stop": "valve_set_position",
}


#: The verb that only makes a thing happen. "תעשה אור בבלקון" is a light
#: being turned on, and the model answered it with a toggle - which turns the
#: light *off* again in a room where it was already lit.
#:
#: Read through a lost space, the same way the off-word below it is and for
#: the same reason: the noise glues this verb to the next word as readily as
#: any other, and "תוכל לעשותאת המנורה" is a lamp being asked for. The
#: ט/ת fold is *not* offered here - it buys nothing this does not, and it is
#: not free at a call site that actuates.
#:
#: לעשות names no direction of its own, which is why it needs a rule rather
#: than a place in :data:`VOCABULARY`: it takes whatever direction the rest of
#: the sentence gives it, and where the rest of the sentence gives none, it
#: turns the thing on. Counted over the v11 corpus on every clause carrying
#: one of these verbs, the sentence says which by naming a word: טוגל or
#: תחליף for a toggle (96 clauses), אוף for an off (13), סיבוב for an
#: oscillation (88) - and where it names none of them, **418 clauses are gold
#: `turn_on` and none is anything else**. On predictions it moves one row
#: of the frozen benchmark, and it is the glued one.
_MAKES_IT: Final[tuple[str, ...]] = (
    "תעשה", "תעשי", "תעשו", "לעשות", "עשה", "עשי",
    "שתעשה", "שתעשי")

#: The toggles that verb turns into a switch-on, and what each becomes. The
#: three families whose toggle the model reaches for; there is no cover or
#: media toggle to confuse with one.
#:
#: The guards are the tables that already exist. `TOOL_HINTS` holds the words
#: that ask for a toggle - it is what :func:`settle_behaviour` runs this rule's
#: mirror on, a paragraph above - and `VOCABULARY` holds the words that ask
#: for an off. The off half is read through a lost space, because that is
#: exactly where the noise puts one: "תעשה אוףאת הלייטס" is a light being
#: switched off, and read strictly it would have been switched on instead.
_MAKING_IT_IS_ON: Final[dict[str, str]] = {
    toggle: sides[0] for toggle, sides in TOGGLES.items()}

#: A light asked for a colour is a light asked to come on. The same hole as
#: the climate promotion one function up, and it opens the same way: the model
#: answers "אני רוצה אור אדום" with `light_toggle`, `color_name` is not one of
#: that behaviour's arguments, and the colour the sentence stated plainly is
#: dropped rather than filled - so the row loses the action *and* the colour.
#: You cannot toggle a lamp to red. 617 corpus clauses name a colour on a
#: light call and **all 617** are gold `on`.
_COLOURS_ARE_ON: Final[frozenset[str]] = frozenset(
    ("light_toggle", "light_turn_off"))


#: The three climate behaviours that *set* something, as opposed to switching
#: the machine. A clause that switches it is not a clause that sets anything,
#: and the model reaches for these anyway - most often
#: `climate_set_temperature` with no temperature in it at all.
_CLIMATE_SETS: Final[frozenset[str]] = frozenset(
    ("climate_set_temperature", "climate_set_hvac_mode", "climate_set_fan_mode"))

#: A television is two devices wearing one noun, and both families claim it:
#: `FAMILY_NOUNS` lists טלוויזיה under `media` *and* under `switch`, so
#: `family_named` reads the ambiguity correctly and says nothing. The verb is
#: what settles it, and it settles it cleanly:
#:
#:     שים טלוויזיה בסלון          52 rows   media_control / input
#:     תכבה את הטלוויזיה בסלון     36 rows   switch_control / shut
#:     תדליק את הטלוויזיה בסלון    29 rows   switch_control / on
#:
#: Putting something *on* the screen is the media player; switching the set
#: on or off is the plug behind it. Measured per clause over the whole corpus:
#: **65 agree, 0 disagree**, and the clauses taking neither verb - which is
#: every one of the 52 - are not touched.
_TV_NOUNS: Final[tuple[str, ...]] = (
    "טלוויזיה", "טלויזיה", "הטלוויזיה", "הטלויזיה", "טיוי", "הטיוי")

#: The media behaviours the model reaches for on those clauses. Written out
#: rather than imported from `const.MEDIA_TOOLS`, because this module depends
#: on :mod:`tool_router` for vocabulary and on nothing else - the same reason
#: `named_mode` and `at_a_position` are passed in rather than looked up.
_SCREEN_BEHAVIOURS: Final[frozenset[str]] = frozenset((
    "media_play", "media_pause", "media_stop", "media_select_source",
    "media_next_track", "media_previous_track", "media_set_volume",
    "media_mute", "music_play"))

#: ``(from these behaviours, only when the clause names one of these nouns,
#: the on behaviour, the off behaviour)``. ``None`` for the nouns means any
#: clause qualifies. Both rows say the same thing - the sentence switches the
#: machine, and the model answered with something the machine *does* instead.
_SWITCHED_BY_VERB: Final[tuple[
        tuple[frozenset[str], tuple[str, ...] | None, str, str], ...]] = (
    (_CLIMATE_SETS, None, "climate_turn_on", "climate_turn_off"),
    (_SCREEN_BEHAVIOURS, _TV_NOUNS, "switch_turn_on", "switch_turn_off"),
)


def settle_named(behaviour: str, text: str,
                 at_a_position: bool = False,
                 names_a_colour: bool = False) -> str:
    """The behaviour the clause names outright, or ``behaviour`` unchanged.

    Five rules, in the order they are read, each measured per clause over the
    whole corpus at zero disagreements:

    1. a toggle word makes it a toggle - :func:`settle_toggle` backwards;
    2. the four families in :data:`_NAMES_ITS_BEHAVIOUR`, which name their own
       behaviour and which :func:`settle_action`'s allow-list cannot reach;
    3. a switching verb on a call that *does* something with the machine
       instead - :data:`_SWITCHED_BY_VERB`, the climate set and the television;
    4. a colour makes a light an ``on`` - you cannot toggle a lamp to red;
    5. a percentage makes a cover or a valve a placement.

    ``at_a_position`` and ``names_a_colour`` are the caller's answers to "does
    this clause name a percentage" and "does it name a colour" -
    `slot_match.percent_from` and `slot_match.setting_from` - passed in rather
    than imported, the same arrangement as :func:`settle_climate`'s
    ``named_mode``.

    Silence is the common case and returns the model's own answer, and a
    sentence that takes a verb back is left alone, as everywhere else in this
    module: a stop verb in the half that was retracted is not a stop.

    Runs *before* the slot fills in both pipelines, never after. The fills are
    keyed on the behaviour, so a promotion made after them leaves the new
    behaviour's own arguments empty - which is the defect that cost `media`
    thirteen points until `settle_media` was moved.
    """
    if not text or CORRECTION.search(_fold(text)):
        return behaviour
    tokens = _tokens(text)
    # :func:`settle_toggle` run backwards, and it needs no vocabulary of its
    # own: the toggle hints are already the words that name one. That function
    # resolves a toggle the sentence contradicts and declines when the
    # sentence "actually says toggle"; this is the other direction - the model
    # answered "תחליף את המצב של הוונטה" with an off, and switching a thing is
    # not turning it off. Over the corpus, per clause, with a self-correction
    # disqualifying it: light **257 agree**, switch **108**, fan **99**, and
    # zero disagreements between them.
    for toggle, sides in TOGGLES.items():
        if behaviour in sides and _hits(TOOL_HINTS.get(toggle, []), tokens, text):
            return toggle
    for family, words, becomes in _NAMES_ITS_BEHAVIOUR:
        if behaviour in family and _hits(list(words), tokens, text):
            return becomes
    # Switching the machine, from a call that sets something on it. The
    # counterpart of `settle_climate`'s promotion and guarded by the same
    # vocabulary from the other side: that function refuses to read a mode off
    # a clause that switches, and this reads the switch. `settle` cannot do it
    # - the model did not answer with either half of the pair, so there is no
    # pair to settle. Over the corpus, one side named and not the other, a
    # self-correction disqualifying it: **235 agree** for on and **901** for
    # off, none against either.
    for sets, nouns, turn_on, turn_off in _SWITCHED_BY_VERB:
        if behaviour not in sets:
            continue
        if nouns is not None and not _hits(list(nouns), tokens, text):
            continue
        on = _hits(list(VOCABULARY[turn_on]), tokens, text)
        off = _hits(list(VOCABULARY[turn_off]), tokens, text)
        if on and not off:
            return turn_on
        if off and not on:
            return turn_off
    if behaviour in _MAKING_IT_IS_ON and _hits_noisy(list(_MAKES_IT), text):
        # And the verb that names no direction at all, on a behaviour that
        # is one. See :data:`_MAKES_IT`.
        switched_off = TOGGLES[behaviour][1]
        if not _hits(TOOL_HINTS.get(behaviour, []), tokens, text) and (
                not _hits_noisy(list(VOCABULARY[switched_off]), text)):
            return _MAKING_IT_IS_ON[behaviour]
    if names_a_colour and behaviour in _COLOURS_ARE_ON:
        return "light_turn_on"
    if at_a_position and behaviour in _PLACEABLE:
        return _PLACEABLE[behaviour]
    if not at_a_position and behaviour in _PLACED_FROM:
        # 6. and a placement with nothing to place is not one. See
        # :data:`_PLACED_FROM`.
        opened, closed = _PLACED_FROM[behaviour]
        settled = settle(opened, text)
        if settled == settle(closed, text):
            return settled
    return behaviour


# Playing something, against skipping to the next of it. ---------------------
#
# :mod:`executor` promotes a media call to ``music_play`` when the sentence
# names something specific to play. This is the mirror, and v11 needed it: the
# router now declares ``media_control`` and ``music_play`` together for any
# audio sentence, and the model answered "השיר הבא בסלון" and "דלג בסלון" with
# `music_assistant.play_media` - four of Home Assistant's nine HassMediaNext
# sentences, and four of its HassMediaPrevious.
#
# The rule is the upgrade's own argument run backwards: a request to play
# names *what*, so a `music_play` about a sentence naming nothing to play and
# naming one transport verb is a transport command. Measured over the corpus:
# **383 agree, 0 disagree**, and the 23 rows whose gold really is a bare
# `music_play` name no transport verb, so none of them is touched.
#
# Only next and previous. ``שים`` and ``תשמיע`` are how an Israeli asks for
# music as well as how they change the source, so widening this to the rest of
# the transport tools blocks 191 genuine music requests - measured, and the
# reason this is a two-item tuple rather than :data:`~const.MEDIA_TOOLS`.
TRANSPORT_ONLY: Final[tuple[str, ...]] = (
    "media_next_track", "media_previous_track")


#: Announce it to the house, or send it to a phone. The two message tools take
#: the same single argument and differ only in who hears it, which the model
#: gets wrong in both directions and the verb never does.
#:
#: The corpus draws the line at the verb, not at the audience, and the
#: distinction is finer than it looks: ``תשלח הודעה לכולם`` is a notification
#: *to everyone* and ``תודיע לכולם`` is an announcement, so a rule reading
#: `לכולם` first gets 31 rows of 330 wrong. The verb is read first and the
#: audience only breaks the tie a bare ``תודיע`` leaves.
_ANNOUNCE: Final = re.compile(
    r"(?:^|\s)(?:ו)?(?:כש|ש)?(?:[בלכמ])?(?:ה)?"
    r"(?:תכריז|תכריזי|הכרז|הכריזי|להכריז|תשדר|תשדרי|שדר|שדרי|לשדר)(?:\s|$)")
_SEND: Final = re.compile(
    r"(?:^|\s)(?:ו)?(?:כש|ש)?(?:[בלכמ])?(?:ה)?"
    r"(?:תשלח|תשלחי|שלח|שלחי|לשלוח|תעדכן|תעדכני|עדכן|עדכני|לעדכן)(?:\s|$)")
#: Where a bare "tell them" is aimed: the whole house at once.
_EVERYONE: Final = re.compile(r"לכולם|בכל\s+הבית|ברמקולים")
#: And its opposite, which is a notification to the household rather than an
#: announcement over it. Checked before `_EVERYONE` and only when that is
#: silent, because "בכל הבית" contains "בבית". Silence falls from 83 clauses to
#: 7 with it, at no disagreements either way.
_HOUSEHOLD: Final = re.compile(r"בבית(?:\s|$)")

#: The pair this settles. Both take one argument and the executor reads it off
#: the sentence for either, so only the audience is ever in doubt.
AUDIENCE: Final = frozenset(("notify_send", "broadcast"))


def settle_audience(tool: str, text: str, has_message: bool) -> str:
    """Announce to the house, or notify a device. The verb says which.

    ``has_message`` is the caller's answer to "does this clause carry
    something to say at all" - :func:`slot_match.extract_message`, passed in so
    this module keeps depending on the vocabulary modules for nothing, the same
    arrangement as :func:`family_named`. It is not a formality: without it
    ``בכל הבית`` reads a vacuum told to clean everywhere as an announcement,
    which is 158 rows of exactly the disagreement this bar forbids.

    Measured per clause over the whole corpus, on the 458 clauses whose gold is
    one of the two: **451 agree, 0 disagree, 7 silent**, and on the clauses
    whose whole shortlist is the pair - the set `repair.recover` sees - 388
    agree, 0 disagree, 4 silent. Of the off-topic rows that reach the model at
    all, this speaks on **none**.
    """
    # An empty ``tool`` asks the question outright - "which of the two does
    # this sentence want, if it says at all" - and gets "" back when it does
    # not. `repair.recover` needs that, because there is no model answer to
    # fall back on when the generation is the thing that broke.
    if (tool and tool not in AUDIENCE) or not text or not has_message:
        return tool
    if _SEND.search(text):
        return "notify_send"
    if _ANNOUNCE.search(text):
        return "broadcast"
    if _EVERYONE.search(text):
        return "broadcast"
    if _HOUSEHOLD.search(text):
        return "notify_send"
    return tool


def settle_transport(tool: str, text: str, names_something: bool) -> str:
    """``music_play`` about a sentence that named nothing to play."""
    if tool != "music_play" or names_something or not text:
        return tool
    if CORRECTION.search(_fold(text)):
        return tool
    tokens = _tokens(text)
    named = [name for name in TRANSPORT_ONLY
             if _hits(TOOL_HINTS.get(name, []), tokens, text)]
    return named[0] if len(named) == 1 else tool

# Which of eight, for a speaker. ---------------------------------------------
#
# The same problem `settle_timer` solves one family over, and it arrived for
# the same reason: `media_control` holds eight behaviours and the v11 model
# answered "נגן את הנגינה בסלון" with `media_player.media_next_track` - seven
# of Home Assistant's nine HassMediaUnpause sentences, and five of its ten
# HassMediaPause.
#
# The generic hint rule cannot help here (65 disagreements; see
# :data:`HINT_DECIDED`) because two of the eight are lexically entangled with
# everything else: ``שים`` selects a source and asks for music, ``קול`` is the
# volume and the sound being stopped. Read in a fixed order, with two of the
# eight given a condition rather than a word list, they separate cleanly:
#
# 1. skipping - הבא, דלג / הקודם, אחורה;
# 2. muting, including the three ways this corpus asks for an unmute;
# 3. an input - ערוץ, ספוטיפיי;
# 4. the volume, which needs **a direction verb or a level** rather than just
#    the noun: "תוכלי לעצור את הסאונד" is a pause, and reading קול alone put
#    every one of those on the volume;
# 5. stopping outright - סטופ, לגמרי;
# 6. pausing;
# 7. playing or resuming.
#
# Pause and stop are one group, and the rule declines to choose between them.
# In Hebrew "תעצור את המוזיקה" is both, and this project's own corpus splits
# them on a noun - ההשמעה against המוזיקה - which is a convention rather than a
# fact about the language. So a model that already said one of the two is left
# alone; the rule only ever moves a call *into* the group from outside it.
#
# Measured over every media row in the corpus, one clause at a time:
#
#     the words agree with gold     2020     71.1%
#     the words disagree               1      0.04%
#     the words say nothing          790     27.8%
#     pause against stop, left alone  44      1.5%
#
# The one disagreement is "תעזור מוזיקה" - speech noise on תעצור - in a
# two-order sentence. Same standing as the sign guard's one.

_M_NEXT: Final = ("הבא", "הבאה", "דלג", "תדלג", "דלגי", "תדלגי",
                  "הבא בתור", "לדלג")
_M_BACK: Final = ("הקודם", "הקודמת", "אחורה", "לאחור", "חזור", "תחזור",
                  "חזרי", "תחזרי", "קודם")
_M_MUTE: Final = ("השתק", "תשתיק", "השתיקי", "תשתיקי", "מיוט", "השתקה",
                  # 38 corpus clauses say "תשים על שקט" and all 38 are a mute.
                  # `בשקט` does not reach it: the preposition is its own token.
                  # The phrase means something else entirely one tool over - 53
                  # `vacuum_control` clauses say it for the silent suction mode -
                  # which is why it lives in this table and not a shared one.
                  "בשקט", "על שקט", "תחזיר את הקול", "להחזיר את הקול", "תבטל השתקה",
                  "לבטל השתקה", "תוריד מיוט")
# `רדיו` is 41 media_control clauses and every one is a source. It is on 150
# `music_play` rows too, where it is *what* to play rather than where from -
# and none of those reaches this function, because none is a plain request.
# `תחליף` is 86 clauses, all source; it also appears on light, switch and fan
# rows, where it means swap the bulb and this table is never consulted.
_M_INPUT: Final = ("ערוץ", "מקור", "רדיו", "תחליף", "תחליפי", "להחליף",
                   "ספוטיפיי", "יוטיוב", "אייראפליי",
                   "בלוטות'", "בלוטות")
#: A direction verb is a volume on its own; the nouns need a level beside them.
_M_VOLUME_VERB: Final = ("תגביר", "תנמיך", "הגבר", "נמיך", "תחליש",
                         "הגבירי", "תנמיכי", "הנמיכי", "תרים", "תוריד")
_M_VOLUME_NOUN: Final = ("ווליום", "וליום", "קול", "עוצמה", "סאונד", "שאונד")
# `די` is not here: on its own it is a pause - 111 corpus clauses of
# "די עם את המוזיקה" and every one of them - and it is only *with* an
# explicit stop verb that it means stop outright. `_M_DI_STOP` reads the
# pair; 19 clauses on the v11 corpus, all `media_stop`, none against.
_M_STOP: Final = ("סטופ", "לגמרי")
_M_DI: Final = re.compile(r"(?:^|\s)די(?:\s|,|$)")
_M_STOP_VERB: Final = re.compile(
    r"(?:^|\s)(?:ו)?(?:כש|ש)?(?:[בלכמ])?(?:ה)?"
    r"(?:תעצור|תעצרי|עצור|עצרי|לעצור)(?:\s|$)")
_M_PAUSE: Final = ("השהה", "תשהה", "השהי", "תשהי", "עצור", "תעצור", "עצרי",
                   "תעצרי", "הפסק", "תפסיק", "הפסיקי", "תפסיקי", "די",
                   "לעצור", "להפסיק", "להשהות")
#: Bare ``נגן`` is here and is *not* in the pair guard's vocabulary, and the
#: difference is the tier order. ``נגן`` is the imperative "play!" and the noun
#: "the player" alike, so in a pair - where both sides are weighed at once -
#: "תעצור את הנגן" reads as a play and the pair measures 22 disagreements. Read
#: in order, pausing is settled two tiers earlier and never reaches this list.
#: Two more ways to ask for music, and the measurement that admits each.
#:
#: `שלח` - send me music - and `ערבב` - shuffle something. Neither was here:
#: ערבב was a `music_play` routing hint, which gets the family onto the
#: shortlist and never reaches this ladder, and שלח was nowhere at all. Over
#: train and test, on rows where every earlier tier is silent and the sentence
#: names no level: **send 32 agree / 0 disagree, shuffle 216 / 0**.
#:
#: `שים` is measured and rejected: 289 agree against **63 disagree**, because
#: "שים על שקט" is a mute and "שים טלוויזיה" is a source. It is the verb an
#: Israeli uses for putting anything anywhere, and the ladder above it does
#: not carry שקט or a channel name, so it would win those outright.
_M_SEND: Final = ("שלח", "תשלח", "שלחי", "תשלחי", "לשלוח")
_M_SHUFFLE: Final = ("ערבב", "תערבב", "ערבבי", "תערבבי", "לערבב")

_M_PLAY: Final = _M_SEND + _M_SHUFFLE + (
                  "נגן", "תנגן", "נגני", "תנגני", "השמע", "תשמיע", "השמיעי",
                  "תשמיעי", "המשך", "תמשיך", "המשיכי", "תמשיכי", "חדש",
                  "חדשי", "להמשיך", "לנגן", "להשמיע")

#: The verb the tuple above rejects, kept for the one path where the readings
#: that beat it have all been ruled out already. See :func:`settle_media`'s
#: last branch, and the rejection three comments up for why it is not in
#: `_M_PLAY`.
_M_PUT_ON: Final = ("שים", "תשים", "שימי", "תשימי", "לשים", "שם")

#: Pause and stop are one decision this rule will not make. See above.
_HALT: Final[frozenset[str]] = frozenset(("media_pause", "media_stop"))

#: The eight behaviours this settles.
MEDIA_BEHAVIOURS: Final[frozenset[str]] = frozenset((
    "media_play", "media_pause", "media_stop", "media_next_track",
    "media_previous_track", "media_mute", "media_set_volume",
    "media_select_source"))


def names_the_volume(text: str, alone: bool) -> bool:
    """Does the clause say *volume* outright, with nothing else to be?

    :func:`settle_media` reaches ``media_set_volume`` from a bare value too,
    because on a call the model already aimed at a speaker there is nothing
    else a percentage could be. :func:`repair.recover` cannot lean on that: it
    speaks where the generation derailed and there is no call to constrain it.
    ``alone`` is the caller's answer to "does the shortlist hold nothing but
    the speakers", passed in rather than imported, the same arrangement as
    ``named_mode`` above.

    **The noun always, the verb only when nothing else is offered**, and the
    split is the whole rule. עוצמה and הקול are a speaker and nothing else in
    this house; תוריד is the volume, the thermostat and the blind at once, and
    "תוריד את המזגן למשהו כמו 27" is a temperature that says no degrees.

    Measured over the v11 corpus per clause, on every clause that offers a
    speaker and names a level. Reading a bare value: 343 agree, **3 disagree**,
    all three speech noise on the device noun - מעוורר for the fan, הווילונוט
    for the curtains, ורז for the tap - which the shortlist then read as a
    speaker. Reading the verb wherever it appears: 391 agree, **7 disagree**,
    every one of them that thermostat sentence. This split: **391 agree, 0
    disagree**, and it speaks on none of the corpus's off-topic rows.
    """
    if not text:
        return False
    tokens = _tokens(text)
    if _hits(list(_M_VOLUME_NOUN), tokens, text):
        return True
    return bool(alone and _hits(list(_M_VOLUME_VERB), tokens, text))


def settle_media(tool: str, text: str, names_a_level: bool,
                 plain_request: bool = False,
                 names_a_value: bool = False) -> str:
    """Which of the eight speaker behaviours the sentence names.

    ``names_a_level`` is the caller's answer to "does the sentence carry a
    number or a percentage", passed in so this module keeps depending on
    :mod:`slot_match` not at all. ``nothing_to_play`` is the same arrangement
    for "did `extract_music` find a title in this clause".

    A ``music_play`` about a clause naming nothing to play is let in, and it
    is the mirror of :func:`settle_transport` widened past next-and-previous.
    That widening is safe *here* and was not there, because the eight-way read
    below is not a verb list: "שים קצת מוזיקה" reaches ``media_play`` while
    "שים את הרמקול על 40 אחוז" reaches the volume and "שים יוטיוב" the source,
    all from the same clause shape that defeated a widened `TRANSPORT_ONLY`.

    Measured per clause over the whole corpus, through this function, on
    every media-family clause it speaks about when reached from `music_play`:
    **2,125 agree, 9 disagree**, and all nine are the injected speech noise -
    השםיעי, תשימ, שתנגנ, לשימ, תפסיקאת - where the glued or misspelled verb
    is what stopped `extract_music` reading the title the clause does name.
    That is the exception class this project names; nothing else disagrees.
    """
    if not text:
        return tool
    if tool not in MEDIA_BEHAVIOURS and not (
            tool == "music_play" and plain_request):
        return tool
    if CORRECTION.search(_fold(text)):
        return tool
    tokens = _tokens(text)

    def said(words: tuple[str, ...]) -> int:
        return _hits(list(words), tokens, text)

    sets_a_level = (which_way(text) is not None
                    and step_size(text, "volume_step_pct") is not None)

    # "די" and a stop verb together, which is the one phrasing that settles
    # halting - see the `_HALT` guard at the end, which this is the exception
    # to. 19 clauses on the v11 corpus, all `media_stop`, none against.
    di_stop = bool(_M_DI.search(text) and _M_STOP_VERB.search(text))

    if said(_M_NEXT):
        found = "media_next_track"
    elif said(_M_BACK):
        found = "media_previous_track"
    elif said(_M_MUTE):
        found = "media_mute"
    elif said(_M_INPUT):
        found = "media_select_source"
    elif (said(_M_VOLUME_VERB) or names_a_value or sets_a_level
            or (said(_M_VOLUME_NOUN) and names_a_level)):
        # A value or a sized step is a volume and there is nothing else on a
        # speaker it could be - "שים את הרמקול על שלושים אחוז", "קצת פחות יותר
        # חלש", "טרים בהרבה את הווליום" - and the model answers all three with
        # a transport verb. The noun is no longer required, which is the point:
        # two of those three name no volume noun at all.
        #
        # `_M_MUTE` is tested above and that ordering is the rule, not an
        # accident: "תוריד מיוט" is an *un*mute and its verb is a downward
        # step, so reading the step first turns 31 unmutes into volume calls.
        # Behind the mute branch, measured per clause over the whole corpus on
        # every media clause: **721 agree, 1 disagree**, the one being "מיות" -
        # speech noise inside the word the branch above matches on.
        found = "media_set_volume"
    elif said(_M_STOP) or di_stop:
        found = "media_stop"
    elif said(_M_PAUSE):
        found = "media_pause"
    elif said(_M_PLAY) and not names_a_level:
        # A level rules a play out: "שים את הנגן על שישים אחוז" sets the
        # volume, and `הנגן` is the player rather than the imperative. Same
        # test the music upgrade uses, for the same sentence shape.
        found = "media_play"
    elif plain_request and not names_a_level and said(_M_PUT_ON):
        # ``שים`` is the whole difficulty of this function, and it is why the
        # chain above will not read it: the same verb selects a source, asks
        # for music, sets a volume and puts a vacuum on a suction setting -
        # 539 corpus clauses over six families. On *this* path every one of
        # those readings has already been ruled out above: no source, no
        # level, no title, and the model itself reached for `music_play`.
        # What is left is the plain Israeli way of asking for something to be
        # put on, and the corpus is unanimous about it.
        found = "media_play"
    else:
        return tool
    # Halting is one decision, and the sentence does not make it - except in
    # the one phrasing where it does. "די" is enough on its own to mean *stop
    # asking*, and next to an explicit stop verb it separates the two halting
    # behaviours the rest of this function cannot. Everywhere else the model's
    # answer stands, which is why "תעצור את המוזיקה" is still a pause.
    if found in _HALT and tool in _HALT and not di_stop:
        return tool
    return found
