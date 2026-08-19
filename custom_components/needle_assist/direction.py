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
  player", so "תעצור את הנגן" contains a play hint and a pause verb and no
  amount of counting or ordering separates them. Nothing is lost: music that
  keeps playing is not a door that opens.
"""

from __future__ import annotations

from typing import Any, Final

from .clause_split import CORRECTION
from .tool_router import TOOL_HINTS, _fold, _hits, _tokens

#: Tools that come in a pair the Hebrew words settle. Play against pause and
#: volume against play are deliberately absent; see the module docstring.
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("lock_lock", "lock_unlock"),
    ("light_turn_on", "light_turn_off"),
    ("cover_open", "cover_close"),
    ("switch_turn_on", "switch_turn_off"),
    ("fan_turn_on", "fan_turn_off"),
    ("camera_turn_on", "camera_turn_off"),
    ("automation_turn_on", "automation_turn_off"),
    ("input_boolean_turn_on", "input_boolean_turn_off"),
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

# A dim is a brightness, not an off. Routing wants it under light_turn_off;
# this does not.
_DIMMING: Final = frozenset(_fold(w) for w in ("עמעם", "תעמעם", "תעמעמי", "עמעמי"))

#: The router's own per-tool verbs, minus the ones that do not mean a direction.
VOCABULARY: Final[dict[str, list[str]]] = {
    tool: [hint for hint in TOOL_HINTS.get(tool, []) if _fold(hint) not in _DIMMING]
    for tool in OPPOSITE
}


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
    """
    if not text or CORRECTION.search(_fold(text)):
        return None
    tokens = _tokens(text)
    for up_words, down_words in TIERS:
        up = _hits(list(up_words), tokens, text)
        down = _hits(list(down_words), tokens, text)
        if up != down:
            return 1 if up > down else -1
    return None


def settle_steps(arguments: dict[str, Any], text: str) -> dict[str, Any]:
    """``arguments`` with the sign of any relative argument corrected.

    Nothing is added, nothing is removed and no magnitude changes: an argument
    the model did not emit stays absent, because the sentence was measured to
    settle which way a step points and not whether there is one.
    """
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
