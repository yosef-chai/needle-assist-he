"""Which way round: lock or unlock, open or close, on or off.

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

Measured against gold over all 2,379 held-out rows, on the 1,179 calls whose
tool belongs to one of the guarded pairs:

    the words agree with gold      802     68.0%
    the words disagree with gold     0      0.00%
    the words say nothing          377     32.0%

Zero. A signal that is never wrong when it speaks is a signal that can overrule
a model which is wrong one time in eight.

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

from typing import Final

from .clause_split import CORRECTION
from .tool_router import TOOL_HINTS, _fold, _hits, _tokens

#: Tools that come in a pair whose direction a Hebrew verb settles. ``media``
#: is deliberately absent; see the module docstring.
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("lock_lock", "lock_unlock"),
    ("light_turn_on", "light_turn_off"),
    ("cover_open", "cover_close"),
    ("switch_turn_on", "switch_turn_off"),
    ("fan_turn_on", "fan_turn_off"),
    ("camera_turn_on", "camera_turn_off"),
    ("automation_turn_on", "automation_turn_off"),
    ("input_boolean_turn_on", "input_boolean_turn_off"),
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
