"""Split a Hebrew sentence carrying several orders into one clause per order.

Needle emits one call for "turn off the light in the kitchen and close the
blinds in the bedroom". Measured on the 97 multi-call rows of the held-out test
set, the shipped model scores **0.0% tool-set accuracy** and returns a single
call on **94 of 97** of them - usually the verb of one clause with the room of
another. Three explanations were ruled out before this module was written, and
they are worth keeping written down because each one looked convincing:

* *The grammar only permits one call.* It does not - asked directly the engine
  returns two.
* *Generation is truncated.* It is not - a three-call target peaks at 157
  tokens against a 192-token budget.
* *The 256-token window hides the tool declarations.* It does not - exporting
  the same adapter at window 704 and at 160 produced byte-identical metrics.

What is left is that 94% of the training targets contain exactly one call, so
the model has an overwhelming prior to stop after the first one, and raising the
multi-call share of the corpus from 4.5% to 6.5% did not move it off zero.

Rather than spend the model's small capacity on learning to segment a sentence,
the segmentation is done here, deterministically, and the model is asked one
short question at a time - which is the regime it is actually good at. That also
makes every deterministic layer downstream sharper: the router shortlists tools
for one clause instead of a mixture, and `executor` resolves the area from a
clause that contains exactly one room instead of guessing which room belongs to
which call by position.

The rule
--------

Cut at a comma, at a joining phrase (``וגם``, ``ואז``, ``ואחר כך``…), or at a
word whose leading ``ו`` is followed by an action verb - Hebrew glues that
conjunction to the next word, so ``וסגור`` is "and close" in a single token.

Then keep a cut **only if both sides contain an action verb**. That one
condition is what makes the rule safe, and it is doing more work than it looks:

* ``תדליק את האור בסלון ובמטבח`` - "ובמטבח" is a room, not a verb, so the two
  rooms stay in one clause and produce one call, which is what the sentence
  means.
* ``אה, סגור את האור`` - the filler before the comma has no verb, so it is
  folded back into the clause instead of becoming an empty order.
* ``תכבה את האור בסלון, תודה`` - the polite tail has no verb either.

Verbs come from :mod:`tool_router`'s own tables, so a verb added for routing is
a verb this module can split on, and there is no second list to keep in step.
"""

from __future__ import annotations

from typing import Final

from .tool_router import FAMILY_VERBS, TOOL_HINTS, _fold

# Four orders in one breath is already an unusual sentence, and each one costs
# a full forward pass - about 2.5 s on the ARM board this runs on. Five would
# put a single utterance past the point where a voice pipeline stops waiting.
MAX_CLAUSES: Final = 4

_PUNCT: Final = ".,!?;:'\"״׳()[]-–—"

# Phrases that join two orders. Stored folded and space-separated so a
# two-word joiner can be matched by look-ahead.
_JOINERS: Final[tuple[tuple[str, ...], ...]] = tuple(
    tuple(_fold(word) for word in phrase.split())
    for phrase in (
        "וגם", "ואז", "וכן", "ובנוסף", "בנוסף", "וגם כן", "גם כן",
        "ואחר כך", "אחר כך", "ואחרי זה", "אחרי זה", "וכמו כן", "כמו כן",
        "ולאחר מכן", "לאחר מכן", "ואחריו", "ועוד", "וברגע שסיימת",
    )
)


def _action_verbs() -> frozenset[str]:
    """Every verb that names an action, folded.

    ``query`` is excluded on purpose: "מה" and "האם" open a question, and a
    question is answered as one unit even when it mentions two rooms. Read-only
    tools are excluded from the hints for the same reason.
    """
    words: set[str] = set()
    for family, verbs in FAMILY_VERBS.items():
        if family == "query":
            continue
        words.update(_fold(v) for v in verbs)
    for tool, hints in TOOL_HINTS.items():
        if tool.startswith("get_"):
            continue
        words.update(_fold(h) for h in hints)
    return frozenset(words)


ACTION_VERBS: Final[frozenset[str]] = _action_verbs()


def _word(token: str) -> str:
    return _fold(token.strip(_PUNCT))


def _is_action_verb(token: str) -> bool:
    return _word(token) in ACTION_VERBS


def _joiner_length(tokens: list[str], i: int) -> int:
    """How many tokens at ``i`` form a joining phrase, longest first."""
    for phrase in sorted(_JOINERS, key=len, reverse=True):
        if tuple(_word(t) for t in tokens[i:i + len(phrase)]) == phrase:
            return len(phrase)
    return 0


def _has_verb(tokens: list[str]) -> bool:
    return any(_is_action_verb(t) for t in tokens)


def split_clauses(text: str, limit: int = MAX_CLAUSES) -> list[str]:
    """One clause per order, or the sentence unchanged if there is only one.

    The returned clauses are the original words re-joined by single spaces:
    joining phrases are dropped and a leading ``ו`` is peeled off the verb it
    was glued to, because neither carries meaning for the router or for the
    slot resolver, and both would otherwise be the first thing the model reads.
    """
    tokens = text.split()
    if len(tokens) < 4:            # too short to hold two orders
        return [text]

    segments: list[list[str]] = [[]]
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if (n := _joiner_length(tokens, i)) and segments[-1]:
            segments.append([])
            i += n
            continue

        # Hebrew glues the conjunction to the word: וסגור = "and close".
        stripped = token[1:]
        if (token.startswith("ו") and stripped and segments[-1]
                and _is_action_verb(stripped)):
            segments.append([stripped])
            i += 1
            continue

        segments[-1].append(token)
        # A comma ends a clause, but only once something precedes it.
        if token.endswith(",") and i + 1 < len(tokens):
            segments.append([])
        i += 1

    # Fold every verbless fragment into a neighbour. This is the step that
    # makes the cuts safe rather than merely plausible - see the module
    # docstring for the three sentences it rescues.
    merged: list[list[str]] = []
    for segment in segments:
        if not segment:
            continue
        if merged and not (_has_verb(segment) and _has_verb(merged[-1])):
            merged[-1].extend(segment)
        else:
            merged.append(segment)

    if len(merged) < 2:
        return [text]
    return [" ".join(s).strip(" ,") for s in merged[:limit]]
