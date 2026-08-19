"""Turn what the calls did into what the assistant says back.

This lives apart from :mod:`conversation` because that module imports Home
Assistant and this decision does not need it: what to say is a function of the
outcomes alone. Keeping it here is what lets it be tested, and the Hebrew a
household actually hears is worth a test - a sentence carrying four orders can
succeed, fail and announce something all at once, and every combination of
those has a right answer.

The caller passes anything with ``ok``, ``speech`` and ``detail``; nothing here
needs the real :class:`executor.CallOutcome`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, NamedTuple, Protocol

from .const import (
    SPEECH_FAILED,
    SPEECH_NO_TARGET,
    SPEECH_NOTHING,
    SPEECH_OK,
)

#: ``CallOutcome.detail`` for a call that found nothing to act on. Every other
#: failure is something going wrong rather than nothing being there, which is
#: the distinction Home Assistant's two error codes draw.
NO_TARGET: Final = "no matching entities"

#: Error tags, mapped to ``intent.IntentResponseErrorCode`` by the caller.
NO_TARGETS: Final = "no_targets"
FAILED: Final = "failed"

# Hebrew makes the noun and the verb agree with the number, so a count spliced
# into a sentence has to be spelled out to come out as Hebrew: "1 פעולות נכשלו"
# is a machine talking. The table stops at four because `clause_split`
# MAX_CLAUSES is four and a clause rarely yields more than one call; past that
# a digit reads naturally anyway.
_FAILURES: Final[dict[int, str]] = {
    1: "פעולה אחת נכשלה",
    2: "שתי פעולות נכשלו",
    3: "שלוש פעולות נכשלו",
    4: "ארבע פעולות נכשלו",
}


class Reply(NamedTuple):
    """What to say, and whether it is an answer or an error."""

    speech: str
    error: str | None = None


def failures(count: int) -> str:
    """"one action failed", counted the way Hebrew counts."""
    return _FAILURES.get(count, f"{count} פעולות נכשלו")


class Outcome(Protocol):
    """The three fields of an executor result this module reads.

    A structural type, not an import: `executor.CallOutcome` satisfies it, but
    importing that module would drag Home Assistant in behind it, and keeping
    this file free of Home Assistant is the reason it exists separately from
    :mod:`conversation`.
    """

    @property
    def ok(self) -> bool:
        """Did the call do what it was asked to."""

    @property
    def speech(self) -> str | None:
        """What to say about it, if anything."""

    @property
    def detail(self) -> str:
        """Why it failed, machine-readable, empty when it did not."""


def compose(outcomes: Sequence[Outcome],
            engine_error: str | None = None) -> Reply:
    """The spoken reply for one utterance.

    ``engine_error`` is the first engine-level failure, if any; it is only
    spoken when nothing else in the sentence worked, because a partial success
    is more useful to hear than a stack of internals.
    """
    # An empty call list is Needle's refusal for anything no tool serves. It is
    # a valid answer, not a failure.
    if not outcomes:
        return Reply(SPEECH_NOTHING)

    spoken = ". ".join(o.speech for o in outcomes if o.ok and o.speech)
    succeeded = [o for o in outcomes if o.ok]
    failed = [o for o in outcomes if not o.ok]

    if succeeded and failed:
        # Both halves get said. A sentence carrying four orders can have one of
        # them fail while another announces what it is playing, and speaking
        # only the success would leave the dropped order sounding like it ran.
        partial = f"{SPEECH_OK} חלקית, {failures(len(failed))}"
        return Reply(f"{spoken}. {partial}" if spoken else partial)

    if succeeded:
        return Reply(spoken or SPEECH_OK)

    if all(o.detail == NO_TARGET for o in failed):
        return Reply(SPEECH_NO_TARGET, NO_TARGETS)

    return Reply(
        f"{SPEECH_FAILED}: {engine_error}" if engine_error else SPEECH_FAILED,
        FAILED,
    )
