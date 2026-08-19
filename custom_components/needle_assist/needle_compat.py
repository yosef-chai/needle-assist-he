"""Tolerant replacement for ``needle.Needle.complete``.

Works around an upstream bug that makes the engine unusable for Hebrew.

``Needle.complete`` does::

    response = json.loads(self._buffer.value.decode("utf-8"))

and catches only ``json.JSONDecodeError``. The engine echoes spans of the user's
query into the unconstrained ``reasoning`` field and truncates that field at a
byte limit. Hebrew is two bytes per character under the model's byte-fallback
tokenizer, so the cut regularly lands *inside* a character, leaving a dangling
continuation byte. The strict decode then raises ``UnicodeDecodeError``, which
nothing catches.

Measured on the Hebrew test set: **12% of queries** (18/150) died this way.

The fix is to decode with ``errors="replace"``. The damaged bytes are confined
to ``reasoning``, which is free text nobody acts on - ``function_calls`` are
grammar-constrained and structurally intact. So one U+FFFD in a field we
discard is the whole cost, versus losing the entire response.

This mirrors the upstream call exactly, including nulling ``confidence`` for
tuned weights, so behaviour is otherwise identical.
"""

from __future__ import annotations

import json
import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

_warned = False


def safe_complete(agent: Any, text: str,
                  max_new_tokens: int = 192) -> dict[str, Any]:
    """One turn, surviving truncated multi-byte characters in the response."""
    global _warned  # noqa: PLW0603 - one debug line, once per process
    # The vendored engine, not the installed `cactus-needle`. Home Assistant has
    # no `needle` on its path - the package is deliberately not a requirement,
    # see needle_engine/VENDOR.md - and importing it here would work on a
    # developer machine and fail only on the device.
    from . import needle_engine as needle  # noqa: PLC0415

    agent._bind()
    rc: int = needle._lib().needle_complete(
        text.encode("utf-8"), int(max_new_tokens), agent._buffer, len(agent._buffer)
    )
    if rc < 0:
        raise RuntimeError(f"needle_complete failed (code {rc})")

    raw = agent._buffer.value
    try:
        payload = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Engine truncated a Hebrew character mid-sequence.
        payload = raw.decode("utf-8", errors="replace")
        if not _warned:
            _LOGGER.debug(
                "engine returned truncated UTF-8 (a known upstream issue with "
                "non-Latin scripts); recovering with lossy decode"
            )
            _warned = True

    try:
        response: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError as err:
        raise RuntimeError(
            f"engine returned an unparseable envelope ({err})"
        ) from err

    # Upstream nulls confidence for tuned weights because fine-tuning does not
    # update the confidence head. Mirrored so callers see identical behaviour.
    if getattr(agent, "_weights", None):
        response["confidence"] = None
    return response
