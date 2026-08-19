"""Thin wrapper around the Needle engine, safe to call from Home Assistant.

Two constraints drive the design:

* ``needle.Needle`` is a ``ctypes`` binding onto a native library and blocks.
  It must never run on the event loop, so every call goes through
  ``async_add_executor_job``.
* The Python package keeps a **module-level singleton** engine (``_active`` in
  ``needle/__init__.py``) and re-initialises it whenever a different ``Needle``
  instance is used. Two concurrent calls would stomp on each other's engine
  state, so a lock serialises them.

The toolset is chosen per utterance rather than bound once, because declaring
every tool hands tool selection to Needle's retrieval head, and measurement
says that head cannot read Hebrew: 6.8% tool-set accuracy with all of them
declared
against 54.4% with a five-tool shortlist. ``tool_router`` picks the shortlist
from the Hebrew text, which keeps the engine at or below the five-tool threshold
where the retrieval head does not run at all.

One agent is cached per distinct shortlist. Switching between them is cheap by
construction: ``Needle._bind`` reloads the weight blob only when the *weights
path* differs from the one already resident (``_active_weights``), so every
cached agent shares one loaded model and a switch costs a single
``needle_init`` — a grammar rebuild over five schemas. Each instance also holds
its own 64KB response buffer, which is why the cache is bounded rather than
unbounded.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from . import engine_lib, tool_router
from .const import BUNDLED_WEIGHTS
from .needle_compat import safe_complete

_LOGGER = logging.getLogger(__name__)

# Loaded from tools.json shipped beside this module.
_TOOLS_FILE = Path(__file__).parent / "tools.json"


class NeedleRunner:
    """Owns the Needle agents and serialises access to the engine."""

    def __init__(self, weights: str | None = None,
                 config_path: str | None = None) -> None:
        self._weights = weights
        self._config_path = config_path
        self._agents: dict[tuple[str, ...], Any] = {}
        self._needle: Any = None
        self._lock = threading.Lock()
        # Deliberately not read here. __init__ is called from
        # async_setup_entry, which runs on the event loop, and Home Assistant
        # reports any blocking I/O there:
        #   Detected blocking call to read_text ... inside the event loop by
        #   custom integration 'needle_assist'
        # It is one small file and it would work, but the loop belongs to every
        # other integration too. load() runs in an executor; the read goes
        # there.
        self._tools: list[dict[str, Any]] = []

    def load(self) -> None:
        """Read the catalogue, point the engine at its library, warm one agent.

        Blocking on all three counts - the first call may download the native
        library, and binding an agent runs the grammar compiler - so this must
        run in an executor.
        """
        self._tools = json.loads(_TOOLS_FILE.read_text(encoding="utf-8"))

        # Nothing configured means the model that ships with the component, not
        # the untuned base model - the base model does not understand Hebrew,
        # and silently running it would look like a broken installation rather
        # than like a missing file. Checked here because load() runs in an
        # executor; a stat on the event loop is still a stat on the event loop.
        if not self._weights and BUNDLED_WEIGHTS.is_file():
            self._weights = str(BUNDLED_WEIGHTS)
            _LOGGER.debug("using the bundled Hebrew weights at %s", self._weights)

        if self._config_path:
            engine_lib.bind_library(self._config_path)

        # Vendored, not the installed `cactus-needle`: the package depends on
        # the whole JAX training stack, none of which inference touches. See
        # needle_engine/VENDOR.md.
        from . import needle_engine as needle  # noqa: PLC0415

        self._needle = needle
        # Build the fallback shortlist now so a first utterance does not pay
        # for the engine's initial load.
        self._agent_for(tool_router.FALLBACK[:tool_router.MAX_TOOLS])
        _LOGGER.info(
            "Needle engine ready: %d tools in catalogue, %d declared per turn, "
            "weights=%s",
            len(self._tools), tool_router.MAX_TOOLS,
            self._weights or "baked-in base model",
        )

    @property
    def weights(self) -> str | None:
        """The weights file actually in use, once :meth:`load` has run.

        ``None`` only when no tuned weights were found at all, which means the
        untuned base model - it does not understand Hebrew, so anything reading
        this should say so plainly rather than call it a default.
        """
        return self._weights

    # The router can only emit shortlists built from the tool families, so the
    # key space is small in practice. The bound is a safety net against a
    # pathological input stream, not an expected condition.
    _MAX_CACHED_AGENTS = 64

    def _agent_for(self, names: list[str]) -> Any:
        """Agent bound to exactly these tools, cached by shortlist."""
        key = tuple(names)
        agent = self._agents.get(key)
        if agent is not None:
            return agent

        if len(self._agents) >= self._MAX_CACHED_AGENTS:
            self._agents.pop(next(iter(self._agents)))

        by_name = {t["name"]: t for t in self._tools}
        kwargs: dict[str, Any] = {"tools": [by_name[n] for n in names if n in by_name]}
        if self._weights:
            kwargs["weights"] = self._weights
        # No tool_index_path: the index only exists to persist retrieval-head
        # embeddings, and at five or fewer declared tools that head never runs.
        agent = self._needle.Needle(**kwargs)
        self._agents[key] = agent
        return agent

    def complete(self, text: str, max_new_tokens: int = 192) -> dict[str, Any]:
        """One turn. Returns Needle's response dict. Blocking."""
        if self._needle is None:
            raise RuntimeError("NeedleRunner.load() was not called")

        names = tool_router.select_tool_names(text, tool_router.MAX_TOOLS)
        _LOGGER.debug("router shortlist for %r: %s", text, names)

        with self._lock:
            try:
                agent = self._agent_for(names)
                # Each utterance is independent: Home Assistant conversations
                # are turn-based and we execute the calls ourselves, so stale
                # history would only pollute the 256-token window.
                agent.reset()
                # safe_complete, not agent.complete: the engine truncates its
                # reasoning field mid-character on Hebrew, which makes the
                # upstream strict decode raise. See needle_compat.
                result = safe_complete(agent, text, max_new_tokens)
            except Exception:
                _LOGGER.exception("Needle engine failed on %r", text)
                raise

        if not isinstance(result, dict):
            raise RuntimeError(f"unexpected engine response: {result!r}")
        return result

    @staticmethod
    def calls_of(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Tool calls from a response, normalised to a list.

        Needle's refusal for off-topic input is an EMPTY call list, not an
        error and not free text, so an empty list here is a valid answer that
        the caller must handle rather than a failure.
        """
        calls = result.get("function_calls") or []
        return [c for c in calls if isinstance(c, dict) and c.get("name")]

    @staticmethod
    def failed(result: dict[str, Any]) -> str | None:
        """Engine-level failure message, or None if the turn was fine.

        An empty ``function_calls`` is ambiguous on its own: it is both the
        legitimate refusal for off-topic input AND what a truncated generation
        leaves behind. The two are only distinguishable through ``success`` and
        ``error`` — e.g. ``"tool call truncated: token budget exhausted"``.
        Conflating them reports a broken model as a polite refusal, which is
        exactly how a bad export stayed invisible here for a whole run.
        """
        if result.get("success") is False:
            return str(result.get("error") or "engine reported failure")
        return None
