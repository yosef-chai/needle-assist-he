"""The plumbing between Home Assistant and a native library.

No real engine is loaded anywhere here: the native library, the download and
the ctypes call are all replaced. What is under test is the code around them -
which weights get chosen, which tools get declared, and what happens when the
engine returns something broken, which on Hebrew it does.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import zipfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.needle_assist import engine_lib, needle_compat
from custom_components.needle_assist.const import BUNDLED_WEIGHTS
from custom_components.needle_assist.needle_runner import NeedleRunner

RUNNER = "custom_components.needle_assist.needle_runner"
PACKAGE = "custom_components.needle_assist"


def with_engine(engine: Any) -> Any:
    """Stand in for the vendored engine.

    Both callers reach it with `from . import needle_engine`, which resolves
    through the parent package's attribute rather than through `sys.modules`,
    so patching `sys.modules` alone leaves the real ctypes loader in place.
    """
    return patch.object(sys.modules[PACKAGE], "needle_engine", engine)


@pytest.fixture
def fake_engine() -> Any:
    """A stand-in for the vendored `needle_engine` package."""
    engine = MagicMock()
    engine.Needle.return_value = MagicMock()
    return engine


def _loaded(runner: NeedleRunner, engine: Any) -> NeedleRunner:
    with (
        with_engine(engine),
        patch(f"{RUNNER}.engine_lib.bind_library", return_value="/tmp/libneedle.so"),
    ):
        runner.load()
    return runner


# -- which model, and which tools ---------------------------------------------


def test_nothing_configured_means_the_bundled_hebrew_adapter(
    fake_engine: Any
) -> None:
    """Not the base model: the base model does not understand Hebrew at all."""
    runner = _loaded(NeedleRunner(weights=None, config_path="/config"), fake_engine)
    assert runner.weights == str(BUNDLED_WEIGHTS)


def test_a_configured_path_wins(
    fake_engine: Any, tmp_path: pathlib.Path
) -> None:
    mine = tmp_path / "mine.cact"
    mine.write_bytes(b"my own fine-tune")
    runner = _loaded(NeedleRunner(weights=str(mine), config_path=None), fake_engine)
    assert runner.weights == str(mine)


def test_no_weights_anywhere_leaves_the_base_model(
    fake_engine: Any, tmp_path: pathlib.Path
) -> None:
    with patch(f"{RUNNER}.BUNDLED_WEIGHTS", tmp_path / "absent.cact"):
        runner = _loaded(NeedleRunner(weights=None, config_path=None), fake_engine)
    assert runner.weights is None


def test_only_a_shortlist_of_tools_is_ever_declared(fake_engine: Any) -> None:
    """The grammar is built from the declared tools, so this is the safety rail."""
    from custom_components.needle_assist import tool_router

    runner = _loaded(NeedleRunner(weights=None, config_path=None), fake_engine)
    declared = fake_engine.Needle.call_args.kwargs["tools"]

    assert len(declared) <= tool_router.MAX_TOOLS
    assert len(runner._tools) > tool_router.MAX_TOOLS      # out of a bigger catalogue


def test_completing_before_loading_is_an_error_not_a_crash() -> None:
    with pytest.raises(RuntimeError, match="load"):
        NeedleRunner().complete("תדליק את האור")


def test_each_utterance_starts_from_a_clean_window(fake_engine: Any) -> None:
    """A 256-token window has no room for a previous turn's history."""
    runner = _loaded(NeedleRunner(weights=None, config_path=None), fake_engine)
    agent = fake_engine.Needle.return_value

    with patch(f"{RUNNER}.safe_complete", return_value={"function_calls": []}):
        runner.complete("תדליק את האור בסלון")

    assert agent.reset.called


def test_an_engine_that_returns_something_other_than_a_dict_is_refused(
    fake_engine: Any
) -> None:
    runner = _loaded(NeedleRunner(weights=None, config_path=None), fake_engine)
    with (
        patch(f"{RUNNER}.safe_complete", return_value="<html>gateway timeout"),
        pytest.raises(RuntimeError, match="unexpected engine response"),
    ):
        runner.complete("תדליק את האור בסלון")


def test_the_agent_cache_does_not_grow_without_bound(fake_engine: Any) -> None:
    runner = _loaded(NeedleRunner(weights=None, config_path=None), fake_engine)
    for index in range(NeedleRunner._MAX_CACHED_AGENTS + 5):
        runner._agent_for([f"tool_{index}"])
    assert len(runner._agents) <= NeedleRunner._MAX_CACHED_AGENTS


# -- reading the envelope ------------------------------------------------------


def test_an_empty_call_list_is_a_refusal_and_a_failure_is_not() -> None:
    """Both look the same until `success` is read; conflating them hid a bad export."""
    refusal = {"function_calls": [], "success": True}
    broken = {"function_calls": [], "success": False,
              "error": "tool call truncated: token budget exhausted"}

    assert NeedleRunner.calls_of(refusal) == []
    assert NeedleRunner.failed(refusal) is None
    assert NeedleRunner.calls_of(broken) == []
    assert "truncated" in (NeedleRunner.failed(broken) or "")


def test_junk_in_the_call_list_is_dropped() -> None:
    result = {"function_calls": [
        {"name": "light_turn_on", "arguments": {}},
        {"arguments": {}},          # no name at all
        "not a dict",
    ]}
    assert [c["name"] for c in NeedleRunner.calls_of(result)] == ["light_turn_on"]


# -- the ctypes boundary -------------------------------------------------------


def _fake_lib(payload: bytes) -> Any:
    library = MagicMock()

    def _complete(text: bytes, tokens: int, buffer: Any, size: int) -> int:
        buffer.value = payload
        return len(payload)

    library.needle_complete.side_effect = _complete
    return library


def test_a_hebrew_answer_truncated_mid_character_still_parses() -> None:
    """The engine cuts its reasoning field mid-sequence; strict decode raises."""
    envelope = json.dumps(
        {"function_calls": [{"name": "light_turn_on", "arguments": {}}],
         "success": True, "reasoning": "מדליק את הא"},
        ensure_ascii=False,
    ).encode("utf-8")
    mangled = envelope[:-3] + b"\xd7" + envelope[-2:]   # a lone lead byte

    agent = MagicMock()
    agent._buffer = MagicMock()
    agent._weights = None
    engine = MagicMock()
    engine._lib.return_value = _fake_lib(mangled)

    with with_engine(engine):
        result = needle_compat.safe_complete(agent, "תדליק את האור", 192)

    assert result["function_calls"][0]["name"] == "light_turn_on"


def test_confidence_is_nulled_for_tuned_weights() -> None:
    """Upstream does the same: fine-tuning never updates the confidence head."""
    envelope = json.dumps({"function_calls": [], "confidence": 0.9}).encode()
    agent = MagicMock()
    agent._buffer = MagicMock()
    agent._weights = "/config/needle_he.cact"
    engine = MagicMock()
    engine._lib.return_value = _fake_lib(envelope)

    with with_engine(engine):
        result = needle_compat.safe_complete(agent, "תדליק את האור", 192)

    assert result["confidence"] is None


def test_a_negative_return_code_is_raised_not_ignored() -> None:
    agent = MagicMock()
    agent._buffer = MagicMock()
    engine = MagicMock()
    engine._lib.return_value.needle_complete.return_value = -7

    with with_engine(engine), pytest.raises(RuntimeError, match="code -7"):
        needle_compat.safe_complete(agent, "תדליק את האור", 192)


def test_an_unparseable_envelope_says_so() -> None:
    agent = MagicMock()
    agent._buffer = MagicMock()
    engine = MagicMock()
    engine._lib.return_value = _fake_lib(b"{not json at all")

    with with_engine(engine), pytest.raises(RuntimeError, match="unparseable envelope"):
        needle_compat.safe_complete(agent, "תדליק את האור", 192)


# -- fetching the native library ----------------------------------------------


def test_an_engine_already_on_disk_is_not_downloaded_again(
    tmp_path: pathlib.Path
) -> None:
    target = engine_lib.library_path(str(tmp_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"already here")

    with patch("urllib.request.urlopen", side_effect=AssertionError("downloaded!")):
        assert engine_lib.ensure_library(str(tmp_path)) == str(target)


def test_the_library_is_extracted_from_the_wheel_and_renamed_into_place(
    tmp_path: pathlib.Path
) -> None:
    """Written beside the target and renamed, so a broken download is never loaded."""
    from custom_components.needle_assist.needle_engine.agent import fetch

    tag = fetch._platform_tag()
    member = "needle/" + fetch._lib_name_for(tag)
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, b"the native library")

    response = MagicMock()
    response.read.return_value = wheel.getvalue()
    response.__enter__ = lambda self: self
    response.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=response):
        path = engine_lib.ensure_library(str(tmp_path))

    assert pathlib.Path(path).read_bytes() == b"the native library"
    # No .part file survives.
    assert not list(pathlib.Path(path).parent.glob("*.part"))


def test_a_download_that_fails_explains_how_to_do_it_by_hand(
    tmp_path: pathlib.Path
) -> None:
    with (
        patch("urllib.request.urlopen", side_effect=OSError("no route to host")),
        pytest.raises(RuntimeError, match="extract"),
    ):
        engine_lib.ensure_library(str(tmp_path))


def test_binding_points_the_vendored_loader_at_our_copy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = engine_lib.library_path(str(tmp_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"the native library")
    monkeypatch.delenv("NEEDLE_LIB_PATH", raising=False)

    assert engine_lib.bind_library(str(tmp_path)) == str(target)
    import os
    assert os.environ["NEEDLE_LIB_PATH"] == str(target)


# -- fetching it over Home Assistant's own session -----------------------------


async def test_the_only_network_call_goes_through_the_shared_session(
    hass: Any, aioclient_mock: Any, tmp_path: pathlib.Path
) -> None:
    """Once per engine version, on a fresh install, and never again."""
    from custom_components.needle_assist.needle_engine.agent import fetch

    tag = fetch._platform_tag()
    member = "needle/" + fetch._lib_name_for(tag)
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, b"the native library")

    url = engine_lib._wheel_for_this_machine().url
    aioclient_mock.get(url, content=wheel.getvalue())

    path = await engine_lib.async_ensure_library(hass, str(tmp_path))

    assert pathlib.Path(path).read_bytes() == b"the native library"
    assert len(aioclient_mock.mock_calls) == 1


async def test_an_engine_already_on_disk_is_never_fetched(
    hass: Any, aioclient_mock: Any, tmp_path: pathlib.Path
) -> None:
    target = engine_lib.library_path(str(tmp_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"already here")

    assert await engine_lib.async_ensure_library(hass, str(tmp_path)) == str(target)
    assert aioclient_mock.mock_calls == []


async def test_a_failed_fetch_explains_how_to_install_it_by_hand(
    hass: Any, aioclient_mock: Any, tmp_path: pathlib.Path
) -> None:
    aioclient_mock.get(engine_lib._wheel_for_this_machine().url, status=404)

    with pytest.raises(RuntimeError, match="extract"):
        await engine_lib.async_ensure_library(hass, str(tmp_path))
