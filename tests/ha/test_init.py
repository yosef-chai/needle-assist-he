"""Setting up, tearing down, migrating, and the one repair worth raising."""

from __future__ import annotations

import os
import pathlib
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.needle_assist.const import (
    CONF_WEIGHTS,
    DOMAIN,
    ENTRY_VERSION,
    ISSUE_WEIGHTS_MISSING,
)

RUNNER = "custom_components.needle_assist.NeedleRunner"


async def test_it_loads_and_unloads(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    # The engine rides on the entry, not in hass.data.
    assert entry.runtime_data is not None

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.NOT_LOADED


async def test_an_engine_that_will_not_start_is_not_ready(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """A missing native library is transient - Home Assistant should retry."""
    with patch(f"{RUNNER}.load", side_effect=OSError("no libneedle.so here")):
        assert not await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.SETUP_RETRY


async def test_a_weights_file_that_has_gone_missing_raises_a_repair(
    hass: HomeAssistant, tmp_path: pathlib.Path
) -> None:
    """Retrying cannot make a deleted file reappear, so say so instead."""
    gone = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_WEIGHTS: str(tmp_path / "deleted.cact")},
        unique_id=DOMAIN,
        version=ENTRY_VERSION,
    )
    gone.add_to_hass(hass)

    assert not await hass.config_entries.async_setup(gone.entry_id)
    await hass.async_block_till_done()

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_WEIGHTS_MISSING)
    assert issue is not None
    assert issue.severity is ir.IssueSeverity.ERROR
    assert issue.translation_placeholders == {
        "path": str(tmp_path / "deleted.cact")
    }


async def test_the_repair_closes_itself_once_the_model_is_found(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    registry = ir.async_get(hass)
    ir.async_create_issue(
        hass, DOMAIN, ISSUE_WEIGHTS_MISSING,
        is_fixable=False, severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_WEIGHTS_MISSING,
    )
    assert registry.async_get_issue(DOMAIN, ISSUE_WEIGHTS_MISSING) is not None

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert registry.async_get_issue(DOMAIN, ISSUE_WEIGHTS_MISSING) is None


async def test_version_one_loses_the_two_settings_that_stopped_being_settings(
    hass: HomeAssistant, loaded_runner: Any
) -> None:
    """The migration exists so a backup does not carry keys nothing reads."""
    old = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_WEIGHTS: ""},
        options={
            "max_new_tokens": 192,
            "music_player": "media_player.kitchen",
            "min_confidence": 0.4,          # retired
            "refuse_off_topic": False,      # retired, and dangerous to keep
        },
        unique_id=DOMAIN,
        version=1,
    )
    old.add_to_hass(hass)

    assert await hass.config_entries.async_setup(old.entry_id)
    await hass.async_block_till_done()

    assert old.version == ENTRY_VERSION
    assert dict(old.options) == {
        "max_new_tokens": 192,
        "music_player": "media_player.kitchen",
    }


async def test_an_entry_from_the_future_is_refused_rather_than_guessed_at(
    hass: HomeAssistant,
) -> None:
    """Downgrading Home Assistant past an entry it does not understand."""
    ahead = MockConfigEntry(
        domain=DOMAIN, data={CONF_WEIGHTS: ""}, unique_id=DOMAIN,
        version=ENTRY_VERSION + 1,
    )
    ahead.add_to_hass(hass)

    assert not await hass.config_entries.async_setup(ahead.entry_id)
    await hass.async_block_till_done()
    assert ahead.state is ConfigEntryState.MIGRATION_ERROR


async def test_changing_an_option_reloads_the_engine(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    """The engine binds its tool set at init, so an option change has to reload."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    before = loaded_runner.call_count

    hass.config_entries.async_update_entry(entry, options={"max_new_tokens": 256})
    await hass.async_block_till_done()

    assert loaded_runner.call_count > before


# --------------------------------------------------------------- removal
#
# Removing the entry has to leave nothing behind. Home Assistant clears the
# registries; these cover the two things it does not know about - the engine
# on disk and the repair in the panel - and the failure paths of both.


async def test_removing_the_entry_takes_the_downloaded_engine_with_it(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry,
    tmp_path: pathlib.Path,
) -> None:
    """13 MB in a directory nobody named is exactly the litter to clear up."""
    hass.config.config_dir = str(tmp_path)
    engine = tmp_path / "needle_assist_engine" / "2.0.2"
    engine.mkdir(parents=True)
    (engine / "libneedle.so").write_bytes(b"not really a library")

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert not (tmp_path / "needle_assist_engine").exists()


async def test_removing_the_entry_closes_the_repair_it_raised(
    hass: HomeAssistant, entry: ConfigEntry, tmp_path: pathlib.Path
) -> None:
    """An issue is keyed by domain, so nothing else would ever clear it."""
    hass.config.config_dir = str(tmp_path)
    ir.async_create_issue(
        hass, DOMAIN, ISSUE_WEIGHTS_MISSING,
        is_fixable=False, severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_WEIGHTS_MISSING,
    )
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_WEIGHTS_MISSING)

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_WEIGHTS_MISSING) is None


async def test_removal_forgets_the_library_path_it_published(
    hass: HomeAssistant, entry: ConfigEntry, tmp_path: pathlib.Path
) -> None:
    """The variable is read on every load; left set it names a deleted file."""
    hass.config.config_dir = str(tmp_path)
    os.environ["NEEDLE_LIB_PATH"] = str(tmp_path / "needle_assist_engine" / "x.so")

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert "NEEDLE_LIB_PATH" not in os.environ


async def test_nothing_to_remove_is_not_a_failure(
    hass: HomeAssistant, entry: ConfigEntry, tmp_path: pathlib.Path
) -> None:
    """An entry removed before the first download ever finished."""
    hass.config.config_dir = str(tmp_path)

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert hass.config_entries.async_get_entry(entry.entry_id) is None


async def test_a_directory_that_will_not_delete_is_reported_not_raised(
    hass: HomeAssistant, entry: ConfigEntry, tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The entry is already gone, so a traceback would help nobody."""
    hass.config.config_dir = str(tmp_path)
    with patch(
        "custom_components.needle_assist.engine_lib.remove_downloads",
        side_effect=OSError("read-only file system"),
    ):
        await hass.config_entries.async_remove(entry.entry_id)
        await hass.async_block_till_done()

    assert hass.config_entries.async_get_entry(entry.entry_id) is None
    # The warning has to name the directory, or it cannot be acted on.
    assert "needle_assist_engine" in caplog.text
    assert "safe to delete by hand" in caplog.text


async def test_a_symlinked_engine_directory_loses_the_link_and_not_the_target(
    hass: HomeAssistant, entry: ConfigEntry, tmp_path: pathlib.Path
) -> None:
    """Some houses point the engine at a bigger disk. That disk is not ours."""
    elsewhere = tmp_path / "big_disk"
    (elsewhere / "2.0.2").mkdir(parents=True)
    (elsewhere / "2.0.2" / "libneedle.so").write_bytes(b"someone else's copy")

    config = tmp_path / "config"
    config.mkdir()
    hass.config.config_dir = str(config)
    try:
        (config / "needle_assist_engine").symlink_to(
            elsewhere, target_is_directory=True
        )
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not let the test create a symlink")

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert not (config / "needle_assist_engine").exists()
    assert (elsewhere / "2.0.2" / "libneedle.so").is_file()
