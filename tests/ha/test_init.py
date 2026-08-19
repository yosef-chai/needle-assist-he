# -*- coding: utf-8 -*-
"""Setting up, tearing down, migrating, and the one repair worth raising."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import patch

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.needle_assist.const import (
    CONF_WEIGHTS, DOMAIN, ENTRY_VERSION, ISSUE_WEIGHTS_MISSING,
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
