"""The diagnostics report, and the one line in it that used to lie."""

from __future__ import annotations

import pathlib
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.needle_assist.const import BUNDLED_WEIGHTS, CONF_WEIGHTS, DOMAIN
from custom_components.needle_assist.diagnostics import (
    async_get_config_entry_diagnostics,
)


async def test_it_reports_the_model_that_loaded_not_the_one_configured(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    """An empty path means the bundled Hebrew adapter.

    This read `entry.data` and called an empty path "base model" - naming the
    untuned English model while the tuned one was running.
    """
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    # `load()` is patched out, so resolve the path the way it would have.
    entry.runtime_data._weights = str(BUNDLED_WEIGHTS)

    report = await async_get_config_entry_diagnostics(hass, entry)

    assert "bundled Hebrew adapter" in report["engine"]["weights"]
    assert "base model" not in report["engine"]["weights"]


async def test_a_model_of_ones_own_is_named_as_such(
    hass: HomeAssistant, loaded_runner: Any, tmp_path: pathlib.Path
) -> None:
    weights = tmp_path / "mine.cact"
    weights.write_bytes(b"a fine-tune of my own")
    own = MockConfigEntry(
        domain=DOMAIN, data={CONF_WEIGHTS: str(weights)}, unique_id=DOMAIN, version=2
    )
    own.add_to_hass(hass)
    assert await hass.config_entries.async_setup(own.entry_id)
    await hass.async_block_till_done()
    own.runtime_data._weights = str(weights)

    report = await async_get_config_entry_diagnostics(hass, own)
    assert report["engine"]["weights"] == f"configured ({weights})"


async def test_no_tuned_weights_at_all_says_so_plainly(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    """The base model does not understand Hebrew; the report must not imply it does."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entry.runtime_data._weights = None

    report = await async_get_config_entry_diagnostics(hass, entry)
    assert "untuned" in report["engine"]["weights"]


async def test_the_report_explains_which_rooms_are_addressable(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    """The point of the report: answering "why did it not find my room"."""
    areas = ar.async_get(hass)
    areas.async_create("סלון")
    areas.async_create("Kitchen", aliases={"מטבח"})

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    report = await async_get_config_entry_diagnostics(hass, entry)
    reachable = {
        phrase
        for area in report["house"]["areas"]
        for phrase in area["reachable_by"]
    }
    # The area's own name, and an English-named room reached by a Hebrew alias.
    assert "סלון" in reachable
    assert "מטבח" in reachable


async def test_it_names_the_gates_so_a_refusal_can_be_explained(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    report = await async_get_config_entry_diagnostics(hass, entry)
    assert isinstance(report["gates"]["off_topic_threshold"], int)
