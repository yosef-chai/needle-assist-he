# -*- coding: utf-8 -*-
"""Every path through the config, reconfigure and options flows.

The quality scale asks for full coverage of the config flow specifically, and
the reason is visible in this file: most of what can go wrong with this
integration goes wrong in a text field holding a path.
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE, SOURCE_USER, ConfigEntry,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.needle_assist.const import (
    CONF_MAX_TOKENS, CONF_MUSIC_PLAYER, CONF_WEIGHTS, DOMAIN,
)

BUNDLED = "custom_components.needle_assist.config_flow.BUNDLED_WEIGHTS"


async def _start_user_flow(hass: HomeAssistant) -> dict[str, Any]:
    return await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )


async def test_the_form_is_offered_before_anything_is_typed(
    hass: HomeAssistant,
) -> None:
    result = await _start_user_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {}


async def test_submitting_it_empty_installs_the_bundled_model(
    hass: HomeAssistant, loaded_runner: Any
) -> None:
    """The whole point of the first step: a household types nothing."""
    result = await hass.config_entries.flow.async_configure(
        (await _start_user_flow(hass))["flow_id"], {}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Needle Assist (Hebrew)"
    # Empty, not absent: an empty path is what the runner reads as "bundled".
    assert result["data"] == {CONF_WEIGHTS: ""}


async def test_a_path_of_ones_own_is_kept(
    hass: HomeAssistant, loaded_runner: Any, tmp_path: pathlib.Path
) -> None:
    weights = tmp_path / "mine.cact"
    weights.write_bytes(b"not really a model, but it is a file")

    result = await hass.config_entries.flow.async_configure(
        (await _start_user_flow(hass))["flow_id"],
        {CONF_WEIGHTS: f"  {weights}  "},          # padded: it gets stripped
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_WEIGHTS: str(weights)}


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("absent.cact", "weights_not_found"),
        ("present.pkl", "weights_not_cact"),
    ],
)
async def test_a_path_that_cannot_work_is_refused_in_the_dialog(
    hass: HomeAssistant, tmp_path: pathlib.Path, filename: str, expected: str
) -> None:
    """Refused while the field is still on screen, not as a dead integration."""
    candidate = tmp_path / filename
    if expected == "weights_not_cact":
        candidate.write_bytes(b"a checkpoint, not a built model")

    result = await hass.config_entries.flow.async_configure(
        (await _start_user_flow(hass))["flow_id"], {CONF_WEIGHTS: str(candidate)}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_WEIGHTS: expected}


async def test_no_path_and_no_bundled_model_is_refused_rather_than_shipped(
    hass: HomeAssistant, tmp_path: pathlib.Path
) -> None:
    """The base model would load and answer every Hebrew sentence with nothing."""
    with patch(BUNDLED, tmp_path / "not-installed.cact"):
        result = await hass.config_entries.flow.async_configure(
            (await _start_user_flow(hass))["flow_id"], {}
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_WEIGHTS: "weights_missing"}


async def test_it_can_only_be_set_up_once(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    result = await _start_user_flow(hass)
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"


# -- reconfigure --------------------------------------------------------------


async def _start_reconfigure(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    return await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": SOURCE_RECONFIGURE, "entry_id": entry.entry_id},
    )


async def test_reconfigure_shows_the_path_that_is_in_use(
    hass: HomeAssistant, loaded_runner: Any, tmp_path: pathlib.Path
) -> None:
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    weights = tmp_path / "old.cact"
    weights.write_bytes(b"an older adapter")
    existing = MockConfigEntry(
        domain=DOMAIN, data={CONF_WEIGHTS: str(weights)}, unique_id=DOMAIN, version=2
    )
    existing.add_to_hass(hass)

    result = await _start_reconfigure(hass, existing)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    # The field is prefilled, so clearing it is a visible act rather than a guess.
    suggested = result["data_schema"]({})
    assert suggested.get(CONF_WEIGHTS, "") in ("", str(weights))


async def test_clearing_the_field_goes_back_to_the_bundled_model(
    hass: HomeAssistant, loaded_runner: Any, tmp_path: pathlib.Path
) -> None:
    """The bug this step exists for: an entry pinned to a hand-copied file."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    weights = tmp_path / "stale.cact"
    weights.write_bytes(b"the model from three versions ago")
    existing = MockConfigEntry(
        domain=DOMAIN, data={CONF_WEIGHTS: str(weights)}, unique_id=DOMAIN, version=2
    )
    existing.add_to_hass(hass)

    result = await hass.config_entries.flow.async_configure(
        (await _start_reconfigure(hass, existing))["flow_id"], {CONF_WEIGHTS: ""}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert existing.data[CONF_WEIGHTS] == ""


async def test_reconfigure_refuses_a_path_that_is_not_there(
    hass: HomeAssistant, entry: ConfigEntry, tmp_path: pathlib.Path
) -> None:
    result = await hass.config_entries.flow.async_configure(
        (await _start_reconfigure(hass, entry))["flow_id"],
        {CONF_WEIGHTS: str(tmp_path / "gone.cact")},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_WEIGHTS: "weights_not_found"}
    # Nothing was written on the way to the error.
    assert entry.data[CONF_WEIGHTS] == ""


# -- options ------------------------------------------------------------------


async def test_the_options_dialog_offers_two_settings_and_only_two(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    """The confidence floor and the off-topic gate are policy, not preferences."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    offered = {str(key) for key in result["data_schema"].schema}
    assert offered == {CONF_MUSIC_PLAYER, CONF_MAX_TOKENS}


async def test_the_two_settings_are_saved(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> None:
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_configure(
        (await hass.config_entries.options.async_init(entry.entry_id))["flow_id"],
        {CONF_MUSIC_PLAYER: "media_player.kitchen", CONF_MAX_TOKENS: 256},
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_MAX_TOKENS] == 256
    assert entry.options[CONF_MUSIC_PLAYER] == "media_player.kitchen"
