# -*- coding: utf-8 -*-
"""Fixtures for the tests that need a running Home Assistant.

Everything in this directory boots a real ``hass``. The Hebrew logic suite one
level up does not, and stays fast because nothing here is autouse outside it.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from custom_components.needle_assist.const import CONF_WEIGHTS, DOMAIN

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def integration_is_discoverable(
    enable_custom_integrations: None,
) -> Generator[None]:
    """Make this repository's ``custom_components`` the one the loader scans.

    The harness ships its own ``custom_components`` package to hold its test
    fixtures. It is a regular package, so on ``sys.path`` it shadows this
    repository's and Home Assistant finds no integration at all. Appending our
    directory to that same package's search path puts the integration exactly
    where the loader looks, without copying anything into site-packages.

    ``enable_custom_integrations`` is requested first because it drops the
    loader's cache; the append has to happen before the next scan, not after.
    """
    import custom_components  # noqa: PLC0415

    ours = str(REPO / "custom_components")
    if ours not in custom_components.__path__:
        custom_components.__path__.append(ours)
    yield


@pytest.fixture(autouse=True)
async def base_components(hass: HomeAssistant) -> None:
    """The pieces of core the conversation platform leans on.

    `conversation` reads the exposed-entity settings that the `homeassistant`
    integration owns, so without it every flow in this directory dies during
    dependency setup rather than in the code under test.
    """
    from homeassistant.setup import async_setup_component  # noqa: PLC0415

    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})


@pytest.fixture
def loaded_runner() -> Generator[Any]:
    """A runner that reports itself loaded without touching the engine.

    Loading is a native library, a download on first run and a 23 MB model.
    None of that is under test here, and all of it is slow, so the seam is the
    one method that does it.
    """
    with (
        patch(
            "custom_components.needle_assist.NeedleRunner.load",
            return_value=None,
        ) as loader,
        # Setup fetches the native library over Home Assistant's session before
        # it builds the runner. Nothing in these tests should reach the network.
        patch(
            "custom_components.needle_assist.engine_lib.async_ensure_library",
            return_value="/config/needle_assist_engine/libneedle.so",
        ),
    ):
        yield loader


@pytest.fixture
def entry(hass: HomeAssistant) -> ConfigEntry:
    """A config entry as the current version writes it: no path, so bundled."""
    from pytest_homeassistant_custom_component.common import (  # noqa: PLC0415
        MockConfigEntry,
    )

    created = MockConfigEntry(
        domain=DOMAIN,
        title="Needle Assist (Hebrew)",
        data={CONF_WEIGHTS: ""},
        options={},
        unique_id=DOMAIN,
        version=2,
    )
    created.add_to_hass(hass)
    return created
