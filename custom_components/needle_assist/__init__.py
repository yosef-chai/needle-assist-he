"""Needle Assist: a fully local Hebrew conversation agent for Home Assistant."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import CONF_WEIGHTS, DOMAIN
from .needle_runner import NeedleRunner

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Load the engine once and hand it to the conversation platform."""
    weights = entry.data.get(CONF_WEIGHTS) or None
    runner = NeedleRunner(
        weights=weights,
        # The engine keeps its native library under the configuration
        # directory, which is the only path that survives a core update.
        config_path=hass.config.path(),
    )

    try:
        # Loading pulls in a native library and, on first run, downloads it.
        # Both block, so neither may touch the event loop.
        await hass.async_add_executor_job(runner.load)
    except Exception as err:
        raise ConfigEntryNotReady(f"could not start the Needle engine: {err}") from err

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runner
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Tear down."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unloaded


async def _async_reload(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload when options change - the engine binds its toolset at init."""
    await hass.config_entries.async_reload(entry.entry_id)
