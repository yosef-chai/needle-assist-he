"""Needle Assist: a fully local Hebrew conversation agent for Home Assistant."""

from __future__ import annotations

import logging
import os

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import issue_registry as ir

from . import engine_lib
from .const import (
    CONF_WEIGHTS,
    DOMAIN,
    ENTRY_VERSION,
    ISSUE_WEIGHTS_MISSING,
    RETIRED_OPTIONS,
)
from .needle_runner import NeedleRunner

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

# The loaded engine is the entry's runtime object. It used to live in
# `hass.data[DOMAIN][entry.entry_id]`, which every consumer had to index by
# hand and none of them could type; on the entry it is typed once, here, and
# Home Assistant drops it when the entry unloads without anyone remembering to.
type NeedleConfigEntry = ConfigEntry[NeedleRunner]


async def async_setup_entry(hass: HomeAssistant, entry: NeedleConfigEntry) -> bool:
    """Load the engine once and hand it to the conversation platform."""
    weights = entry.data.get(CONF_WEIGHTS) or None

    # A configured path that no longer resolves is not a transient failure:
    # retrying cannot make the file reappear, and the entry would sit in a
    # retry loop saying nothing a household could act on. Raise a repair
    # instead, naming the file and the step that can change it. An empty path
    # never lands here - it means the bundled model, which the runner resolves.
    if weights and not await hass.async_add_executor_job(os.path.isfile, weights):
        ir.async_create_issue(
            hass,
            DOMAIN,
            ISSUE_WEIGHTS_MISSING,
            is_fixable=False,
            severity=ir.IssueSeverity.ERROR,
            translation_key=ISSUE_WEIGHTS_MISSING,
            translation_placeholders={"path": weights},
        )
        raise ConfigEntryNotReady(f"configured weights file is missing: {weights}")

    ir.async_delete_issue(hass, DOMAIN, ISSUE_WEIGHTS_MISSING)

    # The engine's native library, fetched once per version over Home
    # Assistant's own session. Doing it here rather than inside `load()` keeps
    # the only network call this integration makes on the event loop's terms.
    try:
        await engine_lib.async_ensure_library(hass, hass.config.path())
    except Exception as err:
        raise ConfigEntryNotReady(str(err)) from err

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

    entry.runtime_data = runner
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def async_migrate_entry(hass: HomeAssistant, entry: NeedleConfigEntry) -> bool:
    """Strip options that are no longer the household's to choose.

    Version 1 offered a confidence floor and an off-topic switch. Each has one
    measured-correct value - see `const.CONFIDENCE_FLOOR` and the gate's own
    comment in `conversation` - so both were removed from the dialog rather
    than left as a way to switch the safety off by accident. An entry written
    by version 1 still carries them, and an option nothing reads is worse than
    no option: it shows up in diagnostics and in a backup as if it still meant
    something. Drop them, so what is stored is what is actually read.
    """
    if entry.version > ENTRY_VERSION:
        # Downgrading Home Assistant past an entry it does not understand.
        return False
    if entry.version < 2:
        hass.config_entries.async_update_entry(
            entry,
            options={k: v for k, v in entry.options.items()
                     if k not in RETIRED_OPTIONS},
            version=2,
        )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: NeedleConfigEntry) -> bool:
    """Tear down. `runtime_data` goes with the entry, so there is nothing to pop."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_reload(hass: HomeAssistant, entry: NeedleConfigEntry) -> None:
    """Reload when options change - the engine binds its toolset at init."""
    await hass.config_entries.async_reload(entry.entry_id)
