# -*- coding: utf-8 -*-
"""Diagnostics: what this house looks like to the integration.

Home Assistant shows this behind *Settings > Devices & Services > Needle Assist
> Download diagnostics*. It exists because installing on somebody else's house
is where this project's real bugs came from, and every one of them was invisible
until the registries were read the way the integration reads them:

* a room the household calls ``הול``, which no fine-tune has a slug for;
* two rooms, ``מקלחת`` and ``שירותים``, that both answered to one slug, so a
  command for one lit the other;
* a light with no area assigned, invisible to every room command, which reads
  to the user as "the model is broken".

All three are answered by the ``areas`` and ``entities_without_an_area`` keys
below. Nothing here leaves the machine unless the user chooses to share it.
"""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from . import slot_match, tool_router
from .const import CONF_WEIGHTS, DOMAIN
from .needle_engine.agent import fetch


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Everything needed to explain a targeting decision, and nothing else."""
    runner = hass.data.get(DOMAIN, {}).get(entry.entry_id)

    # On the event loop deliberately. The registry and state-machine helpers
    # are not thread-safe, and this is in-memory dictionary walking with no I/O
    # in it - the same thing every core diagnostics platform does.
    house = slot_match.SlotIndex(hass).describe()

    return {
        "engine": {
            "version": fetch.ENGINE_VERSION,
            "weights": entry.data.get(CONF_WEIGHTS) or "base model",
            "tools_declared_per_turn": tool_router.MAX_TOOLS,
            "tools_in_catalogue": len(getattr(runner, "_tools", []) or []),
        },
        "house": house,
        # The two deterministic gates, so a "why did it refuse that" question
        # can be answered without a debug log.
        "gates": {
            "off_topic_threshold": tool_router.REFUSE_BELOW,
            "families": sorted(tool_router.FAMILY_TOOLS),
        },
    }
