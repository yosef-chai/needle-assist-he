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
below. The ``music`` key answers the fourth question this integration gets
asked - why a request to play something resumed playback instead - which has
exactly one common cause: no Music Assistant player, so there is no library
to search and nothing to search it with.

Nothing here leaves the machine unless the user chooses to share it.
"""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from . import direction, slot_match, tool_router
from .clause_split import MAX_CLAUSES
from .const import (
    BUNDLED_WEIGHTS,
    CONF_MUSIC_PLAYER,
    MUSIC_INTEGRATION,
)
from .needle_engine.agent import fetch


def _weights_label(runner: Any) -> str:
    """Which model is loaded, phrased so it cannot be misread.

    This reported `entry.data["weights_path"]` and called an empty one "base
    model", which is exactly backwards: empty means the tuned Hebrew adapter
    that ships inside the component, and the base model is the one that does
    not understand Hebrew at all. The runner resolves the path at load time, so
    ask it instead of asking the entry.
    """
    resolved = getattr(runner, "weights", None)
    if not resolved:
        return "base model - untuned, does not understand Hebrew"
    if resolved == str(BUNDLED_WEIGHTS):
        return f"bundled Hebrew adapter ({resolved})"
    return f"configured ({resolved})"


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Everything needed to explain a targeting decision, and nothing else."""
    runner = entry.runtime_data

    # On the event loop deliberately. The registry and state-machine helpers
    # are not thread-safe, and this is in-memory dictionary walking with no I/O
    # in it - the same thing every core diagnostics platform does.
    house = slot_match.SlotIndex(hass).describe()

    registry = er.async_get(hass)
    players = sorted(
        item.entity_id
        for item in registry.entities.values()
        if item.domain == "media_player"
        and item.platform == MUSIC_INTEGRATION
        and not item.disabled_by
    )

    return {
        "engine": {
            "version": fetch.ENGINE_VERSION,
            "weights": _weights_label(runner),
            "tools_declared_per_turn": tool_router.MAX_TOOLS,
            "tools_in_catalogue": len(getattr(runner, "_tools", []) or []),
        },
        "house": house,
        # Playing something by name needs a Music Assistant player. With none
        # found, a request to play resumes playback on the room's speaker
        # instead - which is the right answer, and not the one that was asked
        # for, so it is worth being able to see why.
        "music": {
            "players": players,
            "configured_default": entry.options.get(CONF_MUSIC_PLAYER),
        },
        # The deterministic gates, so a "why did it refuse that" question can
        # be answered without a debug log.
        "gates": {
            "off_topic_threshold": tool_router.REFUSE_BELOW,
            "room_weight": tool_router.ROOM_WEIGHT,
            "max_clauses": MAX_CLAUSES,
            "families": sorted(tool_router.FAMILY_TOOLS),
            # Which corrections the sentence is allowed to make to the model.
            # A household reporting "it unlocked instead of locking" wants to
            # know whether the pair was guarded at all.
            "guarded_pairs": [list(pair) for pair in direction.PAIRS],
            "guarded_steps": sorted(direction.RELATIVE),
        },
    }
