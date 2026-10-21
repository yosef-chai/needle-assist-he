"""One click instead of an instruction.

The integration raises exactly one repair issue: a `weights_path` it was
configured with is gone. Retrying cannot bring the file back, so the issue used
to say "open the integration, choose Reconfigure, and either correct the path
or clear the field" - three steps, in a dialog whose only job was to be read.

Clearing the field is the answer in nearly every case: the path was almost
always typed once, before the release layout bundled the model, and the bundled
one is what an update keeps current. So the repair does that itself and the
reconfigure dialog is left for the household that really does want a different
file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    # Home Assistant validates with Probatio since 2026.9 and imports it by
    # that name. Before 2026.9 there is only voluptuous, which has the same API.
    import probatio as vol
except ImportError:
    import voluptuous as vol  # type: ignore[no-redef]
from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import HomeAssistant

from .const import CONF_WEIGHTS, DOMAIN, ISSUE_WEIGHTS_MISSING

if TYPE_CHECKING:
    # 2026.10 gave repair steps their own result type. Earlier versions do not
    # have it, so it is imported for the type checker only; the annotations are
    # never evaluated at runtime.
    from homeassistant.components.repairs import RepairsFlowResult


class WeightsMissingRepairFlow(RepairsFlow):
    """Point the entry back at the bundled model."""

    def __init__(self, path: str) -> None:
        self._path = path

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        entries = self.hass.config_entries.async_entries(DOMAIN)
        if not entries:
            # The household removed the integration while the issue was open.
            return self.async_abort(reason="not_configured")
        if user_input is not None:
            entry = entries[0]
            self.hass.config_entries.async_update_entry(
                entry, data={**entry.data, CONF_WEIGHTS: ""})
            # Deleted here as well as on the next successful setup, because a
            # reload that fails for some other reason must not leave a repair
            # standing for a path nothing reads any more.
            await self.hass.config_entries.async_reload(entry.entry_id)
            return self.async_create_entry(data={})
        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={"path": self._path},
        )


async def async_create_fix_flow(
    hass: HomeAssistant, issue_id: str, data: dict[str, Any] | None
) -> RepairsFlow:
    """Home Assistant's entry point into the flow above."""
    if issue_id == ISSUE_WEIGHTS_MISSING:
        return WeightsMissingRepairFlow(str((data or {}).get("path", "")))
    raise ValueError(f"unknown repair issue {issue_id}")
