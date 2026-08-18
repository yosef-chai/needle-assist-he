"""Config and options flow for Needle Assist.

Setup asks for nothing. The tuned Hebrew weights ship inside the component, so
the first step is a confirmation the user can submit empty, and the path field
is there only for someone running their own fine-tune. A household that has to
copy a file onto its Home Assistant machine and type an absolute path into a
dialog before the assistant works has, in practice, no assistant.
"""

from __future__ import annotations

import os
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector, NumberSelector, NumberSelectorConfig, NumberSelectorMode,
)

from .const import (
    BUNDLED_WEIGHTS, CONF_CONFIDENCE, CONF_MAX_TOKENS, CONF_REFUSE_GATE,
    CONF_WEIGHTS, DEFAULT_CONFIDENCE, DEFAULT_MAX_TOKENS, DEFAULT_REFUSE_GATE,
    DOMAIN,
)

STEP_USER = vol.Schema({
    vol.Optional(CONF_WEIGHTS, default=""): str,
})


class NeedleAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """One step, submittable empty."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        # Whether the bundled model is present decides which description the
        # dialog shows, so it is read before the form is built. os.path.isfile
        # is I/O and the flow runs on the event loop.
        bundled = await self.hass.async_add_executor_job(
            os.path.isfile, str(BUNDLED_WEIGHTS)
        )

        if user_input is not None:
            weights = (user_input.get(CONF_WEIGHTS) or "").strip()
            if weights:
                # Checked here rather than at setup so the user sees the problem
                # in the dialog instead of as a failed integration afterwards.
                if not await self.hass.async_add_executor_job(os.path.isfile, weights):
                    errors[CONF_WEIGHTS] = "weights_not_found"
                elif not weights.endswith(".cact"):
                    errors[CONF_WEIGHTS] = "weights_not_cact"
            elif not bundled:
                # No bundled model and no path: the base model would load and
                # answer every Hebrew sentence with an empty call, which reads
                # as a broken integration. Refuse instead of shipping that.
                errors[CONF_WEIGHTS] = "weights_missing"

            if not errors:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Needle Assist (Hebrew)",
                    data={CONF_WEIGHTS: weights},
                )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return NeedleAssistOptionsFlow()


class NeedleAssistOptionsFlow(OptionsFlow):
    """Runtime knobs. Every default here is the measured-correct value."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        options = self.config_entry.options
        schema = vol.Schema({
            vol.Optional(
                CONF_REFUSE_GATE,
                default=options.get(CONF_REFUSE_GATE, DEFAULT_REFUSE_GATE),
            ): BooleanSelector(),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ): vol.All(
                NumberSelector(
                    NumberSelectorConfig(min=32, max=512, step=8,
                                         mode=NumberSelectorMode.SLIDER)
                ),
                # The slider hands back a float and this reaches a ctypes call.
                vol.Coerce(int),
            ),
            vol.Optional(
                CONF_CONFIDENCE,
                default=options.get(CONF_CONFIDENCE, DEFAULT_CONFIDENCE),
            ): vol.All(
                NumberSelector(
                    NumberSelectorConfig(min=0.0, max=1.0, step=0.05,
                                         mode=NumberSelectorMode.SLIDER)
                ),
                vol.Coerce(float),
            ),
        })
        return self.async_show_form(step_id="init", data_schema=schema)
