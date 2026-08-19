"""Config, reconfigure and options flows for Needle Assist.

Setup asks for nothing. The tuned Hebrew weights ship inside the component, so
the first step is a confirmation the user can submit empty, and the path field
is there only for someone running their own fine-tune. A household that has to
copy a file onto its Home Assistant machine and type an absolute path into a
dialog before the assistant works has, in practice, no assistant.

The same field is offered again through **reconfigure**, which it was not
before. An install that once pointed at a hand-copied ``.cact`` kept loading
that file forever: an update replaces the component's bundled weights but never
touches a path already written into the entry, and there was no way to clear
one short of deleting the integration and adding it again. That is how this
very installation ended up running an older adapter than the one it shipped.

The options dialog is deliberately two fields. A setting belongs there only if
a household can answer it better than a measurement can - which speaker, and
how long a sentence may get. The off-topic gate and the confidence floor used
to sit beside them and no longer do: each has one measured-correct value, so
offering them as choices only invited someone to switch the safety off. They
are constants now, in :mod:`const`, with the measurement written beside them.
"""

from __future__ import annotations

import os
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from .const import (
    BUNDLED_WEIGHTS,
    CONF_MAX_TOKENS,
    CONF_MUSIC_PLAYER,
    CONF_WEIGHTS,
    DEFAULT_MAX_TOKENS,
    DOMAIN,
    ENTRY_VERSION,
)

STEP_USER = vol.Schema({
    vol.Optional(CONF_WEIGHTS, default=""): str,
})


class NeedleAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """One step, submittable empty, and changeable afterwards."""

    VERSION = ENTRY_VERSION

    async def _weights_error(self, weights: str) -> str | None:
        """Why this path cannot be used, or ``None`` if it can.

        Checked in the dialog rather than at setup so the user sees the problem
        while the field is still in front of them, instead of as an integration
        that failed to start. ``os.path.isfile`` is I/O and the flow runs on the
        event loop, so it goes to an executor.
        """
        if weights:
            if not await self.hass.async_add_executor_job(os.path.isfile, weights):
                return "weights_not_found"
            if not weights.endswith(".cact"):
                return "weights_not_cact"
            return None
        # No path given: the bundled model has to be there, or the base model
        # would load and answer every Hebrew sentence with an empty call, which
        # reads as a broken integration. Refuse instead of shipping that.
        if not await self.hass.async_add_executor_job(
                os.path.isfile, str(BUNDLED_WEIGHTS)):
            return "weights_missing"
        return None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            weights = (user_input.get(CONF_WEIGHTS) or "").strip()
            if error := await self._weights_error(weights):
                errors[CONF_WEIGHTS] = error
            else:
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

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the weights path - including back to the bundled model.

        Submitting the field empty is the way back: it stores an empty path,
        and an empty path means the model that came with the component, which
        is the one every update keeps current.
        """
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            weights = (user_input.get(CONF_WEIGHTS) or "").strip()
            if error := await self._weights_error(weights):
                errors[CONF_WEIGHTS] = error
            else:
                # Reloads the entry, which re-resolves the path and reloads the
                # engine - the weights are read once, at setup.
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_WEIGHTS: weights}
                )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema({
                vol.Optional(
                    CONF_WEIGHTS,
                    description={
                        "suggested_value": entry.data.get(CONF_WEIGHTS, "")
                    },
                ): str,
            }),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return NeedleAssistOptionsFlow()


class NeedleAssistOptionsFlow(OptionsFlow):
    """The two settings a household knows better than the measurements do."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        options = self.config_entry.options
        schema = vol.Schema({
            # Filtered to the integration whose service actually accepts the
            # call, so the list cannot offer a speaker that would fail. A
            # household with no Music Assistant sees an empty picker, which
            # reads correctly as "this does not apply here", and one with a
            # single player never needs it - that player is the only answer.
            vol.Optional(
                CONF_MUSIC_PLAYER,
                description={
                    "suggested_value": options.get(CONF_MUSIC_PLAYER)
                },
            ): EntitySelector(
                EntitySelectorConfig(domain="media_player",
                                     integration="music_assistant")
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ): NumberSelector(
                NumberSelectorConfig(min=32, max=512, step=8,
                                     mode=NumberSelectorMode.SLIDER)
            ),
        })
        return self.async_show_form(step_id="init", data_schema=schema)
