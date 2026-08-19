"""The Needle conversation agent.

Sits where an LLM agent would in an Assist pipeline, but is not one: Needle
returns grammar-constrained tool calls, never prose. So this entity does not
use Home Assistant's LLM API, does not send a system prompt, and cannot
hallucinate a service that does not exist — the grammar makes malformed calls
unrepresentable, and :mod:`executor` validates targets against the registry.

Everything runs on the local machine. No network call is made at any point.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final, Literal

from homeassistant.components import conversation
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from homeassistant.helpers import (
    intent,
)
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import clause_split, reply, tool_router
from .const import (
    CONF_MAX_TOKENS,
    CONFIDENCE_FLOOR,
    DEFAULT_MAX_TOKENS,
    DOMAIN,
    SPEECH_NOTHING,
)
from .executor import CallExecutor, CallOutcome
from .needle_engine.agent import fetch

if TYPE_CHECKING:
    from . import NeedleConfigEntry

_LOGGER = logging.getLogger(__name__)

# Nothing here polls or pushes state: a conversation entity answers when it is
# spoken to. The engine serialises itself behind its own lock, so there is no
# second limit worth imposing here.
PARALLEL_UPDATES = 0

# Stands in for the tool name on a clause the engine never got through, so
# that a failure there is counted alongside the calls that did run.
ENGINE: Final = "engine failure"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: NeedleConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([NeedleConversationEntity(hass, entry)])


class NeedleConversationEntity(conversation.ConversationEntity):
    """Hebrew intent handler backed by a fine-tuned Needle 2."""

    _attr_has_entity_name = True
    # None, not a string: the entity is the whole of what the device does, so
    # it takes the device's name rather than carrying an untranslatable one of
    # its own. See the device below.
    _attr_name = None
    _attr_should_poll = False
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, hass: HomeAssistant, entry: NeedleConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._runner = entry.runtime_data
        self._executor = CallExecutor(hass, entry)
        self._attr_unique_id = entry.entry_id
        # A service device, not a hardware one - there is no box to find in the
        # house. It exists so the engine has somewhere to report its version,
        # and so the agent can be addressed as a device like any other.
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            entry_type=DeviceEntryType.SERVICE,
            manufacturer="Needle Assist",
            model="Needle 2, Hebrew adapter",
            name="Needle Assist",
            sw_version=fetch.ENGINE_VERSION,
        )

    async def async_added_to_hass(self) -> None:
        """Keep the slot resolver's view of the house current.

        Targeting is matched against the area and entity registries, so a room
        renamed or an alias added has to invalidate the compiled phrase index -
        otherwise the integration keeps sending commands to a room that was
        renamed an hour ago. Rebuilding is a few milliseconds and only happens
        on the next utterance after a change.
        """
        await super().async_added_to_hass()

        @callback
        def _house_changed(_event: Event[Any]) -> None:
            """Drop the compiled phrase index; the next utterance rebuilds it."""
            self._executor.slots.invalidate()

        for event in (ar.EVENT_AREA_REGISTRY_UPDATED,
                      er.EVENT_ENTITY_REGISTRY_UPDATED):
            self.async_on_remove(
                self.hass.bus.async_listen(event, _house_changed)
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Hebrew is what the model was tuned for.

        MATCH_ALL is deliberately not used: the fine-tune is Hebrew, and
        claiming every language would let Assist route, say, English at a model
        that will answer with an empty call.
        """
        return ["he"]

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Run one utterance through the model and act on the result."""
        response = intent.IntentResponse(language=user_input.language)
        options = self.entry.options
        # Coerced here rather than in the options schema: a slider hands back a
        # float, max_new_tokens reaches a ctypes call, and a selector wrapped in
        # vol.All does not reliably survive the frontend's schema serialisation.
        max_tokens = int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        min_conf = CONFIDENCE_FLOOR

        # Refuse before inference, not after, and unconditionally. The model's
        # own refusal rate is 0.0% and its false-actuation rate is ~100%, so
        # left to itself it will call a tool on "מי ניצח במשחק אתמול". The
        # router's family score already separates the two classes - it puts
        # 73.9% of off-topic utterances below the threshold and keeps 96.7% of
        # genuine commands at or above it; see tool_router.looks_off_topic for
        # the measurement. Running the gate first also skips a 45M-parameter
        # forward pass on utterances no tool serves.
        #
        # This was a switch in the options dialog until entry version 2.
        # Nothing measured ever argued for turning it off, and the only thing
        # the switch could do for a household was let a question about football
        # move a light, so it became policy rather than a preference.
        if tool_router.looks_off_topic(user_input.text):
            _LOGGER.debug("refused off-topic before inference: %r", user_input.text)
            response.async_set_speech(SPEECH_NOTHING)
            return conversation.ConversationResult(
                response=response, conversation_id=user_input.conversation_id
            )

        # "אל תדליק את האור" must not turn the light on. The engine has its own
        # negation check but its word list is English, and the corpus contains
        # no negated command at all, so neither the engine nor the model will
        # catch this. See tool_router.looks_negated.
        if tool_router.looks_negated(user_input.text):
            _LOGGER.debug("refused negated command: %r", user_input.text)
            response.async_set_speech(SPEECH_NOTHING)
            return conversation.ConversationResult(
                response=response, conversation_id=user_input.conversation_id
            )

        # One order per clause. "turn off the light in the kitchen and close
        # the blinds in the bedroom" is two commands, and asking the model for
        # both at once returns one call for 94 of 97 such sentences - see
        # `clause_split` for the measurement and for the three explanations
        # that were ruled out first. Splitting first takes the same rows from
        # 0.0% to 75.3% tool-set accuracy on the same weights.
        clauses = clause_split.split_clauses(user_input.text)
        if len(clauses) > 1:
            _LOGGER.debug("%r split into %s", user_input.text, clauses)

        outcomes: list[CallOutcome] = []
        # The first engine error, kept for the spoken reply. Only reached
        # when nothing else in the sentence succeeded.
        engine_error: str | None = None
        for clause in clauses:
            try:
                result = await self.hass.async_add_executor_job(
                    self._runner.complete, clause, max_tokens
                )
            except Exception as err:
                # One clause failing is not the sentence failing. The
                # orders before this one have already run and the ones
                # after it can still run, so this is recorded as a failed
                # outcome and the loop goes on - a flat return here would
                # both hide what happened and drop what was still to do.
                _LOGGER.exception("Needle inference failed on %r", clause)
                engine_error = engine_error or str(err)
                outcomes.append(CallOutcome(ENGINE, False, ENGINE))
                continue

            calls = self._runner.calls_of(result)
            confidence = float(result.get("confidence") or 0.0)
            _LOGGER.debug("%r -> %s (confidence %.3f)", clause, calls, confidence)

            # Distinguish a genuine refusal from a broken generation. Both
            # arrive as an empty call list; only `success`/`error` tell them
            # apart.
            if (failure := self._runner.failed(result)) is not None:
                _LOGGER.error("engine failure on %r: %s", clause, failure)
                engine_error = engine_error or str(failure)
                outcomes.append(CallOutcome(ENGINE, False, ENGINE))
                continue

            # Off by default: the confidence head is not updated by fine-tuning
            # and reads 0.0 on correct non-English calls. See
            # const.DEFAULT_CONFIDENCE.
            if min_conf > 0 and confidence < min_conf:
                _LOGGER.info("dropped call at confidence %.3f < %.3f",
                             confidence, min_conf)
                continue

            for index, call in enumerate(calls):
                outcomes.append(
                    # The clause goes with every call it produced, and so does
                    # the call's position among its siblings. Targeting is
                    # resolved from the words against this installation's own
                    # areas and entities rather than from the model's slugs -
                    # see the module docstring of `executor` - and a two-room
                    # clause needs to know which room goes with which call.
                    await self._executor.execute(call, user_input.device_id,
                                                 user_input.context, clause,
                                                 index, len(calls))
                )

        # An empty call list is Needle's refusal for anything no tool serves.
        # It is a valid answer, not a failure. `reply.compose` says the same
        # thing; returning here keeps the rest of this function honest about
        # only ever handling calls that ran.
        if not outcomes:
            response.async_set_speech(SPEECH_NOTHING)
            return conversation.ConversationResult(
                response=response, conversation_id=user_input.conversation_id
            )

        answer = reply.compose(outcomes, engine_error)
        if answer.error == reply.NO_TARGETS:
            response.async_set_error(
                intent.IntentResponseErrorCode.NO_VALID_TARGETS, answer.speech)
        elif answer.error:
            response.async_set_error(
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE, answer.speech)
        else:
            response.async_set_speech(answer.speech)

        return conversation.ConversationResult(
            response=response, conversation_id=user_input.conversation_id
        )
