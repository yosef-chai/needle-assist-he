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
from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    intent,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import clause_split, tool_router
from .const import (
    CONF_CONFIDENCE, CONF_MAX_TOKENS, CONF_REFUSE_GATE, DEFAULT_CONFIDENCE,
    DEFAULT_MAX_TOKENS, DEFAULT_REFUSE_GATE, DOMAIN, SPEECH_FAILED,
    SPEECH_NOTHING, SPEECH_NO_TARGET, SPEECH_OK,
)
from .executor import CallExecutor

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([NeedleConversationEntity(hass, entry)])


class NeedleConversationEntity(conversation.ConversationEntity):
    """Hebrew intent handler backed by a fine-tuned Needle 2."""

    _attr_has_entity_name = True
    _attr_name = "Needle Assist"
    _attr_should_poll = False
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._runner = hass.data[DOMAIN][entry.entry_id]
        self._executor = CallExecutor(hass, entry)
        self._attr_unique_id = entry.entry_id

    async def async_added_to_hass(self) -> None:
        """Keep the slot resolver's view of the house current.

        Targeting is matched against the area and entity registries, so a room
        renamed or an alias added has to invalidate the compiled phrase index -
        otherwise the integration keeps sending commands to a room that was
        renamed an hour ago. Rebuilding is a few milliseconds and only happens
        on the next utterance after a change.
        """
        await super().async_added_to_hass()
        for event in (ar.EVENT_AREA_REGISTRY_UPDATED,
                      er.EVENT_ENTITY_REGISTRY_UPDATED):
            self.async_on_remove(
                self.hass.bus.async_listen(event, self._executor.slots.invalidate)
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
        min_conf = float(options.get(CONF_CONFIDENCE, DEFAULT_CONFIDENCE))

        # Refuse before inference, not after. The model's own refusal rate is
        # 0.0% and its false-actuation rate is ~100%, so left to itself it will
        # call a tool on "מי ניצח במשחק אתמול". The router's family score
        # already separates the two classes - see tool_router.looks_off_topic
        # for the measurement and the threshold. Running the gate first also
        # skips a 45M-parameter forward pass on utterances no tool serves.
        if options.get(CONF_REFUSE_GATE, DEFAULT_REFUSE_GATE) and \
                tool_router.looks_off_topic(user_input.text):
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

        outcomes = []
        for clause in clauses:
            try:
                result = await self.hass.async_add_executor_job(
                    self._runner.complete, clause, max_tokens
                )
            except Exception as err:
                _LOGGER.exception("Needle inference failed")
                response.async_set_error(
                    intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                    f"{SPEECH_FAILED}: {err}",
                )
                return conversation.ConversationResult(
                    response=response, conversation_id=user_input.conversation_id
                )

            calls = self._runner.calls_of(result)
            confidence = float(result.get("confidence") or 0.0)
            _LOGGER.debug("%r -> %s (confidence %.3f)", clause, calls, confidence)

            # Distinguish a genuine refusal from a broken generation. Both
            # arrive as an empty call list; only `success`/`error` tell them
            # apart.
            if (failure := self._runner.failed(result)) is not None:
                _LOGGER.error("engine failure on %r: %s", clause, failure)
                response.async_set_error(
                    intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                    f"{SPEECH_FAILED}: {failure}",
                )
                return conversation.ConversationResult(
                    response=response, conversation_id=user_input.conversation_id
                )

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
        # It is a valid answer, not a failure.
        if not outcomes:
            response.async_set_speech(SPEECH_NOTHING)
            return conversation.ConversationResult(
                response=response, conversation_id=user_input.conversation_id
            )

        spoken = [o.speech for o in outcomes if o.ok and o.speech]
        succeeded = [o for o in outcomes if o.ok]
        failed = [o for o in outcomes if not o.ok]

        if spoken:
            response.async_set_speech(". ".join(spoken))
        elif succeeded and not failed:
            response.async_set_speech(SPEECH_OK)
        elif succeeded and failed:
            response.async_set_speech(
                f"{SPEECH_OK} חלקית, {len(failed)} פעולות נכשלו"
            )
        elif all(o.detail == "no matching entities" for o in failed):
            response.async_set_error(
                intent.IntentResponseErrorCode.NO_VALID_TARGETS, SPEECH_NO_TARGET
            )
        else:
            response.async_set_error(
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE, SPEECH_FAILED
            )

        return conversation.ConversationResult(
            response=response, conversation_id=user_input.conversation_id
        )
