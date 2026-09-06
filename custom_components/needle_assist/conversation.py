"""The Needle conversation agent.

Sits where an LLM agent would in an Assist pipeline, but is not one: Needle
returns grammar-constrained tool calls, never prose. So this entity does not
use Home Assistant's LLM API, does not send a system prompt, and cannot
hallucinate a service that does not exist - the grammar makes malformed calls
unrepresentable, and :mod:`executor` validates targets against the registry.

Everything runs on the local machine. No network call is made at any point.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
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
    floor_registry as fr,
)
from homeassistant.helpers import (
    intent,
)
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import clause_split, repair, reply, slot_match, tool_router
from .const import (
    CONF_LOG_UTTERANCES,
    CONF_MAX_TOKENS,
    CONFIDENCE_FLOOR,
    DEFAULT_MAX_TOKENS,
    DOMAIN,
    QUERY_TOOLS,
    ROOM_MEMORY_SECONDS,
    SPEECH_CANCELLED,
    SPEECH_NOTHING,
    SPEECH_WHICH_ROOM,
    UTTERANCE_LOG,
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
        # conversation_id -> (when, the command we asked "באיזה חדר" about).
        # See `_joined_with_pending`.
        self._pending: dict[str, tuple[float, str]] = {}
        # A service device, not a hardware one - there is no box to find in the
        # house. It exists so the engine has somewhere to report its version,
        # and so the agent can be addressed as a device like any other.
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            entry_type=DeviceEntryType.SERVICE,
            manufacturer="Needle Assist",
            model="Needle 2, Hebrew adapter",
            # Which adapter, not merely which engine. The two ship together and
            # a mismatch is a total failure rather than a degraded one - a v10
            # `.cact` against a v11 catalogue emits names the catalogue no
            # longer has - so the device page says both. The digest is what
            # this project actually compares adapters by.
            model_id=self._runner.weights_id,
            name="Needle Assist",
            sw_version=fetch.ENGINE_VERSION,
        )


    def _log_utterance(self, text: str, clauses: list[str],
                       outcomes: list[CallOutcome], speech: str) -> None:
        """Append one line to the household's own ruler, if it asked for one.

        Every number this project has was measured on a corpus it wrote itself.
        A template holdout is not generalisation - the same adapter reads 90%
        on "rows it has not seen" and 43% on rows whose *template* it has not
        seen - and 17% of the frozen benchmark has a filler-variant twin in
        training. What the house actually says is the one ruler that is not
        like that, and it cannot be collected without asking.

        So: off by default, written under the configuration directory, and
        never sent anywhere. `diagnostics` reports that it is on, and not a
        word of what is in it.

        Blocking IO, so it goes to the executor like the engine does. Failures
        are logged and swallowed: a full disk must not cost the household its
        answer, and this file is a research instrument rather than a feature.
        """
        if not self.entry.options.get(CONF_LOG_UTTERANCES):
            return
        line = json.dumps({
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "text": text,
            "clauses": clauses,
            "calls": [{"tool": o.tool, "ok": o.ok, "detail": o.detail}
                      for o in outcomes],
            "speech": speech,
        }, ensure_ascii=False)
        path = Path(self.hass.config.path(UTTERANCE_LOG))

        def _append() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        try:
            self.hass.async_add_executor_job(_append)
        except Exception:  # noqa: BLE001 - a log must never cost an answer
            _LOGGER.exception("could not write the utterance log")

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

        # The floor registry too: a floor renamed or an alias added has to
        # invalidate the compiled index for the same reason an area does.
        for event in (ar.EVENT_AREA_REGISTRY_UPDATED,
                      er.EVENT_ENTITY_REGISTRY_UPDATED,
                      fr.EVENT_FLOOR_REGISTRY_UPDATED):
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

    # -- the "באיזה חדר" round trip -----------------------------------------
    def _joined_with_pending(
        self, user_input: conversation.ConversationInput
    ) -> tuple[str, bool]:
        """The utterance, joined onto the command a follow-up answers.

        Returns ``(text, joined)``. When this agent asked "באיזה חדר" on the
        previous turn and this one names a room, the two become the single
        sentence the speaker meant - "תכבה את האור" plus "בסלון" is run as
        "תכבה את האור בסלון", which every layer below already handles.

        Joining rather than carrying a room slot forward is deliberate. The
        room is one of eight things the sentence supplies, and a follow-up
        could as easily name a colour or a speed; making the *sentence* whole
        means all eight keep working with no second code path, and the model
        sees the distribution it was trained on rather than a fragment.

        The pending command is dropped whether or not it is used, so a stale
        one cannot attach itself to an unrelated sentence a minute later.
        """
        text = user_input.text
        key = user_input.conversation_id or ""
        pending = self._pending.pop(key, None)
        if pending is None:
            return text, False
        when, command = pending
        if monotonic() - when > ROOM_MEMORY_SECONDS:
            return text, False
        if not slot_match.names_a_room(text):
            # The speaker answered with something other than a room, which
            # means they moved on. Their new sentence stands on its own.
            return text, False
        _LOGGER.debug("joining %r onto the pending %r", text, command)
        return f"{command} {text}", True

    def _remember_pending(self, conversation_id: str | None,
                          command: str) -> None:
        """Keep the command a "באיזה חדר" was asked about.

        Expired entries go on the way in: a speaker who is asked which room and
        walks away leaves an entry that nothing would ever pop, and over a
        household's lifetime that is a slow leak of whole sentences.
        """
        now = monotonic()
        for stale in [key for key, (when, _) in self._pending.items()
                      if now - when > ROOM_MEMORY_SECONDS]:
            del self._pending[stale]
        self._pending[conversation_id or ""] = (now, command)

    def _may_ask_which_room(self, text: str, joined: bool) -> bool:
        """Is "באיזה חדר" a useful thing to say about this failure?

        Only when the sentence named no room at all - if it named one and
        nothing matched, the room is not what is missing and asking for it
        again would be a loop. And never twice: a joined sentence that still
        finds nothing has already had its answer, so the honest reply is that
        no device matched.
        """
        return not joined and not slot_match.names_a_room(text)

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Run one utterance through the model and act on the result."""
        response = intent.IntentResponse(language=user_input.language)

        # The answer to a "באיזה חדר" this agent asked a moment ago. Joined
        # onto the command it was asked about and run as one sentence, because
        # a bare room name is not a command and nothing downstream would treat
        # it as one - the refusal gate scores a room at two against a threshold
        # of three, so "בסלון" on its own is discarded before inference.
        text, joined = self._joined_with_pending(user_input)
        options = self.entry.options
        # Coerced here rather than in the options schema: a slider hands back a
        # float, max_new_tokens reaches a ctypes call, and a selector wrapped in
        # vol.All does not reliably survive the frontend's schema serialisation.
        max_tokens = int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        min_conf = CONFIDENCE_FLOOR

        # "Never mind" - `HassNevermind`, one of Home Assistant's built-in
        # intents, whose entire job is to do nothing. It runs before the
        # refusal gate because the two answers differ: a withdrawal succeeded
        # and should be met with silence, while an off-topic sentence failed to
        # be understood and says so.
        #
        # Four of the six sentences the official Hebrew suite uses for this were
        # already answered correctly by accident - they name no device, so the
        # refusal gate discarded them. That stopped being true the moment `עצור`
        # became a decisive verb, because then a bare "עצור" paused the music.
        # See tool_router.looks_like_cancel for what separates the two: a bare
        # imperative governs nothing, and in Hebrew that is a withdrawal.
        if tool_router.looks_like_cancel(text):
            _LOGGER.debug("withdrawn before inference: %r", text)
            response.async_set_speech(SPEECH_CANCELLED)
            return conversation.ConversationResult(
                response=response, conversation_id=user_input.conversation_id
            )

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
        if tool_router.looks_off_topic(text):
            _LOGGER.debug("refused off-topic before inference: %r", text)
            response.async_set_speech(SPEECH_NOTHING)
            return conversation.ConversationResult(
                response=response, conversation_id=user_input.conversation_id
            )

        # "אל תדליק את האור" must not turn the light on. The engine has its own
        # negation check but its word list is English, and the corpus contains
        # no negated command at all, so neither the engine nor the model will
        # catch this. See tool_router.looks_negated.
        if tool_router.looks_negated(text):
            _LOGGER.debug("refused negated command: %r", text)
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
        clauses = clause_split.split_clauses(text)
        if len(clauses) > 1:
            _LOGGER.debug("%r split into %s", text, clauses)
            # The gate above ran once, on the whole sentence. Each piece needs
            # it too: "צריך מים מינרלים, תוסיפי לרשימה" is a statement and an
            # order, and the statement half was being answered with a call.
            # 303 clauses dropped against 2 gold calls lost - see
            # `tool_router.clause_names_nothing`. Never all of them: a sentence
            # that passed the gate holds an order somewhere.
            speaking = [c for c in clauses
                        if not tool_router.clause_names_nothing(c)]
            if speaking and len(speaking) < len(clauses):
                _LOGGER.debug("clauses naming nothing dropped: %s",
                              [c for c in clauses if c not in speaking])
                clauses = speaking

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
                # But a failure is not a refusal, and where the router offered
                # exactly one tool and that tool only reads, the sentence can
                # still be answered without the model. See `repair.recover`.
                if (rebuilt := repair.recover(clause)) is not None:
                    _LOGGER.info("rebuilt %r from the sentence after an "
                                 "engine failure", clause)
                    calls = [rebuilt]
                else:
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
                                                 index, len(calls),
                                                 user_input.conversation_id)
                )

        # An empty call list is Needle's refusal for anything no tool serves.
        # It is a valid answer, not a failure. `reply.compose` says the same
        # thing; returning here keeps the rest of this function honest about
        # only ever handling calls that ran.
        if not outcomes:
            self._log_utterance(text, clauses, outcomes, SPEECH_NOTHING)
            response.async_set_speech(SPEECH_NOTHING)
            return conversation.ConversationResult(
                response=response, conversation_id=user_input.conversation_id
            )

        answer = reply.compose(outcomes, engine_error)
        if answer.error == reply.NO_TARGETS and self._may_ask_which_room(
                text, joined):
            # Nothing matched, and the sentence never said where. "לא מצאתי
            # מכשיר מתאים" is true and leaves the speaker with nothing to do
            # about it, so ask instead - and remember the command, because the
            # answer to "באיזה חדר" is a bare room name that every gate in this
            # module would otherwise throw away: a room alone scores two
            # against a threshold of three. The next turn is joined onto this
            # one and run as the single sentence it should have been.
            self._remember_pending(user_input.conversation_id, text)
            self._log_utterance(text, clauses, outcomes, SPEECH_WHICH_ROOM)
            response.async_set_speech(SPEECH_WHICH_ROOM)
            return conversation.ConversationResult(
                response=response,
                conversation_id=user_input.conversation_id,
                continue_conversation=True,
            )
        if answer.error == reply.NO_TARGETS:
            response.async_set_error(
                intent.IntentResponseErrorCode.NO_VALID_TARGETS, answer.speech)
        elif answer.error:
            response.async_set_error(
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE, answer.speech)
        else:
            # A question is answered, not performed. Home Assistant's
            # conversation API separates the two - `action_done` carries what
            # was changed, `query_answer` carries what was asked about - and
            # everything here was `action_done`, including "מה המצב של האור
            # במטבח". A caller reading the response could not tell that nothing
            # had moved, and the states the answer was built from were thrown
            # away after the sentence was composed.
            #
            # Only when *every* call was read-only. A sentence that both asks
            # and acts is an action that also reported something, and saying it
            # answered a question would be the more misleading of the two.
            # v11 added two read-only tools that answer from no entity at
            # all: `HassGetCurrentTime` and `HassGetCurrentDate` need Home
            # Assistant's configured timezone and nothing else. So the test is
            # "was every call read-only", and the states are attached only when
            # there are any - keying it on `answered` would have sent the clock
            # back as `action_done`, telling a caller that something moved.
            succeeded = [outcome for outcome in outcomes if outcome.ok]
            answered = [entity_id for outcome in succeeded
                        for entity_id in outcome.answered_from]
            if succeeded and all(outcome.tool in QUERY_TOOLS
                                 for outcome in succeeded):
                response.response_type = intent.IntentResponseType.QUERY_ANSWER
                if answered:
                    response.async_set_states(
                        [state for entity_id in answered
                         if (state := self.hass.states.get(entity_id)) is not None]
                    )
            response.async_set_speech(answer.speech)

        self._log_utterance(text, clauses, outcomes, answer.speech)
        return conversation.ConversationResult(
            response=response, conversation_id=user_input.conversation_id
        )
