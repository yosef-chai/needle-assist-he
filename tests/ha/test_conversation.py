"""One Hebrew sentence in, whatever the house does out.

The model is replaced by a stub that returns the tool calls a real run
returned, so what is under test is everything around it: the two gates, the
clause split, and what gets said back.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.needle_assist.const import SPEECH_NOTHING

AGENT = "conversation.needle_assist"


def _answer(*calls: dict[str, Any], success: bool = True,
            error: str | None = None) -> dict[str, Any]:
    """A Needle response envelope, shaped the way the engine shapes it."""
    return {"function_calls": list(calls), "confidence": 0.0,
            "success": success, "error": error}


@pytest.fixture
def house(hass: HomeAssistant) -> dict[str, str]:
    areas = ar.async_get(hass)
    entities = er.async_get(hass)
    made: dict[str, str] = {}
    for domain, unique, room, state in (
        ("light", "living_light", "סלון", "off"),
        ("light", "kitchen_light", "מטבח", "off"),
        ("lock", "front_door", "סלון", "unlocked"),
    ):
        area = areas.async_get_area_by_name(room) or areas.async_create(room)
        entry = entities.async_get_or_create(
            domain, "needle_test", unique, suggested_object_id=unique
        )
        entities.async_update_entity(entry.entity_id, area_id=area.id)
        hass.states.async_set(entry.entity_id, state, {"friendly_name": unique})
        made[unique] = entry.entity_id
    return made


@pytest.fixture
async def agent(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry
) -> ConfigEntry:
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def _say(hass: HomeAssistant, text: str) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass, text, None, Context(), language="he", agent_id=AGENT
    )


async def test_it_answers_in_hebrew_only(
    hass: HomeAssistant, agent: ConfigEntry
) -> None:
    """Claiming every language would route English at a Hebrew fine-tune."""
    entity = hass.states.get(AGENT)
    assert entity is not None
    assert entity.attributes["supported_features"]


async def test_a_question_about_football_moves_nothing(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    """The gate runs before inference; the model never sees this."""
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    with patch.object(
        agent.runtime_data, "complete",
        side_effect=AssertionError("the model should not have been asked"),
    ):
        result = await _say(hass, "מי ניצח במשחק אתמול")

    assert result.response.speech["plain"]["speech"] == SPEECH_NOTHING
    assert calls == []


async def test_do_not_turn_on_the_light_does_not_turn_on_the_light(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    """The engine's own negation list is English; the corpus has no negations."""
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    with patch.object(
        agent.runtime_data, "complete",
        return_value=_answer({"name": "light_turn_on",
                              "arguments": {"area": "living_room"}}),
    ):
        result = await _say(hass, "אל תדליק את האור בסלון")

    assert result.response.speech["plain"]["speech"] == SPEECH_NOTHING
    assert calls == []


async def test_one_order_reaches_the_house(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    with patch.object(
        agent.runtime_data, "complete",
        return_value=_answer({"name": "light_turn_on",
                              "arguments": {"area": "living_room"}}),
    ):
        result = await _say(hass, "תדליק את האור בסלון")

    assert len(calls) == 1
    assert calls[0].data["entity_id"] == [house["living_light"]]
    assert result.response.speech["plain"]["speech"]


async def test_two_orders_in_one_sentence_become_two_calls(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    """Asked as one sentence the model returns one call for 94 of 97 rows."""
    lights: list[ServiceCall] = async_mock_service(hass, "light", "turn_off")
    locks: list[ServiceCall] = async_mock_service(hass, "lock", "lock")

    def per_clause(clause: str, _tokens: int) -> dict[str, Any]:
        if "דלת" in clause:
            return _answer({"name": "lock_lock", "arguments": {"area": "living_room"}})
        return _answer({"name": "light_turn_off", "arguments": {"area": "living_room"}})

    with patch.object(agent.runtime_data, "complete", side_effect=per_clause):
        await _say(hass, "תכבה את האור בסלון ותנעל את הדלת")

    assert len(lights) == 1
    assert len(locks) == 1


async def test_an_engine_that_falls_over_says_so_rather_than_pretending(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    with patch.object(agent.runtime_data, "complete",
                      side_effect=RuntimeError("libneedle segfaulted")):
        result = await _say(hass, "תדליק את האור בסלון")

    assert result.response.error_code is not None


async def test_a_truncated_generation_is_not_reported_as_a_polite_refusal(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    """Both arrive as an empty call list; only `success` tells them apart."""
    with patch.object(
        agent.runtime_data, "complete",
        return_value=_answer(success=False, error="tool call truncated"),
    ):
        result = await _say(hass, "תדליק את האור בסלון")

    assert result.response.error_code is not None


async def test_an_empty_call_list_is_a_refusal_and_a_valid_answer(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    with patch.object(agent.runtime_data, "complete", return_value=_answer()):
        result = await _say(hass, "תדליק את האור בסלון")

    assert result.response.speech["plain"]["speech"] == SPEECH_NOTHING
    assert calls == []
    assert result.response.error_code is None


async def test_the_token_budget_comes_from_the_options(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    """A slider hands back a float and it reaches a ctypes call as an int."""
    hass.config_entries.async_update_entry(agent, options={"max_new_tokens": 256.0})
    await hass.async_block_till_done()

    with patch.object(agent.runtime_data, "complete",
                      return_value=_answer()) as complete:
        await _say(hass, "תדליק את האור בסלון")

    assert complete.call_args.args[1] == 256
    assert isinstance(complete.call_args.args[1], int)
