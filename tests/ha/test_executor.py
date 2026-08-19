"""The half of the integration that actually touches the house.

Everything here goes through `CallExecutor.execute`, which is where a tool call
becomes a Home Assistant service call. The houses are invented.
"""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.needle_assist.const import DOMAIN
from custom_components.needle_assist.executor import CallExecutor


@pytest.fixture
def house(hass: HomeAssistant) -> dict[str, str]:
    """Two rooms, and one of each thing worth speaking to."""
    areas = ar.async_get(hass)
    entities = er.async_get(hass)
    living = areas.async_create("סלון")
    kitchen = areas.async_create("מטבח")

    made: dict[str, str] = {}
    for domain, unique, area, state, attributes in (
        ("light", "living_light", living, "off", {"friendly_name": "אור סלון"}),
        ("light", "kitchen_light", kitchen, "on", {"friendly_name": "אור מטבח",
                                                   "brightness": 128}),
        ("climate", "living_ac", living, "cool", {"friendly_name": "מזגן סלון",
                                                  "temperature": 24}),
        ("lock", "front_door", living, "unlocked", {"friendly_name": "דלת כניסה"}),
        ("media_player", "living_speaker", living, "idle",
         {"friendly_name": "רמקול סלון", "volume_level": 0.3}),
    ):
        entry = entities.async_get_or_create(
            domain, "needle_test", unique, suggested_object_id=unique
        )
        entities.async_update_entity(entry.entity_id, area_id=area.id)
        hass.states.async_set(entry.entity_id, state, attributes)
        made[unique] = entry.entity_id
    return made


@pytest.fixture
def executor(hass: HomeAssistant) -> CallExecutor:
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={}, version=2)
    entry.add_to_hass(hass)
    return CallExecutor(hass, entry)


async def _run(executor: CallExecutor, tool: str, arguments: dict[str, Any],
               utterance: str = "") -> Any:
    return await executor.execute(
        {"name": tool, "arguments": arguments}, None, Context(), utterance
    )


# -- targeting ----------------------------------------------------------------


async def test_the_room_in_the_sentence_picks_the_entity(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    outcome = await _run(executor, "light_turn_on", {"area": "living_room"},
                         "תדליק את האור בסלון")

    assert outcome.ok
    assert len(calls) == 1
    assert calls[0].data["entity_id"] == [house["living_light"]]


async def test_a_room_this_house_does_not_have_fails_instead_of_widening(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """Otherwise "the garage" in a house with no garage lights the whole place."""
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    outcome = await _run(executor, "light_turn_on", {"area": "garage"},
                         "תדליק את האור במוסך")

    assert not outcome.ok
    assert calls == []


async def test_the_whole_house_really_does_mean_the_whole_house(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_off")

    outcome = await _run(executor, "light_turn_off", {"area": "all"},
                         "תכבה את כל האורות בבית")

    assert outcome.ok
    assert set(calls[0].data["entity_id"]) == {
        house["living_light"], house["kitchen_light"]
    }


# -- the sentence overruling the model ----------------------------------------


async def test_the_verb_decides_lock_from_unlock(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """The model reached for unlock; the sentence says lock. 1337 right, 0 wrong."""
    locks: list[ServiceCall] = async_mock_service(hass, "lock", "lock")
    unlocks: list[ServiceCall] = async_mock_service(hass, "lock", "unlock")

    outcome = await _run(executor, "lock_unlock", {"area": "living_room"},
                         "תנעל את הדלת בסלון")

    assert outcome.ok
    assert len(locks) == 1
    assert unlocks == []


async def test_the_verb_decides_which_way_a_step_points(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """"תנמיך" with a positive step would brighten the room."""
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    outcome = await _run(
        executor, "light_turn_on",
        {"area": "kitchen", "brightness_step_pct": 20},
        "תנמיך את האור במטבח",
    )

    assert outcome.ok
    assert calls[0].data["brightness_step_pct"] < 0


async def test_a_colour_said_out_loud_beats_the_one_the_model_guessed(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    outcome = await _run(executor, "light_turn_on",
                         {"area": "living_room", "color_name": "red"},
                         "תעשה את האור בסלון סגול")

    assert outcome.ok
    assert calls[0].data["color_name"] == "purple"


# -- arguments ----------------------------------------------------------------


async def test_a_call_carries_only_what_its_own_tool_declares(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """A stray argument from a previous clause must not reach the service."""
    calls: list[ServiceCall] = async_mock_service(hass, "lock", "lock")

    outcome = await _run(executor, "lock_lock",
                         {"area": "living_room", "temperature": 24,
                          "brightness_pct": 80},
                         "תנעל את הדלת בסלון")

    assert outcome.ok
    assert "temperature" not in calls[0].data
    assert "brightness_pct" not in calls[0].data


async def test_an_absolute_temperature_is_passed_through(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    calls: list[ServiceCall] = async_mock_service(hass, "climate", "set_temperature")

    outcome = await _run(executor, "climate_set_temperature",
                         {"area": "living_room", "temperature": 22},
                         "שים את המזגן בסלון על 22")

    assert outcome.ok
    assert calls[0].data["temperature"] == 22


# -- queries and failures -----------------------------------------------------


async def test_a_question_is_answered_without_touching_anything(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    outcome = await _run(executor, "get_state",
                         {"area": "kitchen", "domain": "light"},
                         "מה המצב של האור במטבח")

    assert outcome.ok
    assert outcome.speech
    assert calls == []


async def test_a_tool_that_does_not_exist_fails_quietly(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    outcome = await _run(executor, "summon_a_butler", {"area": "living_room"})
    assert not outcome.ok
    assert "unknown tool" in outcome.detail


async def test_nothing_of_that_kind_in_that_room_is_a_failure_not_a_no_op(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """There is no vacuum in this house, and saying "done" would be a lie."""
    calls: list[ServiceCall] = async_mock_service(hass, "vacuum", "start")

    outcome = await _run(executor, "vacuum_start", {"area": "kitchen"},
                         "תפעיל את השואב במטבח")

    assert not outcome.ok
    assert calls == []
