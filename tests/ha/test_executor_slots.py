"""Scenes and scripts by name, relative steps, notifications, and the satellite.

These are the arguments the model is worst at and the sentence is best at, so
each of them is a case where the words overrule the answer.
"""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.needle_assist.const import DOMAIN
from custom_components.needle_assist.executor import CallExecutor


@pytest.fixture
def executor(hass: HomeAssistant) -> CallExecutor:
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={}, version=2)
    entry.add_to_hass(hass)
    return CallExecutor(hass, entry)


def _entity(hass: HomeAssistant, domain: str, object_id: str, state: str,
            attributes: dict[str, Any] | None = None,
            area_id: str | None = None) -> str:
    entities = er.async_get(hass)
    entry = entities.async_get_or_create(
        domain, "needle_test", object_id, suggested_object_id=object_id
    )
    if area_id:
        entities.async_update_entity(entry.entity_id, area_id=area_id)
    hass.states.async_set(entry.entity_id, state, attributes or {})
    return entry.entity_id


async def _run(executor: CallExecutor, tool: str, arguments: dict[str, Any],
               utterance: str = "", device_id: str | None = None) -> Any:
    return await executor.execute(
        {"name": tool, "arguments": arguments}, device_id, Context(), utterance
    )


# -- the satellite's own room -------------------------------------------------


async def test_with_no_room_named_the_speaker_is_where_the_satellite_is(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    """"turn on the light" means the room you said it in."""
    areas = ar.async_get(hass)
    kitchen = areas.async_create("מטבח")
    living = areas.async_create("סלון")
    devices = dr.async_get(hass)
    config_entry = MockConfigEntry(domain="esphome")
    config_entry.add_to_hass(hass)
    satellite = devices.async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={("esphome", "voice_kitchen")},
    )
    devices.async_update_device(satellite.id, area_id=kitchen.id)

    kitchen_light = _entity(hass, "light", "kitchen_light", "off", area_id=kitchen.id)
    _entity(hass, "light", "living_light", "off", area_id=living.id)
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    outcome = await _run(executor, "light_turn_on", {}, "תדליק את האור",
                         device_id=satellite.id)

    assert outcome.ok
    assert calls[0].data["entity_id"] == [kitchen_light]


# -- things addressed by name -------------------------------------------------


async def test_a_scene_is_found_by_the_name_the_household_gave_it(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    """The model's `name` slug is invented English; the sentence has the words."""
    scene = _entity(hass, "scene", "evening", "unknown",
                    {"friendly_name": "אווירת ערב"})
    calls: list[ServiceCall] = async_mock_service(hass, "scene", "turn_on")

    outcome = await _run(executor, "scene_activate", {"name": "movie_mode"},
                         "תפעיל את אווירת ערב")

    assert outcome.ok
    assert calls[0].data["entity_id"] == [scene]


async def test_a_scene_that_is_really_a_script_still_runs(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    """The registry knows which of the two this household built it as."""
    script = _entity(hass, "script", "bedtime", "off",
                     {"friendly_name": "שגרת לילה"})
    scripts: list[ServiceCall] = async_mock_service(hass, "script", "turn_on")

    outcome = await _run(executor, "scene_activate", {}, "תפעיל את שגרת לילה")

    assert outcome.ok
    assert scripts[0].data["entity_id"] == [script]


async def test_the_model_slug_is_the_fallback_when_the_words_find_nothing(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    scene = _entity(hass, "scene", "movie_mode", "unknown",
                    {"friendly_name": "Movie Mode"})
    calls: list[ServiceCall] = async_mock_service(hass, "scene", "turn_on")

    outcome = await _run(executor, "scene_activate", {"name": "movie mode"}, "")

    assert outcome.ok
    assert calls[0].data["entity_id"] == [scene]


async def test_cancel_the_timer_with_nothing_named_cancels_all_of_them(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    first = _entity(hass, "timer", "pasta", "active", {"friendly_name": "פסטה"})
    second = _entity(hass, "timer", "laundry", "active", {"friendly_name": "כביסה"})
    calls: list[ServiceCall] = async_mock_service(hass, "timer", "cancel")

    outcome = await _run(executor, "timer_cancel", {}, "תבטל את הטיימר")

    assert outcome.ok
    assert set(calls[0].data["entity_id"]) == {first, second}


# -- arithmetic on the current reading ----------------------------------------


async def test_a_volume_step_is_added_to_what_is_playing_now(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    areas = ar.async_get(hass)
    living = areas.async_create("סלון")
    _entity(hass, "media_player", "living_speaker", "playing",
            {"friendly_name": "רמקול", "volume_level": 0.4}, area_id=living.id)
    calls: list[ServiceCall] = async_mock_service(hass, "media_player", "volume_set")

    outcome = await _run(executor, "media_set_volume",
                         {"area": "living_room", "volume_step_pct": 20},
                         "תגביר את הווליום בסלון")

    assert outcome.ok
    assert calls[0].data["volume_level"] == pytest.approx(0.6)


async def test_a_volume_percentage_becomes_a_fraction(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    areas = ar.async_get(hass)
    living = areas.async_create("סלון")
    _entity(hass, "media_player", "living_speaker", "playing",
            {"friendly_name": "רמקול", "volume_level": 0.4}, area_id=living.id)
    calls: list[ServiceCall] = async_mock_service(hass, "media_player", "volume_set")

    await _run(executor, "media_set_volume",
               {"area": "living_room", "volume_pct": 30},
               "שים את הווליום בסלון על שלושים אחוז")

    assert calls[0].data["volume_level"] == pytest.approx(0.3)


async def test_a_temperature_step_is_added_to_the_thermostat_and_bounded(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    areas = ar.async_get(hass)
    living = areas.async_create("סלון")
    _entity(hass, "climate", "living_ac", "heat",
            {"friendly_name": "מזגן", "temperature": 24}, area_id=living.id)
    calls: list[ServiceCall] = async_mock_service(
        hass, "climate", "set_temperature"
    )

    outcome = await _run(executor, "climate_set_temperature",
                         {"area": "living_room", "temperature_step": 2},
                         "תעלה את המזגן בסלון")

    assert outcome.ok
    assert calls[0].data["temperature"] == 26


async def test_two_hours_is_a_duration_home_assistant_accepts(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    """"00:120:00" is not one, and the corpus says "שעתיים"."""
    _entity(hass, "timer", "kitchen", "idle", {"friendly_name": "טיימר"})
    calls: list[ServiceCall] = async_mock_service(hass, "timer", "start")

    outcome = await _run(executor, "timer_start", {"minutes": 120},
                         "תפעיל טיימר לשעתיים")

    assert outcome.ok
    assert calls[0].data["duration"] == "02:00:00"


# -- notifications -------------------------------------------------------------


async def test_the_message_comes_out_of_the_sentence_not_the_model(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    """Hebrew reaches this slot as escapes and the model drops hex digits."""
    _entity(hass, "notify", "phone", "unknown", {"friendly_name": "טלפון"})
    calls: list[ServiceCall] = async_mock_service(hass, "notify", "send_message")

    # The extractor keys on the ש- complementizer, which is how the corpus
    # says it: "תשלח הודעה ש..." rather than a quoted string.
    outcome = await _run(executor, "notify_send", {"message": "긓" + "4"},
                         "תשלח הודעה שאני בדרך הביתה")

    assert outcome.ok
    assert calls[0].data["message"] == "אני בדרך הביתה"
    assert calls[0].data["title"] == "Home Assistant"


async def test_a_notification_with_no_words_in_the_sentence_is_not_sent(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    _entity(hass, "notify", "phone", "unknown", {"friendly_name": "טלפון"})
    calls: list[ServiceCall] = async_mock_service(hass, "notify", "send_message")

    outcome = await _run(executor, "notify_send", {"message": "whatever survived"},
                         "תשלח הודעה")

    assert not outcome.ok
    assert calls == []


# -- answering about several things -------------------------------------------


async def test_all_on_all_off_and_a_mixture_read_differently(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    areas = ar.async_get(hass)
    living = areas.async_create("סלון")
    _entity(hass, "light", "lamp_one", "on", {"friendly_name": "מנורה א"},
            area_id=living.id)
    _entity(hass, "light", "lamp_two", "off", {"friendly_name": "מנורה ב"},
            area_id=living.id)

    mixed = await _run(executor, "get_state", {"area": "living_room"},
                       "מה המצב של האור בסלון")
    assert mixed.ok
    assert "פעילים" in mixed.speech

    hass.states.async_set("light.lamp_two", "on", {"friendly_name": "מנורה ב"})
    both_on = await _run(executor, "get_state", {"area": "living_room"},
                         "מה המצב של האור בסלון")
    assert both_on.speech.startswith("כן")

    hass.states.async_set("light.lamp_one", "off", {"friendly_name": "מנורה א"})
    hass.states.async_set("light.lamp_two", "off", {"friendly_name": "מנורה ב"})
    both_off = await _run(executor, "get_state", {"area": "living_room"},
                          "מה המצב של האור בסלון")
    assert both_off.speech.startswith("לא")
