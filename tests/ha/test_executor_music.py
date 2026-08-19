"""Playing something by name, answering questions, and naming a scene.

Music is the part of the integration that has to work in three different
houses: one with Music Assistant, one with several of its players, and one
with none at all.
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

from custom_components.needle_assist.const import (
    CONF_MUSIC_PLAYER,
    DOMAIN,
    MUSIC_INTEGRATION,
)
from custom_components.needle_assist.executor import CallExecutor


@pytest.fixture
def rooms(hass: HomeAssistant) -> dict[str, str]:
    areas = ar.async_get(hass)
    return {name: areas.async_create(name).id for name in ("סלון", "מטבח")}


def _add_player(hass: HomeAssistant, unique: str, platform: str,
                area_id: str | None) -> str:
    entities = er.async_get(hass)
    entry = entities.async_get_or_create(
        "media_player", platform, unique, suggested_object_id=unique
    )
    if area_id:
        entities.async_update_entity(entry.entity_id, area_id=area_id)
    hass.states.async_set(entry.entity_id, "idle", {"friendly_name": unique})
    return entry.entity_id


def _executor(hass: HomeAssistant, options: dict[str, Any] | None = None
              ) -> CallExecutor:
    entry = MockConfigEntry(
        domain=DOMAIN, data={}, options=options or {}, version=2
    )
    entry.add_to_hass(hass)
    return CallExecutor(hass, entry)


async def _run(executor: CallExecutor, tool: str, arguments: dict[str, Any],
               utterance: str) -> Any:
    return await executor.execute(
        {"name": tool, "arguments": arguments}, None, Context(), utterance
    )


async def test_a_named_album_is_played_on_the_music_assistant_player(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    player = _add_player(hass, "ma_living", MUSIC_INTEGRATION, rooms["סלון"])
    calls: list[ServiceCall] = async_mock_service(
        hass, MUSIC_INTEGRATION, "play_media"
    )

    outcome = await _run(_executor(hass), "music_play", {},
                         "תנגן את האלבום שבלול בסלון")

    assert outcome.ok
    assert calls[0].data["entity_id"] == [player]
    assert calls[0].data["media_type"] == "album"
    assert "שבלול" in calls[0].data["media_id"]


async def test_the_sentence_beats_the_model_on_which_media_tool_this_is(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    """"ערבב את האלבום" came back as media_set_volume for 12 of 26 failures."""
    _add_player(hass, "ma_living", MUSIC_INTEGRATION, rooms["סלון"])
    played: list[ServiceCall] = async_mock_service(
        hass, MUSIC_INTEGRATION, "play_media"
    )
    volume: list[ServiceCall] = async_mock_service(
        hass, "media_player", "volume_set"
    )

    outcome = await _run(_executor(hass), "media_set_volume",
                         {"volume_level": 0.5},
                         "ערבב את האלבום המסע של עדן חסון בסלון")

    assert outcome.ok
    assert len(played) == 1
    assert volume == []


async def test_a_number_makes_it_a_volume_again(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    """"שים את השיר על שישים" is a level, and says so."""
    _add_player(hass, "ma_living", MUSIC_INTEGRATION, rooms["סלון"])
    played: list[ServiceCall] = async_mock_service(
        hass, MUSIC_INTEGRATION, "play_media"
    )
    volume: list[ServiceCall] = async_mock_service(
        hass, "media_player", "volume_set"
    )

    await _run(_executor(hass), "media_set_volume", {"volume_level": 0.6},
               "תשים את השיר על הדשא על שישים בסלון")

    assert played == []
    assert len(volume) == 1


async def test_the_chosen_speaker_settles_a_house_with_several(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    _add_player(hass, "ma_one", MUSIC_INTEGRATION, None)
    preferred = _add_player(hass, "ma_two", MUSIC_INTEGRATION, None)
    calls: list[ServiceCall] = async_mock_service(
        hass, MUSIC_INTEGRATION, "play_media"
    )

    executor = _executor(hass, {CONF_MUSIC_PLAYER: preferred})
    outcome = await _run(executor, "music_play", {}, "תנגן את האלבום שבלול")

    assert outcome.ok
    assert calls[0].data["entity_id"] == [preferred]


async def test_a_house_with_no_music_assistant_still_resumes_playback(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    """The tool degrades rather than failing: no worse off than before it existed."""
    speaker = _add_player(hass, "sonos_living", "sonos", rooms["סלון"])
    resumed: list[ServiceCall] = async_mock_service(
        hass, "media_player", "media_play"
    )

    outcome = await _run(_executor(hass), "music_play", {},
                         "תנגן את האלבום שבלול בסלון")

    assert outcome.ok
    assert resumed[0].data["entity_id"] == [speaker]


async def test_music_with_nothing_named_resumes_rather_than_searching(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    player = _add_player(hass, "ma_living", MUSIC_INTEGRATION, rooms["סלון"])
    resumed: list[ServiceCall] = async_mock_service(
        hass, "media_player", "media_play"
    )

    outcome = await _run(_executor(hass), "music_play", {}, "תנגן מוזיקה בסלון")

    assert outcome.ok
    assert resumed[0].data["entity_id"] == [player]


async def test_music_in_a_room_this_house_does_not_have_fails(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    _add_player(hass, "ma_living", MUSIC_INTEGRATION, rooms["סלון"])
    calls: list[ServiceCall] = async_mock_service(
        hass, MUSIC_INTEGRATION, "play_media"
    )

    outcome = await _run(_executor(hass), "music_play", {},
                         "תנגן את האלבום שבלול במוסך")

    assert not outcome.ok
    assert calls == []


async def test_a_service_that_throws_is_a_failed_outcome_not_an_exception(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    _add_player(hass, "ma_living", MUSIC_INTEGRATION, rooms["סלון"])

    async def _explode(call: ServiceCall) -> None:
        raise ValueError("the player is not connected")

    hass.services.async_register(MUSIC_INTEGRATION, "play_media", _explode)

    outcome = await _run(_executor(hass), "music_play", {},
                         "תנגן את האלבום שבלול בסלון")

    assert not outcome.ok
    assert "not connected" in outcome.detail


# -- questions ----------------------------------------------------------------


async def test_the_weather_is_read_off_the_weather_entity(
    hass: HomeAssistant
) -> None:
    hass.states.async_set(
        "weather.home", "sunny",
        {"friendly_name": "מזג אוויר", "temperature": 28},
    )

    outcome = await _run(_executor(hass), "get_weather", {}, "מה מזג האוויר")

    assert outcome.ok
    assert outcome.speech


async def test_asking_the_weather_with_no_weather_entity_says_so(
    hass: HomeAssistant
) -> None:
    outcome = await _run(_executor(hass), "get_weather", {}, "מה מזג האוויר")
    assert not outcome.ok


async def test_a_sensor_reading_is_spoken_with_its_unit(
    hass: HomeAssistant, rooms: dict[str, str]
) -> None:
    entities = er.async_get(hass)
    entry = entities.async_get_or_create(
        "sensor", "needle_test", "living_temp", suggested_object_id="living_temp"
    )
    entities.async_update_entity(entry.entity_id, area_id=rooms["סלון"])
    hass.states.async_set(
        entry.entity_id, "23.5",
        {"friendly_name": "טמפרטורה סלון", "unit_of_measurement": "°C"},
    )

    # "מה הטמפרטורה" is routed to `climate` on purpose - asking the temperature
    # of a room usually means the thermostat, not a bare sensor - so this asks
    # the question that does mean the sensor.
    outcome = await _run(_executor(hass), "get_state",
                         {"area": "living_room", "domain": "sensor"},
                         "מה החיישן בסלון")

    assert outcome.ok
    assert "23.5" in outcome.speech
    assert "°C" in outcome.speech
