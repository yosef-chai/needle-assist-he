"""What the household kept away from voice stays away from it.

Every path that finds an entity - a room, a name, a slug, "all the timers",
the satellites, the notifiers, the Music Assistant players, the weather, the
lists - is asked the same question Home Assistant's own agent asks:
`async_should_expose(hass, "conversation", entity_id)`. Each test below sets
up one entity that would be found, leaves it unexposed, and checks that
nothing is called on it.
"""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.needle_assist.const import DOMAIN, MUSIC_INTEGRATION
from custom_components.needle_assist.executor import CallExecutor

ASSISTANT = "conversation"


@pytest.fixture
def executor(hass: HomeAssistant) -> CallExecutor:
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={}, version=2)
    entry.add_to_hass(hass)
    return CallExecutor(hass, entry)


@pytest.fixture
def living(hass: HomeAssistant) -> str:
    return ar.async_get(hass).async_create("סלון").id


def _entity(hass: HomeAssistant, domain: str, object_id: str, state: str,
            attributes: dict[str, Any] | None = None, *,
            area_id: str | None = None, exposed: bool,
            platform: str = "needle_test") -> str:
    entities = er.async_get(hass)
    entry = entities.async_get_or_create(
        domain, platform, object_id, suggested_object_id=object_id
    )
    if area_id:
        entities.async_update_entity(entry.entity_id, area_id=area_id)
    hass.states.async_set(entry.entity_id, state, attributes or {})
    async_expose_entity(hass, ASSISTANT, entry.entity_id, exposed)
    return entry.entity_id


async def _run(executor: CallExecutor, tool: str, arguments: dict[str, Any],
               utterance: str) -> Any:
    return await executor.execute(
        {"name": tool, "arguments": arguments}, None, Context(), utterance
    )


# -- the room match has no second, unfiltered try -----------------------------


async def test_a_lock_that_is_not_exposed_is_not_unlocked(
    hass: HomeAssistant, executor: CallExecutor, living: str
) -> None:
    """Locks are not exposed by default. The match used to retry without the
    assistant filter when nothing exposed was found, and so reached it."""
    _entity(hass, "lock", "front_door", "locked", {"friendly_name": "דלת כניסה"},
            area_id=living, exposed=False)
    calls: list[ServiceCall] = async_mock_service(hass, "lock", "unlock")

    outcome = await _run(executor, "lock_unlock", {"area": "living_room"},
                         "תפתח את המנעול בסלון")

    assert not outcome.ok
    assert calls == []


async def test_a_lock_left_at_its_default_is_not_unlocked_either(
    hass: HomeAssistant, executor: CallExecutor, living: str
) -> None:
    """No setting at all is Home Assistant's default, and for a lock that is
    "not exposed"."""
    entities = er.async_get(hass)
    entry = entities.async_get_or_create(
        "lock", "needle_test", "back_door", suggested_object_id="back_door")
    entities.async_update_entity(entry.entity_id, area_id=living)
    hass.states.async_set(entry.entity_id, "locked", {})
    calls: list[ServiceCall] = async_mock_service(hass, "lock", "unlock")

    outcome = await _run(executor, "lock_unlock", {"area": "living_room"},
                         "תפתח את המנעול בסלון")

    assert not outcome.ok
    assert calls == []


async def test_only_the_exposed_one_of_two_lights_in_a_room_moves(
    hass: HomeAssistant, executor: CallExecutor, living: str
) -> None:
    shown = _entity(hass, "light", "ceiling", "off", {"friendly_name": "תקרה"},
                    area_id=living, exposed=True)
    _entity(hass, "light", "nursery", "off", {"friendly_name": "מנורת לילה"},
            area_id=living, exposed=False)
    calls: list[ServiceCall] = async_mock_service(hass, "light", "turn_on")

    outcome = await _run(executor, "light_turn_on", {"area": "living_room"},
                         "תדליק את האור בסלון")

    assert outcome.ok
    assert calls[0].data["entity_id"] == [shown]


# -- by name ------------------------------------------------------------------


async def test_a_device_named_outright_is_not_reached_unexposed(
    hass: HomeAssistant, executor: CallExecutor, living: str
) -> None:
    """The phrase index is built from exposed entities only, so the name does
    not even shadow the room: the sentence falls back to the room's lights."""
    _entity(hass, "switch", "espresso", "off", {"friendly_name": "מכונת קפה"},
            area_id=living, exposed=False)
    calls: list[ServiceCall] = async_mock_service(hass, "switch", "turn_on")

    outcome = await _run(executor, "switch_turn_on", {}, "תדליק את מכונת הקפה")

    assert not outcome.ok
    assert calls == []


async def test_exposing_the_device_makes_its_name_work(
    hass: HomeAssistant, executor: CallExecutor, living: str
) -> None:
    espresso = _entity(hass, "switch", "espresso", "off",
                       {"friendly_name": "מכונת קפה"},
                       area_id=living, exposed=False)
    calls: list[ServiceCall] = async_mock_service(hass, "switch", "turn_on")
    assert not (await _run(executor, "switch_turn_on", {},
                           "תדליק את מכונת הקפה")).ok

    async_expose_entity(hass, ASSISTANT, espresso, True)
    executor.slots.invalidate()
    outcome = await _run(executor, "switch_turn_on", {}, "תדליק את מכונת הקפה")

    assert outcome.ok
    assert calls[0].data["entity_id"] == [espresso]


async def test_a_script_is_not_run_unexposed_by_its_name_or_its_slug(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    _entity(hass, "script", "bedtime", "off", {"friendly_name": "שגרת לילה"},
            exposed=False)
    calls: list[ServiceCall] = async_mock_service(hass, "script", "turn_on")

    by_words = await _run(executor, "script_run", {}, "תפעיל את שגרת לילה")
    by_slug = await _run(executor, "script_run", {"name": "bedtime"}, "")

    assert not by_words.ok
    assert not by_slug.ok
    assert calls == []


# -- "all of them" means all of the exposed ones ------------------------------


async def test_cancelling_every_timer_leaves_the_unexposed_one_running(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    shown = _entity(hass, "timer", "pasta", "active", {"friendly_name": "פסטה"},
                    exposed=True)
    _entity(hass, "timer", "irrigation", "active", {"friendly_name": "השקיה"},
            exposed=False)
    calls: list[ServiceCall] = async_mock_service(hass, "timer", "cancel")

    outcome = await _run(executor, "timer_cancel", {}, "תבטל את הטיימר")

    assert outcome.ok
    assert calls[0].data["entity_id"] == [shown]


async def test_the_timer_status_does_not_read_an_unexposed_timer(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    _entity(hass, "timer", "irrigation", "active",
            {"friendly_name": "השקיה", "remaining": "0:10:00"}, exposed=False)

    outcome = await _run(executor, "timer_status", {}, "כמה זמן נשאר בטיימר")

    assert outcome.ok
    assert outcome.answered_from == ()
    assert "השקיה" not in (outcome.speech or "")


async def test_a_notification_does_not_reach_an_unexposed_notifier(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    _entity(hass, "notify", "phone", "unknown", {"friendly_name": "טלפון"},
            exposed=False)
    calls: list[ServiceCall] = async_mock_service(hass, "notify", "send_message")

    outcome = await _run(executor, "notify_send", {},
                         "תשלח הודעה שאני בדרך הביתה")

    assert not outcome.ok
    assert calls == []


async def test_an_announcement_skips_an_unexposed_satellite(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    shown = _entity(hass, "assist_satellite", "kitchen_sat", "idle",
                    exposed=True)
    _entity(hass, "assist_satellite", "nursery_sat", "idle", exposed=False)
    calls: list[ServiceCall] = async_mock_service(hass, "assist_satellite",
                                                  "announce")

    outcome = await _run(executor, "broadcast", {},
                         "תכריז בכל הבית שהאוכל מוכן")

    assert outcome.ok, outcome.detail
    assert calls[0].data["entity_id"] == [shown]


# -- music, weather, lists ----------------------------------------------------


async def test_an_unexposed_music_assistant_player_is_not_played_on(
    hass: HomeAssistant, executor: CallExecutor, living: str
) -> None:
    _entity(hass, "media_player", "ma_living", "idle", {"friendly_name": "סלון"},
            area_id=living, exposed=False, platform=MUSIC_INTEGRATION)
    played: list[ServiceCall] = async_mock_service(hass, MUSIC_INTEGRATION,
                                                   "play_media")
    resumed: list[ServiceCall] = async_mock_service(hass, "media_player",
                                                    "media_play")

    outcome = await _run(executor, "music_play", {},
                         "תנגן את האלבום שבלול בסלון")

    assert not outcome.ok
    assert played == []
    assert resumed == []


async def test_the_weather_is_not_read_off_an_unexposed_entity(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    """The same rule Home Assistant's own `HassGetWeather` applies."""
    hass.states.async_set("weather.home", "sunny", {"temperature": 28})
    async_expose_entity(hass, ASSISTANT, "weather.home", False)

    outcome = await _run(executor, "get_weather", {}, "מה מזג האוויר")

    assert not outcome.ok


async def test_an_unexposed_list_is_not_written_to_by_the_back_door(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    """`shopping_list.add_item` writes to the list its `todo` entity shows,
    so it is not a way round that entity being kept from Assist."""
    _entity(hass, "todo", "shopping_list", "0", {"friendly_name": "רשימת קניות"},
            exposed=False, platform="shopping_list")
    legacy = async_mock_service(hass, "shopping_list", "add_item")
    todo = async_mock_service(hass, "todo", "add_item")

    outcome = await _run(executor, "list_edit", {"action": "add"},
                         "תוסיף לחם לרשימת הקניות")

    assert not outcome.ok
    assert legacy == []
    assert todo == []


# -- the index follows the exposure settings ----------------------------------


async def test_changing_what_is_exposed_rebuilds_the_phrase_index(
    hass: HomeAssistant, loaded_runner: Any, entry: ConfigEntry, living: str
) -> None:
    """Hidden after the index was built, the name must stop working at once -
    not on the next restart."""
    espresso = _entity(hass, "switch", "espresso", "off",
                       {"friendly_name": "מכונת קפה"},
                       area_id=living, exposed=True)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    executor = hass.data["entity_components"]["conversation"].get_entity(
        "conversation.needle_assist")._executor

    assert executor.slots.entities("תדליק את מכונת הקפה", "switch") == [espresso]

    async_expose_entity(hass, ASSISTANT, espresso, False)
    await hass.async_block_till_done()

    assert executor.slots.entities("תדליק את מכונת הקפה", "switch") == []
