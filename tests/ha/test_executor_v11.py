"""The v11 wire format, end to end against a real Home Assistant.

Everything the model emits since v11 is ``(tool, action)`` rather than a
per-service tool name, and the whole of `executor` speaks the old per-service
name - what `ha_tools` calls a **virtual id**. One function decodes between
them. These tests are that function's contract, checked where it matters:
against `hass.services`, with real registries, on the behaviours the catalogue
gained.
"""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.needle_assist.const import ACTIONS, DOMAIN, SERVICE_MAP
from custom_components.needle_assist.executor import CallExecutor


@pytest.fixture
def house(hass: HomeAssistant) -> dict[str, str]:
    """A house with the things v11 added: a valve, a button, a list, a timer.

    Two floors, because a floor is the targeting level v11 added and a house
    with one cannot show the difference.
    """
    areas = ar.async_get(hass)
    entities = er.async_get(hass)
    floors = fr.async_get(hass)

    upstairs = floors.async_create("קומה עליונה")
    downstairs = floors.async_create("קומת קרקע")
    living = areas.async_create("סלון")
    bedroom = areas.async_create("חדר שינה")
    areas.async_update(living.id, floor_id=downstairs.floor_id)
    areas.async_update(bedroom.id, floor_id=upstairs.floor_id)

    made: dict[str, str] = {}
    for domain, unique, area, state, attributes in (
        ("light", "living_light", living, "off", {"friendly_name": "אור סלון"}),
        ("light", "bedroom_light", bedroom, "on", {"friendly_name": "אור חדר שינה"}),
        ("valve", "garden_tap", living, "closed", {"friendly_name": "ברז גינה"}),
        ("button", "doorbell", living, "unknown", {"friendly_name": "כפתור הפעמון"}),
        ("cover", "living_blind", living, "open",
         {"friendly_name": "תריס סלון", "device_class": "blind"}),
        ("cover", "living_curtain", living, "open",
         {"friendly_name": "וילון סלון", "device_class": "curtain"}),
        ("media_player", "living_speaker", living, "playing",
         {"friendly_name": "רמקול סלון", "volume_level": 0.3}),
        ("climate", "living_ac", living, "off",
         {"friendly_name": "מזגן סלון", "temperature": 24}),
        ("timer", "pasta", living, "active",
         {"friendly_name": "טיימר פסטה", "remaining": "0:04:30"}),
        ("todo", "shopping", living, "3", {"friendly_name": "רשימת קניות"}),
        # No switch in this room, so `switch_control` finds nothing here and
        # `const.FALLBACK_DOMAINS` has somewhere to widen to.
        ("water_heater", "boiler", bedroom, "off",
         {"friendly_name": "דוד"}),
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


async def _wire(executor: CallExecutor, tool: str, action: str | None,
                arguments: dict[str, Any] | None = None,
                utterance: str = "") -> Any:
    """Call the executor exactly as the model would: tool plus action."""
    args: dict[str, Any] = {} if action is None else {"action": action}
    args.update(arguments or {})
    return await executor.execute(
        {"name": tool, "arguments": args}, None, Context(), utterance
    )


# -- decoding -----------------------------------------------------------------


async def test_every_action_reaches_the_service_the_map_names(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """The whole catalogue, one behaviour at a time.

    A behaviour that decodes to the wrong virtual id runs the wrong service,
    and there are 63 of them - too many to spot-check. Only the ones whose
    domain this house has are asserted to actually fire; the rest are asserted
    not to be *mis*-decoded, which is the failure this guards.
    """
    for tool, mapping in ACTIONS.items():
        for action, virtual in mapping.items():
            if virtual not in SERVICE_MAP:
                continue  # a query behaviour; covered below
            domain, service = SERVICE_MAP[virtual]
            calls = async_mock_service(hass, domain, service)
            outcome = await _wire(executor, tool, action, {"area": "living_room"})
            # Either it ran the right service, or it found nothing to run it
            # on - never a different service.
            assert outcome.tool == virtual, (tool, action, outcome.tool)
            if outcome.ok:
                assert calls, (tool, action, domain, service)


async def test_a_v10_name_still_runs(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """A household running the previous weights against this component."""
    calls = async_mock_service(hass, "light", "turn_off")
    outcome = await executor.execute(
        {"name": "light_turn_off", "arguments": {"area": "living_room"}},
        None, Context(), "תכבה את האור בסלון",
    )
    assert outcome.ok and calls


async def test_a_name_from_no_catalogue_is_refused(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    outcome = await _wire(executor, "light_explode", "on")
    assert not outcome.ok
    assert "unknown tool" in outcome.detail


async def test_an_action_the_tool_does_not_have_is_refused(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """The grammar cannot emit one, but a household's own fine-tune can."""
    outcome = await _wire(executor, "light_control", "explode")
    assert not outcome.ok


# -- what v11 added -----------------------------------------------------------


async def test_a_valve_opens_with_its_own_service(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """`valve` spells open as `open_valve`, not `open_cover`."""
    calls = async_mock_service(hass, "valve", "open_valve")
    outcome = await _wire(executor, "valve_control", "open", {},
                          "תפתח את הברז בסלון")
    assert outcome.ok
    assert calls and calls[0].data["entity_id"] == [house["garden_tap"]]


async def test_a_boiler_is_reached_through_the_switch_tool(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """`water_heater` is exposed to Assist by default and had no route here.

    A boiler takes the same Hebrew verb as a plug - תדליק - so it reaches
    `switch_control`, and `const.FALLBACK_DOMAINS` widens the search when the
    primary domain has nothing to offer. The table's shape is unit-tested one
    level up; this is the only place the redirect actually runs a service.

    It is also the sentence the plan set as one of its ten live checks, and the
    only one of the ten whose answer the offline harness cannot see, because
    widening needs a registry. See `eval/intent_parity.py::PLAN_TEN`.
    """
    switches = async_mock_service(hass, "switch", "turn_on")
    calls = async_mock_service(hass, "water_heater", "turn_on")
    outcome = await _wire(executor, "switch_control", "on", {}, "תדליק את הדוד")
    assert outcome.ok, outcome.detail
    assert calls and calls[0].data["entity_id"] == [house["boiler"]]
    # The primary domain always wins, so widening may only turn "nothing
    # matched" into an action - never one action into a different one.
    assert not switches


async def test_a_button_is_pressed_not_turned_on(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    calls = async_mock_service(hass, "button", "press")
    outcome = await _wire(executor, "routine_run", "press",
                          {"name": "doorbell"}, "תלחץ על כפתור הפעמון")
    assert outcome.ok, outcome.detail
    assert calls


async def test_the_device_class_narrows_a_room_with_two_kinds_of_cover(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """"תפתח את הווילונות בסלון" is not "תפתח את התריסים בסלון".

    Before v11 it was: the executor matched every cover in the area and opened
    the lot.
    """
    calls = async_mock_service(hass, "cover", "close_cover")
    outcome = await _wire(executor, "cover_control", "close", {},
                          "תסגור את הווילון בסלון")
    assert outcome.ok
    assert calls[0].data["entity_id"] == [house["living_curtain"]]

    calls.clear()
    await _wire(executor, "cover_control", "close", {}, "תסגור את התריס בסלון")
    assert calls[0].data["entity_id"] == [house["living_blind"]]


async def test_a_floor_is_a_set_of_rooms(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """Named in the sentence, and by the model's own slug when it is not."""
    calls = async_mock_service(hass, "light", "turn_off")
    outcome = await _wire(executor, "light_control", "shut", {},
                          "תכבה את האורות בקומה העליונה")
    assert outcome.ok
    assert calls[0].data["entity_id"] == [house["bedroom_light"]]

    # The model's slug, for the phrasing the registry index cannot reach.
    calls.clear()
    outcome = await _wire(executor, "light_control", "shut", {"floor": "upper"},
                          "תכבה את האורות")
    assert outcome.ok
    assert calls[0].data["entity_id"] == [house["bedroom_light"]]


async def test_an_item_reaches_the_list(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """The item comes out of the sentence, never out of the model."""
    calls = async_mock_service(hass, "todo", "add_item")
    outcome = await _wire(executor, "list_edit", "add", {"list": "shopping"},
                          "תוסיף חלב לרשימת הקניות")
    assert outcome.ok, outcome.detail
    assert calls[0].data["item"] == "חלב"
    assert calls[0].data["entity_id"] == [house["shopping"]]


async def test_a_list_command_with_no_item_does_nothing(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """A shopping list with a blank entry on it is worse than a refusal."""
    calls = async_mock_service(hass, "todo", "add_item")
    outcome = await _wire(executor, "list_edit", "add", {"list": "shopping"},
                          "תוסיף")
    assert not outcome.ok
    assert not calls


async def test_a_timer_gains_and_loses_time_with_one_service(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """`timer.change` takes a signed duration, and the sign is the behaviour."""
    calls = async_mock_service(hass, "timer", "change")
    await _wire(executor, "timer_control", "add", {"minutes": 5},
                "תוסיף עוד חמש דקות לטיימר")
    assert calls[0].data["duration"] == 300
    calls.clear()
    await _wire(executor, "timer_control", "less", {"minutes": 5},
                "תקצר את הטיימר בחמש דקות")
    assert calls[0].data["duration"] == -300


async def test_a_timer_carries_hours_and_seconds(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    calls = async_mock_service(hass, "timer", "start")
    await _wire(executor, "timer_control", "start", {"hours": 1, "minutes": 30},
                "תעמיד טיימר לשעה וחצי")
    assert calls[0].data["duration"] == "01:30:00"


async def test_a_paused_timer_resumes_with_start(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """Home Assistant has no `timer.resume`; its own intent calls `start`."""
    calls = async_mock_service(hass, "timer", "start")
    outcome = await _wire(executor, "timer_control", "resume", {},
                          "תמשיך את הטיימר")
    assert outcome.ok
    assert calls and "duration" not in calls[0].data


async def test_the_timer_status_answers_instead_of_restarting(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """Answering "כמה זמן נשאר" with `timer.start` restarts the countdown."""
    calls = async_mock_service(hass, "timer", "start")
    outcome = await _wire(executor, "timer_control", "query", {},
                          "כמה זמן נשאר בטיימר")
    assert outcome.ok
    assert not calls
    assert "0:04:30" in outcome.speech


async def test_the_clock_needs_no_device(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """`HassGetCurrentTime` and `HassGetCurrentDate` answer from the timezone."""
    outcome = await _wire(executor, "get_datetime", "time", {}, "מה השעה")
    assert outcome.ok and outcome.speech.startswith("השעה")
    outcome = await _wire(executor, "get_datetime", "date", {}, "מה התאריך")
    assert outcome.ok
    # Hebrew, not an English month name in the middle of a Hebrew sentence.
    assert not any(c.isascii() and c.isalpha() for c in outcome.speech)


async def test_an_announcement_is_not_a_notification(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """`HassBroadcast` speaks out loud; `notify_send` buzzes a phone."""
    entities = er.async_get(hass)
    entry = entities.async_get_or_create(
        "assist_satellite", "needle_test", "kitchen_sat",
        suggested_object_id="kitchen_sat")
    hass.states.async_set(entry.entity_id, "idle", {})
    calls = async_mock_service(hass, "assist_satellite", "announce")
    outcome = await _wire(executor, "broadcast", None, {"message": "garbage"},
                          "תכריז בכל הבית שהאוכל מוכן")
    assert outcome.ok, outcome.detail
    # The model's own string is discarded; the sentence has the words.
    assert calls[0].data["message"] == "האוכל מוכן"


async def test_an_air_conditioner_can_be_started(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """`climate.turn_on` exists and was simply never mapped, so the commonest
    way in Hebrew to start an air conditioner had to arrive as a mode change."""
    calls = async_mock_service(hass, "climate", "turn_on")
    outcome = await _wire(executor, "climate_control", "run", {},
                          "תדליק את המזגן בסלון")
    assert outcome.ok
    assert calls[0].data["entity_id"] == [house["living_ac"]]


async def test_the_sentence_still_overrules_the_action(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """`direction.settle` measured 1337 agreements and 0 disagreements against
    gold, and it now settles an `action` rather than a tool name - the same
    decision, one level in."""
    calls = async_mock_service(hass, "light", "turn_off")
    outcome = await _wire(executor, "light_control", "on", {},
                          "תכבה את האור בסלון")
    assert outcome.tool == "light_turn_off"
    assert outcome.ok and calls


async def test_a_stale_argument_never_reaches_the_service(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """A domain tool declares every argument any of its behaviours takes, so
    the grammar now permits `light_control{action: "shut", brightness_pct: 40}`
    - well formed, and not a thing `light.turn_off` accepts."""
    calls = async_mock_service(hass, "light", "turn_off")
    outcome = await _wire(executor, "light_control", "shut",
                          {"brightness_pct": 40}, "תכבה את האור בסלון")
    assert outcome.ok
    assert "brightness_pct" not in calls[0].data


async def test_a_colour_temperature_reaches_home_assistants_own_slot(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    calls = async_mock_service(hass, "light", "turn_on")
    outcome = await _wire(executor, "light_control", "on", {},
                          "תדליק את האור בסלון בלבן חם")
    assert outcome.ok
    assert calls[0].data["color_temp_kelvin"] == 2700


async def test_the_registry_picks_the_list_integration_not_the_model(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """Home Assistant has two list integrations and they do not share a
    vocabulary: `todo` completes an item with `update_item` and a status,
    `shopping_list` with `complete_item`. A household has one, or the other, or
    both, and the sentence cannot know which - so the same rule
    `ROUTINE_SIBLING` runs on applies here."""
    todo = async_mock_service(hass, "todo", "update_item")
    outcome = await _wire(executor, "list_edit", "done", {"list": "shopping"},
                          "תסמן שקניתי חלב")
    assert outcome.ok, outcome.detail
    assert todo[0].data["status"] == "completed"
    assert todo[0].data["item"] == "חלב"


async def test_a_house_with_no_todo_entity_falls_back_to_shopping_list(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    """The legacy integration has exactly one list and no entity at all."""
    legacy = async_mock_service(hass, "shopping_list", "add_item")
    outcome = await _wire(executor, "list_edit", "add", {"list": "shopping"},
                          "תוסיף לחם לרשימת הקניות")
    assert outcome.ok, outcome.detail
    # Its own field name, which is not `todo`'s.
    assert legacy[0].data["name"] == "לחם"
    assert "entity_id" not in legacy[0].data


async def test_a_house_with_neither_says_so(
    hass: HomeAssistant, executor: CallExecutor
) -> None:
    outcome = await _wire(executor, "list_edit", "add", {"list": "shopping"},
                          "תוסיף לחם לרשימת הקניות")
    assert not outcome.ok
    assert "no list" in outcome.detail

async def test_a_device_inside_a_room_name_is_the_room(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """"תכבה את המפסק בחדר המחשב" is the switch in the computer room.

    `executor` consults the named entities before it consults the areas at all,
    so a house with a switch called "המחשב" turned off the computer instead.
    The room phrase contains the device phrase outright, which settles it with
    nothing to tune - see `slot_match.Slots.entities`. 8 rows over the corpus,
    every one a wrong device.
    """
    areas = ar.async_get(hass)
    entities = er.async_get(hass)
    study = areas.async_create("חדר המחשב")
    made = {}
    for unique, area, name in (("study_switch", study, "מפסק חדר המחשב"),
                               ("desk_computer", None, "המחשב")):
        entry = entities.async_get_or_create(
            "switch", "needle_v11", unique, suggested_object_id=unique)
        if area is not None:
            entities.async_update_entity(entry.entity_id, area_id=area.id)
        hass.states.async_set(entry.entity_id, "on", {"friendly_name": name})
        made[unique] = entry.entity_id

    calls = async_mock_service(hass, "switch", "turn_off")
    outcome = await _wire(executor, "switch_control", "shut", {},
                          "תכבה את המפסק בחדר המחשב")
    assert outcome.ok, outcome.detail
    assert calls[0].data["entity_id"] == [made["study_switch"]]

    # A device named beside a room is still the device.
    calls.clear()
    outcome = await _wire(executor, "switch_control", "shut", {},
                          "תכבה את המחשב בסלון")
    assert outcome.ok, outcome.detail
    assert calls[0].data["entity_id"] == [made["desk_computer"]]


# -- the same thing, kept in another domain -----------------------------------


async def test_a_television_in_the_other_domain_is_refused_not_guessed(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """The one ambiguity the registry cannot settle, kept as a refusal.

    A television is a `switch` in 43 of this corpus's 59 television rows and a
    `media_player` in the rest, so eighteen benchmark rows are the model
    answering `media_control` where gold says `switch_control`. A fallback was
    written for it and taken back out: the sentence names no room, so widening
    from `switch` to `media_player` reaches *every* speaker in the house and
    silences the kitchen with it. See `const.ROUTINE_SIBLING`'s comment for the
    two other routes and why they fail too.
    """
    entities = er.async_get(hass)
    living = ar.async_get(hass).async_get_area_by_name("סלון")
    entry = entities.async_get_or_create("media_player", "needle_test", "tv",
                                         suggested_object_id="tv")
    entities.async_update_entity(entry.entity_id, area_id=living.id)
    hass.states.async_set(entry.entity_id, "on", {"friendly_name": "טלוויזיה"})

    off = async_mock_service(hass, "media_player", "turn_off")
    outcome = await _wire(executor, "switch_control", "shut", {},
                          "תכבה את הטלוויזיה")
    assert not outcome.ok, "a guess here silences the speaker too"
    assert not off


async def test_the_registry_says_whether_a_mode_is_a_helper_or_a_scene(
    hass: HomeAssistant, house: dict[str, str], executor: CallExecutor
) -> None:
    """"מצב חופשה" was `routine_run` in one corpus recipe and `helper_toggle`
    in another. `data/repair_v11.py` settled the corpus on the helper; which
    one it is in *this* house is a question for the registry, and that is what
    `ROUTINE_SIBLING` has always been for.
    """
    entities = er.async_get(hass)
    entry = entities.async_get_or_create("scene", "needle_test", "vacation",
                                         suggested_object_id="vacation_mode")
    hass.states.async_set(entry.entity_id, "scening",
                          {"friendly_name": "מצב חופשה"})

    scene = async_mock_service(hass, "scene", "turn_on")
    outcome = await _wire(executor, "helper_toggle", "on",
                          {"name": "vacation_mode"}, "תפעיל את מצב חופשה")
    assert outcome.ok, outcome.detail
    assert scene[0].data["entity_id"] == [entry.entity_id]

    # There is nothing to turn off. A scene cannot be un-activated, so the
    # refusal is the right answer and not a gap.
    outcome = await _wire(executor, "helper_toggle", "shut",
                          {"name": "vacation_mode"}, "תכבה את מצב חופשה")
    assert not outcome.ok
