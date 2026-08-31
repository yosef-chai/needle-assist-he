"""What changed in the conversation entity after the v11 measurement.

Two rules, both found by reading the pipeline's own error dump rather than by
design, and both written up in the project README:

* the refusal gate ran **once**, on the whole sentence, and then `clause_split`
  cut the sentence up and every piece went to the model - so the half of
  "צריך מים מינרלים, תוסיפי לרשימה" that is context came back as a tool call;
* the state filter read the state word without asking whether the question was
  *which* or *whether*, and turned 127 yes/no questions into lists.

The model is replaced by a stub that returns what a real run returned, exactly
as `test_conversation.py` does, so what is under test is everything around it.
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

from custom_components.needle_assist import clause_split, tool_router

AGENT = "conversation.needle_assist"


def _answer(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"function_calls": list(calls), "confidence": 0.0,
            "success": True, "error": None}


@pytest.fixture
def house(hass: HomeAssistant) -> dict[str, str]:
    areas = ar.async_get(hass)
    entities = er.async_get(hass)
    made: dict[str, str] = {}
    for domain, unique, room, state, attributes in (
        ("light", "living_light", "סלון", "on", {"friendly_name": "אור סלון"}),
        ("light", "kitchen_light", "מטבח", "off", {"friendly_name": "אור מטבח"}),
        ("cover", "living_blind", "סלון", "open",
         {"friendly_name": "תריס סלון", "device_class": "blind"}),
        ("todo", "shopping", "סלון", "0", {"friendly_name": "רשימת קניות"}),
    ):
        area = areas.async_get_area_by_name(room) or areas.async_create(room)
        entry = entities.async_get_or_create(
            domain, "needle_test", unique, suggested_object_id=unique
        )
        entities.async_update_entity(entry.entity_id, area_id=area.id)
        hass.states.async_set(entry.entity_id, state, attributes)
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


async def test_a_statement_beside_an_order_moves_nothing(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    """The clause that names nothing this house controls never reaches the model.

    Measured over all 28,233 corpus rows: 303 clauses dropped, 2 gold calls
    lost, and both of those are word-merge speech noise that swallowed the
    device noun. See `tool_router.clause_names_nothing`.
    """
    covers: list[ServiceCall] = async_mock_service(hass, "cover", "close_cover")
    seen: list[str] = []

    def per_clause(clause: str, _tokens: int) -> dict[str, Any]:
        seen.append(clause)
        if "שוקולד" in clause:
            # What the model really answered the context half with.
            return _answer({"name": "cover_control",
                            "arguments": {"action": "close"}})
        return _answer({"name": "list_edit", "arguments": {"action": "done"}})

    with patch.object(agent.runtime_data, "complete", side_effect=per_clause):
        await _say(hass, "כבר לקחתי שוקולד, תסמן לרשימת הקניות")

    assert not covers, "a blind moved on the half of the sentence that was context"
    assert seen == ["תסמן לרשימת הקניות"], seen


async def test_every_clause_being_silent_keeps_all_of_them(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    """A sentence that passed the gate holds an order somewhere.

    If each piece looks empty on its own the split is what is wrong, not the
    sentence, so nothing is dropped and the model sees what it saw before.
    """
    seen: list[str] = []

    def per_clause(clause: str, _tokens: int) -> dict[str, Any]:
        seen.append(clause)
        return _answer()

    sentence = "תכבה את האור בסלון ותנעל את הדלת"
    clauses = clause_split.split_clauses(sentence)
    assert len(clauses) > 1, clauses

    # Every clause silent is the case the corpus does not contain, so it is
    # made rather than found: the guard is what keeps a bad split from
    # swallowing the whole sentence.
    with (
        patch.object(agent.runtime_data, "complete", side_effect=per_clause),
        patch.object(tool_router, "clause_names_nothing", return_value=True),
    ):
        await _say(hass, sentence)

    assert seen == clauses, seen


async def test_a_question_asks_which_or_whether_and_they_differ(
    hass: HomeAssistant, house: dict[str, str], agent: ConfigEntry
) -> None:
    """"אילו אורות דולקים" lists the lit ones; "תבדוק אם" answers yes or no.

    Reading the state word on its own was measured at 544 right and 127 wrong,
    every one of the 127 a yes/no question turned into a list. See
    `slot_match.state_filter`.
    """
    def answer(_clause: str, _tokens: int) -> dict[str, Any]:
        return _answer({"name": "get_state", "arguments": {"domain": "light"}})

    with patch.object(agent.runtime_data, "complete", side_effect=answer):
        which = await _say(hass, "אילו אורות דולקים")
        whether = await _say(hass, "תבדוק אם האור דולק")

    said_which = which.response.speech["plain"]["speech"]
    said_whether = whether.response.speech["plain"]["speech"]
    # The house has one light on and one off. Filtered, only the lit one is
    # named; unfiltered, the answer counts both.
    assert "אור סלון" in said_which, said_which
    assert "אור מטבח" not in said_which, said_which
    assert said_whether != said_which
    assert "כבוי" in said_whether or "פעיל" in said_whether, said_whether
