"""Constants and the tool -> Home Assistant service map."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

DOMAIN: Final = "needle_assist"

# The tuned Hebrew weights ship inside the component. A household installing
# this should not have to copy a file onto the machine or type a path into a
# dialog to get a working Hebrew assistant, and the untuned base model does not
# understand Hebrew at all - so with nothing configured, "nothing" has to mean
# the bundled model rather than a model that answers everything wrongly.
#
# An explicit path in the config entry always wins, for anyone running their own
# fine-tune. Resolved in an executor, never on the event loop.
BUNDLED_WEIGHTS: Final = Path(__file__).parent / "needle_he.cact"

CONF_WEIGHTS: Final = "weights_path"
CONF_LOG_UTTERANCES: Final = "log_utterances"
CONF_MAX_TOKENS: Final = "max_new_tokens"

# The shape of what the config entry stores. Version 2 removed two options, so
# an entry written by version 1 is migrated on load rather than left carrying
# keys nothing reads. See `__init__.async_migrate_entry`.
#
# Deliberately NOT bumped for v11. The catalogue rewrite changed what the model
# emits, not what the entry stores - the options are still a weights path, a
# token budget and a music player - and a version bump with no shape change is
# a migration that exists to do nothing. The risk it was proposed for is real
# and lives elsewhere: a household pointing `weights_path` at their own v10
# fine-tune gets a model that emits names this catalogue no longer has. That is
# caught where it happens, in `executor.execute`, which names the mismatch
# rather than reporting a generic failure.
ENTRY_VERSION: Final = 2

# The keys that migration strips. Both used to be dialog fields and are now
# fixed policy; they are named here only so the migration can find them.
RETIRED_OPTIONS: Final = ("min_confidence", "refuse_off_topic")

# Repair issue id, raised when a weights path the household typed in has since
# gone missing. One id, reused, so the repair closes itself when setup succeeds.
ISSUE_WEIGHTS_MISSING: Final = "weights_file_missing"

# Which speaker plays music when the sentence names no room. Optional, and
# only meaningful with Music Assistant installed. A house with exactly one
# Music Assistant player does not need it - that player is the only answer -
# so the option earns its place only where there are several and one of them
# is "the" speaker.
CONF_MUSIC_PLAYER: Final = "music_player"

#: Where the utterance log goes, under Home Assistant's configuration
#: directory. Off by default and never sent anywhere: every number this
#: project has is measured on a corpus it wrote itself, and what the household
#: actually says is the one ruler that is not.
UTTERANCE_LOG: Final = "needle_assist/utterances.jsonl"

#: Raised from 192 after the truncation study. A tool call is about 25 tokens,
#: but a Hebrew `message` is written as `\uXXXX` escapes in the model's own
#: training targets - six ASCII characters per letter - so an announcement of a
#: dozen words really does need the room. It buys three of the thirty-nine rows
#: that truncate; `repair.recover` answers the rest, and nothing else pays for
#: it because decoding stops at the end of the grammar either way.
DEFAULT_MAX_TOKENS: Final = 320

# Confidence gating is off, and that is policy rather than a default.
#
# Needle's own finetuning guide states the confidence head "is calibrated for
# the base model on its training mix and finetuning does not update the head,
# so the package disables them for tuned weights", and separately that correct
# non-English calls have been measured at confidence 0.0. Any floor above zero
# would reject every correct Hebrew call. That is why this is a constant and
# not a slider: there is no value a household could usefully move it to, and
# the slider that used to be here could only break a working assistant.
CONFIDENCE_FLOOR: Final = 0.0

# Whole-home marker. The model is fine-tuned to emit this for "בכל הבית".
# It is deliberately distinguishable from an ABSENT area, which means
# "wherever the speaker is" and resolves to the satellite's own area.
AREA_ALL: Final = "all"

# What the model emits, and what this component speaks internally.
#
# v11 collapsed the catalogue from one tool per service to one tool per domain
# with an ``action`` enum inside it, because Needle renders at most five tools
# per turn before a retrieval head that cannot read Hebrew takes over. Twenty
# tools now cover all 38 of Home Assistant's in-scope built-in intents.
#
# Everything below is still keyed by the *old* per-service name, which
# ``tools/ha_tools.py`` calls a **virtual id**. That is deliberate and it is
# what made a rewrite this size safe: every table here, every keyword list in
# `tool_router`, every pair in `direction` and every measurement behind them
# was taken at that granularity. Only the wire format changed, and it changes
# back at the door - `executor` decodes ``(name, action)`` into a virtual id
# before anything else runs.
#
# This table is a hand-kept mirror of ``ha_tools.ACTIONS``;
# `test_the_component_and_the_catalogue_agree_on_actions` fails if the two ever
# drift. It is duplicated rather than imported because the component ships
# without the workstation's ``tools/`` directory.
ACTIONS: Final[dict[str, dict[str, str]]] = {
    "light_control": {
        "on": "light_turn_on",
        "shut": "light_turn_off",
        "flip": "light_toggle",
    },
    "switch_control": {
        "on": "switch_turn_on",
        "shut": "switch_turn_off",
        "flip": "switch_toggle",
    },
    "fan_control": {
        "on": "fan_turn_on",
        "shut": "fan_turn_off",
        "flip": "fan_toggle",
        "rotate": "fan_oscillate",
    },
    "cover_control": {
        "open": "cover_open",
        "close": "cover_close",
        "stop": "cover_stop",
        "place": "cover_set_position",
    },
    "valve_control": {
        "open": "valve_open",
        "close": "valve_close",
        "stop": "valve_stop",
        "place": "valve_set_position",
    },
    "lock_control": {
        "lock": "lock_lock",
        "unlock": "lock_unlock",
    },
    "camera_control": {
        "on": "camera_turn_on",
        "shut": "camera_turn_off",
    },
    "climate_control": {
        "run": "climate_turn_on",
        "off": "climate_turn_off",
        "temp": "climate_set_temperature",
        "mode": "climate_set_hvac_mode",
        "fan": "climate_set_fan_mode",
    },
    "media_control": {
        "resume": "media_play",
        "pause": "media_pause",
        "stop": "media_stop",
        "next": "media_next_track",
        "back": "media_previous_track",
        "mute": "media_mute",
        "volume": "media_set_volume",
        "input": "media_select_source",
    },
    "vacuum_control": {
        "clean": "vacuum_start",
        "dock": "vacuum_return_to_base",
        "pause": "vacuum_pause",
        "suction": "vacuum_set_fan_speed",
    },
    "timer_control": {
        "start": "timer_start",
        "cancel": "timer_cancel",
        "pause": "timer_pause",
        "resume": "timer_resume",
        "add": "timer_add",
        "less": "timer_less",
        "query": "timer_status",
    },
    "list_edit": {
        "add": "list_add_item",
        "done": "list_complete_item",
        "remove": "list_remove_item",
    },
    "routine_run": {
        "scene": "scene_activate",
        "macro": "script_run",
        "enable": "automation_turn_on",
        "disable": "automation_turn_off",
        "press": "button_press",
    },
    "helper_toggle": {
        "on": "input_boolean_turn_on",
        "shut": "input_boolean_turn_off",
    },
    "get_datetime": {
        "date": "get_date",
        "time": "get_time",
    },
}

# The five tools with one behaviour each, whose virtual id is their own name.
ACTIONLESS: Final[tuple[str, ...]] = (
    "music_play", "notify_send", "broadcast", "get_state", "get_weather",
)

# virtual id -> (tool, action or None), and the reverse.
CALL_OF: Final[dict[str, tuple[str, str | None]]] = {
    virtual: (tool, action)
    for tool, mapping in ACTIONS.items()
    for action, virtual in mapping.items()
} | {name: (name, None) for name in ACTIONLESS}

VIRTUAL_OF: Final[dict[tuple[str, str | None], str]] = {
    pair: virtual for virtual, pair in CALL_OF.items()}

# The argument that carries the behaviour. Stripped at the door: it selects
# the virtual id and must never reach a Home Assistant service.
ACTION_ARG: Final = "action"

# tool name -> (domain, service). Every entry is a real Home Assistant service,
# checked against the component's own `services.yaml` by
# `test_every_service_in_the_map_exists`.
SERVICE_MAP: Final[dict[str, tuple[str, str]]] = {
    "light_turn_on": ("light", "turn_on"),
    "light_turn_off": ("light", "turn_off"),
    "light_toggle": ("light", "toggle"),
    "climate_set_temperature": ("climate", "set_temperature"),
    "climate_set_hvac_mode": ("climate", "set_hvac_mode"),
    "climate_set_fan_mode": ("climate", "set_fan_mode"),
    "climate_turn_off": ("climate", "turn_off"),
    # "תדליק את המזגן". `climate.turn_on` exists and was simply not mapped, so
    # the commonest way in Hebrew to start an air conditioner had to arrive as
    # a mode change or as a switch.
    "climate_turn_on": ("climate", "turn_on"),
    "fan_turn_on": ("fan", "turn_on"),
    "fan_turn_off": ("fan", "turn_off"),
    "fan_toggle": ("fan", "toggle"),
    "fan_oscillate": ("fan", "oscillate"),
    "cover_open": ("cover", "open_cover"),
    "cover_close": ("cover", "close_cover"),
    "cover_stop": ("cover", "stop_cover"),
    "cover_set_position": ("cover", "set_cover_position"),
    # `valve` spells none of these the way `cover` does.
    "valve_open": ("valve", "open_valve"),
    "valve_close": ("valve", "close_valve"),
    "valve_stop": ("valve", "stop_valve"),
    "valve_set_position": ("valve", "set_valve_position"),
    "lock_lock": ("lock", "lock"),
    "lock_unlock": ("lock", "unlock"),
    "camera_turn_on": ("camera", "turn_on"),
    "camera_turn_off": ("camera", "turn_off"),
    "vacuum_start": ("vacuum", "start"),
    "vacuum_return_to_base": ("vacuum", "return_to_base"),
    "vacuum_pause": ("vacuum", "pause"),
    "vacuum_set_fan_speed": ("vacuum", "set_fan_speed"),
    "media_play": ("media_player", "media_play"),
    "media_pause": ("media_player", "media_pause"),
    "media_stop": ("media_player", "media_stop"),
    "media_next_track": ("media_player", "media_next_track"),
    "media_previous_track": ("media_player", "media_previous_track"),
    "media_set_volume": ("media_player", "volume_set"),
    "media_mute": ("media_player", "volume_mute"),
    "media_select_source": ("media_player", "select_source"),
    # A core Home Assistant integration's own service, not media_player's.
    # media_player.play_media wants a content id - a URI - which nobody says
    # out loud; this one takes a search string and resolves it against the
    # library Music Assistant already indexes.
    "music_play": ("music_assistant", "play_media"),
    "switch_turn_on": ("switch", "turn_on"),
    "switch_turn_off": ("switch", "turn_off"),
    "switch_toggle": ("switch", "toggle"),
    "scene_activate": ("scene", "turn_on"),
    "script_run": ("script", "turn_on"),
    "automation_turn_on": ("automation", "turn_on"),
    "automation_turn_off": ("automation", "turn_off"),
    "button_press": ("button", "press"),
    "input_boolean_turn_on": ("input_boolean", "turn_on"),
    "input_boolean_turn_off": ("input_boolean", "turn_off"),
    "timer_start": ("timer", "start"),
    "timer_cancel": ("timer", "cancel"),
    "timer_pause": ("timer", "pause"),
    # Home Assistant has no `timer.resume`. A paused countdown is resumed by
    # calling `timer.start` again with no duration, which is what its own
    # HassUnpauseTimer intent does.
    "timer_resume": ("timer", "start"),
    # Adding or removing time is one service with a signed duration.
    "timer_add": ("timer", "change"),
    "timer_less": ("timer", "change"),
    # The `todo` domain. A household running the legacy `shopping_list`
    # integration instead is redirected by `LIST_INTEGRATIONS` below.
    "list_add_item": ("todo", "add_item"),
    "list_complete_item": ("todo", "update_item"),
    "list_remove_item": ("todo", "remove_item"),
    "notify_send": ("notify", "send_message"),
    # `HassBroadcast` maps to `assist_satellite.announce` inside Home
    # Assistant. There is no service called `broadcast`.
    "broadcast": ("assist_satellite", "announce"),
}

# Read-only tools. Not services - these answer rather than actuate.
#
# `timer_status` is here rather than in SERVICE_MAP for the reason the name
# gives: "כמה זמן נשאר בטיימר" asks, and answering it by calling `timer.start`
# restarts the very countdown somebody wanted to know about.
QUERY_TOOLS: Final = {"get_state", "get_weather", "get_date", "get_time",
                      "timer_status"}

# The domain each tool targets, for entity matching.
TOOL_DOMAIN: Final[dict[str, str]] = {
    name: dom for name, (dom, _) in SERVICE_MAP.items()
} | {
    # The tools whose service domain is not their target domain. The music
    # service belongs to the music_assistant integration and acts on an
    # ordinary media_player; the broadcast service belongs to
    # assist_satellite and acts on its own entities. Deriving these from
    # SERVICE_MAP alone would send entity matching looking for domains that
    # hold no entities.
    "music_play": "media_player",
} | {
    virtual: "timer" for virtual in ACTIONS["timer_control"].values()
} | {
    virtual: "todo" for virtual in ACTIONS["list_edit"].values()
}

# Model argument -> service data key, where they differ.
ARG_RENAME: Final[dict[str, str]] = {
    "brightness_step_pct": "brightness_step_pct",
    "volume_pct": "volume_level",       # HA takes 0..1, converted in code
    "temperature_step": "temperature",  # resolved against current state
    "volume_step_pct": "volume_level",  # resolved against current state
}

# Arguments that are targeting metadata or a selector, never service data.
#
# `floor` and `device_class` narrow which entities the call lands on;
# `list` picks which integration serves it; `state` filters a question. None
# of them is a parameter of any service, and Home Assistant rejects a call
# carrying one.
NON_SERVICE_ARGS: Final = {"area", "name", "floor", "device_class", "list",
                           "state", "domain", ACTION_ARG}

# Every argument each virtual id may carry into a service call.
#
# This is needed because the sentence is allowed to change which behaviour
# runs - `direction.settle` and the music upgrade both do it - and the model
# filled its arguments for the one it originally chose. "תעביר את המזגן
# למהירות גבוה" came back as `climate_set_temperature{temperature: 23}`, the
# verb corrected it to `climate_set_fan_mode`, and the stale `temperature`
# would have made Home Assistant reject the whole call: `climate.set_fan_mode`
# does not take one. The household would have been told the command failed, by
# the guard that was there to make it work.
#
# v11 made this table matter more, not less. A domain tool declares every
# argument any of its behaviours can take, so the grammar now permits
# `light_control{action: "shut", brightness_pct: 40}` - well-formed, and not a
# thing `light.turn_off` accepts. The per-behaviour filter is what keeps the
# wider schema from reaching the service.
#
# Kept here rather than read out of tools.json because that file is
# deliberately not read on the event loop (see `needle_runner`), and a test
# asserts the two agree exactly.
TOOL_ARGS: Final[dict[str, frozenset[str]]] = {
    "light_turn_on": frozenset((
        "area", "floor", "brightness_pct", "brightness_step_pct",
        "color_name", "color_temp_k")),
    "light_turn_off": frozenset(("area", "floor")),
    "light_toggle": frozenset(("area", "floor")),
    "switch_turn_on": frozenset(("area", "floor", "name")),
    "switch_turn_off": frozenset(("area", "floor", "name")),
    "switch_toggle": frozenset(("area", "floor", "name")),
    "fan_turn_on": frozenset(("area", "floor", "percentage")),
    "fan_turn_off": frozenset(("area", "floor")),
    "fan_toggle": frozenset(("area", "floor")),
    "fan_oscillate": frozenset(("area", "floor", "oscillating")),
    "cover_open": frozenset(("area", "floor", "device_class")),
    "cover_close": frozenset(("area", "floor", "device_class")),
    "cover_stop": frozenset(("area", "floor", "device_class")),
    "cover_set_position": frozenset((
        "area", "floor", "position", "device_class")),
    "valve_open": frozenset(("area", "floor")),
    "valve_close": frozenset(("area", "floor")),
    "valve_stop": frozenset(("area", "floor")),
    "valve_set_position": frozenset(("area", "floor", "position")),
    "lock_lock": frozenset(("area", "floor")),
    "lock_unlock": frozenset(("area", "floor")),
    "camera_turn_on": frozenset(("area", "floor")),
    "camera_turn_off": frozenset(("area", "floor")),
    "climate_turn_on": frozenset(("area", "floor")),
    "climate_turn_off": frozenset(("area", "floor")),
    "climate_set_temperature": frozenset((
        "area", "floor", "temperature", "temperature_step")),
    "climate_set_hvac_mode": frozenset(("area", "floor", "hvac_mode")),
    "climate_set_fan_mode": frozenset(("area", "floor", "fan_mode")),
    "media_play": frozenset(("area", "floor")),
    "media_pause": frozenset(("area", "floor")),
    "media_stop": frozenset(("area", "floor")),
    "media_next_track": frozenset(("area", "floor")),
    "media_previous_track": frozenset(("area", "floor")),
    "media_mute": frozenset(("area", "floor", "is_volume_muted")),
    "media_set_volume": frozenset((
        "area", "floor", "volume_pct", "volume_step_pct")),
    "media_select_source": frozenset(("area", "floor", "source")),
    "music_play": frozenset(("area", "floor", "media_type")),
    "vacuum_start": frozenset(("area", "floor")),
    "vacuum_return_to_base": frozenset(),
    "vacuum_pause": frozenset(),
    "vacuum_set_fan_speed": frozenset(("fan_speed",)),
    "timer_start": frozenset(("hours", "minutes", "seconds", "name")),
    "timer_cancel": frozenset(("name",)),
    "timer_pause": frozenset(("name",)),
    "timer_resume": frozenset(("name",)),
    "timer_add": frozenset(("hours", "minutes", "seconds", "name")),
    "timer_less": frozenset(("hours", "minutes", "seconds", "name")),
    "timer_status": frozenset(("name",)),
    "list_add_item": frozenset(("list",)),
    "list_complete_item": frozenset(("list",)),
    "list_remove_item": frozenset(("list",)),
    "scene_activate": frozenset(("name",)),
    "script_run": frozenset(("name",)),
    "automation_turn_on": frozenset(("name",)),
    "automation_turn_off": frozenset(("name",)),
    "button_press": frozenset(("name",)),
    "input_boolean_turn_on": frozenset(("name",)),
    "input_boolean_turn_off": frozenset(("name",)),
    "notify_send": frozenset(("message",)),
    "broadcast": frozenset(("message",)),
    "get_state": frozenset(("area", "floor", "domain", "state")),
    "get_weather": frozenset(("day_offset",)),
    "get_date": frozenset(),
    "get_time": frozenset(),
}

# Domains a tool falls back to when its own finds nothing in the room.
#
# Three of Home Assistant's eleven DEFAULT_EXPOSED_DOMAINS - `humidifier`,
# `water_heater` and `todo` - were unreachable here: a household sees the
# entity offered to Assist and gets "לא מצאתי מכשיר מתאים". `valve`, which
# HassTurnOn/Off/SetPosition all support, was in the same position, and so were
# `button` and `input_button`.
#
# v11 gave `valve`, `button`, `todo` and the datetime intents tools of their
# own, so the model can now say them outright. This table stays for the case a
# tool cannot cover: the household whose boiler is only ever called דוד and
# whose valve is only ever called a tap. A boiler and a humidifier are switched
# on with the same Hebrew verb as a plug and answer the same `turn_on`, so the
# sentence reaches the tool it already reaches and the *executor* widens the
# search when the primary domain has nothing in the room. Ordering is
# significant and the primary domain always wins, so this can only ever turn
# "no matching device" into an action - never one action into a different one.
#
# Each entry is (target domain, service domain, service). The service is named
# rather than derived because `valve` does not spell open as `open_cover` and a
# button is pressed rather than turned on.
FALLBACK_DOMAINS: Final[dict[str, tuple[tuple[str, str, str], ...]]] = {
    "switch_turn_on": (
        ("humidifier", "humidifier", "turn_on"),
        ("water_heater", "water_heater", "turn_on"),
    ),
    "switch_turn_off": (
        ("humidifier", "humidifier", "turn_off"),
        ("water_heater", "water_heater", "turn_off"),
    ),
    "switch_toggle": (
        ("humidifier", "humidifier", "toggle"),
    ),
    # A robot mower is a robot vacuum as far as Hebrew is concerned - it is
    # sent out, it comes back, it is paused - and `lawn_mower` mirrors
    # `vacuum`'s three services exactly under different names. Without this,
    # `HassLawnMowerStartMowing` and `HassLawnMowerDock` were the only two of
    # Home Assistant's own intents this integration could not reach at all.
    "vacuum_start": (("lawn_mower", "lawn_mower", "start_mowing"),),
    "vacuum_return_to_base": (("lawn_mower", "lawn_mower", "dock"),),
    "vacuum_pause": (("lawn_mower", "lawn_mower", "pause"),),
    # "תעלה את הדוד ל-60". The boiler already switches on and off through
    # `switch_control` above; its temperature had nowhere to go, and
    # `water_heater.set_temperature` takes the same `temperature` argument
    # under the same name.
    "climate_set_temperature": (
        ("water_heater", "water_heater", "set_temperature"),),
    "cover_open": (("valve", "valve", "open_valve"),),
    "cover_close": (("valve", "valve", "close_valve"),),
    "cover_stop": (("valve", "valve", "stop_valve"),),
    "cover_set_position": (("valve", "valve", "set_valve_position"),),
    # The mirror direction, for the house whose blind somebody called a tap.
    "valve_open": (("cover", "cover", "open_cover"),),
    "valve_close": (("cover", "cover", "close_cover"),),
    "valve_stop": (("cover", "cover", "stop_cover"),),
    "valve_set_position": (("cover", "cover", "set_cover_position"),),
    # `input_button` is `button` with a different domain and the same verb.
    "button_press": (("input_button", "input_button", "press"),),
}

# Which integration serves a list, and how it spells the three operations.
#
# Home Assistant has two and a household can have either: the modern `todo`
# domain, which holds one entity per list, and the legacy `shopping_list`
# integration, which has exactly one list and no entity at all. They do not
# share a vocabulary - `todo` completes an item with `update_item` and a
# status, `shopping_list` with `complete_item` - so the executor picks by what
# the registry actually holds rather than by what the model guessed.
#
# (virtual id) -> (service domain, service, extra service data).
LIST_INTEGRATIONS: Final[dict[str, dict[str, tuple[str, str, dict[str, Any]]]]] = {
    "todo": {
        "list_add_item": ("todo", "add_item", {}),
        "list_complete_item": ("todo", "update_item", {"status": "completed"}),
        "list_remove_item": ("todo", "remove_item", {}),
    },
    "shopping": {
        "list_add_item": ("shopping_list", "add_item", {}),
        "list_complete_item": ("shopping_list", "complete_item", {}),
        "list_remove_item": ("shopping_list", "remove_item", {}),
    },
}

# The service-data key each integration takes the item under.
LIST_ITEM_KEY: Final[dict[str, str]] = {"todo": "item", "shopping": "name"}

# Domains that hold several kinds of thing, where the sentence names which.
# `cover` is the one that matters: a room with blinds and curtains has two
# cover entities and "תפתח את הווילונות" names one of them, but without a
# device class the executor matches both and opens the lot.
#
# Deliberately not `lock` or `binary_sensor`, though the official Hebrew list
# maps דלת and שער for them too. A lock has one thing in it per area in every
# house this has seen, and narrowing a domain that needs no narrowing only adds
# a way to find nothing.
DEVICE_CLASS_DOMAINS: Final = frozenset(("cover",))

# The integration whose media_player entities music_assistant.play_media can
# target. Named once because it is both the registry test and the reason the
# tool degrades gracefully in a house that does not have it.
MUSIC_INTEGRATION: Final = "music_assistant"

# Slots whose value is a word out of a fixed list, which the sentence names and
# the model guesses at. See `slot_match.SETTING_WORDS`.
#
# v11 widened this from three entries to eight, and the reason is the byte-1
# rule in `ha_tools.assert_disjoint_first_byte`. Four of Home Assistant's own
# enums collide at their first byte - heat/heat_cool, silent/standard,
# album/artist, pink/purple - and those values cannot be renamed, because a
# service call carrying anything else is rejected. Two of the four were already
# covered here (colour at 335 right and 0 wrong, vacuum suction at 180/0); the
# other two, `hvac_mode` and `media_type`, were left to a model emitting a
# colliding prefix byte by byte. Now the sentence settles all four.
#
# A tuple, because a tool can carry more than one such slot: "תדליק אור חם
# בסלון" names a colour temperature and "תדליק אור אדום" names a colour.
SETTING_SLOT: Final[dict[str, tuple[str, ...]]] = {
    "vacuum_set_fan_speed": ("fan_speed",),
    "climate_set_fan_mode": ("fan_mode",),
    "climate_set_hvac_mode": ("hvac_mode",),
    "light_turn_on": ("color_name", "color_temp_k"),
    "music_play": ("media_type",),
    "media_select_source": ("source",),
    "list_add_item": ("list",),
    "list_complete_item": ("list",),
    "list_remove_item": ("list",),
}

#: Setting slots whose schema declares an integer. `slot_match.SETTING_WORDS`
#: is one type throughout - strings, so the tables read the same way - and
#: `tools.json` declares `color_temp_k` as ``integer`` with an enum of four.
#: Written straight through, the fill contradicts the tool's own contract:
#: the call carries ``"2700"`` where every gold call and the schema carry
#: 2700, which is a wrong argument on the wire and an exact-match failure on
#: every row that names a colour temperature. `executor._service_data` cast it
#: on the way to Home Assistant, so the household never saw it and nothing
#: pointed at it.
INTEGER_SETTING: Final = frozenset(("color_temp_k",))

# The audio tools a named title can be promoted out of. `music_play` is absent
# because it is the destination, and the query tools because a question about
# music is not a request to play it.
MEDIA_TOOLS: Final = frozenset((
    "media_play", "media_pause", "media_stop", "media_set_volume",
    "media_mute", "media_next_track", "media_previous_track",
    "media_select_source",
))

# Domains whose "name" argument identifies the entity itself rather than a
# device inside an area: scene.evening, script.good_night, timer.pasta.
NAME_ADDRESSED: Final = {
    "scene_activate": "scene",
    "script_run": "script",
    "automation_turn_on": "automation",
    "automation_turn_off": "automation",
    "button_press": "button",
    "input_boolean_turn_on": "input_boolean",
    "input_boolean_turn_off": "input_boolean",
    "timer_start": "timer",
    "timer_cancel": "timer",
    "timer_pause": "timer",
    "timer_resume": "timer",
    "timer_add": "timer",
    "timer_less": "timer",
    "timer_status": "timer",
}

# Home Assistant keeps a household's routines in three domains and offers no
# way to tell from a Hebrew sentence which one a given routine landed in:
# "אווירת ערב" is a scene in one house and a script in the next. Both models
# this project has trained confuse the two - v10 answered 21 of its routine
# failures with script_run where the gold was scene_activate, v9 answered 14
# with automation_turn_on - and the registry knows the answer.
#
# So a routine named in the sentence but absent from the domain the model
# chose is looked for in the sibling domain. Only then: a match in the
# chosen domain always wins, so this can turn a failure into an action and
# never an action into a different one.
#
# Automations are deliberately not in this table. `automation.turn_on`
# *enables* an automation rather than running it, which is not the same act
# as activating a scene, and guessing wrong there would leave a household
# with an automation quietly switched on.
ROUTINE_SIBLING: Final = {
    "scene_activate": "script_run",
    "script_run": "scene_activate",
}

#: Which numeric slot a behaviour's percentage lands in.
#:
#: Hebrew says all four the same way - "על חמישים אחוז", "לחצי", "למקסימום" -
#: and only the behaviour separates a brightness from a volume from how far a
#: blind is open. `slot_match.percent_from` does the reading and this table
#: says where it goes, exactly as `SETTING_SLOT` does for the enums.
#:
#: `fan_turn_on` carries `percentage` because `HassFanSetSpeed` does; the
#: fan's *named* speeds (שקט, טורבו) are a `SETTING_SLOT` entry and reach
#: `fan_speed` on the vacuum instead. The step slots are deliberately absent:
#: the corpus says those as idioms and never as a percentage, measured at
#: zero rows out of 779.
#: Which absolute slot has to be silent before a relative one may be filled
#: from the words. `direction.settle_steps` supplies a step the model left out,
#: and the guard against doing that on "תוריד את המזגן ל-25" is that the
#: sentence names no value for the thing being stepped - which
#: `slot_match.unsupported` answers, for the slot named here.
STEP_SOURCE: Final[dict[str, str]] = {
    "temperature_step": "temperature",
    "brightness_step_pct": "brightness_pct",
    "volume_step_pct": "volume_pct",
}

NUMBER_SLOT: Final[dict[str, str]] = {
    "light_turn_on": "brightness_pct",
    "media_set_volume": "volume_pct",
    "cover_set_position": "position",
    "valve_set_position": "position",
    "fan_turn_on": "percentage",
}

#: Which two-state slot a behaviour carries, for the ones whose value the
#: sentence always settles. The mirror of :data:`NUMBER_SLOT`: that table says
#: where a percentage lands, this one says where a yes-or-no does, and
#: `slot_match.switch_from` does the reading for both.
#:
#: Only these two. Every other boolean in the catalogue is either the
#: behaviour itself - an `on` is not a `shut` with a flag - or something the
#: sentence does not say out loud.
BOOLEAN_SLOT: Final[dict[str, str]] = {
    "fan_oscillate": "oscillating",
    "media_mute": "is_volume_muted",
}

#: The behaviours that carry a countdown, and so may take one from the
#: sentence. `timer_pause`, `timer_cancel`, `timer_resume` and `timer_status`
#: are absent on purpose - a duration in one of those sentences is the name of
#: the countdown being paused, not a new length. See `TOOL_ARGS`, which says
#: the same thing one table over and enforces it at the service call.
DURATION_TOOLS: Final[frozenset[str]] = frozenset((
    "timer_start", "timer_add", "timer_less"))

# How long a conversation's last room stays worth inheriting.
#
# Only ever consulted when the sentence named no room *and* the device that
# heard it belongs to no area - see `executor._remembered_area`. Thirty seconds
# is long enough for "תדליק את האור בסלון" then "וגם את המזגן", and short
# enough that a room named before somebody walked out of it has expired.
ROOM_MEMORY_SECONDS: Final = 30.0

# A helper toggle asked about a countdown is a timer command.
#
# The model answers "בטל את הטיימר" with `input_boolean_turn_off` and does it
# reliably: 0 of the 13 HassCancelTimer sentences in Home Assistant's Hebrew
# suite reached `timer.cancel`, and putting `timer_cancel` first in the
# shortlist did not move it - the prior is the model's, not the router's.
#
# The sentence settles it and the corpus says so without ambiguity: of the 473
# generated rows whose sentence names a countdown, the gold call is a timer
# tool or a state query on every single one, and an `input_boolean` on none.
# See `tool_router.TIMER_NOUNS` for the measurement.
#
# v11 should make this unreachable - timers and helper toggles are two
# different tools now, in two different router families, so the shortlist can
# simply decline to offer the wrong one. It stays because "should" is not
# "does", it costs nothing when it never fires, and a household running the
# previous weights against this component still needs it.
HELPER_IS_A_TIMER: Final[dict[str, str]] = {
    "input_boolean_turn_on": "timer_start",
    "input_boolean_turn_off": "timer_cancel",
}

# Tools where naming nothing means every one of them. "בטל את הטיימר" in a
# house with one timer running is unambiguous, and with three it plainly
# means all three. Nothing else belongs here: "תפעיל סצנה" with no scene
# named is a sentence that failed to say which, and activating every scene
# in the house is not a reading of it.
ALL_WHEN_UNNAMED: Final = {
    "timer_start", "timer_cancel", "timer_pause", "timer_resume",
    "timer_add", "timer_less", "timer_status",
}

# Spoken confirmations. Hebrew, because the user is speaking Hebrew.
SPEECH_OK: Final = "בוצע"

# `HassNevermind` - the speaker withdrew the request. Home Assistant's own
# Hebrew responses file answers this with the empty string, and so does this:
# a withdrawal that gets talked back at has not been honoured. The turn still
# succeeds, because doing nothing is what was asked for.
SPEECH_CANCELLED: Final = ""
SPEECH_NOTHING: Final = "לא הבנתי מה לעשות"
SPEECH_NO_TARGET: Final = "לא מצאתי מכשיר מתאים"

# Asked when a command names no room, the device that heard it is in no area,
# and nothing matched. The alternative is "לא מצאתי מכשיר מתאים", which is true
# and leaves the speaker with nothing to do about it.
SPEECH_WHICH_ROOM: Final = "באיזה חדר"
SPEECH_FAILED: Final = "הפעולה נכשלה"

# `HassGetCurrentDate` answers with a spoken date, and a Hebrew assistant
# saying "17 September 2013" is the same defect as the English weather slug
# below. Gregorian rather than Hebrew-calendar months: Home Assistant's own
# Hebrew response for this intent is "17 בספטמבר 2013".
HEBREW_MONTHS: Final = (
    "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
    "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
)

# Home Assistant's fifteen weather condition states, in Hebrew.
#
# `weather.<entity>.state` is an English slug, so speaking it back produced
# "מזג האוויר sunny, 30 מעלות" on the first live test - an English word in the
# middle of a Hebrew sentence, from an assistant whose entire purpose is
# Hebrew. The keys are read off `homeassistant.components.weather`'s
# ATTR_CONDITION_* constants rather than guessed, and the set is closed: Home
# Assistant validates a weather entity's state against exactly these.
# Anything unrecognised falls through to the raw slug, which is ugly but true.
WEATHER_STATES_HE: Final = {
    "clear-night": "שמיים בהירים",
    "cloudy": "מעונן",
    "exceptional": "מזג אוויר חריג",
    "fog": "ערפל",
    "hail": "ברד",
    "lightning": "סופת ברקים",
    "lightning-rainy": "סופת ברקים וגשם",
    "partlycloudy": "מעונן חלקית",
    "pouring": "גשם שוטף",
    "rainy": "גשום",
    "snowy": "מושלג",
    "snowy-rainy": "שלג וגשם",
    "sunny": "שמשי",
    "windy": "רוחות",
    "windy-variant": "רוחות משתנות",
}
