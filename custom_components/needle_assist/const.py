"""Constants and the tool -> Home Assistant service map."""

from __future__ import annotations

from pathlib import Path
from typing import Final

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
CONF_MAX_TOKENS: Final = "max_new_tokens"

# The shape of what the config entry stores. Version 2 removed two options, so
# an entry written by version 1 is migrated on load rather than left carrying
# keys nothing reads. See `__init__.async_migrate_entry`.
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

DEFAULT_MAX_TOKENS: Final = 192

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

# tool name -> (domain, service). Every entry is a real Home Assistant service.
SERVICE_MAP: Final[dict[str, tuple[str, str]]] = {
    "light_turn_on": ("light", "turn_on"),
    "light_turn_off": ("light", "turn_off"),
    "light_toggle": ("light", "toggle"),
    "climate_set_temperature": ("climate", "set_temperature"),
    "climate_set_hvac_mode": ("climate", "set_hvac_mode"),
    "climate_set_fan_mode": ("climate", "set_fan_mode"),
    "climate_turn_off": ("climate", "turn_off"),
    "fan_turn_on": ("fan", "turn_on"),
    "fan_turn_off": ("fan", "turn_off"),
    "fan_oscillate": ("fan", "oscillate"),
    "cover_open": ("cover", "open_cover"),
    "cover_close": ("cover", "close_cover"),
    "cover_stop": ("cover", "stop_cover"),
    "cover_set_position": ("cover", "set_cover_position"),
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
    "media_next_track": ("media_player", "media_next_track"),
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
    "scene_activate": ("scene", "turn_on"),
    "script_run": ("script", "turn_on"),
    "automation_turn_on": ("automation", "turn_on"),
    "automation_turn_off": ("automation", "turn_off"),
    "input_boolean_turn_on": ("input_boolean", "turn_on"),
    "input_boolean_turn_off": ("input_boolean", "turn_off"),
    "timer_start": ("timer", "start"),
    "timer_cancel": ("timer", "cancel"),
    "notify_send": ("notify", "send_message"),
}

# Read-only tools. Not services - these answer rather than actuate.
QUERY_TOOLS: Final = {"get_state", "get_weather"}

# The domain each tool targets, for entity matching.
TOOL_DOMAIN: Final[dict[str, str]] = {
    name: dom for name, (dom, _) in SERVICE_MAP.items()
} | {
    # The one tool whose service domain is not its target domain: the service
    # belongs to the music_assistant integration, the entity it acts on is an
    # ordinary media_player. Deriving this from SERVICE_MAP alone would send
    # entity matching looking for a "music_assistant" domain that has no
    # entities in it.
    "music_play": "media_player",
}

# Model argument -> service data key, where they differ.
ARG_RENAME: Final[dict[str, str]] = {
    "brightness_step_pct": "brightness_step_pct",
    "volume_pct": "volume_level",       # HA takes 0..1, converted in code
    "temperature_step": "temperature",  # resolved against current state
    "volume_step_pct": "volume_level",  # resolved against current state
}

# Arguments that are targeting metadata, never service data.
NON_SERVICE_ARGS: Final = {"area", "name"}

# Every argument each tool's schema declares, so a call carries nothing the
# service will reject.
#
# This is needed because the sentence is allowed to change which tool runs -
# `direction.settle` and the music upgrade both do it - and the model filled
# its arguments for the tool it originally chose. "תעביר את המזגן למהירות
# גבוה" came back as `climate_set_temperature{temperature: 23}`, the verb
# corrected it to `climate_set_fan_mode`, and the stale `temperature` would
# have made Home Assistant reject the whole call: `climate.set_fan_mode` does
# not take one. The household would have been told the command failed, by the
# guard that was there to make it work.
#
# Kept here rather than read out of tools.json because that file is
# deliberately not read on the event loop (see `needle_runner`), and a test
# asserts the two agree exactly.
TOOL_ARGS: Final[dict[str, frozenset[str]]] = {
    "automation_turn_off": frozenset(("name",)),
    "automation_turn_on": frozenset(("name",)),
    "camera_turn_off": frozenset(("area",)),
    "camera_turn_on": frozenset(("area",)),
    "climate_set_fan_mode": frozenset(("area", "fan_mode",)),
    "climate_set_hvac_mode": frozenset(("area", "hvac_mode",)),
    "climate_set_temperature": frozenset((
        "area", "temperature", "temperature_step"
    )),
    "climate_turn_off": frozenset(("area",)),
    "cover_close": frozenset(("area",)),
    "cover_open": frozenset(("area",)),
    "cover_set_position": frozenset(("area", "position",)),
    "cover_stop": frozenset(("area",)),
    "fan_oscillate": frozenset(("area", "oscillating",)),
    "fan_turn_off": frozenset(("area",)),
    "fan_turn_on": frozenset(("area", "percentage",)),
    "get_state": frozenset(("area", "domain",)),
    "get_weather": frozenset(("day_offset",)),
    "input_boolean_turn_off": frozenset(("name",)),
    "input_boolean_turn_on": frozenset(("name",)),
    "light_toggle": frozenset(("area",)),
    "light_turn_off": frozenset(("area",)),
    "light_turn_on": frozenset((
        "area", "brightness_pct", "brightness_step_pct", "color_name"
    )),
    "lock_lock": frozenset(("area",)),
    "lock_unlock": frozenset(("area",)),
    "media_mute": frozenset(("area", "is_volume_muted",)),
    "media_next_track": frozenset(("area",)),
    "media_pause": frozenset(("area",)),
    "media_play": frozenset(("area",)),
    "media_select_source": frozenset(("area", "source",)),
    "media_set_volume": frozenset(("area", "volume_pct", "volume_step_pct",)),
    "music_play": frozenset(("area", "media_type",)),
    "notify_send": frozenset(("message",)),
    "scene_activate": frozenset(("name",)),
    "script_run": frozenset(("name",)),
    "switch_turn_off": frozenset(("area", "name",)),
    "switch_turn_on": frozenset(("area", "name",)),
    "timer_cancel": frozenset(("name",)),
    "timer_start": frozenset(("minutes", "name",)),
    "vacuum_pause": frozenset(),
    "vacuum_return_to_base": frozenset(),
    "vacuum_set_fan_speed": frozenset(("fan_speed",)),
    "vacuum_start": frozenset(("area",)),
}

# The integration whose media_player entities music_assistant.play_media can
# target. Named once because it is both the registry test and the reason the
# tool degrades gracefully in a house that does not have it.
MUSIC_INTEGRATION: Final = "music_assistant"

# Tools whose one argument is a word out of a fixed list, which the sentence
# names and the model guesses at. See `slot_match.SETTING_WORDS`.
SETTING_SLOT: Final = {
    "vacuum_set_fan_speed": "fan_speed",
    "climate_set_fan_mode": "fan_mode",
    "light_turn_on": "color_name",
}

# The audio tools a named title can be promoted out of. `music_play` is absent
# because it is the destination, and the query tools because a question about
# music is not a request to play it.
MEDIA_TOOLS: Final = frozenset((
    "media_play", "media_pause", "media_set_volume", "media_mute",
    "media_next_track", "media_select_source",
))

# Domains whose "name" argument identifies the entity itself rather than a
# device inside an area: scene.evening, script.good_night, timer.pasta.
NAME_ADDRESSED: Final = {
    "scene_activate": "scene",
    "script_run": "script",
    "automation_turn_on": "automation",
    "automation_turn_off": "automation",
    "input_boolean_turn_on": "input_boolean",
    "input_boolean_turn_off": "input_boolean",
    "timer_start": "timer",
    "timer_cancel": "timer",
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

# Tools where naming nothing means every one of them. "בטל את הטיימר" in a
# house with one timer running is unambiguous, and with three it plainly
# means all three. Nothing else belongs here: "תפעיל סצנה" with no scene
# named is a sentence that failed to say which, and activating every scene
# in the house is not a reading of it.
ALL_WHEN_UNNAMED: Final = {"timer_start", "timer_cancel"}

# Spoken confirmations. Hebrew, because the user is speaking Hebrew.
SPEECH_OK: Final = "בוצע"
SPEECH_NOTHING: Final = "לא הבנתי מה לעשות"
SPEECH_NO_TARGET: Final = "לא מצאתי מכשיר מתאים"
SPEECH_FAILED: Final = "הפעולה נכשלה"

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
