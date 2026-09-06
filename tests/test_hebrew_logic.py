"""The deterministic half of the integration, tested without Home Assistant.

Everything here runs before or after the model, and every case is one that was
wrong at some point. The houses are invented; nothing in this file describes a
real installation.

Kept in step with the workshop tree's `eval/test_integration_logic.py`, which
is where these are written. The tests that live only there are the ones whose
subject does not ship: the corpus generator, the tool-catalogue source and the
Hebrew lexicon.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest  # noqa: E402
from component_loader import COMPONENT, load  # noqa: E402

CONST = load("const")
AREA_MAP = load("area_map")
SLOT = load("slot_match")
HEB = load("hebrew_text")
ROUTER = load("tool_router")
CLAUSE = load("clause_split")
REPLY = load("reply")
DIRECTION = load("direction")
NUM = load("hebrew_numbers")

# `executor` is the one module here that imports Home Assistant, and one CI
# job runs this file with nothing installed at all - that job exists precisely
# to prove the Hebrew logic can be checked without a running HA. A plain
# module-level load therefore fails at *collection*, before any test-level
# skip can run, and takes all 330 tests with it. So the import is allowed to
# fail and the handful of tests that reach into the executor skip instead.
try:
    EXEC = load("executor")
except ModuleNotFoundError:  # pragma: no cover - only in the bare CI job
    EXEC = None

needs_executor = pytest.mark.skipif(
    EXEC is None, reason="reaches into executor, which imports homeassistant")


# --- tool / service coverage ------------------------------------------------

def test_every_service_in_the_map_exists():
    """A behaviour mapped to a service Home Assistant does not have fails at
    the last step, after the sentence was read correctly and the entities were
    found - so the household is told the command failed by the map that was
    supposed to run it.

    Checked against the components' own `services.yaml`, which is where the
    names came from: `valve` spells open as `open_valve`, `todo` completes an
    item with `update_item`, and there is no `timer.resume` at all.
    """
    # Home Assistant is not installed in the bare CI job, and this test
    # reaches it. Skipping beats failing: the job exists to prove the
    # *Hebrew* logic needs nothing installed, not to check this.
    pytest.importorskip("homeassistant")
    import homeassistant.components as components
    import yaml

    root = pathlib.Path(components.__file__).parent
    for virtual, (domain, service) in CONST.SERVICE_MAP.items():
        path = root / domain / "services.yaml"
        if not path.exists():
            # A service from an integration that is not part of core - only
            # `music_assistant` and `notify` reach here.
            assert domain in ("music_assistant", "notify"), (virtual, domain)
            continue
        declared = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert service in declared, (virtual, domain, service)


# --- area resolution --------------------------------------------------------

def test_alias_lookup_is_punctuation_insensitive():
    assert AREA_MAP.slug_for_name('ממ"ד') == "safe_room"
    assert AREA_MAP.slug_for_name("ממד") == "safe_room"
    assert AREA_MAP.slug_for_name("Living Room") == "living_room"
    assert AREA_MAP.slug_for_name("living_room") == "living_room"


# --- spoken Hebrew ----------------------------------------------------------

# --- dataset contract -------------------------------------------------------

def test_router_never_exceeds_the_retrieval_threshold():
    """More than five declared tools hands tool choice back to the retrieval head."""
    queries = [
        "תדליק את האור בסלון",
        "תכבה את המזגן ותנעל את הדלת ותוריד את התריס בחדר שינה",
        "מה השעה",
        "",
        "אהלן מה נשמע איך הולך היום שלך בכלל",
    ]
    for q in queries:
        assert len(ROUTER.select_tool_names(q)) <= ROUTER.MAX_TOOLS, q


def test_router_always_offers_something():
    """An empty shortlist would leave the model nothing to refuse against."""
    for q in ["", "בלה בלה", "תספר לי בדיחה", "xyzzy"]:
        assert ROUTER.select_tool_names(q), repr(q)


@pytest.mark.parametrize("query,expected", [
    ("תדליק את האור בסלון", "light_turn_on"),
    ("תכבה את האור במטבח", "light_turn_off"),
    ("סגור את המזגן בחדר שינה", "climate_turn_off"),
    ("תוריד את התריס בסלון", "cover_close"),
    ("תנעל את הדלת", "lock_lock"),
    ("תגביר את הקול בסלון", "media_set_volume"),
    ("מה מזג האוויר מחר", "get_weather"),
    ("שיחזור לבסיס", "vacuum_return_to_base"),
    ("תדליק את הדוד במקלחת", "switch_turn_on"),
])
def test_router_reaches_the_right_tool(query, expected):
    assert expected in ROUTER.select_virtual_names(query), query


def test_router_keeps_both_domains_of_a_two_domain_command():
    names = ROUTER.select_virtual_names("תכבה את האור ותנעל את הדלת")
    assert "light_turn_off" in names
    assert "lock_lock" in names


def test_router_does_not_confuse_guests_with_light():
    """"אור" is a substring of "אורחים", and חדר האורחים is the living room."""
    names = ROUTER.select_virtual_names("תנעל את הדלת בחדר האורחים")
    assert "lock_lock" in names


# --- deterministic off-topic gate ------------------------------------------
#
# The model refuses 0.0% of off-topic utterances and actuates on ~100% of them,
# so refusal is enforced before inference instead. See
# tool_router.looks_off_topic for the threshold and the measurement behind it.

@pytest.mark.parametrize("query", [
    "מה השעה בטוקיו",
    "תספר לי בדיחה",
    "מי ניצח במשחק אתמול",
    "כמה זה שלוש כפול ארבע",
])
def test_off_topic_is_refused(query):
    assert ROUTER.looks_off_topic(query)


@pytest.mark.parametrize("query", [
    "תדליק את האור בסלון",
    "תסגור את התריס במטבח",
    "תעצור את המוזיקה",
    "שים את המזגן על עשרים ושתיים",
    "תנעל את הדלת",
    "תפעיל את השואב",
])
def test_real_commands_pass_the_gate(query):
    assert not ROUTER.looks_off_topic(query)


def test_gate_threshold_is_not_a_knife_edge():
    """A single device noun scores 3, which is why the gate sits at 3 and not 4.

    Measured on the held-out set, 67.2% of genuine commands score exactly 3, so
    a threshold of 4 would refuse most of them.
    """
    assert ROUTER.score_families("תדליק את האור")[0][1] >= ROUTER.REFUSE_BELOW


# --- the vendored engine ----------------------------------------------------
#
# needle_engine/ is a byte-identical copy of the inference half of
# cactus-needle. The integration ships it instead of declaring the package as a
# requirement, because the package drags in JAX - 314MB that inference never
# touches, and that pip cannot install at all on a musl-based Home Assistant.
# needle_engine/VENDOR.md has the full argument.
#
# Vendored code rots silently. These tests are what makes it noisy instead.

VENDORED = COMPONENT / "needle_engine"
_VENDORED_FILES = ["__init__.py", "agent/__init__.py", "agent/tools.py",
                   "agent/fetch.py"]


def test_vendored_engine_needs_only_the_standard_library():
    """The point of vendoring: no third-party import anywhere in the tree.

    A new upstream release that reaches for `requests` or `numpy` at import
    time would silently reintroduce the dependency this directory exists to
    remove, and it would only surface as a crash on a user's Home Assistant.
    """
    import ast
    import sys

    allowed = set(sys.stdlib_module_names)
    for name in _VENDORED_FILES:
        tree = ast.parse((VENDORED / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, i.e. within the vendored tree.
                roots = [] if node.level else [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                # huggingface_hub is imported inside fetch_library, which
                # engine_lib bypasses via NEEDLE_LIB_PATH. Nothing else may.
                if root == "huggingface_hub" and name == "agent/fetch.py":
                    continue
                assert root in allowed, f"{name} imports {root!r}"


def test_manifest_declares_no_requirements():
    """The vendoring is pointless if the manifest still pulls the package in."""
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["requirements"] == []


def test_engine_cache_lives_under_the_config_directory():
    """`~/.cache` does not survive a Home Assistant core update; /config does."""
    # Home Assistant is not installed in the bare CI job, and this test
    # reaches it. Skipping beats failing: the job exists to prove the
    # *Hebrew* logic needs nothing installed, not to check this.
    pytest.importorskip("homeassistant")
    import importlib.util
    import types

    # engine_lib does `from .needle_engine.agent import fetch`, so it needs a
    # package to be relative *to*. Importing the real one would execute the
    # component's __init__.py, which imports Home Assistant. A stand-in package
    # pointed at the same directory gives the relative import somewhere to
    # resolve without dragging the framework in.
    pkg = types.ModuleType("na_pkg")
    pkg.__path__ = [str(COMPONENT)]
    sys.modules["na_pkg"] = pkg
    spec = importlib.util.spec_from_file_location(
        "na_pkg.engine_lib", COMPONENT / "engine_lib.py")
    engine_lib = importlib.util.module_from_spec(spec)
    sys.modules["na_pkg.engine_lib"] = engine_lib
    try:
        spec.loader.exec_module(engine_lib)
        path = engine_lib.library_path("/config")
    finally:
        sys.modules.pop("na_pkg.engine_lib", None)
        sys.modules.pop("na_pkg", None)
    # Compared as path objects, not strings: this suite runs on Windows, where
    # str(Path("/config/x")) is "\\config\\x".
    assert pathlib.Path("/config") in path.parents, path
    assert path.name.startswith("libneedle."), path


def test_component_never_imports_the_real_package():
    """No module may `import needle`. It only fails on the device.

    Home Assistant has no `cactus-needle` on its path, but a developer machine
    does — so this mistake runs fine locally and raises ModuleNotFoundError on
    the first Hebrew utterance a household speaks. It happened once, in
    needle_compat.safe_complete, and was caught by running the component inside
    the real Home Assistant container rather than by any test. This is that
    test.
    """
    import ast

    offenders = []
    for path in sorted(COMPONENT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "needle" or name.startswith("needle."):
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, (
        "use `from . import needle_engine` instead:\n  " + "\n  ".join(offenders))


# --- two rooms, one slug -----------------------------------------------------
#
# The model has twelve area slugs. A real home does not partition into the same
# twelve. On the first installation this was tested against, מקלחת and שירותים
# are separate rooms and both fold onto `washroom`, so "תדליק את האור
# בשירותים" lit the shower room - whichever area the registry happened to list
# first. Slug equality cannot separate them; the sentence can.

class _FakeArea:
    def __init__(self, id, name, aliases=()):
        self.id, self.name, self.aliases = id, name, list(aliases)


def _with_registry(areas):
    """Point area_map at a stand-in registry and hand back the module."""
    import importlib.util
    import types

    ha = types.ModuleType("homeassistant")
    helpers = types.ModuleType("homeassistant.helpers")
    ar = types.ModuleType("homeassistant.helpers.area_registry")
    ar.async_get = lambda hass: types.SimpleNamespace(
        async_list_areas=lambda: areas)
    helpers.area_registry = ar
    ha.helpers = helpers
    for name, mod in (("homeassistant", ha), ("homeassistant.helpers", helpers),
                      ("homeassistant.helpers.area_registry", ar)):
        sys.modules.setdefault(name, mod)

    spec = importlib.util.spec_from_file_location(
        "na_area_live", COMPONENT / "area_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The real registry of the installation where this was found, in its real order.
_REAL = [
    _FakeArea("hvl", "הול"),
    _FakeArea("khdr_hvrym", "חדר הורים", ["parents room", "חדר שינה"]),
    _FakeArea("khdr_yldym", "חדר ילדים"),
    _FakeArea("kl_hbyt", "כל הבית"),
    _FakeArea("mtbkh_2", "מטבח"),
    _FakeArea("mtbkh", "מסדרון"),      # id says kitchen, room is a corridor
    _FakeArea("khdr_shynh", "מקלחת"),  # id says bedroom, room is a shower
    _FakeArea("slvn", "סלון"),
    _FakeArea("shyrvtym", "שירותים"),
]


def _index(areas):
    """The production area index, built over a simulated installation."""
    return SLOT.build_area_index((a.id, a.name, a.aliases) for a in areas)


def _room(areas, utterance):
    index = _index(areas)
    if SLOT.mentions_whole_home(utterance):
        return "all"
    found = index.find_all(utterance)
    return max(found, key=lambda m: m.rank).value if found else None


def test_two_rooms_sharing_one_slug_are_separable():
    """The bug a live installation found.

    מקלחת and שירותים are two rooms in this house and both fold onto the single
    ``washroom`` slug, so the model cannot tell them apart at all - it has one
    token for both. The sentence can, because the speaker said which one.
    """
    assert _room(_REAL, "תדליק את האור בשירותים") == "shyrvtym"
    assert _room(_REAL, "תדליק את האור במקלחת") == "khdr_shynh"


def test_a_room_outside_the_lexicon_is_reachable():
    """הול is in no table here and in no training example. It still resolves.

    This is the whole point of matching against the installation rather than
    against twelve baked-in slugs: the model has no token that can express this
    room, so under the old design it was unreachable by voice in this house.
    """
    assert _room(_REAL, "תדליק את האור בהול") == "hvl"


def test_transliterated_area_ids_never_win():
    """An area_id is not a name.

    In this registry `mtbkh` is a corridor and `khdr_shynh` is a shower - the
    ids were minted from names the rooms no longer have. Matching on them would
    turn "the light in the kitchen" into the corridor light.
    """
    assert _room(_REAL, "תדליק את האור במטבח") == "mtbkh_2"       # not מסדרון
    assert _room(_REAL, "תדליק את האור בחדר שינה") == "khdr_hvrym"  # not מקלחת


def test_home_assistant_alias_reaches_a_differently_named_room():
    """The documented escape hatch has to actually work.

    Nothing in our tables maps חדר הורים to a bedroom; the household added
    "חדר שינה" as a Home Assistant alias, and that is what makes it resolve.
    """
    assert _room(_REAL, "כבה את האור בחדר השינה") == "khdr_hvrym"


def test_an_area_named_for_the_whole_house_does_not_swallow_rooms():
    """This installation really does have an area called כל הבית.

    It must not capture "בכל הבית", which means every room, and it must not
    capture sentences that name a different room either.
    """
    assert _room(_REAL, "תכבה את האורות בכל הבית") == "all"
    assert _room(_REAL, "תכבה את האור בסלון") == "slvn"


def test_no_room_named_resolves_to_nothing():
    """Silence is not a room.

    A live engine answers "תכבה את האור" with ``area="terrace"``. Resolving from
    the sentence returns nothing, which is what lets the executor fall back to
    the room the speaker is standing in.
    """
    assert _room(_REAL, "תכבה את האור") is None
    assert _room(_REAL, "תדליק את המזגן") is None


def test_device_nouns_do_not_look_like_rooms():
    """The failure mode a substring search would have.

    ``גן`` (garden) is inside ``מזגן`` (air conditioner) and ``אור`` (light) is
    inside ``מאוורר`` (fan). Either one matching would send a command to a room
    nobody mentioned.
    """
    house = [_FakeArea("garden", "גינה"), _FakeArea("living", "סלון")]
    assert _room(house, "תכבה את המזגן") is None
    assert _room(house, "תדליק את המאוורר") is None
    assert _room(house, "תפתח את הווילון") is None
    assert _room(house, "מה המצב של החלון") is None


def test_hebrew_prefixes_attach_to_room_names():
    """A room's name never appears bare - it arrives fused to a preposition."""
    house = [_FakeArea("kitchen", "מטבח")]
    for said in ("תדליק במטבח", "ובמטבח תדליק", "שבמטבח", "מהמטבח",
                 "כשבמטבח יש אור"):
        assert _room(house, said) == "kitchen", said


def test_the_houses_own_name_beats_our_synonym_table():
    """A house with both a סלון and a חדר אורחים means two different rooms.

    Our lexicon lists חדר אורחים as a synonym for the living room. The house
    disagrees, and the house is right.
    """
    house = [_FakeArea("living", "סלון"), _FakeArea("guest", "חדר אורחים")]
    assert _room(house, "תדליק בחדר אורחים") == "guest"
    assert _room(house, "תדליק בסלון") == "living"


def test_a_synonym_two_rooms_claim_resolves_to_neither():
    """Ambiguity has to fail, not guess.

    Both מקלחת and שירותים pull in אמבטיה through the synonym table and neither
    owns it. Picking whichever came first in the registry is exactly how the
    live installation lit the wrong room.
    """
    assert _room(_REAL, "תדליק באמבטיה") is None


def test_a_room_this_house_lacks_is_still_recognised_as_a_room():
    """Knowing a room was named is a separate question from knowing which.

    Without this the executor cannot tell "turn on the light in the garage" in a
    house with no garage from "turn on the light", and would widen a one-room
    command to the whole building.
    """
    assert SLOT.names_a_room("תדליק את האור במוסך")
    assert not SLOT.names_a_room("תדליק את האור")
    assert _room(_REAL, "תדליק את האור במוסך") is None


def test_speech_to_text_damage_still_resolves():
    """Real transcripts lose spaces, gain them, and misspell."""
    house = [_FakeArea("parking", "חניה"), _FakeArea("hallway", "מסדרון"),
             _FakeArea("office", "משרד"), _FakeArea("kitchen", "מטבח")]
    assert _room(house, "אפשר להתחיל את שואב האבקבגראז'") == "parking"
    assert _room(house, "סגור את השאטרס בפרוז דור") == "hallway"
    assert _room(house, "תדליק את האור במשרדד") == "office"
    assert _room(house, "תדליק את האור במתבח") == "kitchen"


def test_whole_home_never_comes_from_a_typo():
    """One edit must not turn one room into every room.

    "בכניסה לבית" (at the entrance) is one edit from "בבית" (in the house), and
    the fuzzy pass is switched off for whole-home markers because of it.
    """
    assert not SLOT.mentions_whole_home("תדליק את האור בכניסה לבית")
    assert SLOT.mentions_whole_home("תכבה את האורות בכל הבית")


def test_two_rooms_two_calls_pair_off_in_order():
    index = _index(_REAL)
    said = "תדליק את האור בסלון ותכבה את האור במטבח"
    assert [m.value for m in index.find_occurrences(said)] == ["slvn", "mtbkh_2"]


def test_negation_blocks_actuation_but_correction_does_not():
    """Bare לא is how Israelis correct themselves mid-sentence.

    327 rows of the corpus use it that way and every one of them must still
    actuate. Only לא and אל *governing a verb* are negations.
    """
    assert ROUTER.looks_negated("אל תדליק את האור בסלון")
    assert ROUTER.looks_negated("בלי להדליק את האור")
    assert ROUTER.looks_negated("לא צריך להדליק")
    assert not ROUTER.looks_negated("תדליק את המנורה, לא לא, תכבה את המנורה")
    assert not ROUTER.looks_negated("לא, בעצם תכבה לי את המנורה בחוץ")


def test_notification_text_comes_out_of_the_sentence():
    """The model cannot spell Hebrew into an argument; the sentence already has it."""
    assert SLOT.extract_message("תודיע בבית שהאוכל מוכן") == "האוכל מוכן"
    assert SLOT.extract_message(
        "תשלח הודעה לכולם שהכביסה מוכנה בבקשה") == "הכביסה מוכנה"
    assert SLOT.extract_message("תעדכן את כולם שתרדו למטה") == "תרדו למטה"
    assert SLOT.extract_message("תדליק את האור בסלון") is None


def test_a_generic_device_name_cannot_capture_a_room_command():
    """An entity called "אור" must not swallow "turn on the light".

    Naming a device is more specific than naming its room, so a name match
    overrides the room - which makes a device named after its device class
    dangerous: every room command would collapse onto one bulb.
    """
    blocked = SLOT._blocked_forms()
    for generic in ("אור", "מזגן", "תריס", "דלת", "מוזיקה", "סלון", "מטבח"):
        assert not SLOT._addressable(generic, blocked), generic
    for specific in ("מנורת קריאה", "לד מתחת לארון", "Reading Lamp"):
        assert SLOT._addressable(specific, blocked), specific


def test_a_name_that_is_not_a_string_is_ignored():
    """Home Assistant does not promise a string where a name goes.

    ``entry.original_name`` is a ``ComputedNameType`` sentinel for entities that
    derive their name. It is truthy and it is not a str, and passing it to
    ``unicodedata.normalize`` took down a live installation's first state query
    with a TypeError.
    """
    class Sentinel:
        def __bool__(self):
            return True

    blocked = SLOT._blocked_forms()
    assert not SLOT._addressable(Sentinel(), blocked)
    assert not SLOT._addressable(None, blocked)
    assert SLOT._addressable("מנורת קריאה", blocked)


def test_lost_spaces_still_score_the_right_family():
    """Speech-to-text drops spaces, and a lost space used to lose the domain.

    "תסגור אתהתאורה" is one token, matched no device noun at all, and the verb
    sent it to cover - so a light command closed the blinds.
    """
    assert ROUTER.select_virtual_names(
        "תקשיבי, תסגור אתהתאורה במוסך")[0].startswith("light_")
    assert "climate_turn_off" in ROUTER.select_virtual_names(
        "בא לי שתכבי אתהמזגן במרפסת")
    assert "timer_cancel" in ROUTER.select_virtual_names("אפשר שתעצור את הספירה")


def test_weather_states_cover_home_assistants_full_set():
    """Every condition Home Assistant can report must have Hebrew.

    The list is closed - Home Assistant validates a weather entity's state
    against exactly these fifteen - so a gap here is a guaranteed English word
    in a Hebrew sentence, not a hypothetical one. The first live test said
    "מזג האוויר sunny".
    """
    ha_conditions = {
        "clear-night", "cloudy", "exceptional", "fog", "hail", "lightning",
        "lightning-rainy", "partlycloudy", "pouring", "rainy", "snowy",
        "snowy-rainy", "sunny", "windy", "windy-variant",
    }
    assert set(CONST.WEATHER_STATES_HE) == ha_conditions
    for state, hebrew in CONST.WEATHER_STATES_HE.items():
        assert any("\u0590" <= c <= "\u05ff" for c in hebrew), state


def test_a_question_is_never_given_a_tool_that_can_actuate():
    """The live installation turned three kitchen lights on when asked their state.

    Only 5.0% of the generated rows are state questions, so with four actuation
    tools in a five-tool shortlist the model answers a question by acting. The
    shortlist is where that gets fixed: a tool the router does not declare
    cannot be emitted, because the grammar is built from the declared set.
    """
    for question in ("מה המצב של האור במטבח",
                     "מה קורה עם המזגן בסלון",
                     "האם החלון בממד סגור",
                     "כמה מעלות בחדר השינה",
                     "תבדוק את התאורה בממד",
                     "תגיד לי מה קורה עם הווילונות בסלון",
                     # The abbreviation an Israeli actually says, and the one
                     # Home Assistant's own suite uses. Without it "מה טמפ"
                     # was read as an order and answered by *setting* a
                     # temperature.
                     "מה טמפ",
                     "מה טמפ בסלון"):
        assert ROUTER.looks_like_question(question), question
        assert not [t for t in ROUTER.select_tool_names(question)
                    if t not in ("get_state", "get_weather")], question


def test_an_order_is_never_mistaken_for_a_question():
    """The expensive direction is the other one, so this is the test that matters.

    Measured over all 18,071 actuation rows, the question patterns fire on
    zero. These are the near misses: a state adjective inside an order, and the
    polite forms that read like questions in English.
    """
    orders = ("תשאיר את האור דולק",
              "אפשר קצת אור בסלון",
              "תוכל לפתוח את התריס",
              "אני רוצה שיהיה חם בחדר",
              "תדליק את האור במטבח",
              # The same abbreviation in an order. Only "מה" in front of it
              # asks anything, which is why the pattern requires it.
              "תעלה את הטמפ ל22")
    for order in orders:
        assert not ROUTER.looks_like_question(order), order

    # ...and something that can act on them survives into the shortlist.
    #
    # "אני רוצה שיהיה חם בחדר" is left out on purpose: it names no device, and
    # חם is a query keyword, so the scorer has offered it read-only tools since
    # long before the gate existed. That is the safe direction - a missed
    # command, not a moved device - and it belongs to the keyword table rather
    # than to this change.
    for order in orders[:3] + orders[4:]:
        assert [t for t in ROUTER.select_tool_names(order)
                if t not in ("get_state", "get_weather")], order


def test_a_device_question_is_not_answered_with_the_weather():
    """Both read-only tools declared, and the model picks the wrong one.

    "מה המצב של האור בשירותים" came back as the weather from the live install.
    The distinction is lexical, so the router makes it and declares one tool.
    A weather question wins the tie: weather words appear in 2.0% of device
    questions, but a weather sentence names a device noun far more often than
    that ("יהיה חם מחר" scores climate), and no weather row may be sent to a
    device lookup.
    """
    assert ROUTER.select_tool_names("מה המצב של האור בשירותים") == ["get_state"]
    assert ROUTER.select_tool_names("מה קורה עם החלון במטבח") == ["get_state"]
    assert ROUTER.select_tool_names("מה מזג האוויר") == ["get_weather"]
    assert ROUTER.select_tool_names("תגיד לי אם ירד גשם מחר") == ["get_weather"]
    # Nothing named at all: the model still gets to choose between the two.
    assert set(ROUTER.select_tool_names("מה המצב")) == {"get_state", "get_weather"}


def test_a_state_question_is_typed_from_the_sentence():
    """The domain slot, resolved the same way the area and the message are.

    Measured at 98.0% against the gold domain of every generated get_state row,
    where the model scores 0.9% argument F1. The two Home Assistant domains
    that own no service - a humidity sensor and a window contact - have no
    family in the router, so they carry their own nouns.
    """
    assert ROUTER.query_domain("מה המצב של האור במטבח") == "light"
    assert ROUTER.query_domain("מה קורה עם המזגן בסלון") == "climate"
    assert ROUTER.query_domain("מה רמת הלחות במרפסת") == "sensor"
    assert ROUTER.query_domain("מה המצב של החיישן בלובי") == "sensor"
    # חלון is a window contact; וילון and תריס are a blind. One letter apart.
    assert ROUTER.query_domain("האם החלון בממד סגור") == "binary_sensor"
    assert ROUTER.query_domain("מה המצב של הווילונות בסלון") == "cover"
    # Nothing recognisable leaves the model's own guess in place.
    assert ROUTER.query_domain("מה המצב") is None


# --- several orders in one sentence -----------------------------------------

@pytest.mark.parametrize("sentence,clauses", [
    ("תדליק את האור בסלון וסגור את התריסים בחדר שינה",
     ["תדליק את האור בסלון", "סגור את התריסים בחדר שינה"]),
    ("כבה את האור במטבח וגם תנעל את הדלת",
     ["כבה את האור במטבח", "תנעל את הדלת"]),
    ("סגור את התריסים ואז תכבה את המזגן",
     ["סגור את התריסים", "תכבה את המזגן"]),
    ("שים את המזגן על 23 ותדליק את האור",
     ["שים את המזגן על 23", "תדליק את האור"]),
    ("תפעיל את הרובוט, נעל את המנעול בחדר המחשב, תסגור לי את התריסים בחניון",
     ["תפעיל את הרובוט", "נעל את המנעול בחדר המחשב",
      "תסגור לי את התריסים בחניון"]),
])
def test_a_sentence_with_two_orders_becomes_two_clauses(sentence, clauses):
    """The measurement behind this module.

    On the 97 multi-call rows of the held-out set the model answers the whole
    sentence with a single call 94 times and scores 0.0% tool-set accuracy.
    Routing one clause at a time, on the same weights, scores 75.3%.
    """
    assert CLAUSE.split_clauses(sentence) == clauses


@pytest.mark.parametrize("sentence", [
    # Two rooms, one order. "ובמטבח" is a place, not a verb.
    "תדליק את האור בסלון ובמטבח",
    # Two devices, one order.
    "תדליק את האור והמזגן בסלון",
    # A filler before a comma is not an order.
    "אה, סגור את האור בסלון",
    # Neither is a polite tail.
    "תכבה את האור בסלון, תודה",
    # A question is answered as one thing even when it names two rooms.
    "מה המצב של האור בסלון ובמטבח",
    "תדליק את האור בסלון",
])
def test_coordination_that_is_not_a_second_order_is_left_alone(sentence):
    """The cut only survives if both sides carry an action verb.

    Every sentence here contains a joining word and none of them contains two
    orders. Splitting any of them would turn one command into two, and the
    second would have no verb to act on.
    """
    assert CLAUSE.split_clauses(sentence) == [sentence]


def test_a_clause_is_still_a_command_the_router_can_read():
    """Each piece has to survive on its own.

    A clause that lost its verb or its room to the cut would route to nothing,
    which is worse than not splitting at all.
    """
    sentence = "תדליק את האור בסלון וסגור את התריסים בחדר שינה וגם תנעל את הדלת"
    clauses = CLAUSE.split_clauses(sentence)
    assert len(clauses) == 3
    assert [ROUTER.select_virtual_names(c)[0] for c in clauses] == [
        "light_turn_on", "cover_close", "lock_lock"]


def test_the_number_of_clauses_is_capped():
    """Every clause is a forward pass, about 2.5 s on the target hardware.

    Five of them is past the point where a voice pipeline stops waiting, so
    the cap is a real limit rather than a defensive constant.
    """
    sentence = " וגם ".join(["תדליק את האור בסלון"] * 8)
    assert len(CLAUSE.split_clauses(sentence)) == CLAUSE.MAX_CLAUSES


def test_the_splitter_reads_the_router_verb_tables():
    """One list of verbs, not two.

    A verb added to the router for routing has to become a cut point too,
    otherwise a sentence that routes correctly silently stops being splittable.
    """
    for family, verbs in ROUTER.FAMILY_VERBS.items():
        if family == "query":
            continue
        for verb in verbs:
            assert ROUTER._fold(verb) in CLAUSE.ACTION_VERBS, verb
    # And the interrogatives must not be cut points.
    assert "מה" not in CLAUSE.ACTION_VERBS
    assert "האם" not in CLAUSE.ACTION_VERBS


# --- one order, several rooms -----------------------------------------------

def _areas(areas, utterance, index=0, total=1, floors=None):
    """`SlotIndex.areas_for_call` over a stand-in registry.

    ``floors`` is the household's floor index, which `areas_for_call` consults
    to drop a room whose name sits inside a floor's - "כניסה" inside "קומת
    הכניסה". A house with no floors passes an empty one, which is what most of
    these tests want.
    """
    import types
    index_obj = _index(areas)
    floor_obj = floors if floors is not None else SLOT.PhraseIndex()
    fake = types.SimpleNamespace(_area_index=lambda: index_obj,
                                 _floor_index=lambda: floor_obj)
    return SLOT.SlotIndex.areas_for_call(fake, utterance, index, total)


def test_a_room_inside_a_floor_phrase_is_the_floor():
    """"בקומת הכניסה" is the entrance *floor*, not the hallway.

    `כניסה` is one of this project's words for the hallway, so the room index
    claimed the phrase and `executor._target_area` never got as far as asking
    about floors - "תכבה את האורות בקומת הכניסה" lit one corridor. The floor
    phrase contains the room phrase outright, which is the same span rule that
    keeps a switch called "המחשב" out of "חדר המחשב". 16 rows over the corpus,
    every one a floor read as a room and none the other way.
    """
    floors = SLOT.PhraseIndex()
    for slug, phrases in SLOT.FLOOR_PHRASES.items():
        for phrase in phrases:
            floors.add(phrase, slug)
    rooms = [_FakeArea("hallway", "מסדרון", ["כניסה"]),
             _FakeArea("kitchen", "מטבח")]
    assert _areas(rooms, "תכבה את האורות בקומת הכניסה", floors=floors) == []
    # A room really named still resolves, floor index or no floor index.
    assert _areas(rooms, "תכבה את האורות בכניסה", floors=floors) == ["hallway"]
    assert _areas(rooms, "תכבה את האורות במטבח", floors=floors) == ["kitchen"]


def test_one_order_naming_two_rooms_targets_both():
    """"Turn off the light in the living room and in the kitchen" is two rooms.

    The clause splitter deliberately leaves this as one clause - "ובמטבח" is a
    place, not a second order - so the room list has to be what makes both
    lights go out. Answering with one room leaves the other on.
    """
    assert _areas(_REAL, "תכבה את האור בסלון ובמטבח") == ["slvn", "mtbkh_2"]
    assert _areas(_REAL, "תדליק את האור בסלון") == ["slvn"]


def test_two_readings_of_one_room_phrase_are_not_two_rooms():
    """The dangerous case, and the reason overlap is checked.

    This house has חדר הורים and חדר ילדים. "בחדר ילדים" matches both indexes
    over overlapping spans - it is one room named once, read two ways - and
    treating it as two would switch on the parents' room because the children's
    room was asked for.
    """
    assert _areas(_REAL, "תדליק את האור בחדר ילדים") == ["khdr_yldym"]
    assert _areas(_REAL, "תדליק את האור בחדר הורים") == ["khdr_hvrym"]


def test_as_many_rooms_as_calls_still_pairs_them_off():
    """The existing rule is untouched: n rooms and n calls go in order."""
    sentence = "תדליק את האור בסלון וסגור את התריסים במטבח"
    assert _areas(_REAL, sentence, index=0, total=2) == ["slvn"]
    assert _areas(_REAL, sentence, index=1, total=2) == ["mtbkh_2"]


# --- music ------------------------------------------------------------------

@pytest.mark.parametrize("sentence,media_id,media_type,artist", [
    ("תנגן לי את אם ננעלו של עומר אדם", "אם ננעלו", None, "עומר אדם"),
    ("תנגן לי מוזיקה של עומר אדם", "עומר אדם", "artist", None),
    ("שים לי את האלבום שבלול של כוורת", "שבלול", "album", "כוורת"),
    ("תשמיע את השיר יש בי אהבה", "יש בי אהבה", "track", None),
    ("הפעל פלייליסט לילה טוב בסלון", "לילה טוב", "playlist", None),
    ("תנגן רדיו גלגלצ במטבח", "גלגלצ", "radio", None),
    ("תנגן את הזמרת נועה קירל", "נועה קירל", "artist", None),
    ("תנגן לי כוורת", "כוורת", None, None),
])
def test_music_is_read_out_of_the_sentence(sentence, media_id, media_type, artist):
    """The title never comes from the model, for the same reason a message does not.

    Hebrew reaches a tool argument as escape sequences, six exact characters
    per letter, and the model gets them wrong. The sentence has the words, so
    the tool schema has no slot for a title at all - the model is asked only
    for ``media_type``, one value from a five-item enum.
    """
    found = SLOT.extract_music(sentence)
    assert found is not None
    assert found.media_id == media_id
    assert found.media_type == media_type
    assert found.artist == artist


@pytest.mark.parametrize("sentence", [
    "תנגן מוזיקה",
    "תנגן קצת מוזיקה בסלון",
    "תנגן את השיר הבא",
    "תדליק את האור בסלון",
    "תפעיל את השואב",
])
def test_a_request_that_names_nothing_is_not_a_search(sentence):
    """None means "resume", not "search for the word music".

    Music Assistant needs something to look up. When the sentence never said
    what, the honest answer is to start playing rather than to search the
    library for "מוזיקה" - and "the next song" is transport control wearing a
    title's clothes.
    """
    assert SLOT.extract_music(sentence) is None


def test_the_room_is_not_part_of_the_search():
    """"Play Kaveret in the living room" searches for Kaveret.

    Leaving the room in would ask the library for a record called "Kaveret in
    the living room". Stripped before the ``של`` split, because a room can
    contain one: בחדר של הילדים.
    """
    assert SLOT.extract_music("תנגן לי כוורת בסלון").media_id == "כוורת"
    found = SLOT.extract_music("ערבב לי שירים של היהודים בחדר של הילדים")
    assert (found.media_id, found.media_type) == ("היהודים", "artist")


def test_the_router_offers_the_music_tool_when_a_kind_is_named():
    """A word for the kind of thing is what separates playing from resuming."""
    for sentence in ("שים לי את האלבום שבלול של כוורת", "תנגן רדיו גלגלצ",
                     "תשמיע לי פלייליסט רגוע", "תנגן את הלהקה כוורת"):
        assert ROUTER.select_virtual_names(sentence)[0] == "music_play", sentence
    # A bare play verb still offers both, and the model chooses.
    assert "music_play" in ROUTER.select_virtual_names("תנגן לי כוורת")
    # "Stop the song" is transport control; the noun must not hijack it.
    assert ROUTER.select_virtual_names("תעצור את השיר")[0] == "media_pause"
    # And nothing about music reaches a light or a lock.
    assert "music_play" not in ROUTER.select_virtual_names("תדליק את האור בסלון")


def test_music_play_targets_a_media_player():
    """Its service lives in one domain and its entities in another.

    ``music_assistant.play_media`` is served by media_player entities, so
    deriving the target domain from the service string alone would look for a
    "music_assistant" domain that holds no entities at all.
    """
    assert CONST.SERVICE_MAP["music_play"] == ("music_assistant", "play_media")
    assert CONST.TOOL_DOMAIN["music_play"] == "media_player"
    assert CONST.TOOL_DOMAIN["media_play"] == "media_player"


@pytest.mark.parametrize("sentence,media_id", [
    # Israelis wrap an order in a frame far more often than they bark it.
    # Anchoring the verb at the head of the sentence looked tidy and lost 72 of
    # 97 held-out music requests.
    ("אני רוצה שתנגן לי את אם ננעלו של עומר אדם", "אם ננעלו"),
    ("בבקשה תוכל לשים את הלהקה קולדפליי", "קולדפליי"),
    ("אפשר להפעיל לנו את הפלייליסט שירים ישראלים בבקשה", "שירים ישראלים"),
    # Hebrew builds the infinitive from a stem that is not the imperative:
    # להפעיל is ל + הפעיל, not ל + הפעל.
    ("את יכולה להפעיל את הפלייליסט רגוע", "רגוע"),
    # ...and "listen to X" puts a preposition where nothing else does.
    ("תוכל להאזין לפינק פלויד", "פינק פלויד"),
    # The definite article lands on the second noun of a construct chain.
    ("אני צריך שתפעילי לנו את תחנת הרדיו אקו 99", "אקו 99"),
    # A room can follow "על" as easily as "ב".
    ("תנגן את פינק פלויד על המרפסת", "פינק פלויד"),
])
def test_a_framed_request_still_names_its_music(sentence, media_id):
    found = SLOT.extract_music(sentence)
    assert found is not None, sentence
    assert found.media_id == media_id


@pytest.mark.parametrize("sentence", [
    # A mood is not a title, and speech-to-text damage to the word for "music"
    # must not become a search for the misspelling.
    "תפעיל לנו משהו טוב בחדר ההורים",
    "אני רוצה שתנגן לי מוזיכה",
    # "set the vacuum to quiet" is a setting. שקט is also a media noun, which
    # is what lets the router offer a play tool here at all.
    "שים את הרובוט על שקט",
    "שים את הפן של המיזוג במטבחון על חזק",
])
def test_a_device_setting_is_not_a_record(sentence):
    assert SLOT.extract_music(sentence) is None


def test_a_kind_word_protects_a_title_that_looks_vague():
    """"רגוע" is a mood and also the name of a playlist.

    The vague-word test is switched off once the sentence has said what kind of
    thing it wants, which is what keeps a playlist actually called רגוע
    reachable while "משהו טוב" stays a mood.
    """
    assert SLOT.extract_music("תשמיע פלייליסט רגוע").media_id == "רגוע"
    # An adjective that is not in the vague list stays a search term, and that
    # is the right answer rather than a gap: with no library in hand there is
    # nothing to tell "something calm" apart from a playlist called רגוע, and
    # searching for it is what a listener meant either way.
    assert SLOT.extract_music("תשמיע לי משהו רגוע").media_id == "רגוע"
    # What is filtered is the mood with no content at all.
    assert SLOT.extract_music("תשמיע לי משהו טוב") is None
    # And a grammar word is never a title, whatever kind was said.
    assert SLOT.extract_music("תנגן את השיר הבא") is None


# --- what the assistant says back -------------------------------------------

class Outcome:
    """Stand-in for executor.CallOutcome, which needs Home Assistant."""

    def __init__(self, ok, speech=None, detail=""):
        self.ok, self.speech, self.detail = ok, speech, detail


def test_nothing_to_do_is_an_answer_not_a_failure():
    """An empty call list is Needle refusing, which is a valid outcome."""
    assert REPLY.compose([]).error is None
    assert REPLY.compose([]).speech == CONST.SPEECH_NOTHING


def test_a_plain_success_confirms_and_a_speaking_one_speaks():
    assert REPLY.compose([Outcome(True)]).speech == CONST.SPEECH_OK
    assert REPLY.compose([Outcome(True, "מנגן שבלול")]).speech == "מנגן שבלול"
    # Several orders, several confirmations, one sentence.
    both = REPLY.compose([Outcome(True, "מנגן שבלול"), Outcome(True, "22 מעלות")])
    assert both.speech == "מנגן שבלול. 22 מעלות"


def test_a_partly_successful_sentence_says_so():
    """Speaking only the success leaves a dropped order sounding like it ran."""
    said = REPLY.compose([Outcome(True, "מנגן שבלול"), Outcome(False, detail="boom")])
    assert said.error is None                      # something did happen
    assert said.speech.startswith("מנגן שבלול. ")
    assert "נכשלה" in said.speech
    # With nothing to announce, the partial report stands on its own.
    quiet = REPLY.compose([Outcome(True), Outcome(False, detail="boom")])
    assert quiet.speech == CONST.SPEECH_OK + " חלקית, פעולה אחת נכשלה"


def test_hebrew_counts_the_failures_the_way_hebrew_counts():
    """"1 פעולות נכשלו" is a machine talking; the noun and verb agree here."""
    assert REPLY.failures(1) == "פעולה אחת נכשלה"
    assert REPLY.failures(2) == "שתי פעולות נכשלו"
    assert REPLY.failures(4) == "ארבע פעולות נכשלו"
    # Past the clause limit a digit reads naturally again.
    assert REPLY.failures(7) == "7 פעולות נכשלו"


def test_finding_nothing_is_a_different_error_from_going_wrong():
    """Home Assistant has two codes and they mean different things.

    "no lamp in the study" is the house not having one; anything else is the
    integration failing, and only the second is worth putting internals into.
    """
    empty = REPLY.compose([Outcome(False, detail=REPLY.NO_TARGET)])
    assert empty.error == REPLY.NO_TARGETS
    assert empty.speech == CONST.SPEECH_NO_TARGET

    broke = REPLY.compose([Outcome(False, detail="engine failure")], "bad export")
    assert broke.error == REPLY.FAILED
    assert broke.speech.endswith("bad export")

    # One of each is not "nothing was there", so it reports the failure.
    mixed = REPLY.compose([Outcome(False, detail=REPLY.NO_TARGET),
                           Outcome(False, detail="engine failure")])
    assert mixed.error == REPLY.FAILED


# --- what the miss list taught the router -----------------------------------
#
# Each of these was a family of misses in `eval/router_recall.py` before the
# word behind it went into a table. Together they took recall on the held-out
# set from 97.6% to 99.1%, and they are here so a table edit cannot quietly
# undo one.

@pytest.mark.parametrize("query,expected", [
    # A scene is asked for by mood as often as by name.
    ("תעשה לי אווירת ערב בבקשה", "scene_activate"),
    ("אפשר לעשות לי אווירת שינה", "scene_activate"),
    # ...and an automation named outright keeps its slot even though its own
    # name ("מצב חופשה") is also a helper.
    ("תשבית את האוטומציה מצב חופשה", "automation_turn_off"),
    ("תדליק את האוטומציה מצב חופשה", "automation_turn_on"),
    # "next in line" is how the next track gets asked for.
    ("הבא בתור בחדר השינה בבקשה", "media_next_track"),
    # A playlist has a Hebrew name as well as a borrowed one.
    ("הפעל לנו את רשימת ההשמעה לילה טוב", "music_play"),
    # Speech-to-text writes a borrowed word the way it sounds.
    ("שים לי תיימר של עשר דקות", "timer_start"),
])
def test_router_reaches_the_tool_the_sentence_named(query, expected):
    assert expected in ROUTER.select_virtual_names(query), query


def test_a_weather_word_inside_a_room_name_is_not_a_forecast():
    """"מרפסת שמש" is the balcony and "בחוץ" is the garden.

    Both were weather words on equal footing with "מזג אוויר", so a question
    about a device standing in either room was answered with the forecast. Of
    the 34 sentences in the held-out set containing שמש, none ask about the
    sky.
    """
    assert ROUTER.select_tool_names("מה קורה עם המאוורר במרפסת שמש") == ["get_state"]
    assert ROUTER.select_tool_names("מה קורה עם התאורה בחוץ") == ["get_state"]
    # The sky still answers for itself, including when what is asked for is a
    # number rather than a thing - a thermostat is a מזגן, not a טמפרטורה.
    assert ROUTER.select_tool_names("מה מזג האוויר היום") == ["get_weather"]
    assert ROUTER.select_tool_names("מה הטמפרטורה בחוץ") == ["get_weather"]
    # A statement with no interrogative in it never reaches the weather
    # test at all - it falls through to the family scorer, which keeps a
    # query slot for exactly this case. get_weather still has to be in
    # the shortlist; it just does not have it to itself.
    assert "get_weather" in ROUTER.select_tool_names("חם בחוץ")


def test_a_vague_request_reaches_audio_without_unlocking_the_gate():
    """"תפעיל לי משהו" is a request to play; "תספר לי משהו" is chatter.

    The word carries a weak weight for exactly this reason. Entered as a device
    noun it also scored three points on the four off-topic rows that say
    "תספר לי משהו על הפירמידות", which then stopped being refused.
    """
    assert "media_play" in ROUTER.select_virtual_names("תפעיל לי משהו בסלון")
    assert ROUTER.looks_off_topic("תספר לי משהו על הפירמידות")
    assert ROUTER.looks_off_topic("ספר לי משהו מעניין")


@pytest.mark.parametrize("sentence,media_id", [
    # על says where to play it...
    ("תנגן את פינק פלויד על המרפסת", "פינק פלויד"),
    # ...or what to set it to, and neither is part of the name.
    ("שים את העוצמה בחדר שינה על שישים אחוז", None),
    ("שים את שואב האבק על מצב שקט", None),
    # A speaker, a volume and a source are where the music comes out, not
    # what comes out of them.
    ("תנגן את הרמקול על הדשא", None),
    ("שים ספוטיפיי בחדר האורחים", None),
    ("תשים את הקול במוסך על 15 אחוז", None),
    # ...but a playlist really called שקט survives all of it, which is why the
    # equipment words are listed one by one instead of taken as a family.
    ("נגן את הפלייליסט שקט", "שקט"),
    ("הפעל את רשימת ההשמעה לילה טוב", "לילה טוב"),
])
def test_where_to_play_it_is_not_what_to_play(sentence, media_id):
    found = SLOT.extract_music(sentence)
    assert (found.media_id if found else None) == media_id, sentence


def test_naming_a_room_says_the_sentence_is_about_the_house():
    """A room is worth two of the three the gate asks for, and never three.

    "תפעיל לי משהו בסלון" scores one for the play verb and was being thrown
    away before the model ever saw it - 34 real commands were, on the held-out
    set. Nobody says "בסלון" about the pyramids.

    Two rather than three because a room names a place and not a thing to act
    on: at three a room passes the gate on its own, and "הגינה של השכנים
    מוזנחת" is not a question, so it would be handed a shortlist that can
    actuate.
    """
    assert not ROUTER.looks_off_topic("תפעיל לי משהו בסלון")
    assert not ROUTER.looks_off_topic("תנגן לנו משהו טוב בחדר השינה")
    # Chatter stays refused, with a room in it or without one.
    assert ROUTER.looks_off_topic("תספר לי משהו על הפירמידות")
    assert ROUTER.looks_off_topic("הגינה של השכנים מוזנחת")
    assert ROUTER.looks_off_topic("לאיזה מוסך כדאי לקחת את האוטו")


def test_a_device_in_the_car_is_not_a_device_this_house_has():
    """The noun is real, the thing it names is somewhere else.

    A car has an air conditioner and a bag has a lock, so these score the full
    three a device noun is worth and reach the gate looking exactly like an
    order. Naming the container is what tells them apart, and it is the
    preposition that does the naming - see NOT_THIS_HOUSE.
    """
    assert ROUTER.looks_off_topic("המזגן ברכב מטפטף מים")
    assert ROUTER.looks_off_topic("המנעול של התיק נשבר")
    assert ROUTER.looks_off_topic("המצלמה בטלפון שלי מטושטשת")
    assert ROUTER.looks_off_topic("המאוורר במחשב עושה רעש")
    assert ROUTER.looks_off_topic("איפה קונים מנעול לאופניים")
    assert ROUTER.looks_off_topic("הדוד בבית של אמא לא מתחמם")


def test_an_order_survives_a_word_that_would_otherwise_cancel_it():
    """The guard on the rule above, and the whole of its safety.

    Nothing in the corpus says "תפתח את השער לאופניים", and it is a perfectly
    ordinary thing to say to a house that has a gate. What spares it is the
    shape of the verb - second-person future, or the ה-imperative - which is
    how Israeli Hebrew gives an order and is not how "קונים" and "מתחמם"
    report one. `names_a_domain` is not enough here: "תפתח" is claimed by
    covers and by locks, so it is not decisive.
    """
    assert not ROUTER.looks_off_topic("תפתח את השער לאופניים")
    assert not ROUTER.looks_off_topic("תסגור את התריס ברכב")
    assert not ROUTER.looks_off_topic("הדלק את האור ברכב")
    # And the words themselves must never have been the thing that decides:
    # "בגן" (the garden) and "במשרד" (the office) are rooms in this house, and
    # they measure 274 and 275 genuine orders against zero off-topic ones, so
    # they are not on the list at all.
    assert not ROUTER.looks_off_topic("תכבה את האור בגן")
    assert not ROUTER.looks_off_topic("תדליק את המזגן במשרד")


# --- which way round --------------------------------------------------------

@pytest.mark.parametrize("said,sentence,settled", [
    # The worst thing a home assistant can do, and the reason this exists: the
    # shipped model answered 45 of these with lock_unlock.
    ("lock_unlock", "נעל את הדלת בחדר ההורים", "lock_lock"),
    ("lock_unlock", "תנעל לי את המנעול בכניסה", "lock_lock"),
    ("lock_lock", "תפתח את המנעול במטבח", "lock_unlock"),
    ("lock_lock", "תשחרר את הנעילה בגינה", "lock_unlock"),
    # ...and the rest of the pairs.
    ("cover_close", "תפתח את התריס בסלון", "cover_open"),
    ("cover_open", "תוריד את הווילון בחדר שינה", "cover_close"),
    ("light_turn_off", "תדליק את האור במטבח", "light_turn_on"),
    ("light_turn_on", "תכבה את המנורה בסלון", "light_turn_off"),
    ("switch_turn_off", "תדליק את הדוד במקלחת", "switch_turn_on"),
    ("camera_turn_on", "תכבי את המצלמה בחניון", "camera_turn_off"),
    # A tool with no opposite is returned untouched.
    ("vacuum_start", "תפעיל את השואב", "vacuum_start"),
    ("get_state", "מה המצב של האור בסלון", "get_state"),
])
def test_the_verb_decides_which_way_round(said, sentence, settled):
    assert DIRECTION.settle(said, sentence) == settled


def test_a_speaker_taking_a_verb_back_is_left_alone():
    """"תכבה את המנורה, לא לא, תעשה את המנורה" says the wrong verb first.

    Sixteen of the eighteen disagreements with gold were this shape, and every
    one of them is a light the speaker wanted *on*. The correction markers are
    what the corpus's own `correction` family is built from.
    """
    sentence = "תכבה את המנורה בחדר האוכל, לא לא, תעשה את המנורה בחדר האוכל"
    assert DIRECTION.settle("light_turn_on", sentence) == "light_turn_on"
    assert DIRECTION.settle("light_turn_off", sentence) == "light_turn_off"


def test_silence_leaves_the_model_alone():
    """No direction verb, no correction. The common case and the safe one."""
    assert DIRECTION.settle("cover_open", "את התריס בסלון בבקשה") == "cover_open"
    assert DIRECTION.settle("cover_close", "את התריס בסלון בבקשה") == "cover_close"


def test_dimming_is_a_brightness_and_not_an_off():
    """עמעם sits in light_turn_off's routing hints, which is right there.

    "תעמעם קצת פחות" is light_turn_on carrying a brightness argument, so the
    guard does not read that word as a direction.
    """
    assert DIRECTION.settle(
        "light_turn_on", "תעמעם קצת פחות את הנורה בחדר הביטחון") == "light_turn_on"


def test_play_and_pause_are_guarded_with_the_noun_dropped():
    """נגן is both "play!" and "the player", and dropping it separates them.

    This pair was left out for a long time, and the reason was sound as far as
    it went: "די עם את הנגן בסלון" carries a pause verb and a play hint, and
    counting them reads 731 agree against **22 disagree** - every one of those
    that shape.

    What changed is not the ambiguity but the remedy. `_NOT_A_DIRECTION`
    already drops ``עמעם`` from the direction vocabulary while leaving it in
    the router's, because a dim routes as a reduction and reads as a
    brightness; ``נגן`` is the same case one family over. Dropped, the pair
    measures **795 agree and 0 disagree**, which is this module's bar, so it
    ships. Home Assistant's own Hebrew suite is what forced the re-look: six
    of its HassMediaPause and HassMediaUnpause sentences ran the opposite
    service.

    Only the bare noun. ``תנגן`` and ``נגני`` are unambiguously verbs and
    dropping them too costs 84 agreements for the same zero.
    """
    assert DIRECTION.OPPOSITE["media_play"] == "media_pause"
    assert "נגן" not in DIRECTION.VOCABULARY["media_play"]
    assert "תנגן" in DIRECTION.VOCABULARY["media_play"]

    # The sentence that kept the pair out: still no evidence either way, so
    # the model's answer stands rather than being overruled on a noun.
    assert DIRECTION.settle("media_play", "די עם את הנגן בסלון") == "media_play"

    # The six the official suite asks for, in both directions.
    for verb in ("הפסק את המוזיקה", "עצור את המוזיקה", "השהה את המוזיקה"):
        assert DIRECTION.settle("media_play", verb) == "media_pause", verb
    for verb in ("המשך את המוזיקה", "חדשי את הטלוויזיה", "תנגן את המוזיקה"):
        assert DIRECTION.settle("media_pause", verb) == "media_play", verb

    # A volume is still not a play: measured with the same drop and still one
    # disagreement, and one is not zero.
    assert DIRECTION.OPPOSITE.get("media_set_volume") is None


def test_a_routine_falls_back_to_its_sibling_domain_and_no_further():
    """A scene and a script are the same act; an automation is not.

    Home Assistant keeps a household's routines in three domains and a Hebrew
    sentence cannot say which one "אווירת ערב" landed in. Both models trained
    here confuse them. The registry knows, so a name absent from the domain the
    model chose is looked for in the sibling one.

    Automations are excluded on purpose: `automation.turn_on` *enables* an
    automation rather than running it, so guessing wrong there would leave a
    household with one quietly switched on.
    """
    assert CONST.ROUTINE_SIBLING == {"scene_activate": "script_run",
                                     "script_run": "scene_activate"}
    for tool, sibling in CONST.ROUTINE_SIBLING.items():
        assert CONST.ROUTINE_SIBLING[sibling] == tool
        assert tool in CONST.NAME_ADDRESSED and sibling in CONST.NAME_ADDRESSED
        assert CONST.NAME_ADDRESSED[tool] != CONST.NAME_ADDRESSED[sibling]
    assert not any(t.startswith("automation") for t in CONST.ROUTINE_SIBLING)


def test_naming_nothing_means_all_of_them_only_for_timers():
    """"בטל את הטיימר" means every timer. "תפעיל סצנה" does not mean every scene."""
    assert CONST.ALL_WHEN_UNNAMED == set(CONST.ACTIONS["timer_control"].values())
    assert CONST.ALL_WHEN_UNNAMED <= set(CONST.NAME_ADDRESSED)
    assert all(CONST.NAME_ADDRESSED[t] == "timer" for t in CONST.ALL_WHEN_UNNAMED)
    # Nothing that is not a countdown. Activating every scene in the house is
    # not a reading of "תפעיל סצנה".
    for virtual in CONST.ACTIONS["routine_run"].values():
        assert virtual not in CONST.ALL_WHEN_UNNAMED, virtual
    for virtual in CONST.ACTIONS["helper_toggle"].values():
        assert virtual not in CONST.ALL_WHEN_UNNAMED, virtual


def test_every_name_addressed_tool_targets_a_real_domain():
    for tool, domain in CONST.NAME_ADDRESSED.items():
        if tool in CONST.QUERY_TOOLS:
            # `timer_status` answers rather than actuates, so it has no
            # service - but it still finds its entity by name.
            assert CONST.TOOL_DOMAIN[tool] == domain, tool
            continue
        assert tool in CONST.SERVICE_MAP, tool
        assert CONST.SERVICE_MAP[tool][0] == domain, tool


@pytest.mark.parametrize("sentence,clauses", [
    # Taking an order back is not giving two orders. Cutting these in two
    # turned the light off and then on again - 33 of the 74 correction rows of
    # the held-out set were being split that way.
    ("תכבה את המנורה בחדר האוכל, לא לא, תעשה את המנורה בחדר האוכל",
     ["תעשה את המנורה בחדר האוכל"]),
    ("אה, תסגור אור בחדר העבודה, לא, בעצם תעשי אור בחדר העבודה",
     ["תעשי אור בחדר העבודה"]),
    ("נו תכבה לי את המנורה על הבלקון, טעות, פתחי את המנורה על הבלקון",
     ["פתחי את המנורה על הבלקון"]),
    ("תכבה לי את האור בחצר, אה לא פתחי את האור בחצר",
     ["פתחי את האור בחצר"]),
    # An afterthought is not a retraction: dropping the first half here would
    # drop the only verb in the sentence, so the rule stands down.
    ("תדליק את האור בסלון ובעצם גם במטבח",
     ["תדליק את האור בסלון ובעצם גם במטבח"]),
    # And two real orders are still two.
    ("תכבה את האור בסלון וסגור את התריס במטבח",
     ["תכבה את האור בסלון", "סגור את התריס במטבח"]),
])
def test_an_order_taken_back_is_not_a_second_order(sentence, clauses):
    assert CLAUSE.split_clauses(sentence) == clauses


def test_the_correction_pattern_is_shared_with_the_direction_guard():
    """One definition, because both modules need the same sentences.

    `clause_split` must not cut them and `direction` must not read a verb off
    them, and for the same reason: the verb before the retraction is the one
    the speaker withdrew.
    """
    assert DIRECTION.CORRECTION is CLAUSE.CORRECTION


@pytest.mark.parametrize("said,sentence,settled", [
    # Not a direction, the same question: which of two tools in one domain.
    ("vacuum_start", "שים את הרובוט על שקט", "vacuum_set_fan_speed"),
    ("climate_set_temperature", "תעביר את המזגן למהירות נמוכה",
     "climate_set_fan_mode"),
    # ...and the other way round, so the rule is not a one-way ratchet.
    ("vacuum_set_fan_speed", "תפעיל את שואב האבק", "vacuum_start"),
    ("climate_set_fan_mode", "תוריד את המזגן ל-22 מעלות",
     "climate_set_temperature"),
])
def test_a_setting_is_not_a_start(said, sentence, settled):
    assert DIRECTION.settle(said, sentence) == settled


def test_volume_against_play_is_left_out_on_purpose():
    """65 agree and one disagrees, and one is not zero.

    The bar for overruling a model is that the rule is never wrong on the
    held-out set, not that it is usually right.
    """
    assert "media_set_volume" not in DIRECTION.OPPOSITE


@pytest.mark.parametrize("sentence,way", [
    # The verb decides.
    ("בסלון הגדול חם מדי, תנמיך משמעותית", -1),
    ("במרפסת קר מדי, תגביר קצת", 1),
    ("תעמעם את האור בסלון", -1),
    ("תרים את העוצמה במטבח", 1),
    # No verb, so the adjective decides - and "יותר חלש" is more *quiet*,
    # not more. Reading the comparative first gets all 32 of these backwards.
    ("יותר חלש בחוץ", -1),
    ("משמעותית יותר חלש בפינת המטבח", -1),
    ("קצת יותר חזק בסלון", 1),
    # Neither, so the bare comparative decides. 170 brightness rows of the
    # corpus say only this much.
    ("קצת יותר בסלון", 1),
    ("טיפה פחות במטבח", -1),
    # And silence is allowed.
    ("תעשה משהו בסלון", None),
])
def test_the_sentence_says_which_way(sentence, way):
    assert DIRECTION.which_way(sentence) == way


def test_a_retraction_is_not_a_direction():
    """The verb before "לא לא" is the one the speaker withdrew."""
    assert DIRECTION.which_way("תנמיך, לא לא, תגביר") is None


@pytest.mark.parametrize("sentence,before,after", [
    # An eight-degree error from a one-character one.
    ("בסלון הגדול חם מדי, תנמיך משמעותית",
     {"area": "living_room", "temperature_step": 4},
     {"area": "living_room", "temperature_step": -4}),
    ("יותר חלש בחוץ",
     {"area": "garden", "volume_step_pct": 20},
     {"area": "garden", "volume_step_pct": -20}),
    ("תגביר קצת יותר את האור",
     {"area": "salon", "brightness_step_pct": -10},
     {"area": "salon", "brightness_step_pct": 10}),
    # Already right: left alone, magnitude and all.
    ("תנמיך קצת", {"area": "salon", "temperature_step": -1},
     {"area": "salon", "temperature_step": -1}),
    # Nothing is invented: an absolute temperature is not a step, an argument
    # the model did not emit stays absent, and a silent sentence changes
    # nothing.
    ("תנמיך", {"area": "salon", "temperature": 24},
     {"area": "salon", "temperature": 24}),
    ("תכבה את האור", {"area": "salon"}, {"area": "salon"}),
    ("תעשה משהו", {"area": "salon", "temperature_step": 3},
     {"area": "salon", "temperature_step": 3}),
])
def test_the_sentence_settles_the_sign_and_nothing_else(sentence, before, after):
    assert DIRECTION.settle_steps(before, sentence) == after


def test_settling_a_sign_does_not_touch_the_caller_s_arguments():
    args = {"area": "salon", "temperature_step": 4}
    DIRECTION.settle_steps(args, "חם מדי, תנמיך")
    assert args == {"area": "salon", "temperature_step": 4}


def test_every_relative_argument_is_guarded():
    """The set must match what `executor._service_data` resolves against a
    current reading; a fourth one added there without being added here would
    ship with an unguarded sign."""
    assert DIRECTION.RELATIVE == {
        "temperature_step", "brightness_step_pct", "volume_step_pct"}


@pytest.mark.parametrize("sentence,settled", [
    # The infinitive is how a Hebrew speaker asks politely, and it was the
    # last shape the guard could not read: every lock inversion left in the
    # shipped evaluation was one of these.
    ("אתה יכול לנעול את הדלת בחדר הילדים", "lock_lock"),
    ("תוכל לנעול את המנעול בחדר העבודה", "lock_lock"),
    ("אפשר לסגור את התריס בסלון", "cover_close"),
    ("אתה יכול לכבות את האור במטבח", "light_turn_off"),
    ("אפשר להדליק את המאוורר בחדר", "fan_turn_on"),
])
def test_the_polite_infinitive_is_still_an_order(sentence, settled):
    opposite = DIRECTION.OPPOSITE[settled]
    assert DIRECTION.settle(opposite, sentence) == settled
    assert DIRECTION.settle(settled, sentence) == settled


def test_a_hint_list_names_a_tool_that_exists():
    """A hint under a name no tool has never ranks anything, and quietly
    stops `direction` guarding the pair it was written for - which is how an
    invented `vacuum_stop` reached a commit."""
    import json
    catalogue = {t["name"] for t in json.loads(
        (COMPONENT / "tools.json").read_text(encoding="utf-8"))}
    # Hints name behaviours, and every behaviour has to belong to a tool the
    # shipped catalogue actually declares.
    assert set(ROUTER.TOOL_HINTS) <= set(CONST.CALL_OF)
    assert {ROUTER.TOOL_OF[h] for h in ROUTER.TOOL_HINTS} <= catalogue


@pytest.mark.parametrize("sentence,clauses", [
    # A room being hot is not an instruction. Found on the device: this was
    # cut at the comma into "בסלון חם מדי" and "תנמיך משמעותית", because
    # `חם` is a climate hint and so looked like a verb. The first half then
    # set a temperature in the wrong room and the second opened a blind.
    ("בסלון חם מדי, תנמיך משמעותית", ["בסלון חם מדי, תנמיך משמעותית"]),
    ("במרפסת קר מדי, תגביר קצת", ["במרפסת קר מדי, תגביר קצת"]),
    # 17 rows of the held-out set, and none of them stops being one order.
    ("בחדר האוכל חם מדי, תוריד בהרבה", ["בחדר האוכל חם מדי, תוריד בהרבה"]),
    # Two real orders separated by a comma are still two.
    ("תכבה את האור בסלון, סגור את התריס במטבח",
     ["תכבה את האור בסלון", "סגור את התריס במטבח"]),
])
def test_a_room_being_hot_is_not_an_order(sentence, clauses):
    assert CLAUSE.split_clauses(sentence) == clauses


def test_the_state_words_stay_in_the_router():
    """They are excluded from the *verbs*, not from routing - "חם" really is
    evidence that a sentence is about the air conditioner, and dropping it
    there would send "בסלון חם מדי, תנמיך" to no family at all."""
    assert CLAUSE.NOT_ORDERS & CLAUSE.ACTION_VERBS == frozenset()
    assert ROUTER.score_families("בסלון חם מדי, תנמיך משמעותית")[0][0] == "climate"


def test_every_excluded_state_word_is_one_the_router_holds():
    """A word the router never had, excluded from the verbs, is fiction
    dressed as a rule: it changes nothing and reads as though it does."""
    known = {CLAUSE._fold(h) for hints in ROUTER.TOOL_HINTS.values() for h in hints}
    known |= {CLAUSE._fold(v) for verbs in ROUTER.FAMILY_VERBS.values() for v in verbs}
    assert CLAUSE.NOT_ORDERS <= known, CLAUSE.NOT_ORDERS - known


@pytest.mark.parametrize("sentence,kind,level", [
    # A kind and a title and no number: strong enough to overrule a model that
    # reached for the wrong media tool. 12 of the 26 music failures were
    # "ערבב את האלבום" - shuffle - answered with media_set_volume.
    ("תשמע, ערבבי את התקליט סיפורי פוגי בסטודיו בבקשה", "album", False),
    ("אני צריכה שתערבב לי את האלבום המסע של עדן חסון", "album", False),
    ("אה, בבקשה נגן את הפלייליסט ילדים", "playlist", False),
    # A number is a level, not part of a name: this one really is a volume.
    ("אתה יכול לשים את השיר על הדשא על שישים", "track", True),
    # No kind named, so the sentence is not strong enough on its own - which
    # is what keeps a mute a mute and a source a source.
    ("תשים על שקט בחדר האורחים", None, False),
    ("שים בלוטות' בחדר אוכל", None, False),
])
def test_how_strongly_the_sentence_names_something_to_play(sentence, kind, level):
    request = SLOT.extract_music(sentence)
    assert (request.media_type if request else None) == kind
    assert SLOT.names_a_level(sentence) is level


def test_the_upgrade_never_leaves_the_audio_domain():
    """`music_play` is the destination and the query tools are not audio: a
    question about music is not a request to play it."""
    assert "music_play" not in CONST.MEDIA_TOOLS
    assert not (CONST.MEDIA_TOOLS & set(CONST.QUERY_TOOLS))
    assert CONST.MEDIA_TOOLS <= set(CONST.SERVICE_MAP)


@pytest.mark.parametrize("sentence,slot,value", [
    # The vacuum's suction and the air conditioner's fan are both a word out
    # of a fixed list, so the sentence has the answer and the model guesses.
    ("אוקיי, שים את הרובוט על שקט בבקשה", "fan_speed", "silent"),
    ("בבקשה את יכולה לשים את השואב על רגיל", "fan_speed", "standard"),
    ("תעביר את המזגן במרפסת למהירות גבוה", "fan_mode", "high"),
    ("שים את הפן של המזגן על אוטומטי", "fan_mode", "auto"),
    # Nothing named, so the model keeps its answer.
    ("תדליק את האור בסלון", "fan_speed", None),
    # Two values named settles nothing.
    ("שים את השואב בין שקט לחזק", "fan_speed", None),
])
def test_the_sentence_names_the_speed_and_the_mode(sentence, slot, value):
    assert SLOT.setting_from(sentence, slot) == value


def test_the_same_word_means_different_things_in_the_two_tables():
    """`חזק` is the vacuum's strongest suction and the air conditioner's
    strongest fan, and they are different enum values - which is why the
    tables are keyed by slot instead of shared."""
    assert SLOT.SETTING_WORDS["fan_speed"]["חזק"] == "turbo"
    assert SLOT.SETTING_WORDS["fan_mode"]["חזק"] == "high"


def test_every_setting_value_is_one_the_tool_accepts():
    """A value outside the schema's enum is a service call Home Assistant
    rejects, and the sentence would have been read correctly."""
    import json
    catalogue = {t["name"]: t for t in json.loads(
        (COMPONENT / "tools.json").read_text(encoding="utf-8"))}
    for virtual, slots in CONST.SETTING_SLOT.items():
        tool = CONST.CALL_OF[virtual][0]
        for slot in slots:
            spec = catalogue[tool]["parameters"]["properties"][slot]
            allowed = {str(v) for v in spec["enum"]}
            said = set(SLOT.SETTING_WORDS[slot].values())
            assert said <= allowed, (virtual, slot, sorted(said - allowed))
            # And the behaviour has to be allowed to carry it, or
            # `_service_data` drops the value the sentence read correctly.
            assert slot in CONST.TOOL_ARGS[virtual], (virtual, slot)


def test_a_verb_is_not_a_colour():
    """`להוריד` is `ל` + `ה` + `ורוד`, and `ורוד` is pink. The general prefix
    chain - the one that lets "בסלון" find the living room - turns "להוריד את
    התאורה" into a request for a pink light, on 21 corpus calls that name no
    colour at all. A value is a noun in a prepositional phrase and never
    carries a verb's prefixes, which is why these slots do not go through
    PhraseIndex."""
    assert SLOT.setting_from("תוכל להוריד את התאורה ליד האוטו", "color_name") is None
    assert SLOT.setting_from("תעשה את האור בסלון ורוד", "color_name") == "pink"
    assert SLOT.setting_from("תעשה את האור בסלון בורוד", "color_name") == "pink"
    assert "color_name" in CONST.SETTING_SLOT["light_turn_on"]


@pytest.mark.parametrize("sentence,colour", [
    ("תעשה את האור בסלון סגול", "purple"),
    ("תדליק את הנורה במטבח בצבע לבן", "white"),
    ("אני רוצה אור קר בבלקון", "cool_white"),
    ("תעשה את הנורה בסלון חמים", "warm_white"),
])
def test_the_sentence_names_the_colour(sentence, colour):
    assert SLOT.setting_from(sentence, "color_name") == colour


def test_the_service_data_carries_only_arguments_the_tool_declares():
    """The sentence is allowed to change which tool runs - `direction.settle`
    and the music upgrade both do it - and the model filled its arguments for
    the tool it originally chose. "תעביר את המזגן למהירות גבוה" came back as
    `climate_set_temperature{temperature: 23}`, the verb corrected it to
    `climate_set_fan_mode`, and the stale `temperature` would have made Home
    Assistant reject the call. Found on the device."""
    import json
    catalogue = {t["name"]: t for t in json.loads(
        (COMPONENT / "tools.json").read_text(encoding="utf-8"))}

    # v11 made this table matter more, not less. A domain tool declares every
    # argument any of its behaviours can take, so the grammar now permits
    # `light_control{action: "shut", brightness_pct: 40}` - well formed, and
    # not something `light.turn_off` accepts. So the relation is no longer
    # equality: each behaviour's arguments must be a *subset* of what its tool
    # declares, and between them the behaviours must cover the whole tool, or
    # the schema declares an argument nothing can ever use.
    assert set(CONST.TOOL_ARGS) == set(CONST.CALL_OF)
    for virtual, allowed in CONST.TOOL_ARGS.items():
        tool = CONST.CALL_OF[virtual][0]
        declared = set(catalogue[tool]["parameters"]["properties"]) - {"action"}
        assert allowed <= declared, (
            virtual, sorted(allowed - declared))
    by_tool: dict[str, set[str]] = {}
    for virtual, allowed in CONST.TOOL_ARGS.items():
        by_tool.setdefault(CONST.CALL_OF[virtual][0], set()).update(allowed)
    for name, tool in catalogue.items():
        declared = set(tool["parameters"]["properties"]) - {"action"}
        assert by_tool.get(name, set()) == declared, (
            name, sorted(declared - by_tool.get(name, set())))


# --- what Home Assistant's own Hebrew suite taught the router ---------------
#
# `eval/intent_parity.py` runs the 192 sentences of `OHF-Voice/intents`'
# Hebrew test suite through this stack. They are the one measurement here that
# no change to this project's corpus can move, and the first run scored 75.0%.
# Everything in this section is something that run turned up.

@pytest.mark.parametrize("query,family", [
    # Verbs that name a domain with no device noun anywhere in the sentence.
    # Every one is a sentence the official suite requires and the refusal gate
    # was discarding before inference.
    ("נקה כאן", "vacuum"),
    ("שאב אבק פה", "vacuum"),
    ("נקה בחדר הזה", "vacuum"),
    ("דלג", "media"),
    ("דלגי", "media"),
])
def test_a_verb_only_one_family_claims_names_a_domain(query, family):
    assert ROUTER.names_a_domain(query) == family
    assert not ROUTER.looks_off_topic(query)


@pytest.mark.parametrize("verb", [
    # Measured over all 20,814 generated rows, these fire on off-topic ones and
    # are therefore excluded by hand. Three are the prefix-stripping trap this
    # module already documents for מזגן/גן: "מקרר" strips to "קרר" and
    # "מתחמם" to "תחמם".
    "סגור", "תזכיר", "תשלח", "תעמיד", "תחמם", "תפתח", "תדליק", "קרר",
    "תצלם", "שלח",
])
def test_ambiguous_verbs_are_not_decisive(verb):
    assert verb not in ROUTER.DECISIVE_VERBS


def test_decisive_verbs_are_derived_not_listed():
    """Adding a verb to FAMILY_VERBS must make it decisive on its own.

    The alternative is a second table beside FAMILY_VERBS, and the two would
    drift the first time somebody added a synonym. Every decisive verb has to
    be a verb some family actually declares, and no family may claim two.
    """
    for verb, family in ROUTER.DECISIVE_VERBS.items():
        claimants = [f for f, verbs in ROUTER.FAMILY_VERBS.items()
                     if any(ROUTER._fold(v).strip(ROUTER._PUNCT) == verb
                            for v in verbs)]
        assert claimants == [family], (verb, claimants)
    # An interrogative is not an imperative. "מה" must never carry a sentence
    # past the refusal gate - that is what looks_like_question is for, and it
    # declares read-only tools rather than letting anything act.
    for verb in ROUTER.FAMILY_VERBS["query"]:
        assert ROUTER._fold(verb).strip(ROUTER._PUNCT) not in ROUTER.DECISIVE_VERBS


@pytest.mark.parametrize("query", [
    # The six sentences Home Assistant's Hebrew suite uses for HassNevermind.
    "לא משנה", "ביטול", "עצור", "עצרי", "בטל", "בטלי",
    # And the ordinary Israeli forms, with the politeness the corpus wraps
    # every utterance in.
    "שכח מזה", "אה, לא משנה עזוב תודה", "עזוב", "די", "מספיק",
])
def test_a_bare_withdrawal_does_nothing(query):
    assert ROUTER.looks_like_cancel(query)


@pytest.mark.parametrize("query", [
    # The same verbs governing something. A bare imperative is a withdrawal;
    # one with an object is an order, and reading these as cancels would make
    # "stop the music" silently do nothing.
    "עצור את המוזיקה",
    "בטל את הטיימר",
    "עצרי את השיר בסלון",
    "תעצור את השואב",
    "תדליק את האור בסלון",
    # Politeness alone is not a withdrawal. It is not anything, and the
    # refusal gate is the honest answer for it.
    "בבקשה",
    "תודה",
])
def test_a_governed_verb_is_still_a_command(query):
    assert not ROUTER.looks_like_cancel(query)


@pytest.mark.parametrize("query", [
    # "אילו אורות דולקים" names a light, so it scores three and passes the
    # refusal gate - and without an interrogative matching it, the shortlist
    # filled with light_*. A question that can turn the lights on is the exact
    # failure the read-only gate exists to prevent.
    "אילו אורות דולקים",
    "אילו מאווררים פועלים",
    # A timer is asked after by name rather than with an interrogative, and
    # both of these were reaching timer_start - restarting the very countdown
    # somebody just asked about.
    "מצב הטיימר",
    "מצב הטיימרים במטבח",
    "כמה זמן נשאר",
])
def test_a_question_never_reaches_an_actuation_tool(query):
    assert ROUTER.looks_like_question(query)
    for name in ROUTER.select_virtual_names(query):
        assert name in CONST.QUERY_TOOLS, (query, name)


@pytest.mark.parametrize("query,tool", [
    # Vocabulary the official suite uses and this project had no word for.
    ("מה טמפ", "get_state"),
    ("כמה חם בתרמוסטט", "get_state"),
    ("האם המתגים דולקים", "get_state"),
    ("האם יש חלונות פתוחים", "get_state"),
    ("קבע את הבהירות של מנורת שינה ל50%", "light_turn_on"),
    ("בטל את כל הטיימרים", "timer_cancel"),
])
def test_the_official_vocabulary_reaches_its_tool(query, tool):
    assert not ROUTER.looks_off_topic(query)
    assert tool in ROUTER.select_virtual_names(query)


# --- the official Hebrew lists, and not drifting from them ------------------

@pytest.mark.parametrize("sentence,want", [
    # A room with both must be able to tell them apart. Without device_class
    # the executor matches every cover in the area and opens the lot.
    ("תפתח את התריסים בסלון", "blind"),
    ("תפתח את הווילונות בסלון", "curtain"),
    ("תפתח את דלת החניה", "garage"),
    ("תפתח את דלתות החניה", "garage"),
    ("תסגור את החלון במטבח", "window"),
    ("תפתח את השער", "gate"),
    ("תפתח את תריסי ההצללה", "shutter"),
    # Two kinds named settles nothing, and the executor keeps its wider match.
    ("תפתח את התריסים והווילונות", None),
    # A sentence about something else must not name a cover kind.
    ("תדליק את האור בסלון", None),
])
def test_the_sentence_names_which_kind_of_cover(sentence, want):
    assert SLOT.setting_from(sentence, "device_class") == want


@pytest.mark.parametrize("sentence,want", [
    ("אילו אורות דולקים", "on"),
    ("אילו מאווררים פועלים", "on"),
    ("האם כל החלונות סגורים", "closed"),
    ("אילו חלונות פתוחים", "open"),
    ("האם המתגים כבויים", "off"),
    # An order is not a state filter.
    ("תדליק את האור", None),
])
def test_the_sentence_names_the_state_asked_about(sentence, want):
    assert SLOT.setting_from(sentence, "state") == want


def test_multi_word_keys_do_not_disturb_the_measured_slots():
    """The three slots measured at 0 wrong must be byte-for-byte unaffected.

    `setting_from` gained a substring pass for multi-word keys. colour, fan
    speed and fan mode have no key with a space in it, so that pass cannot fire
    for them - this asserts the premise rather than trusting it, because the
    measurement those three carry (335 / 180 / 293 right, 0 wrong) was made
    before the pass existed.
    """
    for slot in ("color_name", "fan_speed", "fan_mode"):
        assert not any(" " in key for key in SLOT.SETTING_WORDS[slot]), slot


@pytest.mark.parametrize("utterance,named,total,expected", [
    # Home Assistant's own Hebrew responses answer these three differently for
    # the same tool and the same entities, and a household hears the
    # difference. The expected strings are the shapes `responses/he` uses.
    ("האם האורות דולקים", ["אור א"], 2, "כן, אור א"),
    ("אילו אורות דולקים", ["אור א"], 2, "אור א"),
    ("כמה אורות דולקים", ["אור א"], 2, "1"),
    # "all" is a different question from "any": two of five being on is not a
    # yes, and answering one for the other is how a state question misleads.
    ("האם כל האורות דולקים", ["אור א"], 2, "לא, אור א"),
    ("האם כל המאווררים פועלים", ["מאוורר א"], 1, "כן"),
    # Nothing in the asked-for state still has to be an answer.
    ("אילו חלונות פתוחים", [], 3, "אף אחד"),
    ("האם החלונות סגורים", [], 3, "לא"),
])
@needs_executor
def test_a_state_question_is_answered_in_the_shape_it_was_asked(
        utterance, named, total, expected):
    assert EXEC.CallExecutor._say_which(utterance, named, total) == expected


def test_device_class_narrowing_is_only_for_domains_that_need_it():
    """Narrowing a domain that holds one kind of thing can only lose entities.

    Every value in the table has to be a device class Home Assistant actually
    defines for `cover`, or the narrowed match silently finds nothing and the
    fallback quietly does all the work.
    """
    # Home Assistant is not installed in the bare CI job, and this test
    # reaches it. Skipping beats failing: the job exists to prove the
    # *Hebrew* logic needs nothing installed, not to check this.
    pytest.importorskip("homeassistant")
    assert CONST.DEVICE_CLASS_DOMAINS == frozenset(("cover",))
    from homeassistant.components.cover import CoverDeviceClass
    real = {c.value for c in CoverDeviceClass}
    assert set(SLOT.SETTING_WORDS["device_class"].values()) <= real


def test_every_fallback_service_exists():
    """A fallback that names a service Home Assistant does not have is worse
    than no fallback: the primary domain found nothing, the widened search
    finds an entity, and then the call fails - so the household is told the
    command failed by the code that was there to make it work.

    Checked against the components' own `services.yaml`, which is where the
    names came from: `valve` spells open as `open_valve`, not `open_cover`.
    """
    # Home Assistant is not installed in the bare CI job, and this test
    # reaches it. Skipping beats failing: the job exists to prove the
    # *Hebrew* logic needs nothing installed, not to check this.
    pytest.importorskip("homeassistant")
    import homeassistant.components as components
    import yaml

    root = pathlib.Path(components.__file__).parent
    for tool, entries in CONST.FALLBACK_DOMAINS.items():
        assert tool in CONST.SERVICE_MAP, tool
        for target, service_domain, service in entries:
            path = root / service_domain / "services.yaml"
            assert path.exists(), (tool, service_domain)
            declared = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            assert service in declared, (tool, service_domain, service)
            # The entity domain a fallback searches has to be a real one too,
            # or the widened match silently finds nothing every time.
            assert (root / target).is_dir(), (tool, target)


def test_fallbacks_are_reached_by_words_the_router_knows():
    """A fallback domain nothing routes to is unreachable.

    `valve` rides the cover family and `humidifier` rides switch, so the words
    that name them have to be in those families - otherwise the executor's
    widening never runs, because the tool it widens is never shortlisted.
    """
    for word in ("ברז", "שסתום"):
        assert "cover_open" in ROUTER.select_virtual_names(f"תפתח את ה{word}")
    for word in ("מאדה", "מייבש"):
        assert "switch_turn_on" in ROUTER.select_virtual_names(f"תדליק את ה{word}")
    # A robot mower rides the vacuum family the same way, and needed its own
    # noun to do it: "תפעיל את המכסחת" scored nothing for vacuum and reached
    # the automations instead, which left the lawn_mower fallback unreachable.
    for word in ("מכסחת", "מכסחה"):
        assert "vacuum_start" in ROUTER.select_virtual_names(f"תפעיל את ה{word}")


# --- the "באיזה חדר" round trip --------------------------------------------
#
# `conversation.py` imports Home Assistant's conversation component, which
# needs `hassil`; it is present in the QA venv and not on a plain workstation.
# The two methods under test touch nothing but `self._pending` and
# `slot_match`, so they are exercised against a stub rather than a live entity.

def _agent():
    pytest.importorskip("hassil")
    CONV = load("conversation")

    class Stub(CONV.NeedleConversationEntity):
        def __init__(self):          # noqa: D107 - no hass, no entry
            self._pending = {}

    return Stub()


class _Input:
    def __init__(self, text, conversation_id="c1"):
        self.text = text
        self.conversation_id = conversation_id


@needs_executor
def test_a_room_answers_the_question_it_was_asked():
    """"תכבה את האור" then "בסלון" has to run as one sentence.

    Carrying a room slot forward instead would need a second code path for
    each of the eight things the sentence supplies. Joining gives the layers
    below exactly the shape they already handle.
    """
    agent = _agent()
    assert agent._may_ask_which_room("תכבה את האור", joined=False)
    agent._pending["c1"] = (EXEC.monotonic(), "תכבה את האור")
    assert agent._joined_with_pending(_Input("בסלון")) == (
        "תכבה את האור בסלון", True)


@needs_executor
def test_the_pending_command_is_used_once_and_dropped():
    """A stale command must not attach itself to an unrelated sentence."""
    agent = _agent()
    agent._pending["c1"] = (EXEC.monotonic(), "תכבה את האור")
    agent._joined_with_pending(_Input("בסלון"))
    # Consumed. The next turn stands on its own.
    assert agent._joined_with_pending(_Input("בסלון")) == ("בסלון", False)


@needs_executor
def test_a_follow_up_that_is_not_a_room_stands_on_its_own():
    """The speaker moved on. Their new sentence is not part of the old one."""
    agent = _agent()
    agent._pending["c1"] = (EXEC.monotonic(), "תכבה את האור")
    assert agent._joined_with_pending(_Input("תנגן מוזיקה")) == (
        "תנגן מוזיקה", False)


def test_which_room_is_never_asked_twice():
    """A joined sentence that still finds nothing has had its answer.

    Asking again would be a loop, and the honest reply is that no device
    matched. Nor is it asked about a sentence that did name a room - the room
    is not what was missing.
    """
    agent = _agent()
    assert not agent._may_ask_which_room("תכבה את האור בסלון", joined=True)
    assert not agent._may_ask_which_room("תכבה את האור בסלון", joined=False)


def test_a_timer_is_cancelled_by_the_timer_tool():
    """"בטל את הטיימר" reached `input_boolean_turn_off`, and the ceiling
    measurement could not see it.

    Three tools scored one hint each - `בטל` here and under the helper toggle,
    `טיימר` under timer_start - so the tie broke on position in FAMILY_TOOLS,
    where the helper toggle sits second and timer_cancel fourth. The right tool
    was declared, just not first, and the model took the first: 0 of the 13
    HassCancelTimer sentences in Home Assistant's Hebrew suite reached
    `timer.cancel`.
    """
    for sentence in ("בטל את הטיימר", "בטלי את הטיימר", "עצור את הטיימר",
                     "בטל את כל הטיימרים", "הפסק את הספירה"):
        assert ROUTER.select_virtual_names(sentence)[0] == "timer_cancel", sentence
    # And the other direction still leads with the tool that starts one.
    for sentence in ("קבע טיימר ל5 דקות", "טיימר ל2 שעות",
                     "תעמיד טיימר לעשר דקות"):
        assert ROUTER.select_virtual_names(sentence)[0] == "timer_start", sentence
    # The helper toggles are not disturbed by any of it.
    assert ROUTER.select_virtual_names("תכבה את מצב אורחים")[0] == \
        "input_boolean_turn_off"
    assert ROUTER.select_virtual_names("תפעיל את מצב חופשה")[0] == \
        "input_boolean_turn_on"


def test_a_helper_toggle_about_a_countdown_is_a_timer():
    """The model answers "בטל את הטיימר" with `input_boolean_turn_off`.

    Reliably, and not because of the shortlist: putting `timer_cancel` first
    did not move it, so the prior is the model's. 0 of the 13 HassCancelTimer
    sentences in Home Assistant's Hebrew suite reached `timer.cancel`.

    The corpus settles it without ambiguity - of the 473 generated rows whose
    sentence names a countdown, every gold call is a timer tool or a state
    query and none is an `input_boolean` - so the sentence overrules the model,
    and `direction` then picks which end of the countdown, because the toggle
    only carries on or off.
    """
    assert CONST.HELPER_IS_A_TIMER == {
        "input_boolean_turn_on": "timer_start",
        "input_boolean_turn_off": "timer_cancel",
    }

    def corrected(tool, sentence):
        if tool in CONST.HELPER_IS_A_TIMER and ROUTER.names_a_timer(sentence):
            tool = CONST.HELPER_IS_A_TIMER[tool]
        return DIRECTION.settle(tool, sentence)

    assert corrected("input_boolean_turn_off", "בטל את הטיימר") == "timer_cancel"
    assert corrected("input_boolean_turn_off", "בטל את כל הטיימרים") == "timer_cancel"
    # The toggle said *on* and the sentence says stop. Reading the toggle would
    # start a countdown somebody just asked to end.
    assert corrected("input_boolean_turn_on", "עצור את הטיימר") == "timer_cancel"
    assert corrected("input_boolean_turn_on", "הפסק את הטיימר") == "timer_cancel"
    assert corrected("input_boolean_turn_on", "קבע טיימר ל5 דקות") == "timer_start"

    # A helper that is not a countdown is untouched, in both directions.
    assert corrected("input_boolean_turn_off", "תכבה את מצב אורחים") == \
        "input_boolean_turn_off"
    assert corrected("input_boolean_turn_on", "תפעיל את מצב חופשה") == \
        "input_boolean_turn_on"


def test_a_transport_verb_about_a_countdown_is_a_timer():
    """The same reach, one family over: v12 answers "השהה את הטיימר" with a
    media pause.

    `עצור` and `השהה` halt a track and a countdown with the same word, and the
    shortlist is not the problem - the router declares `timer_control` first
    and `media_control` second on every one of these. The corpus licenses the
    override at least as strongly as the helper toggle above: 1,463 of 30,613
    rows have a transport verb as their gold and **not one names a countdown**,
    and on the rows whose sentence does name one the gold is a media behaviour
    **zero** times against 1,249 that are a timer.

    `settle_timer` then picks which of the seven, exactly as it does for a
    toggle - so mapping pause onto pause is a starting point, not a verdict.
    """
    assert DIRECTION.TRANSPORT_IS_A_TIMER == {
        "media_pause": "timer_pause",
        "media_stop": "timer_cancel",
        "media_play": "timer_resume",
    }
    # One name for the three call sites, so they cannot drift on which tools
    # are worth offering.
    assert DIRECTION.SETTLES_A_TIMER == (
        DIRECTION.TIMER_BEHAVIOURS | {"media_pause", "media_stop", "media_play"})

    def corrected(tool, sentence, args=None):
        return DIRECTION.settle_timer(
            tool, args or {}, sentence,
            ROUTER.looks_like_question(sentence),
            ROUTER.names_a_timer(sentence))

    assert corrected("media_pause", "השהה את הטיימר") == "timer_pause"
    assert corrected("media_pause", "תשהה את הספירה") == "timer_pause"
    assert corrected("media_pause", "עצור את הטיימר") == "timer_cancel"
    assert corrected("media_stop", "עצור את הטיימר") == "timer_cancel"
    assert corrected("media_stop", "בטל את הטיימר") == "timer_cancel"
    assert corrected("media_play", "תמשיך את הטיימר") == "timer_resume"
    # The verb wins over the pairing, which is the whole point of arbitrating
    # after the mapping rather than instead of it.
    assert corrected("media_pause", "כמה זמן נשאר בטיימר") == "timer_status"
    assert corrected("media_stop", "עוד חמש דקות לטיימר",
                     {"minutes": 5}) == "timer_add"

    # A speaker is still a speaker. None of the three moves without the noun.
    for tool in ("media_pause", "media_stop", "media_play"):
        assert corrected(tool, "תשהה את המוזיקה בסלון") == tool
        assert corrected(tool, "עצור את הנגן") == tool
    # And the five that overlap in nothing stay put even when it is there.
    for tool in ("media_next_track", "media_previous_track", "media_mute",
                 "media_set_volume", "media_select_source"):
        assert corrected(tool, "תשהה את הטיימר") == tool


def test_naming_a_countdown_is_measured_not_guessed():
    """Every timer noun has to be one the corpus actually uses that way."""
    assert ROUTER.names_a_timer("בטל את הטיימר")
    assert ROUTER.names_a_timer("תעצור את הספירה")
    assert not ROUTER.names_a_timer("תכבה את האור בסלון")
    assert not ROUTER.names_a_timer("תפעיל את מצב לילה")


# --- v11: the wire format ---------------------------------------------------
#
# One tool per domain, one `action` per behaviour. The internal currency stays
# the per-service **virtual id**, and the tests below guard the two places the
# translation happens: the corpus writes it, the executor reads it.

def test_the_shortlist_stays_inside_the_token_budget():
    """v11's tools are three to five times the size of the ones they replaced.

    Needle decodes through a 256-token sliding window with nothing pinned, so a
    long tools block pushes its own head out before generation starts. The
    router trims from the back, which drops the lowest-scoring speculative
    family and never the one the sentence named.
    """
    catalogue = json.loads((COMPONENT / "tools.json").read_text(encoding="utf-8"))
    biggest = sorted(catalogue,
                     key=lambda t: -len(json.dumps(t, separators=(",", ":"))))
    chosen = ROUTER.select_tools("תדליק את האור בסלון", biggest, 5)
    rendered = len(json.dumps(chosen, separators=(",", ":")))
    assert rendered <= ROUTER.MAX_TOOL_CHARS + len(
        json.dumps(biggest[0], separators=(",", ":")))
    # One tool always survives, whatever it costs: a sentence with nothing
    # declared can only be refused.
    assert chosen


# --- v11: the new capabilities ---------------------------------------------

def test_a_countdown_question_is_a_timer_status():
    """"כמה זמן נשאר" names no device and no room, so the refusal gate threw
    it away - the one sentence in the whole official Hebrew suite this stack
    could not reach. Answering it with `timer.start` would restart the very
    countdown that was asked about, which is why it is a query tool."""
    for sentence in ("כמה זמן נשאר", "כמה זמן נשאר בטיימר",
                     "מה מצב הטיימר", "כמה נשאר בספירה"):
        assert not ROUTER.looks_off_topic(sentence), sentence
        assert ROUTER.select_tool_names(sentence) == ["timer_control"], sentence
        assert ROUTER.select_virtual_names(sentence) == ["timer_status"], sentence
    assert "timer_status" in CONST.QUERY_TOOLS
    assert "timer_status" not in CONST.SERVICE_MAP


def test_the_list_item_comes_out_of_the_sentence():
    """Hebrew never reaches a tool argument - it arrives as escape sequences,
    six characters per letter, and the model gets them wrong. A shopping list
    with a blank entry on it is worse than a command that said it did not
    understand, so `None` means "do not act"."""
    assert SLOT.extract_item("תוסיף חלב לרשימת קניות") == "חלב"
    assert SLOT.extract_item("תוסיף לי גבינה צהובה לרשימה") == "גבינה צהובה"
    assert SLOT.extract_item("תמחק ביצים מהרשימה") == "ביצים"
    assert SLOT.extract_item("תסמן שקניתי חלב") == "חלב"
    assert SLOT.extract_item("תוסיף לקחת את הכלב לווטרינר לרשימת המטלות") == \
        "לקחת את הכלב לווטרינר"
    # Nothing named is not an empty item.
    assert SLOT.extract_item("תדליק את האור בסלון") is None
    assert SLOT.extract_item("תוסיף") is None


def test_the_four_colliding_enums_are_read_from_the_sentence():
    """Home Assistant fixes these values and two pairs of them share a first
    byte, which v4 measured as the failure mode of a byte-level decode. The
    sentence has no such problem."""
    assert SLOT.setting_from("תעביר את המזגן לקירור", "hvac_mode") == "cool"
    assert SLOT.setting_from("שים את המזגן על חימום", "hvac_mode") == "heat"
    assert SLOT.setting_from("תנגן את האלבום שבלול", "media_type") == "album"
    assert SLOT.setting_from("תנגן את הזמר שלמה ארצי", "media_type") == "artist"
    assert SLOT.setting_from("שים את הרובוט על שקט", "fan_speed") == "silent"
    assert SLOT.setting_from("תעשה את האור אדום", "color_name") == "red"
    # And the ones v11 added beside them.
    assert SLOT.setting_from("תוסיף חלב לרשימת הקניות", "list") == "shopping"
    assert SLOT.setting_from("תוסיף לרשימת המטלות", "list") == "todo"
    assert SLOT.setting_from("תדליק אור בלבן חם", "color_temp_k") == "2700"
    # Two different values named is no value at all.
    assert SLOT.setting_from("בין קירור לחימום", "hvac_mode") is None


def test_the_sentence_supplies_the_temperature_the_model_drops():
    """"שנה את הטמפרטורה ל20 מעלות" turned the air conditioner *off*.

    Three of the four sentences Home Assistant's own Hebrew suite still got
    wrong were this one shape. The routing is right and `settle_climate` is
    right; what fails is that the model answers with no argument at all, so
    the rule that turns a climate call carrying a temperature into a call to
    set one has nothing to fire on.

    Read per clause over all 30,613 corpus rows: 442 agree, **0 disagree**,
    zero fires on the 463 rows whose gold is a step, and zero on the 2,513
    climate rows that want no number.
    """
    assert SLOT.temperature_from("שנה את הטמפרטורה ל20 מעלות") == 20
    assert SLOT.temperature_from("קבע את הטמפרטורה ל20 מעלות") == 20
    assert SLOT.temperature_from("שים את המזגן על 24") == 24
    assert SLOT.temperature_from("תעשה 22 מעלות בסלון") == 22
    assert SLOT.temperature_from("שים את המזגן על 23 בבקשה") == 23

    # `ב` is the preposition for a step, and a step reaches another service
    # with another argument. Reading it as a target moves the thermostat to
    # two degrees.
    assert SLOT.temperature_from("תוריד את המזגן ב2 מעלות") is None
    assert SLOT.temperature_from("תעלה ב3 מעלות") is None

    # Numbers wearing another argument's unit.
    assert SLOT.temperature_from("שים את התריסים על 30 אחוז") is None
    assert SLOT.temperature_from("תפעיל טיימר ל5 דקות") is None
    assert SLOT.temperature_from("טיימר ל20 שניות") is None
    assert SLOT.temperature_from("תעמיד טיימר לשעה") is None

    # Two numbers is no number, and a reading outside what a thermostat takes
    # is a misread rather than an instruction.
    assert SLOT.temperature_from("על 20 ועל 24") is None
    assert SLOT.temperature_from("שים את המזגן על 90") is None
    assert SLOT.temperature_from("תדליק את האור בסלון") is None

    # And the whole point: with the number attached, the existing rule fires.
    args = {"temperature": SLOT.temperature_from("שנה את הטמפרטורה ל20 מעלות")}
    assert DIRECTION.settle_climate(
        "climate_turn_off", args, "שנה את הטמפרטורה ל20 מעלות") == \
        "climate_set_temperature"


@needs_executor
def test_a_timer_duration_carries_hours_and_seconds():
    """"תעמיד טיימר לשעה וחצי" could only be said in minutes before v11, and
    `timer.change` takes a signed duration in seconds rather than a clock
    string - a negative "HH:MM:SS" is not something its selector accepts."""
    seconds = EXEC.CallExecutor._seconds
    assert seconds({"hours": 1, "minutes": 30}) == 5400
    assert seconds({"minutes": 90}) == 5400
    assert seconds({"seconds": 45}) == 45
    # No duration at all is not a duration of zero: that is what separates
    # "resume the timer" from "start a timer for nothing".
    assert seconds({"name": "pasta"}) is None


def test_the_accusative_object_is_the_device_not_the_value():
    """"את הקירור במטבחון רק מאוורר" switches the cooling to fan-only.

    Reading the first value gets it exactly backwards, and it did: four of the
    corpus's hvac rows, three of them with the target word corrupted by
    injected speech noise so that only the object was legible. The rule is
    per-slot rather than general, because the grammar is identical and the
    semantics are opposite - "תנגן את האלבום שבלול" plays the album, and the
    same rule applied to `media_type` threw away 487 correct readings.
    """
    assert SLOT.setting_from("את הקירור במטבחון רק מאוורר", "hvac_mode") == "fan_only"
    # A real preposition makes it a target wherever it stands.
    assert SLOT.setting_from("תעביר את המזגן לקירור", "hvac_mode") == "cool"
    assert SLOT.setting_from("שים את המזגן על חימום", "hvac_mode") == "heat"
    # And the content slots are untouched.
    assert SLOT.setting_from("תנגן את האלבום שבלול", "media_type") == "album"
    assert SLOT.setting_from("שים את הרדיו", "media_type") == "radio"
    assert SLOT.SETTING_WORDS["hvac_mode"].get("אוטו") is None, (
        "אוטו is the Hebrew for *car*: "
        "'הדליקי את המיזוג ליד האוטו' read as a request for automatic mode")


def test_a_toggle_the_sentence_contradicts_is_settled():
    """A toggle has no opposite, so `settle` cannot reach it.

    v11 made that matter: until then the router ranked `light_toggle` third in
    its family and a tight shortlist usually cut it, so the model rarely
    emitted one. Now all three are values of one enum, and on an epoch-4
    checkpoint "כבי את המנורה בחדר הביטחון" came back as a toggle with nothing
    to correct it.

    Narrow on purpose: no toggle word at all, and exactly one side named.
    335 agree and 0 disagree on the rows whose gold really is a toggle.
    """
    assert DIRECTION.settle_toggle("light_toggle", "כבי את המנורה") == "light_turn_off"
    assert DIRECTION.settle_toggle("light_toggle", "תדליק את האור") == "light_turn_on"
    assert DIRECTION.settle_toggle("fan_toggle", "תכבה את המאוורר") == "fan_turn_off"
    # A sentence that really does say "toggle" keeps it.
    assert DIRECTION.settle_toggle(
        "light_toggle", "תחליף מצב של האור") == "light_toggle"
    assert DIRECTION.settle_toggle("light_toggle", "תהפוך את המצב") == "light_toggle"
    # A self-correction settles nothing, and neither does silence.
    assert DIRECTION.settle_toggle("light_toggle", "תדליק לא לא תכבה") == "light_toggle"
    assert DIRECTION.settle_toggle("light_toggle", "האור בסלון") == "light_toggle"
    # Anything that is not a toggle is untouched.
    assert DIRECTION.settle_toggle("light_turn_on", "כבי את המנורה") == "light_turn_on"
    assert set(DIRECTION.TOGGLES) == (
        {v for v in CONST.CALL_OF if v.endswith("_toggle")}
        - {"light_toggle"} | {"light_toggle"})


def test_a_gerund_names_a_light_behaviour_as_plainly_as_an_imperative():
    """"הדלקה של אור ראשי" is how the official suite asks, and it is a noun.

    Nothing named a light behaviour, so the model's `flip` stood and the
    sentence ran `light.toggle`. `הדלקה` and `כיבוי` are the nouns of two
    verbs already in the table, added as a pair; neither occurs in any of the
    30,613 corpus rows, which is what says they cannot move a shortlist.
    """
    assert DIRECTION.settle_toggle(
        "light_toggle", "הדלקה של אור ראשי") == "light_turn_on"
    assert DIRECTION.settle_toggle(
        "light_toggle", "כיבוי של האור בסלון") == "light_turn_off"
    # Naming both settles nothing, which is the table's standing rule.
    assert DIRECTION.settle_toggle(
        "light_toggle", "הדלקה או כיבוי, מה שתחליט") == "light_toggle"


def test_a_climate_call_carrying_a_temperature_is_a_call_to_set_one():
    """The argument settles the action, and no word has to be read to see it.

    `climate_control` cannot be hint-decided - קירור names the machine as well
    as one of its modes - but `temp` is the only one of its five behaviours
    with anywhere to put a temperature. 1,469 gold calls carry one and every
    one of them is `temp`.
    """
    settle = DIRECTION.settle_climate
    assert settle("climate_set_hvac_mode", {"temperature": 20}, "") == \
        "climate_set_temperature"
    assert settle("climate_turn_off", {"temperature_step": -2}, "") == \
        "climate_set_temperature"
    # A call with nothing to set is left exactly as it came.
    assert settle("climate_turn_off", {"area": "salon"}, "") == "climate_turn_off"
    assert settle("climate_set_hvac_mode", {"hvac_mode": "cool"}, "") == \
        "climate_set_hvac_mode"
    # And a leftover argument from the half of a sentence that was retracted
    # must not drag the call back to it.
    assert settle("climate_turn_off", {"temperature": 22},
                  "תעלה ל22, לא, תכבה") == "climate_turn_off"
    # A clause that asked for a fan speed asked for a fan speed, whatever the
    # model attached to the call. This is the guard that took the rule from
    # three losses to one, and it is the *clause* that decides and not the
    # call: 15 moves whose call also named a mode were all correct, while 3 of
    # the 4 whose clause named one were not.
    assert settle("climate_set_fan_mode", {"temperature": 20},
                  "שים את הפן של המיזוג בפינת האוכל על נמוך",
                  True) == "climate_set_fan_mode"
    assert settle("climate_set_fan_mode", {"temperature": 20},
                  "שים את המיזוג בפינת האוכל על 20 מעלות",
                  False) == "climate_set_temperature"
    # Nothing outside the family is touched, whatever it carries.
    assert settle("light_turn_on", {"temperature": 20}, "") == "light_turn_on"


def test_the_general_hint_rule_is_allow_listed_because_it_fails():
    """Taking the best-*scoring* sibling for any multi-behaviour tool measures
    14,161 agree and 647 disagree - 4.4% wrong, which is not the bar. What
    ships is narrower: exactly one sibling's hints must appear at all. Only the
    tools that measure exactly zero under *that* rule are used.

    `routine_run` is the ninth, and the only one admitted on a shorter list
    than the router's: its three siblings all carry להפעיל, so the strict rule
    was silent on almost every scene. `DISCRIMINATING` gives it the nouns that
    separate them, and it measures 609 agree against 3 disagree - all three a
    negative verb speech noise glued to its neighbour. See `HINT_DECIDED`."""
    assert DIRECTION.HINT_DECIDED == frozenset((
        "list_edit", "vacuum_control", "switch_control", "lock_control",
        "helper_toggle", "camera_control", "get_datetime", "timer_control",
        "routine_run"))
    # The ones it would have broken are the ones this project already documents
    # as lexically ambiguous - and the two of them that needed settling got a
    # rule written for their own structure instead.
    for tool in ("climate_control", "media_control", "light_control",
                 "cover_control"):
        assert tool not in DIRECTION.HINT_DECIDED
    assert DIRECTION.MEDIA_BEHAVIOURS and DIRECTION.TIMER_BEHAVIOURS

    def settle(virtual, sentence):
        tool = CONST.CALL_OF[virtual][0]
        if tool not in DIRECTION.HINT_DECIDED:
            return virtual
        return DIRECTION.settle_action(
            virtual, list(CONST.ACTIONS[tool].values()), sentence)

    assert settle("list_complete_item",
                  "תסיר לי נייר טואלט מהרשימה") == "list_remove_item"
    assert settle("list_remove_item", "תוסיף חלב לרשימת קניות") == "list_add_item"
    assert settle("vacuum_start", "תחזיר את השואב לבסיס") == "vacuum_return_to_base"
    # A climate sentence is left to the model and to the pair guard, because
    # קירור is the machine as well as the mode. A structural rule was written
    # for it and measured at 270 disagreements against the corpus's own
    # convention - "הדלק את הקירור" is a mode change and "כבה את הקירור" is a
    # turn-off - and rejected.
    assert settle(
        "climate_set_temperature", "תעביר את המזגן לקירור") == "climate_set_temperature"


def test_a_clause_that_names_nothing_is_not_an_order():
    """The sentence gate runs once; every clause needs it too.

    "צריך מים מינרלים, תוסיפי לרשימה" is a statement and an order, and the
    statement half was reaching the model and coming back as a call - a device
    moving because somebody said what they needed. Measured over all 28,233
    corpus rows: 303 clauses dropped, 2 gold calls lost, both of them word-merge
    speech noise that swallowed the device noun.

    Narrower than `looks_off_topic` on purpose. A clause scoring one or two
    still named something - "תזכיר לי בעוד שעה" scores one for the countdown -
    and those have to reach the model.
    """
    assert ROUTER.clause_names_nothing("צריך מים מינרלים")
    assert ROUTER.clause_names_nothing("כבר לקחתי שוקולד")
    assert ROUTER.clause_names_nothing("חצי אם אפשר")
    # Named something: not this gate's business, whatever it scores.
    assert not ROUTER.clause_names_nothing("תוסיפי לרשימה")
    assert not ROUTER.clause_names_nothing("תדליק את האור בסלון")
    assert not ROUTER.clause_names_nothing("תזכיר לי בעוד שעה")
    # And it is strictly inside the sentence gate: a clause the sentence gate
    # would refuse can still name something, and then it is not this gate's.
    assert ROUTER.looks_off_topic("תזכיר לי בעוד שעה")
    assert not ROUTER.clause_names_nothing("תזכיר לי בעוד שעה")


def test_a_question_asks_which_or_whether_and_they_differ():
    """"אילו אורות דולקים" filters the answer; "תבדוק אם האור דולק" does not.

    Reading the state word on its own was measured at 544 right and 127 wrong,
    every one of the 127 a yes/no question turned into a list. The interrogative
    test takes it to 420 right and **0** wrong, with 132 silent.
    """
    assert SLOT.state_filter("אילו אורות דולקים") == "on"
    assert SLOT.state_filter("איזה תריסים סגורים") == "closed"
    assert SLOT.state_filter("כמה מנורות כבויות") == "off"
    assert SLOT.state_filter("האם כל הרמקולים מופעלים") == "on"
    # Whether, not which.
    assert SLOT.state_filter("תבדוק אם האור בגן דולק") is None
    assert SLOT.state_filter("האם התריס בגינה סגור") is None
    assert SLOT.state_filter("") is None
    # The word is still readable on its own; only the filter is gated.
    assert SLOT.setting_from("תבדוק אם האור בגן דולק", "state") == "on"


def test_a_door_asked_about_is_a_lock():
    """`דלת` names both `cover` and `lock`, and a question has to break the tie.

    For a command the registry breaks it - a house with no cover doors finds
    nothing and falls through. A question types the domain and answers about it,
    with nothing to fall through to, and every one of the 98 disagreements this
    slot had was this word. 1,371 right / 98 wrong becomes 1,467 / 2.
    """
    assert ROUTER.query_domain("כמה הדלתות סגורות") == "lock"
    assert ROUTER.query_domain("תגיד לי מה קורה עם הדלת בממד") == "lock"
    assert ROUTER.query_domain("איזה הדלתות סגורות") == "lock"
    # A door that names its class outright keeps it.
    assert ROUTER.query_domain("מה המצב של דלת החניה") == "cover"
    # And nothing else moved.
    assert ROUTER.query_domain("אילו אורות דולקים") == "light"
    assert ROUTER.query_domain("מה המצב של החלון") == "binary_sensor"


# ---------------------------------------------------------------------------
# The sentence decides, in both directions.
#
# Ten repairs measured on predictions rather than on gold, which the replay
# tape in the workshop tree made affordable. Together they moved exact match
# from 60.7% to 70.7% on the frozen benchmark with the weights unchanged, and
# no family regressed. See "The twelfth problem" in the project README.
# ---------------------------------------------------------------------------

def test_an_israeli_opens_a_boiler_the_way_they_open_a_blind():
    """`light_turn_on` has carried פתח since v4; `switch_turn_on` never did,
    though `switch_turn_off` has carried סגור all along.

    With neither side of the pair named, `settle_toggle` falls silent and the
    call ships as a toggle - so "תפתח את הדוד" turned the boiler *off* when it
    was on. 50 of the 396 failures in one release dump were this one verb.

    Measured over the corpus on single-clause single-call switch rows: an
    open-form verb means `switch_turn_on` 124 times and its opposite none.

    Deliberately not given to the router's own hint table, which also scores
    the shortlist: see `direction.DIRECTION_ONLY`.
    """
    assert "פתח" not in ROUTER.TOOL_HINTS["switch_turn_on"]
    assert "פתח" in DIRECTION.VOCABULARY["switch_turn_on"]
    for text in ("תפתח את הדוד בסלון", "פתחי את המפסק בחדר האורחים",
                 "אתה יכול לפתוח לי את הבוילר"):
        assert DIRECTION.settle_toggle("switch_toggle", text) == "switch_turn_on"
    # And the direction it already read stays read: אוף still wins, and a
    # sentence that really does say toggle is still a toggle.
    assert DIRECTION.settle_toggle(
        "switch_toggle", "תעשה אוף את הפלאג בממד") == "switch_turn_off"
    assert DIRECTION.settle_toggle(
        "switch_toggle", "תעשה טוגל לשקע במקלחת") == "switch_toggle"


def test_a_scene_is_not_a_script_and_the_noun_says_which():
    """`routine_run` was the worst tool in the catalogue - 24.3% - and the
    failure was almost always a scene answered as a macro.

    Its three siblings all carry להפעיל, because all three genuinely are
    things you run, so `settle_action`'s strict "exactly one sibling named"
    rule settled nothing every time. The nouns separate them completely:
    609 agree, 3 disagree over the corpus. See `direction.DISCRIMINATING`.
    """
    siblings = list(CONST.ACTIONS["routine_run"].values())
    settle = DIRECTION.settle_action
    assert settle("script_run", siblings, "תפעיל סצנת סרט") == "scene_activate"
    assert settle("scene_activate", siblings, "תריץ את הסקריפט השקיה") == "script_run"
    assert settle("scene_activate", siblings, "תלחץ על הכפתור בכניסה") == "button_press"
    # Two halves of one pair count as one naming, and `settle` says which half.
    assert settle("scene_activate", siblings,
                  "תפעיל את האוטומציה תריסים בבוקר") == "automation_turn_on"
    assert settle("scene_activate", siblings,
                  "תבטל את האוטומציה תריסים בבוקר") == "automation_turn_off"
    # A sentence naming no routine noun still settles nothing.
    assert settle("scene_activate", siblings, "תפעיל את זה") == "scene_activate"


def test_stopping_a_countdown_for_a_moment_stays_a_pause():
    """The reason `settle_timer` runs *after* the allow-list rather than before.

    עצור is a cancel hint and "לעצור **רגע**" is a pause, a phrase only
    `_T_PAUSE` carries. With the timer settled first, `settle_action` then
    overruled it on eleven rows of one template.
    """
    text = "אפשר לעצור רגע את הספירה"
    siblings = list(CONST.ACTIONS["timer_control"].values())
    # The allow-list on its own gets this wrong, and is allowed to.
    assert DIRECTION.settle_action("timer_start", siblings, text) == "timer_cancel"
    # The specialist speaks last, so the pipeline gets it right.
    assert DIRECTION.settle_timer(
        "timer_cancel", {}, text, False, True) == "timer_pause"


def test_a_bare_list_is_the_shopping_list():
    """Israelis say "תוסיף לרשימה חלב" far more often than they say which list.

    Over train and test the bare forms take the slot from 538 right to 623
    right at 0 wrong. The construct form רשימת stays out: it opens "רשימת
    מטלות" as readily as "רשימת קניות".
    """
    assert SLOT.setting_from("תוסיפי לרשימה פיתות", "list") == "shopping"
    assert SLOT.setting_from("תמחק את הביצים מהרשימה", "list") == "shopping"
    # The multi-word keys are read first and win outright.
    assert SLOT.setting_from("תוסיף לרשימת מטלות לשטוף כלים", "list") == "todo"
    assert SLOT.setting_from("סמן את החלב כבוצע ברשימת הקניות", "list") == "shopping"
    assert "רשימת" not in SLOT.SETTING_WORDS["list"]


def test_a_number_said_out_loud_is_still_a_number():
    """Half of every number in the corpus is a word, and nothing read one.

    `data/hebrew_speech.py` has spelled values both ways since v2, because
    Whisper emits "22 מעלות" or "עשרים ושתיים מעלות" depending on the model.
    Every reader on the other side was a two-digit regex. Counting gold
    values whose digits appear nowhere in the sentence: temperature 347,
    brightness_pct 319, minutes 308, position 277, volume_pct 177,
    percentage 113 - 1,558 rows the deterministic layer could not see.

    `hebrew_numbers` is the inverse of the tables the corpus is generated
    from, so it reads exactly what the generator writes and what a person
    says. Measured over the training corpus: degrees 485 right / 2 wrong,
    percentage 1,175 / 3, countdown 553 / 7 - and every one of the twelve is
    a value speech noise corrupted.
    """
    # Both genders, the teens, and the compound that carries the conjunction.
    assert NUM.degrees_in("תכוון את המזגן לעשרים ושמונה מעלות") == 28
    assert NUM.degrees_in("שמונה עשרה מעלות בשירותים") == 18
    assert NUM.percent_in("שים את האורות על שבעים וחמישה אחוז") == 75
    assert NUM.percent_in("תפתח את התריס לחצי") == 50
    # Particles glue to the number itself, digits included.
    assert NUM.duration_in("טיימר ל5 דקות") == {"minutes": 5}
    assert NUM.duration_in("טיימר לשעה ו15 דקות") == {"hours": 1, "minutes": 15}
    # Canonical, so ninety seconds is a minute and a half.
    assert NUM.duration_in("טיימר לתשעים שניות") == {"minutes": 1, "seconds": 30}
    # Unit-anchored: a bare number is not a reading.
    assert NUM.degrees_in("תדליק את האור בחדר שתיים") is None
    assert NUM.percent_in("תדליק את האור בחדר שתיים") is None
    assert NUM.duration_in("תדליק את האור בחדר שתיים") == {}
    # "all the way" is not a percentage; it was 114 media rows read as 100%.
    assert NUM.percent_in("תפסיק לגמרי בחדר האוכל") is None


def test_the_sentence_wins_the_temperature_it_names():
    """It used to fill only where the model was silent, so a model that said
    23 for a sentence saying 19 kept the 23.

    The sentence is right about this whenever it speaks - 485 against 2 - and
    a wrong temperature is the commoner failure of the two. `ב` still marks a
    step and is still not read as a target.
    """
    assert SLOT.temperature_from("תעמיד את המזגן על עשרים ושתיים מעלות") == 22
    assert SLOT.temperature_from("תוריד את המזגן ב2 מעלות") is None
    assert SLOT.temperature_from("תדליק את המזגן") is None


def test_doing_the_light_means_turning_it_on():
    """"תעשה את האור" - *do* the light - is how an Israeli asks for it on.

    A last tier in `settle_toggle`, reached only by a sentence naming neither
    side of the pair and no toggle either: 385 agree, 1 disagree over train and
    test, and the one glued its אוף to the next word.

    It could not ship until `light_toggle` gained טוגל, which `switch_toggle`
    and `fan_toggle` had all along. Without it "תעשה טוגל לאור" reached this
    tier and 50 genuine toggles were switched on instead.
    """
    assert "טוגל" in ROUTER.TOOL_HINTS["light_toggle"]
    assert DIRECTION.settle_toggle(
        "light_toggle", "תעשה את האור בסלון") == "light_turn_on"
    assert DIRECTION.settle_toggle("fan_toggle", "תעשי את המאוורר") == "fan_turn_on"
    # A real toggle stays a toggle, and a named side still wins.
    assert DIRECTION.settle_toggle(
        "light_toggle", "תעשה טוגל לאור בסלון") == "light_toggle"
    assert DIRECTION.settle_toggle(
        "light_toggle", "תכבה את האור בסלון") == "light_turn_off"


def test_asking_for_a_mode_is_asking_to_set_one():
    """The setting slots are read for the behaviour the model chose, so when it
    answers "תעביר את המזגן לאוורור" with a set-temperature, nothing ever reads
    the mode - the behaviour that carries the slot was never selected.

    `hvac_target` is the strict reading that closes it: preposition-bound, or
    bare only after רק, never an adjective, never the noun in "מהירות
    המאוורר". 119 agree, 0 disagree over the corpus.
    """
    settle = DIRECTION.settle_climate
    target = SLOT.hvac_target("תעביר את המזגן בסלון לאוורור")
    assert target == "fan_only"
    assert settle("climate_set_temperature", {"temperature": 23},
                  "תעביר את המזגן בסלון לאוורור", False,
                  target) == "climate_set_hvac_mode"
    # A room that is too hot is a temperature, not a request for the heater.
    assert SLOT.hvac_target("בחדר של הקטן חם מדי, תוריד שמץ") is None
    # And a fan speed is a fan speed, whatever noun carries it.
    assert SLOT.hvac_target("תכוון את מהירות המאוורר של המזגן לנמוך") is None


def test_a_garage_door_is_a_blind_not_a_bolt():
    """"תפתח את דלת החניה" is the garage; דלת on its own is what a lock has.

    So the router scores lock, the model answers `lock_control`, and a
    household that asked for the garage gets a bolt - 20 of 400 failures in
    one dump. All 147 corpus rows naming one of the eight compound cover
    nouns are cover rows. `settle` still says open or closed.
    """
    assert SLOT.names_a_compound_cover("תפתח את דלת החניה")
    assert not SLOT.names_a_compound_cover("תפתח את הדלת")
    settled = DIRECTION.NAMES_A_COVER["lock_unlock"]
    assert settled == "cover_open"
    assert DIRECTION.settle(settled, "תסגור את דלת החניה") == "cover_close"


def test_the_verbs_an_israeli_uses_to_ask_for_music():
    """`שלח` - send me music - and `ערבב` - shuffle something.

    Neither reached `settle_media`: ערבב was a routing hint, which gets the
    family onto the shortlist and never reaches the ladder, and שלח was
    nowhere. Send 32 agree / 0 disagree, shuffle 216 / 0.

    `שים` is measured and rejected at 289 against 63: "שים על שקט" is a mute
    and "שים טלוויזיה" is a source, and neither tier above carries those words.
    """
    for verb in ("שלח", "תשלח", "ערבב", "תערבב"):
        assert verb in DIRECTION._M_PLAY
    assert "שים" not in DIRECTION._M_PLAY
    assert DIRECTION.settle_media(
        "media_pause", "תשלח מוזיקה בסלון", False) == "media_play"
    assert DIRECTION.settle_media(
        "media_next_track", "תערבב לי משהו במטבח", False) == "media_play"


def test_a_slot_the_sentence_does_not_support_is_dropped():
    """Every rule here could only ever add. The model invented 459 arguments
    in one release run and nothing removed them.

    Nine slots are measured *complete*: over train and test, 34 rows out of
    roughly four thousand carry one of them without the sentence saying so,
    and every one of the 34 is a value speech noise corrupted. Four more are
    deliberately excluded - see `slot_match.SENTENCE_OWNS`.
    """
    assert SLOT.unsupported("תדליק את המזגן בסלון", "hvac_mode") is False
    assert SLOT.unsupported("תדליק את המזגן בסלון", "temperature") is True
    assert SLOT.unsupported("תדליק את האור בסלון", "brightness_pct") is True
    assert SLOT.unsupported("תדליק את האור על חמישים אחוז", "brightness_pct") is False
    assert SLOT.unsupported("תדליק את האור בסלון", "color_name") is True
    assert SLOT.unsupported("תדליק אור אדום בסלון", "color_name") is False
    # position is not owned: "לגמרי" is a position without being a number.
    assert SLOT.unsupported("תסגור את התריס לגמרי", "position") is False


def test_the_noun_says_which_family():
    """The general form of the two rules below it, and the largest of the ten.

    A sentence whose nouns name exactly one family belongs to that family,
    whatever the model answered: 19,410 agree and **7 disagree** over train and
    test on rows whose gold is an actuation. Six of the seven are a word speech
    noise glued to its neighbour.

    Questions are excluded, because a question names a device and asks *about*
    it - `tool_router.query_domain` owns those. A sentence naming two families
    settles nothing, which is what keeps an automation whose *name* is about
    blinds from becoming a cover command.
    """
    assert ROUTER.family_named("כבה את המאוורר במטבח") == "fan"
    assert DIRECTION.family_named("light_turn_off", "fan") == "fan_turn_on"
    # And the verb still says which behaviour inside the family.
    assert DIRECTION.settle(
        DIRECTION.family_named("light_turn_off", "fan"),
        "כבה את המאוורר במטבח") == "fan_turn_off"
    # Two families named settles nothing.
    assert ROUTER.family_named("תדליק את האוטומציה תריסים בבוקר") is None
    # A family the model already agreed with is left alone.
    assert DIRECTION.family_named("light_turn_off", "light") == "light_turn_off"
    # A question is not this rule's business.
    assert DIRECTION.family_named("get_state", "fan") == "get_state"


def test_a_plug_has_no_pause():
    """A television is a smart plug in 43 of this corpus's 59 television rows,
    so the model answers "השהה את הטלוויזיה" with `switch_control{flip}` and it
    is not wrong about the noun.

    It is wrong about the verb. A transport verb *and* a media noun together
    are decisive where each half is not: 1,204 rows over train and test carry
    both, and every one is a media or music behaviour. The halting verbs were
    measured apart, because הפסק and עצור can mean switching something off -
    paired with a media noun they do not, in 342 rows out of 342.

    Five of the nine sentences Home Assistant's own Hebrew suite still ran the
    wrong service for were these. It now runs none wrong at all.
    """
    assert ROUTER.names_a_transport("השהה את הטלוויזיה")
    assert ROUTER.names_a_transport("הפסיקי את הטלוויזיה")
    # A verb a plug *can* answer is not this rule's business.
    assert not ROUTER.names_a_transport("כבה את הטלוויזיה")
    # And neither is a question about the speaker, or a countdown.
    assert not ROUTER.names_a_transport("האם המוזיקה מנגנת")
    assert not ROUTER.names_a_transport("השהה את הטיימר")
    assert "switch_toggle" in DIRECTION.TRANSPORT_IS_MEDIA
    assert DIRECTION.settle_media(
        "media_pause", "המשך את הטלוויזיה", False) == "media_play"


def test_a_temperature_call_with_no_temperature_is_the_mode_it_named():
    """The largest single failure shape left on the frozen benchmark, and it
    cost two arguments per row rather than one.

    The model answers "תכוון את המזגן לרק מאוורר" with `climate_set_temperature`.
    `settle_climate` used to return that untouched - the early return fired
    before `named_mode` was ever read - and keeping the temperature behaviour
    also *discards the mode*, because `hvac_mode` is not one of its arguments,
    so the value `slot_match` had already resolved had nowhere to go.
    """
    settle = DIRECTION.settle_climate
    assert settle("climate_set_temperature", {"area": "salon"},
                  "תכוון את המזגן בסלון לרק מאוורר", "hvac_mode") == \
        "climate_set_hvac_mode"
    assert settle("climate_set_temperature", {"area": "salon"},
                  "תכוון את מהירות המאוורר של המזגן בסלון לנמוך", "fan_mode") == \
        "climate_set_fan_mode"
    # With something to set, the older reading stands: this really is a
    # temperature call that also names a fan speed.
    assert settle("climate_set_temperature", {"temperature": 24},
                  "תעלה את המזגן ל24 ובמהירות גבוהה", "fan_mode") == \
        "climate_set_temperature"
    # And a clause that switches the machine is not a clause that sets its
    # mode, however many mode words it carries. Without this guard the rule
    # reads 114 disagreements against gold instead of none.
    assert settle("climate_set_temperature", {"area": "salon"},
                  "תכבה את כל הקירור בסלון", "hvac_mode") == \
        "climate_set_temperature"
    assert settle("climate_set_temperature", {"area": "salon"},
                  "תוכלי לעשות מזגן בסלון", "hvac_mode") == \
        "climate_set_temperature"


def test_the_fan_is_a_device_as_well_as_a_mode():
    """``אוטומטי`` is a machine mode and a fan speed both, so which device the
    clause is built around decides - and ``רק`` flips it back."""
    assert SLOT.mode_slot("תכוון את מהירות המאוורר של המזגן לנמוך") == "fan_mode"
    assert SLOT.mode_slot("לשים את הפן של המזגן על אוטומטי") == "fan_mode"
    # The preposition is glued on in both directions: "למהירות", "לרק".
    assert SLOT.mode_slot("להעביר את הקירור למהירות גבוהה") == "fan_mode"
    assert SLOT.mode_slot("להעביר את המזגן לרק מאוורר") == "hvac_mode"
    # A fan sentence naming no speed settles nothing rather than reading the
    # noun as a request for fan-only mode.
    assert SLOT.mode_slot("תכוון את מהירות המאוורר של המזגן") is None


def test_a_clause_can_name_its_own_behaviour():
    """Four families `settle_action`'s allow-list cannot take. Each measured
    per clause over the whole corpus at zero disagreements."""
    named = DIRECTION.settle_named
    assert named("fan_turn_on", "תעשה שהמאוורר בגן עם סיבוב") == "fan_oscillate"
    assert named("cover_set_position", "עצרי את הרפפות בחדר הביטחון") == "cover_stop"
    assert named("valve_open", "תפסיקי ברז בחדר ילדים") == "valve_stop"
    assert named("media_pause", "שתעצור לגמרי בשירותים") == "media_stop"
    # A stop verb alone does not settle a speaker: "תעצור את המוזיקה" is a
    # pause, and only לגמרי/סטופ/השמעה separate the two.
    assert named("media_pause", "תעצור את המוזיקה בסלון") == "media_pause"
    # The percentage, and the colour.
    assert named("cover_open", "את הווילונות במרפסת על שבעים אחוז",
                 at_a_position=True) == "cover_set_position"
    assert named("light_toggle", "אני רוצה אור אדום בפינת האוכל",
                 names_a_colour=True) == "light_turn_on"
    # And a sentence that takes its verb back is left alone, as everywhere
    # else in this module.
    assert named("fan_turn_on", "תעשה עם סיבוב, לא, תכבה") == "fan_turn_on"


def test_the_sentence_says_which_way_a_two_state_slot_points():
    """``בלי סיבוב`` is still a request to oscillate - the negation lands on
    the argument, not on the behaviour - and an unmute is said as undoing
    one, never with a negative particle."""
    assert SLOT.switch_from("תעשה שהמאוורר בגן עם סיבוב", "oscillating") is True
    assert SLOT.switch_from("בלי סיבוב של מאוורר בפינת המטבח", "oscillating") is False
    assert SLOT.switch_from("תבטל את הסיבוב של מאוורר בשירותים", "oscillating") is False
    assert SLOT.switch_from("תעשה מיוט במטבחון", "is_volume_muted") is True
    assert SLOT.switch_from("תשים על שקט במטבח", "is_volume_muted") is True
    assert SLOT.switch_from("תוריד מיוט על הדשא", "is_volume_muted") is False
    assert SLOT.switch_from("שתבטל השתקה בחדר שלנו", "is_volume_muted") is False
    # A slot with no table is left to the model.
    assert SLOT.switch_from("תעשה מיוט במטבחון", "brightness_pct") is True


def test_every_two_state_slot_is_an_argument_of_its_behaviour():
    """`BOOLEAN_SLOT` writes straight into the call, so a typo in either half
    would put an argument the schema rejects onto every one of those rows."""
    for behaviour, slot in CONST.BOOLEAN_SLOT.items():
        assert behaviour in CONST.CALL_OF, behaviour
        assert slot in CONST.TOOL_ARGS[behaviour], (behaviour, slot)


def test_a_camera_is_closed_as_readily_as_it_is_switched_off():
    """`camera_control` was in the hint-decided allow-list with only the on and
    off verbs in its vocabulary, so "תוכל לסגור לי מצלמה" came back as `on`.
    107 close-form verbs on camera rows and 82 open-form ones, none of either
    against its opposite."""
    assert DIRECTION.settle("camera_turn_on", "נו תוכל לסגור לי מצלמה בחדר הילדים") == \
        "camera_turn_off"
    assert DIRECTION.settle("camera_turn_off", "תפתח את המצלמה בחצר") == \
        "camera_turn_on"


def test_asking_who_sang_it_is_not_asking_for_it():
    """The gate scores words, and "מי שר את השיר על האור בקצה המנהרה" names a
    song and a light, so it sailed past the bar and got `media_control{next}` -
    the assistant skipping a track at someone asking a trivia question. 208
    corpus rows match the category and every one of them is a refusal."""
    assert ROUTER.looks_off_topic("מי שר את השיר על האור בקצה המנהרה")
    assert ROUTER.looks_off_topic("אה, מי כתב את השיר הזה")
    # Still an order, and still reaches the house.
    assert not ROUTER.looks_off_topic("תשים את השיר הבא בסלון")
    assert not ROUTER.looks_off_topic("תדליק את האור בסלון")


def test_a_step_the_model_left_out_is_taken_from_the_adverb():
    """The module header rejected filling a missing step, and the reason was
    the 106 corpus calls carrying an *absolute* value under a directional verb
    - "תוריד את המזגן לבערך 25" lowers it **to** 25. Every one of those names a
    number, so the fill is guarded on the clause naming none. 389 agree, 1
    disagree.

    A verb that names a direction and *no* size at all fills two, the unmarked
    reading the adverbs above modify in both directions. Measured per clause
    over the corpus on climate calls that set a temperature, take a direction
    verb and name neither a number nor a size word: 186 agree, 4 disagree, and
    all four say `הרבה` or `משמעותית` through the injected noise - "הרוה פחות",
    "משמאותית", "הרבהיותר" - where a readable adverb would have sized it four.
    """
    settle = DIRECTION.settle_steps
    assert settle({}, "בגן חם מדי, תנמיך קצת",
           ("temperature_step",)) == {"temperature_step": -1}
    assert settle({}, "אפשר שתגביר משמעותית את המיזוג", ("temperature_step",)) == \
        {"temperature_step": 4}
    # Without the caller's leave - the clause names a number - nothing is added.
    assert settle({}, "תוריד את המזגן לבערך 25", ()) == {}
    # No adverb: the unmarked step, and its sign still comes from the verb.
    assert settle({}, "תנמיך את המזגן",
           ("temperature_step",)) == {"temperature_step": -2}
    assert settle({}, "תגביר את המזגן",
           ("temperature_step",)) == {"temperature_step": 2}
    # A direction with no verb behind it sizes nothing: יותר and חלש modify a
    # step, they do not make one.
    assert settle({}, "אני רוצה יותר", ("temperature_step",)) == {}
    # And a step the model did emit is still only re-signed, never resized.
    assert settle({"temperature_step": 3}, "חם מדי, תנמיך קצת",
           ("temperature_step",)) == \
        {"temperature_step": -3}


def test_switching_a_thing_is_not_turning_it_off():
    """`settle_toggle` run backwards, and with no vocabulary of its own — the
    toggle hints already name one. The model answered "תחליף את המצב של הוונטה"
    with a `shut`. light 257 agree, switch 108, fan 99, none against."""
    named = DIRECTION.settle_named
    assert named("fan_turn_off", "תחליף את המצב של הוונטה בחדר המגורים") == "fan_toggle"
    assert named("light_turn_on", "תעשה טוגל לנורה במקלחת") == "light_toggle"
    assert named("switch_turn_off", "החלף את המצב של השקע במטבח") == "switch_toggle"
    # And the direction still wins when the sentence names one.
    assert named("light_turn_on", "תדליק את האור בסלון") == "light_turn_on"


def test_switching_the_machine_is_not_setting_anything_on_it():
    """The counterpart of `settle_climate`'s promotion, from the other side.
    That function refuses to read a mode off a clause that switches; this one
    reads the switch. `settle` cannot: the model answered with neither half of
    the pair, so there is no pair to settle. 235 agree for on, 901 for off."""
    named = DIRECTION.settle_named
    assert named("climate_set_temperature", "את יכולה להדליק מזגן בחדר המגורים") == \
        "climate_turn_on"
    assert named("climate_set_temperature", "תכבי את הקירור בשירותים") == \
        "climate_turn_off"
    assert named("climate_set_fan_mode", "לכבות את כל הקירור בבית") == \
        "climate_turn_off"
    # Both sides named settles nothing, as everywhere else in this module.
    assert named("climate_set_temperature", "תדליק ותכבה") == \
        "climate_set_temperature"
    # And a call that really does set something is left alone.
    assert named("climate_set_temperature", "תעלה את המזגן ל24") == \
        "climate_set_temperature"


# --- the third round of deterministic repairs -------------------------------

def test_a_question_about_the_world_is_not_an_order():
    """The gate's own docstring named these as the work that was left: "every
    one of them is a sentence whose device noun is real and whose meaning is
    not". Scoring words cannot separate "כמה עולה מזגן חדש לסלון" from a
    command - the noun really is this house's - so the shape answers the gate
    outright, the way `ASKS_ABOUT_CONTENT` already did for songs.

    Measured over all 40,631 corpus rows: the category is matched by 700 of the
    4,462 refusals and by **zero** of the 36,169 rows that want a call.
    """
    off = ROUTER.looks_off_topic
    assert off("כמה עולה מזגן חדש לסלון")
    assert off("כמה זמן לוקח להתקין תריסים חשמליים")
    assert off("איך מכבים מחשב שנתקע")
    assert off("מה ההבדל בין מזגן אינוורטר לרגיל")
    assert off("כדאי לקנות שואב אבק רובוטי")
    assert off("תמליץ לי על מוזיקה לריצה")
    assert off("מה מזג האוויר הטיפוסי באלסקה בחורף")
    assert off("מה יש בטלוויזיה הערב")
    assert off("שלח הודעה לדני בוואטסאפ")
    assert off("תוסיף פגישה ליומן מחר בעשר")
    # And the orders that share their nouns are still orders.
    assert not off("תדליק את המזגן בסלון")
    assert not off("תסגור את התריסים בחדר השינה")
    # `תכתוב לי` is how this corpus adds to a list, on 37 rows, and stays out.
    assert not off("תכתוב לי בננות לרשימת המצרכים")


def test_asking_for_something_names_no_device_and_is_still_an_order():
    """The mirror of the category above. Israelis ask for music by asking for
    *something*, and `משהו` is a pronoun - the sentence scores nothing for any
    family and 22 genuine requests were refused.

    The verb is what makes it safe: 196 corpus utterances pair `משהו` with a
    play-or-put verb and every one wants a call, while the 84 that say `משהו`
    and want none are all "תספר לי משהו על הדינוזאורים".
    """
    assert ROUTER.asks_for_something("שים משהו")
    assert ROUTER.asks_for_something("תשימי לנו משהו טוב")
    assert ROUTER.asks_for_something("תפעילי משהו בחניון")
    assert not ROUTER.asks_for_something("תספר לי משהו על הדינוזאורים")
    assert not ROUTER.looks_off_topic("שימי לנו משהו")
    assert ROUTER.looks_off_topic("תספר לי משהו על מגדל אייפל")


def test_an_invented_temperature_does_not_outrank_a_mode_that_was_spoken():
    """The guard on the mode promotion asked whether the *call* carried a
    temperature. The model attaches `temperature: 23` to "תעביר את המזגן לרק
    מאוורר" out of habit, so the guard held on a value nobody said and the
    clause kept a temperature behaviour with nowhere to put its mode.

    Now the sentence has to back the argument up - with a number, or with a
    verb that asks for a relative move. Measured per clause over the corpus on
    every climate call whose clause names a mode, names no number and takes no
    switching verb: 911 agree, 1 disagrees, and the one is `תקררשמץ`.
    """
    settle = DIRECTION.settle_climate
    # No number in the sentence: the mode wins over the invented temperature.
    assert settle("climate_set_temperature", {"temperature": 23},
                  "תעביר את המזגן לרק מאוורר", "hvac_mode",
                  None, False) == "climate_set_hvac_mode"
    # A number in the sentence: it really is a temperature call.
    assert settle("climate_set_temperature", {"temperature": 23},
                  "שים את המזגן על 23 במהירות גבוהה", "fan_mode",
                  None, True) == "climate_set_temperature"
    # And a relative verb is something to set even with no number at all.
    assert settle("climate_set_temperature", {"temperature_step": -2},
                  "בחדר שינה חם מדי, תקרר", "hvac_mode",
                  None, False) == "climate_set_temperature"


def test_the_car_is_not_the_automatic_mode():
    """`hvac_mode` dropped the clipped `אוטו` for this reason and its sibling
    table kept it: it is also the Hebrew for *car*, and "ליד האוטו" - beside
    the car - is how this corpus says the garage.

    357 corpus clauses say a bare אוטו and **not one** wants auto as a mode.
    """
    assert SLOT.setting_from("שים את המזגן ליד האוטו על עשרים",
                             "fan_mode") is None
    assert SLOT.mode_slot("תעמיד את המזגן ליד האוטו על עשרים") is None
    # The word itself still reads, spelled out.
    assert SLOT.setting_from("שים את המאוורר על אוטומטי", "fan_mode") == "auto"


def test_a_television_is_switched_and_played_by_different_verbs():
    """Both families claim טלוויזיה - `FAMILY_NOUNS` lists it under `media` and
    under `switch` - so `family_named` reads the ambiguity and says nothing.
    The verb settles it, and the corpus is unanimous: 52 rows put something
    *on* the screen and 65 switch the set, with no overlap.
    """
    named = DIRECTION.settle_named
    assert named("media_select_source", "תכבה את הטלוויזיה בסלון") == \
        "switch_turn_off"
    assert named("media_play", "תדליק את הטלוויזיה בחדר השינה") == \
        "switch_turn_on"
    # Putting something on the screen is still the media player.
    assert named("media_select_source", "שים טלוויזיה בסלון") == \
        "media_select_source"


def test_a_request_that_names_nothing_to_play_asks_for_no_particular_thing():
    """`settle_transport` says this already for next-and-previous, and the
    widening it refuses - reading `שים` as a transport verb - is safe here
    because the eight-way read is not a verb list: a source, a level and a
    title are all ruled out above before the verb is reached.

    Measured through the function over the whole corpus, on every media-family
    clause it speaks about when reached from `music_play`: 2,125 agree, 9
    disagree, and all nine are the injected speech noise.
    """
    settle = DIRECTION.settle_media
    assert settle("music_play", "שים קצת מוזיקה", False, True) == "media_play"
    assert settle("music_play", "תשימי לנו משהו", False, True) == "media_play"
    # Not a plain request and this branch never opens: a `music_play` about a
    # clause naming a level or a source is left exactly as the model sent it.
    assert settle("music_play", "שים את הרמקול על 40 אחוז", True, False) == \
        "music_play"
    # Reached from a media behaviour, the eight-way read settles what it
    # always has, and `שים` still never gets that far - the level on that
    # clause is `NUMBER_SLOT`'s to fill, not this function's to promote on.
    assert settle("media_play", "שים יוטיוב בסלון", False, False) == \
        "media_select_source"
    assert settle("media_play", "תגביר את הווליום בסלון", False, False) == \
        "media_set_volume"
    assert SLOT.a_plain_request("שים קצת מוזיקה")
    assert not SLOT.a_plain_request("שים את תחנת הרדיו אקו 99")


def test_a_station_is_named_with_a_number_and_that_is_not_a_level():
    """"שים את השיר על שישים" is a volume, which is why a number rules a play
    out - but an Israeli radio station *is* named with one. אקו 99, כאן 88,
    and songs too: חורף 73. 47 corpus clauses turn on this and every one is
    `music_play`.

    Only a title whose number is written in digits, because a level said out
    loud is a word: without that, `extract_music` scraping "על הדשא על שישים"
    out of a volume request would take the level away with it.
    """
    level = SLOT.names_a_level
    assert level("ערבב את תחנת הרדיו אקו 99")
    assert not level("ערבב את תחנת הרדיו אקו 99", "אקו 99")
    assert not level("תשמיע את השיר חורף 73", "חורף 73")
    # A level inside the scraped title is still a level.
    assert level("תשים את השיר על הדשא על שישים", "על הדשא על שישים")


def test_whether_asks_which_when_the_adjective_is_plural():
    """`האם` asks yes-or-no about one thing and *which* about several, and
    Hebrew marks the difference on the adjective rather than on the question
    word. 171 corpus clauses turn on it, every one a `get_state`.
    """
    assert SLOT.state_filter("האם השקעים כבויים בסלון") == "off"
    assert SLOT.state_filter("האם המאווררים דולקים") == "on"
    # The singular still wants yes or no, which is the guard this began as.
    assert SLOT.state_filter("תבדוק אם האור בגן דולק") is None


def test_a_colour_temperature_outranks_the_colour_inside_it():
    """"לבן חם" is 2700K and the "לבן" in it is not a colour. The multi-word
    rule settles that inside one table and cannot see across two, so
    `SETTING_SLOT["light_turn_on"]` asked the lamp for warm white *and* plain
    white in the same call. 32 corpus clauses name both; all 32 want the
    temperature alone.
    """
    assert SLOT.setting_from("אני רוצה לבן חם בסלון", "color_temp_k") == "2700"
    assert SLOT.setting_from("אני רוצה לבן חם בסלון", "color_name") is None
    # A colour on its own is untouched.
    assert SLOT.setting_from("תדליק אור לבן בסלון", "color_name") == "white"


def test_a_setting_the_schema_types_as_an_integer_is_sent_as_one():
    """`SETTING_WORDS` is strings throughout so the tables read the same way,
    and `tools.json` declares `color_temp_k` as an integer with an enum of
    four. Written straight through, the call carried "2700" where gold and the
    schema carry 2700 - a wrong argument on the wire, and an exact-match
    failure on every row naming a colour temperature.
    """
    assert CONST.INTEGER_SETTING == frozenset(("color_temp_k",))
    schema = json.loads((COMPONENT / "tools.json").read_text(encoding="utf-8"))
    light = next(t for t in schema if t["name"] == "light_control")
    assert light["parameters"]["properties"]["color_temp_k"]["type"] == "integer"
    for slot in CONST.INTEGER_SETTING:
        for value in SLOT.SETTING_WORDS[slot].values():
            assert int(value)  # every one of them casts


def test_taking_something_off_a_list_still_names_the_list():
    """`מ` and `מה` are the prefix a *removal* takes and they were missing, so
    every sentence that took something off a list lost which list it meant.
    119 corpus calls, and `list` was the worst-read slot in the table by an
    order of magnitude because of it: 218 wrong, now 98.
    """
    assert SLOT.setting_from("תוריד תפוחים מהקניות", "list") == "shopping"
    assert SLOT.setting_from("תסיר לי לתאם פגישה מהמשימות", "list") == "todo"
    # And the prefix an addition takes still reads, as it always did.
    assert SLOT.setting_from("תוסיף חלב לקניות", "list") == "shopping"


def test_a_room_inside_a_device_name_is_part_of_the_device():
    """"דלת החניה" is the garage door and the חניה in it is part of what the
    device is called. "תסגור את דלת החניה בסלון" closes the garage door, and
    the sentence puts it in the living room: the resolver answered `parking`
    for 19 of the 230 corpus clauses naming one of these.

    Blanked rather than filtered, because `find_occurrences` returns one hit
    per value - on "דלת החניה ליד האוטו" the only `parking` hit is the one
    inside the device name, and dropping it loses the room the sentence names.
    """
    blank = SLOT.without_device_names
    assert "חניה" not in blank("תסגור את דלת החניה בסלון")
    assert "סלון" in blank("תסגור את דלת החניה בסלון")
    # The room named twice, once inside the device's name and once outside it.
    assert "האוטו" in blank("תפתח את דלת החניה ליד האוטו")
    # A sentence with no such phrase is returned unchanged.
    assert blank("תסגור את התריס בסלון") == "תסגור את התריס בסלון"


def test_wait_no_takes_the_order_back_like_every_other_retraction():
    """"רגע לא" - wait, no - is how this corpus says it on 183 rows, and it
    was the one member of the category missing. The retained half names the
    gold behaviour on 156 of the 165 it cuts, and the nine are `תעשה`, whose
    `light_turn_on` vocabulary this check cannot see rather than the rule
    failing.
    """
    after = CLAUSE.after_a_correction
    assert after("תדליק את הנורה... רגע לא תסגור את הנורה בסלון") == \
        "תסגור את הנורה בסלון"
    # Nothing after the retraction leaves the sentence whole, as it always did.
    assert after("כאילו אה רגע לא בבקשה תודה") == "כאילו אה רגע לא בבקשה תודה"

def test_broadcasting_over_the_speakers_is_a_notification_verb():
    """`תשדר` was missing from the notification openers, and the cost was not a
    lost benchmark row: `executor.execute` refuses `broadcast` outright when no
    message can be read out of the sentence, so "תשדר ברמקולים שהאוכל מוכן"
    announced nothing at all. 35 corpus clauses say it.

    Over every notify and broadcast clause, `extract_message` read 357 right
    and 14 wrong before this and its two companions below; after, **403 and
    7**, and all seven are noise inside the message body itself.
    """
    read = SLOT.extract_message
    assert read("תשדר ברמקולים שהאוכל מוכן") == "האוכל מוכן"
    assert read("תשמעי, יאללה תשדר ברמקולים שתרדו למטה תודה") == "תרדו למטה"
    # The openers that always worked, unchanged.
    assert read("תודיע בבית שהאוכל מוכן") == "האוכל מוכן"
    # No trigger at all is still no message, which the executor reads as
    # "do not send" rather than as an empty notification.
    assert read("תדליק את האור בסלון") is None


def test_a_final_letter_belongs_at_the_end_of_a_word_and_nowhere_else():
    """The one place in the project where the *output* is repaired instead of
    the input. Everywhere else `normalise` folds the five final forms away and
    the difference stops mattering; a notification is read by a person, and
    "האוכל םוכן" is what they would have seen.

    Both directions, because speech-to-text produces both: a final form with a
    letter after it, and a plain form with none.
    """
    read = SLOT.extract_message
    assert read("תכריז בכל הבית שהאוכל םוכן") == "האוכל מוכן"
    assert read("תודיע לכולם שהאוכל מוכנ") == "האוכל מוכן"
    # And the courtesy comes off even with a space dropped into it.
    assert read("תכריז בכל הבית שהכביסה מוכנה בבק שה") == "הכביסה מוכנה"
    assert read("תעדכן את כולם שהאוכל מוכן טודה") == "האוכל מוכן"


def test_a_broken_verb_still_names_its_direction():
    """`_hits_noisy`, and the reason it is not `_hits`.

    "כבהאת הנורה" lost a space and "קבי אור" swapped ק for כ - the same sound -
    so neither side of the toggle was found and the model's `flip` stood. This
    reads again with the boundaries given up, which is affordable only because
    the router has already chosen the family: a false match here costs a
    direction and cannot cost a device.

    Measured per clause where the strict reader is silent on both sides:
    **49 agree, 0 disagree**.
    """
    settle = DIRECTION.settle_toggle
    assert settle("light_toggle", "כבהאת הנורה בשרותים") == "light_turn_off"
    assert settle("light_toggle", "קבי אור באמבטייה") == "light_turn_off"
    assert settle("switch_toggle", "תדליכ את המפסק במבואה") == "switch_turn_on"
    # A real toggle is still a toggle.
    assert settle("light_toggle", "תהפוך את המצב") == "light_toggle"


def test_a_level_is_a_turn_on_because_nothing_else_can_hold_one():
    """"אור בחדר שינה בעשרים וחמישה אחוז" is a brightness, and `light.toggle`
    takes no brightness. The model answers `flip` because it read a light and
    no verb; the percentage was in the sentence all along.

    An upward step counts and a downward one does not, and the asymmetry is
    Hebrew rather than fitting: raising a thing that is off turns it on, while
    "תוריד את התקע" takes the plug down and means switch it off. Measured per
    clause where both sides are silent - level alone **467/297/0 agree and 0
    disagree** on light, fan and switch; with the upward step **868/297/21 and
    still 0**; with the downward step as well, 74 break.
    """
    settle = DIRECTION.settle_toggle
    assert settle("light_toggle", "אור בחדר שינה", names_a_level=True) == \
        "light_turn_on"
    assert settle("light_toggle", "תחזק את האורות במטבח הרבה יותר") == \
        "light_turn_on"
    # Down, with no level: the model's answer stands.
    assert settle("switch_toggle", "תוריד את התקע בחדר אוכל") == "switch_toggle"


def test_a_value_needs_no_unit_when_only_one_slot_could_hold_it():
    """"את הווליום על עשרים", "תפתח את הוילונות לשבעים וחמישה", "שים את המזגן
    על עשרים וארבע" - Hebrew introduces a target with a preposition and very
    often names no unit at all, and every reader here wanted one.

    On the five behaviours of `const.NUMBER_SLOT` and on a thermostat there is
    nothing else a bare number in range can be. Measured per clause over the
    corpus: the level slots go from **1,645 right, 1 wrong, 423 silent** to
    **1,942, 2 and 125**; the temperature from **699, 4 and 205** to **748, 5
    and 155**, firing on none of the corpus's `temperature_step` rows.
    """
    assert SLOT.level_from("את הווליום בחדר הילדים על עשרים") == 20
    assert SLOT.level_from("תפתח את הוילונות לשבעים וחמישה") == 75
    assert SLOT.temperature_from("שים את אינוורטר על עשרים וארבע") == 24
    # Two of them settle nothing, and the conjunction counts as the
    # preposition - without the ו stripped, only the first would be seen.
    assert SLOT.temperature_from("על 20 ועל 24") is None
    # A number wearing another slot's unit is not this one, and this reading is
    # the last one tried precisely so that it yields.
    assert SLOT.temperature_from("שים את התריסים על 30 אחוז") is None
    assert SLOT.temperature_from("תפעיל טיימר ל5 דקות") is None
    # One character past the preposition is enough: a blind opened to nothing.
    assert SLOT.level_from("תפתח את התריסים ל0") == 0


def test_a_degree_and_a_percent_are_not_the_same_size_of_step():
    """`_STEP_SCALE`. The unmarked step is 2 on a thermostat and 20 on a light,
    and reading one table for both is why `brightness_step_pct` and
    `volume_step_pct` were never filled at all.

    Three readings, none of them guessed: **עוד raises a small step and only a
    small one** (15 on both percent slots, unanimous); **the smaller adverb
    wins a compound** - "טיפה יותר חזק" is 10, not 30; and **חזק sizes a light
    and not a speaker**, where gold is 30 on twenty clauses and 20 on sixteen,
    so it stays silent rather than guessing.

    Measured per clause: temperature **389 agree, 1 disagree**, brightness
    **572 and 6**, volume **311 and 2**, every one of the nine speech noise
    inside the adverb.
    """
    size = DIRECTION.step_size
    assert size("תנמיך קצת", "temperature_step") == 1
    assert size("תנמיך קצת", "brightness_step_pct") == 10
    assert size("תנמיך עוד קצת", "brightness_step_pct") == 15
    assert size("תגביר בהרבה", "volume_step_pct") == 35
    assert size("תגביר חזק", "brightness_step_pct") == 30
    assert size("תגביר טיפה יותר חזק", "volume_step_pct") == 10
    # The coin-flip cell stays quiet rather than falling through to unmarked.
    assert size("תגביר חזק", "volume_step_pct") is None
    # No adverb at all is the unmarked step, and it needs a verb behind it.
    assert size("תנמיך את המזגן", "temperature_step") == 2
    assert size("אני רוצה יותר", "temperature_step") is None


def test_the_slot_the_sentence_names_settles_the_behaviour():
    """The recurring shape, twice more. A step is a thermostat setting and a
    level is a volume, and in both families the model answers with a behaviour
    that cannot hold what the sentence said - a mode, an off, a transport verb.
    `settle_steps` runs after these, so it has nothing to fill.

    Each exclusion is the rule rather than a caveat. On climate, "שים את הפן של
    המזגן על חזק" is a fan speed and חזק is also how a step is sized: without
    that guard 72 fan rows read as temperatures. On media, "תוריד מיוט" is an
    unmute whose verb is a downward step: without the mute branch running
    first, 31 unmutes become volume calls.

    Measured per clause over the corpus: climate **1,318 agree, 0 disagree**;
    media **721 and 1**, the one being "מיות".
    """
    climate = DIRECTION.settle_climate
    assert climate("climate_turn_on", {}, "תקרר את המזגן במטבח קצת",
                   None, None, False) == "climate_set_temperature"
    assert climate("climate_turn_off", {}, "בגן חם מדי, תנמיך",
                   None, None, False) == "climate_set_temperature"
    # A fan speed is not a step, however it is sized.
    assert climate("climate_turn_on", {}, "שים את הפן של המזגן על חזק",
                   "fan_mode", None, False) != "climate_set_temperature"

    media = DIRECTION.settle_media
    assert media("media_pause", "קצת פחות יותר חלש בגינה", False) == \
        "media_set_volume"
    assert media("media_play", "שים את הרמקול על שלושים אחוז", False,
                 names_a_value=True) == "media_set_volume"
    # The mute branch runs first and takes its own downward verb with it.
    assert media("media_play", "תוריד מיוט בגינה", False) == "media_mute"


def test_a_speaker_is_put_on_quiet_and_a_blind_is_stopped_with_the_same_word():
    """Two vocabulary gaps, and one word that means different things one tool
    apart.

    "תשים על שקט" is a mute on 38 clauses and `בשקט` could not reach it: the
    preposition is a separate token. `רדיו` is a source on all 41 of its
    `media_control` clauses - and what to play on 150 `music_play` rows, none
    of which reaches this function.

    `די` is a **stop** on all 33 cover and valve clauses and a **pause** on all
    111 media ones, which is why it is in `_STOPS` and deliberately not in
    `_MEDIA_STOPS`.
    """
    assert DIRECTION.settle_media("media_play", "תשים על שקט במטבח", False) == \
        "media_mute"
    assert DIRECTION.settle_media("media_next_track", "שים רדיו במטבח", False) == \
        "media_select_source"
    assert "די" in DIRECTION._STOPS
    assert "די" not in DIRECTION._MEDIA_STOPS


def test_a_placement_with_nothing_to_place_is_not_a_placement():
    """The mirror of the promotion above it. Over the corpus **684 of 684**
    cover and valve clauses naming a level are gold `place`, and of the 5,116
    naming none only 21 are - so a `cover_set_position` on a silent clause is
    the model reaching for the behaviour rather than the sentence asking.

    Handed back to the verb, and `settle` is asked from both ends so a clause
    whose verbs point two ways stays silent. **4,558 agree, 0 disagree.**
    """
    named = DIRECTION.settle_named
    assert named("cover_set_position", "הרם את הווילונות בחדר ילדים") == \
        "cover_open"
    assert named("valve_set_position", "תסגור את הברז במטבח") == "valve_close"
    # With a level it is a placement, which is the rule this mirrors.
    assert named("cover_set_position", "תפתח את הוילונות לשבעים וחמישה",
                 at_a_position=True) == "cover_set_position"
    # Verbs pointing two ways settle nothing.
    assert named("cover_set_position", "הוילונות בסלון") == "cover_set_position"


def test_a_scene_is_named_by_its_mode_when_it_is_not_named_by_its_noun():
    """`DISCRIMINATING_WEAK`, and its order is the rule rather than an accident
    of the literal: "תפעיל מצב שבת" says both markers and is a scene.

    `מצב` is what an Israeli calls a scene when they do not call it a scene -
    222 of the 234 silent scene clauses - and the run-verbs are the mirror, a
    script being a thing you activate. The ladder takes routine_run from **655
    agree, 0 disagree, 966 silent** to **1,074, 4 and 543**.

    A run-verb needs something to run: all 194 clauses this reads as a script
    name one, and "תפעיל את זה" names nothing, so the model's answer stands.
    Not one of the 194 says a demonstrative, so the guard is free.
    """
    siblings = ["scene_activate", "script_run", "automation_turn_on",
                "automation_turn_off", "button_press"]
    settle = DIRECTION.settle_action
    assert settle("script_run", siblings, "מצב שבת") == "scene_activate"
    assert settle("script_run", siblings, "תפעיל מצב שבת") == "scene_activate"
    assert settle("scene_activate", siblings, "תפעיל יציאה מהבית") == "script_run"
    assert settle("scene_activate", siblings, "תפעיל את זה") == "scene_activate"


def test_the_weather_question_names_its_own_day():
    """The slot existed and nothing read it, so "ירד גשם מחר" was answered with
    today's sky - a wrong answer rather than a missing one.

    Measured per clause over every `get_weather` call: **50 agree, 0 disagree,
    0 silent**, and quiet on all 81 clauses whose gold names no day. Today is
    spelled by saying nothing, so a zero is never written into the call.
    """
    day = SLOT.day_offset_from
    assert day("ירד גשם מחר") == 1
    assert day("ירד גשם מחרתיים") == 2
    assert day("מה הטמפרטורה היום") == 0
    assert day("מה מזג האוויר") is None


def test_one_wrong_letter_in_the_politeness_is_still_the_bare_question():
    """`names_a_clock` requires everything left over to be politeness, which is
    what keeps "מה השעה בניו יורק" out. One edit of tolerance on that closed
    set rescues "מה השעה כרגא" and "עוקיי מה השעה כרגע" without touching the
    guard: a city is not one edit from a word meaning please.

    Over the corpus this reads a clock on **182** genuine datetime rows against
    173 before, and on **zero** rows that are not one, unchanged.
    """
    assert ROUTER.names_a_clock("מה השעה כרגא")
    assert ROUTER.names_a_clock("עוקיי, מה השעה כרגע תודה")
    assert not ROUTER.names_a_clock("מה השעה בניו יורק")


def test_a_robot_mower_is_a_robot_vacuum_and_a_boiler_takes_a_temperature():
    """The last four of Home Assistant's own intents this integration could not
    reach. `lawn_mower` mirrors `vacuum`'s three services exactly under
    different names, and `water_heater.set_temperature` takes the same argument
    under the same name as `climate`'s.

    A fallback fires only when the primary domain found nothing in the room, so
    it can turn a refusal into an action and never one action into another.
    """
    fallback = CONST.FALLBACK_DOMAINS
    assert ("lawn_mower", "lawn_mower", "start_mowing") in fallback["vacuum_start"]
    assert ("lawn_mower", "lawn_mower", "dock") in fallback["vacuum_return_to_base"]
    assert ("water_heater", "water_heater", "set_temperature") in \
        fallback["climate_set_temperature"]


def test_a_phone_is_not_a_speaker_and_the_verb_says_which():
    """`notify_send` and `broadcast` take one argument each and it is the same
    one, so the model has nothing to go on but the verb - and it picks wrong,
    or refuses outright, on 14 of the benchmark's notification rows.

    The verbs separate them completely and in both directions: **199 agree, 0
    disagree** for the notification words and **102 and 0** for the broadcast
    ones. Not one corpus clause says both, so the order they are read in is
    free rather than load-bearing.
    """
    named = DIRECTION.settle_named
    assert named("broadcast", "תשלח הודעה לכולם שתרדו למטה") == "notify_send"
    assert named("broadcast", "תעדכן את כולם שיוצא מהבית") == "notify_send"
    assert named("notify_send", "תכריז בכל הבית שהאוכל מוכן") == "broadcast"
    assert named("notify_send", "תשדר ברמקולים שהאוכל מוכן") == "broadcast"
    # A verb neither list carries settles nothing: "תודיע" is how both are said.
    assert named("broadcast", "תודיע בבית שהאוכל מוכן") == "broadcast"


def test_hebrew_asks_a_yes_no_question_with_no_interrogative_at_all():
    """"החלון פתוח" is a question and, written down, is exactly a statement.
    Hebrew forms it with intonation, so `_QUESTION` - which looks for האם, איזה,
    מה - finds nothing, and fifteen benchmark rows reached the model with the
    whole catalogue in front of them and came back as `cover_control`,
    `lock_control`, `light_control`: a question about a window answered by
    opening it.

    What separates the readings is the **verb**, or rather its absence. פתוח is
    a passive participle and describes a window; תפתח is an imperative and
    opens it.

    Measured per clause over the corpus: **981 agree, 1 disagree**, and 80 of
    the 981 are clauses the question gate does not reach today.
    """
    asks = ROUTER.looks_like_question
    assert asks("החלון פתוח")
    assert asks("הדלת בסלון נעול")
    assert asks("התאורה בפינת העבודה דולק")
    # An imperative in the same sentence makes it an order again.
    assert not asks("תפתח את החלון")
    assert not asks("תנעל את הדלת בסלון")


def test_an_adjective_is_not_introduced_by_a_preposition():
    """The bare-token rule, and why it is a rule rather than an optimisation.

    `_tokens` strips the Hebrew clitics, and stripping the ל of the infinitive
    לפתוח leaves פתוח. Matching the adjectives through it read **1,051** plain
    orders as questions - "אתה יכול לפתוח את האורות" among them - against 981
    genuine ones.

    And a level is an instruction, not a question: nobody asks whether a blind
    is half open. Ten corpus clauses say "חצי פתוח" and all ten are covers.
    """
    asks = ROUTER.looks_like_question
    assert not asks("אתה יכול לפתוח את האורות בחדר האוכל")
    assert not asks("תקשיבי, אפשר לפתוח לי את המצלמות בפינת המטבח")
    assert not asks("את התריס בחדר שינה שלנו חצי פתוח")


def test_a_state_adjective_is_not_a_verb():
    """`tool_router._STATE_ADJECTIVES` and `slot_match.SETTING_WORDS["state"]`
    are two halves of one list kept in different modules, because `slot_match`
    imports the router and the dependency cannot run the other way.

    This is what keeps them in step. Every word the slot resolver knows has to
    be one the question gate recognises, or a sentence naming it would be read
    as an order; and no adjective may sit in the imperative list, or the gate
    would never fire.
    """
    known = {ROUTER._fold(w) for w in SLOT.SETTING_WORDS["state"]}
    assert known <= ROUTER._STATE_ADJECTIVES
    # The query family's own adjectives, which carry the singular forms.
    for word in ("דולק", "כבוי", "פתוח", "נעול", "סגורה"):
        assert ROUTER._fold(word) in ROUTER._STATE_ADJECTIVES
    # The query family is excluded from `_IMPERATIVES` precisely because its
    # "verbs" are these words, so the two lists barely meet. סגור is the one
    # word in both, and in Hebrew it really is both - "close!" and "closed",
    # spelled the same. A clause carrying it therefore reaches neither reading
    # and the model's own answer stands, which is the safe direction.
    assert (ROUTER._STATE_ADJECTIVES
            & {ROUTER._fold(v) for v in ROUTER._IMPERATIVES}) == {ROUTER._fold("סגור")}
    # And "leave it on" is an order, though no family claims the verb.
    assert not ROUTER.looks_like_question("תשאיר את האור דולק")


def test_asking_when_a_countdown_ends_does_not_start_one():
    """`מתי` was not in the interrogative table, so `settle_timer` never saw a
    question and answered "מתי הטיימר נגמר" by *starting* a countdown - the
    worst reading available, since the household asked how long was left.

    Measured over the whole corpus: 19 clauses say it and every one is gold
    `timer_control{query}`; **not one row that would actuate anything**. It is
    also on 122 off-topic rows - "מתי נולד רמברנדט" - and all 122 stay refused,
    because they name no device and so take no room bonus.
    """
    assert ROUTER.looks_like_question("מתי הטיימר נגמר")
    assert not ROUTER.looks_off_topic("מתי הטיימר נגמר")
    assert ROUTER.looks_off_topic("אה, מתי נולד רמברנדט בבקשה")
    settled = DIRECTION.settle_timer(
        "timer_start", {}, "נו מתי הטיימר נגמר", True,
        ROUTER.names_a_timer("נו מתי הטיימר נגמר"))
    assert settled == "timer_status"


# ---------------------------------------------------------------------------
# `repair`: one chain for the harness and the household, and what it does with
# a generation the engine could not finish. See the workshop tree's
# `eval/test_integration_logic.py`, where these are written.
# ---------------------------------------------------------------------------

REPAIR = load("repair")


def test_the_noun_settles_the_family_before_the_verb_settles_the_pair():
    """"תסגור את המצלמה" switches a camera off, not on.

    `direction.family_named` answers with the family's *anchor*, so it throws
    the direction away: run after `direction.settle`, a sentence saying סגור
    that the model answered with a cover came back as `camera_turn_on`. Eleven
    rows of the frozen benchmark, eight cameras and three fans, every one
    scored correct by the evaluation harness and wrong by the house.
    """
    for query, expected in (
            ("תסגור את המצלמות בגינה", "camera_turn_off"),
            ("תסגרי את הפן בשירותים", "fan_turn_off"),
            ("תפתח את המצלמה בסלון", "camera_turn_on"),
    ):
        tool, _ = REPAIR.settle("cover_control", {"action": "close"}, query)
        assert tool == expected, f"{query!r} -> {tool}"


def test_a_failed_generation_is_rebuilt_only_where_nothing_was_in_doubt():
    """An engine failure is not a refusal, and the scope is the argument.

    The router offered exactly one tool, so there was nothing to choose, and
    that tool cannot move anything. Measured per clause over the corpus behind
    the pre-inference gates: 2,252 agree, 2 disagree.
    """
    assert REPAIR.recover("האם המתגים מכובים בחדר המוגן") == {
        "name": "get_state", "arguments": {"domain": "switch", "state": "off"}}

    # One tool, but it moves something.
    assert REPAIR.recover("תדליק את האור בסלון") is None
    # A question this cannot type is a question it cannot answer: `domain` is
    # required and nothing may be invented for it.
    assert REPAIR.recover("מה קורה") is None
    # Off-topic sentences get a wide shortlist, which is the property that
    # makes the narrow scope safe rather than merely cautious.
    for query in ("רגע, מי כתב את הספר מלחמה ושלום תודה",
                  "אה, שלח הודעה לדני בוואטסאפ"):
        assert REPAIR.recover(query) is None, query


def test_the_verb_says_who_hears_a_message():
    """451 agree and none disagree over the corpus.

    The line is at the verb and not at the audience: "תשלח הודעה לכולם" is a
    notification *to everyone*, so reading the audience first is 31 clauses of
    330 wrong.
    """
    for query, expected in (
            ("תכריז בכל הבית שהאוכל מוכן", "broadcast"),
            ("תשדר ברמקולים שהכביסה מוכנה", "broadcast"),
            ("הכרז שתרדו למטה", "broadcast"),
            ("תודיע לכולם שהאוכל מוכן", "broadcast"),
            ("תשלח הודעה לכולם שתרדו למטה", "notify_send"),
            ("תעדכן את כולם שיוצא מהבית", "notify_send"),
            ("תודיע בבית שהאוכל מוכן", "notify_send"),
    ):
        for started_as in ("notify_send", "broadcast"):
            tool, _ = REPAIR.settle(started_as, {}, query)
            assert tool == expected, f"{query!r} from {started_as} -> {tool}"


def test_the_message_verbs_reach_the_plain_imperative():
    """`הכרז שהכביסה מוכנה` used to extract nothing, so it was refused."""
    assert SLOT.extract_message("הכרז שהכביסה מוכנה") == "הכביסה מוכנה"
    assert SLOT.extract_message("תעדכן את כולם שיוצא מהבית") == "יוצא מהבית"
    assert SLOT.extract_message("שלח הודעה לכולם שבואו לאכול") == "בואו לאכול"


def test_an_automation_is_named_after_what_it_does():
    """Which is why the family vote cannot read the one family stated outright.

    "האוטומציה תריסים בבוקר" names blinds and an automation, so
    `tool_router.family_named` sees two and answers None. 1,172 agree and 3
    disagree over the corpus.
    """
    for query, expected in (
            ("תשמע, תכבה את האוטומציה תריסים בבוקר", "automation_turn_off"),
            ("תפעיל את האוטומציה אורות בלילה", "automation_turn_on"),
            ("תפעיל את הסצנה ערב", "scene_activate"),
            ("תריץ את הסקריפט בוקר טוב", "script_run"),
    ):
        for started_as, args in (("cover_control", {"action": "open"}),
                                 ("light_control", {"action": "on"})):
            tool, _ = REPAIR.settle(started_as, dict(args), query)
            assert tool == expected, f"{query!r} from {started_as} -> {tool}"


def test_a_routine_verb_alone_does_not_pull_the_family_in():
    """`לחצי` is both "press" and "to a half", and a volume is not a button."""
    tool, _ = REPAIR.settle("media_control", {"action": "volume"},
                            "תכוון את הסאונד בכניסה לבית לחצי")
    assert tool == "media_set_volume", tool


def test_di_is_an_intensifier_and_not_an_order():
    """It sat in the media hints, so it was cut off and thrown away.

    "די, תעצור בפינת אוכל" came apart into "די" - which names nothing and is
    dropped - and "תעצור בפינת אוכל", which is a pause. Four sentences of
    44,043 cut differently without it and all four are that shape.
    """
    assert CLAUSE.split_clauses("די, תעצור בפינת אוכל תודה") == [
        "די, תעצור בפינת אוכל תודה"]
    for query in ("די, תעצור בפינת אוכל תודה",
                  "תקשיב, די, תעצור בחדר של הקטנה תודה"):
        tool, _ = REPAIR.settle("media_control", {"action": "pause"}, query)
        assert tool == "media_stop", f"{query!r} -> {tool}"
    # And alone it is still a pause: 111 clauses of "די עם את המוזיקה".
    for query in ("די עם את המוזיקה בחדר ילדים", "תעצור את המוזיקה בסלון"):
        tool, _ = REPAIR.settle("media_control", {"action": "pause"}, query)
        assert tool == "media_pause", f"{query!r} -> {tool}"


def test_the_one_room_the_fuzzy_pass_cannot_reach_spells_its_typo_out():
    """`ממד` is three characters and the fuzzy pass needs four.

    That floor is right - `חצר` and `חדר` are one edit apart and both are
    ordinary words - so the safe room is the one room a misspelling cannot
    reach, and the shape speech-to-text produces for it is listed instead. Over
    the whole corpus the room goes from 24,809 right / 12 wrong / 126 silent to
    24,840 / 12 / 95.
    """
    assert "ממדד" in AREA_MAP.AREA_ALIASES["safe_room"]
    assert AREA_MAP.slug_for_name("ממדד") == "safe_room"
    claimed = [slug for slug, forms in AREA_MAP.AREA_ALIASES.items()
               if "ממדד" in forms]
    assert claimed == ["safe_room"], claimed
