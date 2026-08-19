"""What ships, checked as data rather than by eye.

None of this needs Home Assistant. It exists because every one of these files
is read by something other than Python - HACS reads the manifest, the frontend
reads the translations, Home Assistant serves the brand images - so a mistake
in one is invisible until an installation fails. `quality_scale.yaml` shipped
for a while without parsing at all, which is what prompted the file.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
COMPONENT = REPO / "custom_components" / "needle_assist"


def _keys(node: dict, prefix: str = "") -> set[str]:
    """Every key in a nested mapping, as dotted paths."""
    found = set()
    for key, value in node.items():
        found.add(prefix + key)
        if isinstance(value, dict):
            found |= _keys(value, prefix + key + ".")
    return found


def test_every_json_file_in_the_repository_parses():
    for path in REPO.rglob("*.json"):
        if ".git" in path.parts:
            continue
        json.loads(path.read_text(encoding="utf-8"))


def test_the_manifest_says_what_hacs_and_hassfest_require():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["domain"] == "needle_assist"
    assert manifest["version"].count(".") == 2
    assert manifest["config_flow"] is True
    assert manifest["single_config_entry"] is True
    # The engine is vendored and its library is fetched at runtime, so a
    # requirement here would be a pip install on a machine that may have no
    # compiler. Empty is deliberate; see needle_engine/VENDOR.md.
    assert manifest["requirements"] == []


def test_every_translation_carries_every_string():
    base = _keys(json.loads((COMPONENT / "strings.json").read_text(encoding="utf-8")))
    for path in sorted((COMPONENT / "translations").glob("*.json")):
        have = _keys(json.loads(path.read_text(encoding="utf-8")))
        assert have == base, f"{path.name}: {base ^ have}"


def test_the_hebrew_strings_keep_their_direction_marks():
    """A left-to-right run inside a Hebrew sentence needs one, or the dot of
    ".cact" renders on the wrong side of the word. The marks are invisible, so
    the source spells them as escapes - including here, because a linter is
    right to object to an invisible character in a file it is reading.
    """
    lrm, rlm, backslash = chr(0x200E), chr(0x200F), chr(92)
    raw = (COMPONENT / "translations" / "he.json").read_text(encoding="utf-8")
    hebrew = json.loads(raw)

    assert lrm not in raw and rlm not in raw
    assert backslash + "u200e" in raw
    assert hebrew["config"]["error"]["weights_not_cact"].count(lrm) == 3
    assert hebrew["issues"]["weights_file_missing"]["title"].startswith(rlm)


def test_the_brand_images_are_all_eight_and_are_all_png():
    names = {
        f"{dark}{kind}{scale}.png"
        for dark in ("", "dark_")
        for kind in ("icon", "logo")
        for scale in ("", "@2x")
    }
    brand = COMPONENT / "brand"
    assert {p.name for p in brand.glob("*.png")} == names
    for path in brand.glob("*.png"):
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", path.name


def test_the_quality_scale_record_parses_and_says_something_about_every_rule():
    yaml = pytest.importorskip("yaml", reason="PyYAML ships with Home Assistant")
    doc = yaml.safe_load((COMPONENT / "quality_scale.yaml").read_text(encoding="utf-8"))
    rules = doc["rules"]
    assert len(rules) > 50
    for name, entry in rules.items():
        status = entry if isinstance(entry, str) else entry["status"]
        assert status in ("done", "exempt", "todo"), name
        # An exemption without a reason is a claim, not a record.
        if status == "exempt":
            assert entry["comment"].strip(), name
