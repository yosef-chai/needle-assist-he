"""Import the integration's Hebrew logic without a running Home Assistant.

``needle_assist/__init__.py`` imports Home Assistant, so importing the package
the ordinary way fails on a bare CI runner. Registering a stand-in parent
package skips that one file while leaving the relative imports *inside* the
modules working - so these tests exercise the shipped files byte for byte,
rather than a copy kept in step by hand.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

COMPONENT = (pathlib.Path(__file__).resolve().parents[1]
             / "custom_components" / "needle_assist")

_PACKAGE = "needle_assist"


def load(name: str):
    """Import ``needle_assist.<name>`` without running the package __init__."""
    if _PACKAGE not in sys.modules:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(COMPONENT)]
        sys.modules[_PACKAGE] = package
    return importlib.import_module(f"{_PACKAGE}.{name}")
