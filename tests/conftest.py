# -*- coding: utf-8 -*-
"""Root conftest.

The component loader the Hebrew logic suite uses lives in
`component_loader.py`, not here: `tests/ha/conftest.py` shadows the plain
name `conftest` on pytest's path, so importing the helper by that name
reached the wrong file.
"""

# The Home Assistant suite under `tests/ha/` needs the harness plugin, and
# pytest only honours `pytest_plugins` in the root conftest. Declaring it here
# costs the logic tests nothing: nothing boots until a test asks for `hass`.
pytest_plugins = "pytest_homeassistant_custom_component"
