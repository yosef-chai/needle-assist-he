"""Root conftest.

The component loader the Hebrew logic suite uses lives in
`component_loader.py`, not here: `tests/ha/conftest.py` shadows the plain
name `conftest` on pytest's path, so importing the helper by that name
reached the wrong file.
"""

from importlib.util import find_spec

# The Home Assistant suite under `tests/ha/` needs the harness plugin, and
# pytest only honours `pytest_plugins` in the root conftest - so it has to be
# declared here or not at all.
#
# Conditionally, though. The Hebrew logic suite is meant to run on a machine
# with no Home Assistant installed at all; that is what `component_loader`
# exists for, and it is how the fast job on CI gets an answer in thirty
# seconds instead of five minutes. Declared unconditionally, this line made
# that impossible - pytest fails to start rather than fails a test, so the
# suite that needs nothing could no longer run with nothing.
if find_spec("pytest_homeassistant_custom_component") is not None:
    pytest_plugins = "pytest_homeassistant_custom_component"
