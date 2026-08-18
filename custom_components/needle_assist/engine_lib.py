"""Locate the native Needle engine, and put it somewhere it will survive.

The Python half of Needle is vendored in ``needle_engine/`` (see its
``VENDOR.md`` for why). The other half is ``libneedle.so`` — about 13 MB of
compiled inference engine, published per platform as a wheel alongside the
model on Hugging Face. This module is the only thing standing between the two.

Three decisions are worth stating, because none of them is the obvious one.

**The library lives under ``/config``, not under ``~/.cache``.**
Upstream's ``_library_path`` caches it in the home directory. In a Home
Assistant container that is ``/root``, which is *not* preserved across a core
update — so a household that updates Home Assistant on a bad network day would
find its voice assistant unable to start. Home Assistant's configuration
directory is the one location guaranteed to persist, so that is where it goes.

**The override is upstream's own, not a patch.**
``cactus-needle`` 2.0.5 checks ``NEEDLE_LIB_PATH`` before anything else in
``_library_path``. Setting it is what lets the vendored files stay byte-identical
to the package, which in turn lets the drift test be a plain byte comparison.

**The download uses ``urllib``, not ``huggingface_hub``.**
Upstream's ``fetch_library`` pulls in ``huggingface_hub``, whose own dependency
chain now includes ``hf-xet`` — a compiled Rust extension with no guarantee of a
wheel on the architecture and libc combination a given Home Assistant runs on.
Depending on it to fetch one file over HTTPS would reintroduce, in miniature,
exactly the packaging problem that vendoring solved. The resolve URL below is
the same public endpoint ``hf_hub_download`` would call.

Nothing here reaches the network if the library is already in place, which is
the normal case after first setup. An installation that must stay offline from
the start can have the file copied in by hand: put the right
``libneedle.{so,dll,dylib}`` at the path :func:`library_path` reports and this
module will simply find it.
"""

from __future__ import annotations

import logging
import os
import urllib.request
import zipfile
from pathlib import Path

from .needle_engine.agent import fetch

_LOGGER = logging.getLogger(__name__)

# Where the engine wheels are published. Same host, repo and path that
# `huggingface_hub` resolves to; `resolve/main` is the raw-file endpoint.
_WHEEL_URL = "https://huggingface.co/{repo}/resolve/main/python/{wheel}"

# One subdirectory per engine version, because a `.cact` archive is only
# loadable by the engine version it was exported with. Keeping old versions
# addressable makes a rollback a matter of pointing at the other directory
# rather than re-downloading.
_CACHE_DIR = "needle_assist_engine"

_TIMEOUT = 120


def cache_dir(config_path: str) -> Path:
    """Directory holding the engine for the version this component vendors."""
    return Path(config_path) / _CACHE_DIR / fetch.ENGINE_VERSION


def library_path(config_path: str) -> Path:
    """Where the native library is expected, whether or not it exists yet."""
    return cache_dir(config_path) / fetch._lib_name()


def ensure_library(config_path: str) -> str:
    """Return the path to the engine, downloading it once if it is missing.

    Blocking: does file I/O and possibly an HTTPS download, so Home Assistant
    must call this from an executor, never on the event loop.
    """
    target = library_path(config_path)
    if target.exists():
        _LOGGER.debug("engine already present at %s", target)
        return str(target)

    # `_platform_tag` reads `platform.machine()` and sniffs /proc/self/maps for
    # musl, so it identifies Alpine-based Home Assistant OS correctly without
    # us having to guess. Reusing it rather than reimplementing it means the
    # architecture logic has exactly one definition.
    tag = fetch._platform_tag()
    wheel = "cactus_needle-{}-py3-none-{}.whl".format(fetch.ENGINE_VERSION, tag)
    url = _WHEEL_URL.format(repo=fetch.HF_REPO, wheel=wheel)
    member = "needle/" + fetch._lib_name_for(tag)

    target.parent.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("fetching Needle engine %s for %s", fetch.ENGINE_VERSION, tag)

    archive = target.parent / (wheel + ".part")
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as response:
            archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as zf:
            payload = zf.read(member)
    except Exception as err:
        raise RuntimeError(
            f"could not fetch the Needle engine for this platform ({tag}) from "
            f"{url}: {err}. Home Assistant needs it once; afterwards it runs "
            f"offline. To install it by hand, extract {member} from that wheel "
            f"to {target}."
        ) from err
    finally:
        archive.unlink(missing_ok=True)

    # Write beside the target and rename, so an interrupted download can never
    # leave a half-written .so that ctypes would try to load on next start.
    staged = target.with_suffix(target.suffix + ".part")
    staged.write_bytes(payload)
    os.replace(staged, target)
    _LOGGER.info("engine ready at %s (%d bytes)", target, len(payload))
    return str(target)


def bind_library(config_path: str) -> str:
    """Make the vendored loader use our copy, and return its path.

    ``NEEDLE_LIB_PATH`` is process-wide. That is acceptable because this
    component is the only thing in a Home Assistant process that binds Needle;
    it is documented here so the coupling is not a surprise if that ever stops
    being true.
    """
    path = ensure_library(config_path)
    os.environ["NEEDLE_LIB_PATH"] = path
    return path
