"""Locate the native Needle engine, and put it somewhere it will survive.

The Python half of Needle is vendored in ``needle_engine/`` (see its
``VENDOR.md`` for why). The other half is ``libneedle.so`` - about 13 MB of
compiled inference engine, published per platform as a wheel alongside the
model on Hugging Face. This module is the only thing standing between the two.

Three decisions are worth stating, because none of them is the obvious one.

**The library lives under ``/config``, not under ``~/.cache``.**
Upstream's ``_library_path`` caches it in the home directory. In a Home
Assistant container that is ``/root``, which is *not* preserved across a core
update - so a household that updates Home Assistant on a bad network day would
find its voice assistant unable to start. Home Assistant's configuration
directory is the one location guaranteed to persist, so that is where it goes.

**The override is upstream's own, not a patch.**
``cactus-needle`` 2.0.5 checks ``NEEDLE_LIB_PATH`` before anything else in
``_library_path``. Setting it is what lets the vendored files stay byte-identical
to the package, which in turn lets the drift test be a plain byte comparison.

**The download uses ``urllib``, not ``huggingface_hub``.**
Upstream's ``fetch_library`` pulls in ``huggingface_hub``, whose own dependency
chain now includes ``hf-xet`` - a compiled Rust extension with no guarantee of a
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

import io
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Final, NamedTuple

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

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

# The engine release this component vendors. `fetch` is upstream code kept
# byte-identical and therefore untyped, so its values are named here - once,
# with their types - and nothing below reads it directly.
ENGINE_VERSION: Final[str] = fetch.ENGINE_VERSION
HF_REPO: Final[str] = fetch.HF_REPO

_TIMEOUT = 120


def cache_dir(config_path: str) -> Path:
    """Directory holding the engine for the version this component vendors."""
    return Path(config_path) / _CACHE_DIR / ENGINE_VERSION


def library_path(config_path: str) -> Path:
    """Where the native library is expected, whether or not it exists yet."""
    # The vendored engine is untyped upstream code; name the type here, at
    # the boundary, rather than letting Any leak into a Path.
    name: str = fetch._lib_name()
    return cache_dir(config_path) / name


class _Wheel(NamedTuple):
    """Where this machine's engine lives, and what to take out of it."""

    url: str
    member: str
    tag: str


def _wheel_for_this_machine() -> _Wheel:
    """The wheel to fetch and the file inside it, for this architecture.

    `_platform_tag` reads `platform.machine()` and sniffs /proc/self/maps for
    musl, so it identifies Alpine-based Home Assistant OS correctly without us
    having to guess. Reusing it rather than reimplementing it means the
    architecture logic has exactly one definition.
    """
    tag: str = fetch._platform_tag()
    wheel = f"cactus_needle-{ENGINE_VERSION}-py3-none-{tag}.whl"
    member: str = "needle/" + fetch._lib_name_for(tag)
    return _Wheel(_WHEEL_URL.format(repo=HF_REPO, wheel=wheel), member, tag)


def _by_hand(wheel: _Wheel, target: Path, err: object) -> str:
    """The message for a machine that could not fetch its own engine."""
    return (
        f"could not fetch the Needle engine for this platform ({wheel.tag}) "
        f"from {wheel.url}: {err}. Home Assistant needs it once; afterwards it "
        f"runs offline. To install it by hand, extract {wheel.member} from that "
        f"wheel to {target}."
    )


def _install(target: Path, archive_bytes: bytes, wheel: _Wheel) -> str:
    """Take the library out of a downloaded wheel and put it in place.

    Blocking: unzips and writes. Written beside the target and renamed, so an
    interrupted install can never leave a half-written .so for ctypes to load
    on the next start.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            payload = archive.read(wheel.member)
    except Exception as err:
        raise RuntimeError(_by_hand(wheel, target, err)) from err

    staged = target.with_suffix(target.suffix + ".part")
    staged.write_bytes(payload)
    os.replace(staged, target)
    _LOGGER.info("engine ready at %s (%d bytes)", target, len(payload))
    return str(target)


async def async_ensure_library(hass: HomeAssistant, config_path: str) -> str:
    """Return the path to the engine, fetching it once over Home Assistant's
    own HTTP session if it is missing.

    This is the only network call the integration ever makes, and it happens
    once per engine version on a fresh install; everything after it is local.
    Going through `async_get_clientsession` rather than through `urllib` means
    it inherits the instance's proxy settings, its SSL context and its
    connection pool, which is what the quality scale asks for and is also
    simply the right way to make an HTTP request inside Home Assistant.
    """
    target = library_path(config_path)
    if await hass.async_add_executor_job(target.exists):
        _LOGGER.debug("engine already present at %s", target)
        return str(target)

    wheel = _wheel_for_this_machine()
    _LOGGER.info("fetching Needle engine %s for %s", ENGINE_VERSION, wheel.tag)
    try:
        response = await async_get_clientsession(hass).get(
            wheel.url, timeout=aiohttp.ClientTimeout(total=_TIMEOUT)
        )
        response.raise_for_status()
        archive_bytes = await response.read()
    except Exception as err:
        raise RuntimeError(_by_hand(wheel, target, err)) from err

    return await hass.async_add_executor_job(
        _install, target, archive_bytes, wheel
    )


def ensure_library(config_path: str) -> str:
    """Return the path to the engine, downloading it once if it is missing.

    The fallback for anything running without a `hass` - the test suite, and a
    first utterance that somehow arrives before setup finished the download.
    Blocking on both counts, so Home Assistant calls it from an executor.
    :func:`async_ensure_library` is the one that runs on a live instance.
    """
    target = library_path(config_path)
    if target.exists():
        _LOGGER.debug("engine already present at %s", target)
        return str(target)

    wheel = _wheel_for_this_machine()
    _LOGGER.info("fetching Needle engine %s for %s", ENGINE_VERSION, wheel.tag)
    try:
        with urllib.request.urlopen(wheel.url, timeout=_TIMEOUT) as response:
            archive_bytes = response.read()
    except Exception as err:
        raise RuntimeError(_by_hand(wheel, target, err)) from err

    return _install(target, archive_bytes, wheel)


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


def engine_root(config_path: str) -> Path:
    """The one directory this component creates outside its own folder.

    Everything downloaded lives under it, one subdirectory per engine version,
    so uninstalling is a single tree to remove rather than a list of files to
    keep in step with :func:`library_path`.
    """
    return Path(config_path) / _CACHE_DIR


def remove_downloads(config_path: str) -> str | None:
    """Delete every engine this component ever fetched.

    Called when the config entry is removed, which - the manifest declares
    ``single_config_entry`` - means the last one. Nothing else can be using the
    directory, and leaving 13 MB per engine version behind after an uninstall
    is exactly the litter a household cannot be expected to find.

    Blocking: walks a directory and unlinks files. Returns the path that was
    removed, or ``None`` if there was nothing there.
    """
    root = engine_root(config_path)
    # A symlink is somebody deliberately putting the engine elsewhere - on a
    # bigger disk, usually. Removing the link is fine; following it out of the
    # configuration directory to delete whatever is at the other end is not,
    # and `shutil.rmtree` refuses to anyway.
    if root.is_symlink():
        root.unlink()
        return str(root)
    if not root.is_dir():
        return None
    shutil.rmtree(root)
    return str(root)


def unbind_library() -> None:
    """Forget the path :func:`bind_library` published.

    The variable is read every time the vendored engine loads its library, so
    leaving it pointing at a file that has just been deleted turns a later
    re-install into a confusing failure rather than a fresh download.
    """
    os.environ.pop("NEEDLE_LIB_PATH", None)
