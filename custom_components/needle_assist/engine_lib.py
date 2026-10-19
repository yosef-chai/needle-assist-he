"""Locate the native Needle engine, and put it somewhere it will survive.

The Python half of Needle is vendored in ``needle_engine/`` (see its
``VENDOR.md`` for why). The other half is ``libneedle.so`` - about 13 MB of
compiled inference engine, published per platform as a wheel alongside the
model on Hugging Face. This module is the only thing standing between the two.

Four decisions are worth stating, because none of them is the obvious one.

**What is loaded is what was released, byte for byte.**
The library is loaded into the Home Assistant process with ``ctypes``, so it
runs with every permission Home Assistant has. The wheel is fetched from one
fixed Hugging Face commit rather than from ``main``, and both the wheel and the
library inside it are checked against SHA-256 digests kept in this file before
anything is written. The library already on disk is checked again before every
load, so a file swapped or truncated after install is replaced rather than run.
``main`` is not a release: upstream has replaced the binaries of an already
published engine version in place more than once.

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
module will find it - provided it is the released file, which the digest
decides.
"""

from __future__ import annotations

import hashlib
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
# `huggingface_hub` resolves to, at one fixed commit: the one that last wrote
# the 2.0.2 wheels (2026-08-14). `resolve/main` would be whatever the repository
# holds on the day of install, and it has replaced published binaries since.
_REVISION: Final = "17a803d95928ba33d3e9a0160e024d9565b5c3f2"
_WHEEL_URL = "https://huggingface.co/{repo}/resolve/{revision}/python/{wheel}"

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


class _Pin(NamedTuple):
    """SHA-256 of one platform's wheel, and of the library inside it."""

    wheel: str
    library: str


# The engine version the digests below were taken from. A vendored engine bump
# without new digests fails `tests/test_packaging.py` rather than shipping a
# component that refuses every download.
PINNED_VERSION: Final = "2.0.2"

# Computed from the wheels at `_REVISION`; the wheel digests are also the LFS
# object ids Hugging Face reports for those files. A platform missing from this
# table gets no engine at all rather than an unchecked one.
_PINNED: Final[dict[str, _Pin]] = {
    "manylinux2014_x86_64": _Pin(
        "ba42809fadb7b2fe1aebc0af6c7560ddbe5ddea84b7875d12c0750770eb39f14",
        "dc9c89ec3038ad60b0e6e8162a9c7d2053d251e4846f40ef9b1db061556314a5"),
    "manylinux2014_aarch64": _Pin(
        "002aa4653a2152725ad040289915912dd62afedfbb485b7ffd73cd9ce7abe82a",
        "119e887de48672a69600344c24b0ea0447895bb083c53698a5b44c84943f7f94"),
    "musllinux_1_2_x86_64": _Pin(
        "c48873e3c2a19d4c421b48b2bcb6ae7ff7001d2ae705f6cb3508e8784c4a994b",
        "33860917572dfa0b74fd08e4be5da8296bc7075a10d31bcd521a8ff0c580cb43"),
    "musllinux_1_2_aarch64": _Pin(
        "7135f89d4d1586e3034c3f59dcbc76a3e4e5b3b3d142f5c7fc54c65f5adaf732",
        "56bbe718fe9b3597a41ce450df103b20cb119b8d9a3f8496b87819a182f14295"),
    "macosx_11_0_arm64": _Pin(
        "5ef0caf9b1bbef91e3e6c3eef180440a4d53e4a06dda7b409247efcc4cb55ad8",
        "f8ff35e0ceea5812f3f90bddbb40ca3a8d03af07c84555e116bf0096fa994afd"),
    "macosx_11_0_x86_64": _Pin(
        "ceb0cd52abf3a20c0e5196c2e09a55532fc1bf6a931aca02d133c4bf1883b7e2",
        "ba057bc297de4f06e6c4fd8e7a477da38bdbdd613a95342c8eb858df4511237f"),
    "win_amd64": _Pin(
        "718955a433f4c16c193266a37627dd34167389fa55d0c385a2a38064a5691403",
        "ec6f681d86e7a31ee81d02930b8bc5abf7b1a8da25b6c24dd3188fd62e950f13"),
    "win_arm64": _Pin(
        "a4b13c0959c7dac5e789cd9da1a09a76fa6b19be570fa17556e869300872bd6e",
        "f79ae4297b4cea7a1b85f2827f96c23bf1f7cd12d455067d6c4dcc1b27e82c9b"),
}


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
    pin: _Pin


def _wheel_for_this_machine() -> _Wheel:
    """The wheel to fetch and the file inside it, for this architecture.

    `_platform_tag` reads `platform.machine()` and sniffs /proc/self/maps for
    musl, so it identifies Alpine-based Home Assistant OS correctly without us
    having to guess. Reusing it rather than reimplementing it means the
    architecture logic has exactly one definition.
    """
    tag: str = fetch._platform_tag()
    if (pin := _PINNED.get(tag)) is None:
        raise RuntimeError(
            f"no verified Needle engine {ENGINE_VERSION} for this platform "
            f"({tag}); supported: {', '.join(sorted(_PINNED))}")
    wheel = f"cactus_needle-{ENGINE_VERSION}-py3-none-{tag}.whl"
    member: str = "needle/" + fetch._lib_name_for(tag)
    url = _WHEEL_URL.format(repo=HF_REPO, revision=_REVISION, wheel=wheel)
    return _Wheel(url, member, tag, pin)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _present(target: Path) -> bool:
    """Is the released library already at ``target``?

    Blocking: reads about 14 MB. A file that is there and does not match is
    removed, so the caller fetches the released one in its place - a library
    that differs from the release is never handed to ``ctypes``, whether it
    was swapped, truncated or copied in by hand from the wrong wheel.
    """
    if not target.is_file():
        return False
    wheel = _wheel_for_this_machine()
    if _sha256(target.read_bytes()) == wheel.pin.library:
        return True
    _LOGGER.warning(
        "%s does not match the released Needle engine %s for %s; replacing it",
        target, ENGINE_VERSION, wheel.tag)
    target.unlink()
    return False


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

    Blocking: unzips and writes. Both digests are checked before anything
    touches the disk, and the file is written beside the target and renamed,
    so neither a tampered download nor an interrupted install can leave a
    library for ctypes to load on the next start.
    """
    if (got := _sha256(archive_bytes)) != wheel.pin.wheel:
        raise RuntimeError(_by_hand(
            wheel, target,
            f"the wheel's SHA-256 is {got}, expected {wheel.pin.wheel}"))
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            payload = archive.read(wheel.member)
    except Exception as err:
        raise RuntimeError(_by_hand(wheel, target, err)) from err
    if (got := _sha256(payload)) != wheel.pin.library:
        raise RuntimeError(_by_hand(
            wheel, target,
            f"the library's SHA-256 is {got}, expected {wheel.pin.library}"))

    target.parent.mkdir(parents=True, exist_ok=True)

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
    if await hass.async_add_executor_job(_present, target):
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
    if _present(target):
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
