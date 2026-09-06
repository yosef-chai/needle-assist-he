# `needle_engine/` - vendored from `cactus-needle` 2.0.5

The four Python files in this directory are **byte-identical copies** of the
`needle` package from `cactus-needle==2.0.5`:

| here | upstream |
|---|---|
| `__init__.py` | `needle/__init__.py` |
| `agent/__init__.py` | `needle/agent/__init__.py` |
| `agent/tools.py` | `needle/agent/tools.py` |
| `agent/fetch.py` | `needle/agent/fetch.py` |

`eval/test_integration_logic.py::test_vendored_engine_matches_upstream` compares
them against the installed package and fails on any drift, so this copy cannot
quietly fall behind. Nothing here is edited - see *Why not just edit it* below.

## Why the integration does not simply depend on the package

`manifest.json` used to declare `requirements: ["cactus-needle==2.0.5"]`, which
is the normal way for a Home Assistant integration to get a library. It is the
wrong way here, because the package declares:

    huggingface_hub, numpy, jax, jaxlib, flax, optax, sentencepiece

That is roughly **314 MB**, of which `jaxlib` alone is 240 MB. Home Assistant
would install all of it into its own container on first setup.

None of it is used at inference time. That was measured, not assumed:
`import needle` was run with `sys.modules` snapshotted either side, and **no**
heavy module is pulled in. JAX exists in the dependency list for
`needle.model.finetune`, the training path, which never runs inside Home
Assistant - training happens on a workstation and ships a `.cact` file.

The inference path is 395 lines of pure standard library (`ctypes`, `json`,
`os`, `sys`, `enum`, `inspect`, `re`, `types`, `typing`, `zipfile`) plus a
native `libneedle.so`.

There is also a correctness argument, not just a size one. `jaxlib` publishes no
`musllinux` wheels. On an Alpine-based Home Assistant OS, pip cannot satisfy the
requirement at all and would try to build JAX from source - so the integration
would not merely be bloated, it would **fail to install**.

With this directory in place `manifest.json` declares no requirements at all and
the component runs on any platform Home Assistant runs on.

## Why not just edit it

Two things upstream does are inconvenient here:

1. `fetch.fetch_library` downloads the native library with `huggingface_hub`,
   which is the dependency we are trying to avoid.
2. `_library_path` caches it under `~/.cache`, which in a Home Assistant
   container is not persistent across core updates.

Neither is patched. Upstream 2.0.5 added a `NEEDLE_LIB_PATH` environment
override at the top of `_library_path`, and `engine_lib.py` in the parent
directory uses exactly that: it places the library under `/config` - which does
survive updates - downloads it with `urllib` if it is missing, and points the
override at it. `fetch_library` is therefore never called and its
`huggingface_hub` import never executes.

Keeping these files untouched is what makes the drift test a plain byte
comparison. A patched copy would need a diff-aware test, and the patch would be
re-applied by hand on every upgrade, which is how vendored code rots.

## Upgrading

1. `pip install --upgrade cactus-needle`
2. Copy the four files over again from the installed package.
3. Run `pytest eval/test_integration_logic.py`. The drift test then passes by
   construction; what matters is that the other tests still do.
4. Check whether `ENGINE_VERSION` in `agent/fetch.py` changed. If it did, the
   native library and every exported `.cact` have to be rebuilt - the archive
   format is tied to the engine version, and `Needle._bind` raises a fairly
   clear error when they disagree.

## Why this is still 2.0.5, checked against 2.0.12

Measured 2026-09-06, because the package is now eight releases ahead.

`needle/model/export.py` is **byte-identical** between the two, so `needle
build` writes the same archive and the `.cact` format has not moved. What did
move is the native library the archive is read by: `ENGINE_VERSION` is a map
now, and generation 2 points at `2.0.4` rather than `2.0.2`.

So the whole question is whether the newer engine answers differently, and it
does not. The shipping v11 `.cact` was loaded under both and asked 120 rows of
the frozen benchmark:

| | engine 2.0.2 | engine 2.0.4 |
|---|---:|---:|
| answered | 118 | 118 |
| engine failures | 2 | 2 |
| **answers that differ** | | **0** |

Upstream's own changes in that range are real but not ours: the constraint
preservation on optional fields and the integer-enum inference are in
`agent/tools.py`'s schema *builder*, which turns decorated Python functions
into schemas. This component passes the raw JSON schema out of `tools.json`
and never calls it.

Against zero measurable gain, the upgrade costs a re-vendor of three changed
files (297 lines in `__init__.py` alone, where `complete` gained telemetry and
a generation dispatch that `needle_compat.safe_complete` re-implements), a new
library download for every household, and a release. Left at 2.0.5.

One change there **would** matter and is a reason to look again before any
retrain: `render_example` moved to `ensure_ascii=False` and put newlines around
the think block. That changes the supervised target - a Hebrew argument value
stops costing six ASCII characters per letter - and it is the defect behind the
truncation study in `repair.recover`. `finetune_he.load_shaped` asserts
byte-identity with the installed renderer, so the assert is what will say so.
