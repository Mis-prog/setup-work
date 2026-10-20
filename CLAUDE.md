# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`setup-work` bootstraps a developer workstation inside a closed corporate network, in three sequential stages:
**infrastructure (pip/npm registries, `.venv`, Node.js, JARs) → coding-agent harnesses (internal Git forks + their `setup.sh`, GigaChat backend) → optional local LLM (engine + model + OpenAI-compatible server)**. It also collects agent token-usage metrics and uploads them to HDFS.

Environment constraints that shape the code: no public internet (artifacts come from HDFS/NFS/HTTPS-internal, Python packages only from the corporate PyPI), no sudo, no shell-profile edits, the system Python (3.11+, department uses 3.12.1) is reused rather than installed. User-facing text, docstrings and comments are in **Russian** — keep new messages/hints in Russian to match.

## Commands

The Makefile is the primary UI; every target is a thin wrapper over `.venv/bin/python -m setup_work <subcommand>`. Make variables (`MODEL=`, `ENGINE=`, `DRY_RUN=1`, `JSON=1`, …) are exported as `SETUP_WORK_*` env vars and read by Typer `envvar=` options — they are never interpolated into shell commands. When adding a CLI option that Make should drive, add both the `export SETUP_WORK_X = $(X)` line in the Makefile and `envvar="SETUP_WORK_X"` in `cli.py`.

```bash
make sync                 # create/refresh .venv and install deps (incl. dev) via scripts/bootstrap.sh --sync-only
make test                 # .venv/bin/python -m pytest
make lint                 # ruff check . && ruff format --check .
make format               # ruff format . && ruff check --fix .
make check                # lint + test + config-validate
make config-validate      # validate built-in YAML; CI also runs it with SETUP_WORK_USER_CONFIG_DIR=examples/corp
shellcheck -s sh scripts/bootstrap.sh   # CI lints the POSIX-sh stage-0 script

.venv/bin/python -m pytest tests/test_recommend.py::test_name   # single test
```

Read-only / dry-run smoke commands (as in CI): `make doctor`, `make recommend TASK=coding HARDWARE=examples/hardware/4x-a100-80gb.json`, `make bootstrap DRY_RUN=1`, `make serve MODEL=qwen3.8-27b:full ENGINE=vllm PLAN=1 HARDWARE=examples/hardware/4x-a100-80gb.json`. `HARDWARE=<json>` (→ `SETUP_WORK_HARDWARE_FILE`) substitutes a saved hardware profile for live detection; profiles live in `examples/hardware/`.

Tests need no GPU, HDFS or internal repos; server tests need to bind a local port (`tests/fake_server.py` is a stdlib OpenAI-compatible stub). Ruff: line length 100, py311 target, `S`/`B`/`UP`/`RUF` rules enabled.

## Architecture

**Two-stage startup.** `scripts/bootstrap.sh` (POSIX sh, stage 0) runs before any Python dependency exists: reads `bootstrap.env` as `KEY=VALUE` (never executed; secret-looking keys are rejected — secrets go in auth files), writes `pip.conf`/`uv.toml`/`npmrc` into `$SETUP_WORK_HOME/registries/`, creates `.venv`, repairs pip via `ensurepip`, verifies the corporate index, then `pip install -e .[dev]`. `make setup` = `bootstrap.sh --setup`, which hands off to `python -m setup_work setup` → `pipeline.run`. `.venv/.setup-work-synced` is the Make stamp that re-syncs when `pyproject.toml`, `bootstrap.sh` or `Makefile` change.

**`scripts/setup-lite.sh`** is a separate, self-contained POSIX-sh reimplementation of the setup flow (registries, Node.js from npm registry, agents via `setup.sh`, and on GPU machines llama.cpp / vLLM venv / model artifacts from HDFS, NFS or HTTPS — no recommend/serve) with settings embedded in its header block; it must not depend on Python or other repo files. Keep its behaviour in sync with `harness.py`/`node_registry.py`/`artifacts.py`/`bootstrap.repair_venv` when those change; tests in `tests/test_setup_lite.py` rewrite the settings block of a copy.

**Pipeline** (`pipeline.py`): `infrastructure` → harnesses → `local_model`, each stage's result saved to `<state_dir>/setup-state.json` so reruns resume/reuse. Local model stage is skipped when no GPU.

**Plan-then-execute.** Mutating work is expressed as `steps.Step` lists (`command` argv, `download`, `extract` with path-safety checks, `link`, `manual`) and run via `steps.execute` through `runner.Runner`, which centralises `--dry-run`, `--yes`, `--json`, confirmation prompts, secret masking and `events.jsonl` logging. No arbitrary shell strings — commands are argv lists (hence ruff `S603`/`S607` ignored).

**Configuration** (`config.py`, pydantic): a built-in layer (`SETUP_WORK_CONFIG_DIR`, else repo `config/`, else `setup_work/_config` packaged via hatch `force-include`) merged with a user layer (`SETUP_WORK_USER_CONFIG_DIR` → `$SETUP_WORK_HOME/config` → XDG `~/.config/setup-work`). Settings merge by key, but entries in the model and harness catalogs are replaced whole. Files: `defaults.yaml` (tool pins, setup options), `models.yaml` (catalog + architecture params for memory estimates), `harnesses.yaml` (Git recipes), `corporate.yaml` (artifact sources, HDFS, metrics), `bootstrap.env`. `CHANGE-ME` placeholders must produce errors naming the file/field (see `tests/test_localization.py`). `examples/corp/` is the filled corporate template.

**Directories**: `$SETUP_WORK_HOME` (persistent, often NFS: registries, harnesses, config, state, `env.sh`) vs `$SETUP_WORK_RUNTIME_DIR` (container-local tmp by default: model weights, engines, downloads, server state, Git credential-cache socket — must not be on NFS).

**Model selection & serving**: `hardware.py` (detection that degrades to unknown fields, never crashes) → `memory.py` (weights/KV-cache byte estimates) → `engines.py` (llama.cpp / vLLM adapters: compat, fit, TP/offload, launch command) → `models.py` (rank candidates; rejection reasons kept for `recommend --explain`; deterministic ordering) → `inference.py` (`plan_launch`, `ServerManager` start/stop/ready/logs; signals only processes whose PID *and* start time match). Artifacts (`artifacts.py`, `hub.py`) come from HDFS (cluster client + Kerberos), local/NFS paths or HTTPS, unpacked via temp dir with a `.setup-work-artifact.json` reuse marker; `zip-parts` handles `<name>-1-of-N.zip` sets.

**Harnesses** (`harness.py`): clone internal Git repo into `<data_dir>/harnesses/<name>`, run its `setup.sh` with a TTY and `SETUP_WORK_BACKEND`, `SETUP_WORK_HARNESS_*`, `SETUP_WORK_METRICS_*` env; record installed commit; on rerun check for new revision and ask before updating; `install.start` restarts an already-installed API without reinstall. Interactive Git auth uses an in-memory credential cache (8h); `GIT_CONFIG_GLOBAL` points into `$SETUP_WORK_HOME`.

**Metrics** (`metrics.py`, `metrics_proxy.py`): usage recorded per real API request into SQLite queue `<state_dir>/metrics/usage.sqlite3` (dedup by `request_id`), uploaded as JSONL batches to `hdfs_uri/dt=…/device_id=…/` (write hidden `.pending`, then rename; retries are idempotent). The proxy forwards Chat Completions/Responses/Messages (incl. SSE), injects `stream_options.include_usage`, listens only on 127.0.0.1, and never logs prompt/response text. Token semantics: cache/reasoning counts are nested inside input/output, not additive.

**Exit codes** (`errors.py`) are a scripting contract: 0 ok, 1 failure, 2 usage/config, 3 missing dependency, 4 unsupported, 5 aborted, 6 not found, 7 check failed. Raise the matching `SetupWorkError` subclass with a concrete `hint=`.

## Conventions

- Releases: a separate commit `Release X.Y.Z: <summary>` bumping the version in both `pyproject.toml` and `src/setup_work/__init__.py`.
- `tests/conftest.py` autouse fixture isolates every test (strips `SETUP_WORK_*` env, temp XDG/bin/runtime/user-config dirs, builtin config = repo `defaults.yaml` + `tests/fixtures/{models,harnesses}.yaml` + empty `corporate.yaml`). Hardware fixtures (`hw_a100x4`, `hw_cpu`, …) load `examples/hardware/*.json`.
- CI (`.github/workflows/ci.yml`) runs outside the corporate network with `PIP_INDEX_URL=https://pypi.org/simple`, on Ubuntu + macOS × Python 3.11–3.13.
- When README behaviour changes (setup flow, config keys, metrics format), update `README.md` (Russian) alongside the code.
