# Bootstrapping a fresh remote GPU server without sudo

Scenario: a shared Linux GPU server (Ubuntu 22.04, NVIDIA driver already installed by
the admins), an unprivileged account, home directory on a small disk and a large
scratch volume at `/data/$USER`. Python 3.12 is installed system-wide, there is no Node.js, and
you cannot run `sudo`.

> Status: the commands below are exercised by the test-suite with mocked hardware and
> in dry-run mode. Running them on real NVIDIA hardware has **not** been verified by the
> authors yet — please report results.

## 0. Requirements on the server

- `git`, `make`, `tar`, and `curl` *or* `wget` (present on almost every server).
- Access to the corporate storage (HDFS client + Kerberos ticket, NFS or internal HTTPS)
  and the corporate PyPI / npm registries. Nothing is fetched from public hosts.
- A team config directory (`SETUP_WORK_TEAM_CONFIG_DIR`) with `corporate.yaml` mapping
  uv, Node.js, Git LFS, the engines and the model weights, and `bootstrap.env` with the
  registries and the uv source — see [corp/](corp/README.md).

## 1. Clone and point heavy data at the big volume

```bash
git clone https://<bitbucket-host>/scm/<PROJECT>/setup-work.git ~/setup-work
cd ~/setup-work
export SETUP_WORK_TEAM_CONFIG_DIR=~/setup-work-team

# Keep weights and engines off the small home disk (optional).
export XDG_DATA_HOME=/data/$USER/share      # models, engines, harnesses, Node.js, uv
export PATH="$HOME/.local/bin:$PATH"         # where setup-work links binaries
```

Put those `export` lines into your shell profile yourself — setup-work never edits
startup files.

## 2. Look before you leap

```bash
sh scripts/bootstrap.sh --dry-run   # what stage 0 would do (no Python needed)
make bootstrap                      # uv from the corporate source (sha256-checked) + project venv
make doctor                         # system, tools, GPUs, engines
make hardware-json > hw.json        # keep a record; use it to plan from your laptop
```

`make bootstrap` installs uv into `~/.local/bin`, builds the project venv with the
machine's Python (3.11+ required; there is no Python mirror, so uv never downloads
interpreters), then installs Node.js 24 and Git LFS from the archives mapped in
`corporate.yaml` `tools`.
Packages that need a system package manager (e.g. `build-essential`) are *reported*
with the exact command for an administrator; nothing is attempted with sudo unless
you pass `ALLOW_SUDO=1` and confirm.

## 3. Pick a model that actually fits

```bash
make recommend TASK=coding CONTEXT=32768 EXPLAIN=1
make recommend TASK=agents CONTEXT=32768 CONCURRENCY=8 ENGINE=vllm
```

The recommender reads per-GPU *free* memory, compute capability and the KV-cache
geometry of each model. Busy GPUs reduce the budget; V100s (compute capability 7.0)
will not be offered AWQ or FP8 checkpoints.

## 4. Install an engine and the weights

```bash
make engine-install ENGINE=vllm          # venv zip from corporate.yaml, repaired for this machine
make engine-install ENGINE=llama.cpp     # corporate CUDA build from corporate.yaml

make model-pull MODEL=qwen3-coder-30b-a3b:fp8   # from the artifact mapped in corporate.yaml
```

## 5. Serve on localhost and reach it through SSH

```bash
make serve MODEL=qwen3-coder-30b-a3b:fp8 ENGINE=vllm CONTEXT=32768
make serve-test
make serve-logs FOLLOW=1
```

From your laptop:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@gpu-server
curl http://127.0.0.1:8000/v1/models
```

Binding to `0.0.0.0` (`HOST=0.0.0.0`) requires confirmation; set
`SETUP_WORK_API_KEY` first so the endpoint is not unauthenticated.

## 6. Coding agents against the local endpoint

```bash
make harness-install NAME=qwen-code
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=local OPENAI_MODEL=qwen3-coder-30b-a3b
qwen
make serve-stop
```
