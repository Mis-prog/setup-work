# Bootstrapping a fresh remote GPU server without sudo

Scenario: a shared Linux GPU server (Ubuntu 22.04, NVIDIA driver already installed by
the admins), an unprivileged account, home directory on a small disk and a large
scratch volume at `/data/$USER`. Python on the box is 3.8, there is no Node.js, and
you cannot run `sudo`.

> Status: the commands below are exercised by the test-suite with mocked hardware and
> in dry-run mode. Running them on real NVIDIA hardware has **not** been verified by the
> authors yet — please report results.

## 0. Requirements on the server

- `git`, `make`, `tar`, and `curl` *or* `wget` (present on almost every server).
- Outbound HTTPS to github.com, nodejs.org, pypi.org and huggingface.co (or a proxy).

## 1. Clone and point heavy data at the big volume

```bash
git clone https://github.com/<you>/setup-work.git ~/setup-work
cd ~/setup-work

# Keep weights and engines off the small home disk (optional).
export HF_HOME=/data/$USER/hf
export XDG_DATA_HOME=/data/$USER/share      # engines, harnesses, Node.js, uv
export PATH="$HOME/.local/bin:$PATH"         # where setup-work links binaries
```

Put those `export` lines into your shell profile yourself — setup-work never edits
startup files.

## 2. Look before you leap

```bash
sh scripts/bootstrap.sh --dry-run   # what stage 0 would do (no Python needed)
make bootstrap                      # installs pinned uv (sha256-checked) + project venv
make doctor                         # system, tools, GPUs, engines
make hardware-json > hw.json        # keep a record; use it to plan from your laptop
```

`make bootstrap` installs uv into `~/.local/bin`, lets uv download Python 3.12 into
user space, then installs Node.js 24 and Git LFS from pinned official archives.
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
make engine-install ENGINE=vllm          # Python 3.12 venv in $XDG_DATA_HOME/setup-work/engines/vllm
make engine-install ENGINE=llama.cpp     # prebuilt CUDA build matching the driver

make model-pull MODEL=qwen3-coder-30b-a3b:fp8   # asks before large downloads; resumable
```

Gated models (Llama, Gemma) need `hf auth login` or `HF_TOKEN` in the environment.

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
