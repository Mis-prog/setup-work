# Corporate networks: proxies, mirrors and air-gapped machines

## Proxies

Every download uses the standard environment variables, which curl, wget, uv, npm,
git, huggingface_hub and setup-work's own downloader all honour:

```bash
export HTTPS_PROXY=http://proxy.corp:3128 HTTP_PROXY=http://proxy.corp:3128
export NO_PROXY=localhost,127.0.0.1,.corp
```

Requests to local inference servers (`serve-test`, readiness checks) always bypass the
proxy. `make doctor` shows the active proxy variables with credentials masked.

Custom CA bundles: set `SSL_CERT_FILE` (Python, uv), `NODE_EXTRA_CA_CERTS` (npm) and
`REQUESTS_CA_BUNDLE` / `HF_HUB_*` as your organisation requires.

## Mirrors

- PyPI: `UV_INDEX_URL=https://pypi.corp/simple` (uv) — affects `uv sync`, Aider, vLLM.
- npm: `npm config set registry https://npm.corp/` or `NPM_CONFIG_REGISTRY`.
- Hugging Face: `HF_ENDPOINT=https://hf-mirror.corp`.
- Release archives (uv, Node.js, Git LFS, llama.cpp): override the `url` templates in
  `~/.config/setup-work/defaults.yaml`; checksums stay pinned, so a mirror cannot
  substitute different bytes.

## Air-gapped installs

Installer archives are looked up in the download cache before any network access and
are used if their sha256 matches the pinned digest:

```text
$XDG_DATA_HOME/setup-work/tools/downloads/uv-0.12.13-<target>.tar.gz
$XDG_DATA_HOME/setup-work/tools/downloads/node-v24.20.0-<target>.tar.xz
$XDG_DATA_HOME/setup-work/tools/downloads/git-lfs-<target>-v3.8.0.tar.gz
$XDG_DATA_HOME/setup-work/engines/llama.cpp/downloads/llama-b11080-bin-<asset>.tar.gz
```

Copy them from a connected machine (`make bootstrap` there fills the same cache) and run
`make bootstrap` / `make engine-install` on the offline machine.

Models: copy a populated `HF_HUB_CACHE` directory and set `HF_HUB_OFFLINE=1`.
`make model-list` shows what is available; `make serve` never downloads anything.

Planning without access to the target: run `make hardware-json > hw.json` on the target,
bring the file back and use `make recommend HARDWARE=hw.json`.
