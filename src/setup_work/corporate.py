"""Corporate environment: `corporate.yaml` env and a read-only status for `doctor`.

pip, npm and uv are configured by `make registries` (scripts/bootstrap.sh), which
writes pip.conf, .npmrc and uv.toml. Each tool then reads its own file, so nothing
has to be bridged here.
"""

from __future__ import annotations

import configparser
import os
import re
import tomllib
from pathlib import Path
from typing import Any

from .config import Config
from .runner import probe, which


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def pip_index() -> str | None:
    """index-url from pip.conf (later files win, PIP_INDEX_URL wins over all)."""
    if os.environ.get("PIP_INDEX_URL"):
        return os.environ["PIP_INDEX_URL"]
    files = [
        Path("/etc/pip.conf"),
        Path.home() / ".pip" / "pip.conf",
        _config_home() / "pip" / "pip.conf",
    ]
    if os.environ.get("PIP_CONFIG_FILE"):
        files.append(Path(os.environ["PIP_CONFIG_FILE"]))
    found = None
    for f in files:
        cp = configparser.RawConfigParser()
        try:
            cp.read(f, encoding="utf-8")
        except (configparser.Error, OSError):
            continue
        if cp.has_option("global", "index-url"):
            found = cp.get("global", "index-url").strip()
    return found


def uv_index() -> str | None:
    """Default index from uv's user config (uv.toml) or UV_DEFAULT_INDEX."""
    if os.environ.get("UV_DEFAULT_INDEX"):
        return os.environ["UV_DEFAULT_INDEX"]
    f = Path(os.environ.get("UV_CONFIG_FILE") or _config_home() / "uv" / "uv.toml")
    try:
        data = tomllib.loads(f.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    for idx in data.get("index", []):
        if idx.get("default"):
            return idx.get("url")
    return data.get("index-url")


def apply_environment(cfg: Config) -> None:
    """Export `corporate.yaml` env for every subprocess (variables you set win)."""
    os.environ.setdefault("UV_PYTHON_DOWNLOADS", "never")  # there is no Python mirror
    try:
        env = cfg.corporate.env
    except Exception:  # invalid corporate.yaml is reported by `config validate`
        return
    for k, v in env.items():
        os.environ.setdefault(k, os.path.expandvars(os.path.expanduser(v)))


def mask(value: str) -> str:
    return re.sub(r"//[^/@\s]+@", "//***@", value)


def status(cfg: Config) -> dict[str, Any]:
    """Read-only snapshot for `doctor`."""
    from .artifacts import kerberos_ok

    corp = cfg.corporate
    info: dict[str, Any] = {"configured": corp.enabled}
    hdfs = which(corp.hdfs.cli)
    info["hdfs_cli"] = hdfs
    if corp.models or corp.engines or corp.tools or hdfs:
        ok, why = kerberos_ok(corp.hdfs)
        info["kerberos"] = {"ok": ok, "detail": why}
    pip, uv = pip_index(), uv_index()
    info["pip_index"] = mask(pip) if pip else None
    info["uv_index"] = mask(uv) if uv else None
    npm = which("npm")
    if npm:
        res = probe([npm, "config", "get", "registry"], timeout=15)
        info["npm_registry"] = mask(res.stdout.strip()) if res.ok else None
    info["artifacts"] = {
        "models": len(corp.models),
        "engines": sorted(corp.engines),
        "tools": sorted(corp.tools),
    }
    return info
