"""Corporate environment integration.

* ``apply_environment`` exports ``corporate.env`` (non-secret settings such as
  ``UV_PYTHON_INSTALL_MIRROR`` or ``NPM_CONFIG_REGISTRY``) for every subprocess,
  without overriding variables the user already set.
* uv ignores ``pip.conf``. When corporate scripts configure pip (index-url,
  extra-index-url, trusted-host, cert), ``pip_conf_bridge`` maps them to the
  equivalent ``UV_*`` variables so uv-based installs use the same mirrors.
* ``plan_scripts`` turns ``corporate.scripts`` into confirmed, idempotent steps.
"""

from __future__ import annotations

import configparser
import os
import re
from pathlib import Path
from typing import Any

from .config import Config, expand_path
from .runner import probe, which
from .steps import Step


def pip_conf_files() -> list[Path]:
    env = os.environ.get("PIP_CONFIG_FILE")
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    cands = [
        Path("/etc/pip.conf"),
        Path("/etc/xdg/pip/pip.conf"),
        Path.home() / ".pip" / "pip.conf",
        Path(xdg) / "pip" / "pip.conf",
        Path.home() / "Library" / "Application Support" / "pip" / "pip.conf",
    ]
    if env:
        cands.append(Path(env))
    return [p for p in cands if p.is_file()]


def read_pip_settings() -> dict[str, str]:
    """Later files win (pip's own precedence: global < user < PIP_CONFIG_FILE)."""
    out: dict[str, str] = {}
    for f in pip_conf_files():
        cp = configparser.RawConfigParser()
        try:
            cp.read(f, encoding="utf-8")
        except (configparser.Error, OSError):
            continue
        for section in ("global", "install"):
            if cp.has_section(section):
                for key in ("index-url", "extra-index-url", "trusted-host", "cert"):
                    if cp.has_option(section, key):
                        out[key] = cp.get(section, key).strip()
    for env_key, key in (
        ("PIP_INDEX_URL", "index-url"),
        ("PIP_EXTRA_INDEX_URL", "extra-index-url"),
        ("PIP_TRUSTED_HOST", "trusted-host"),
        ("PIP_CERT", "cert"),
    ):
        if os.environ.get(env_key):
            out[key] = os.environ[env_key]
    return out


def pip_conf_bridge(settings: dict[str, str] | None = None) -> dict[str, str]:
    """UV_* variables equivalent to the pip settings, excluding ones already set."""
    s = read_pip_settings() if settings is None else settings
    env: dict[str, str] = {}
    has_index = any(os.environ.get(k) for k in ("UV_DEFAULT_INDEX", "UV_INDEX_URL"))
    if s.get("index-url") and not has_index:
        env["UV_DEFAULT_INDEX"] = s["index-url"]
    if s.get("extra-index-url") and not os.environ.get("UV_INDEX"):
        env["UV_INDEX"] = " ".join(s["extra-index-url"].split())
    if s.get("trusted-host") and not os.environ.get("UV_INSECURE_HOST"):
        env["UV_INSECURE_HOST"] = " ".join(s["trusted-host"].split())
    if s.get("cert") and not os.environ.get("SSL_CERT_FILE"):
        env["SSL_CERT_FILE"] = str(Path(s["cert"]).expanduser())
    return env


def apply_environment(cfg: Config) -> dict[str, str]:
    """Export corporate settings into os.environ (user-set values win). Returns what was set."""
    applied: dict[str, str] = {}
    try:
        corp = cfg.corporate
    except Exception:  # invalid corporate.yaml is reported by `config validate`
        return applied
    for k, v in corp.env.items():
        if k not in os.environ:
            os.environ[k] = os.path.expandvars(os.path.expanduser(v))
            applied[k] = os.environ[k]
    if corp.bridge_pip_conf:
        for k, v in pip_conf_bridge().items():
            os.environ[k] = v
            applied[k] = v
    return applied


def mask(value: str) -> str:
    return re.sub(r"//[^/@\s]+@", "//***@", value)


def plan_scripts(cfg: Config, only: list[str] | None = None) -> list[Step]:
    steps: list[Step] = []
    for sc in cfg.corporate.scripts:
        if only and sc.name not in only:
            continue
        creates = [expand_path(c) for c in sc.creates]
        argv = [os.path.expanduser(a) for a in sc.argv]
        steps.append(
            Step(
                "command",
                f"corporate setup: {sc.name}" + (f" — {sc.description}" if sc.description else ""),
                argv=argv,
                confirm=True,
                timeout=1800,
                when=(lambda c=creates: not all(p.exists() for p in c)) if creates else None,
            )
        )
    return steps


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
    info["scripts"] = [
        {
            "name": s.name,
            "done": bool(s.creates) and all(expand_path(c).exists() for c in s.creates),
        }
        for s in corp.scripts
    ]
    pip = read_pip_settings()
    info["pip_index"] = mask(pip["index-url"]) if pip.get("index-url") else None
    uv_index = os.environ.get("UV_DEFAULT_INDEX") or os.environ.get("UV_INDEX_URL")
    info["uv_index"] = mask(uv_index) if uv_index else None
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
