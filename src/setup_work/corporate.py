"""Корпоративное окружение из corporate.yaml и диагностика для doctor.

make registries записывает pip.conf, uv.toml и npmrc; инструменты самостоятельно
читают эти файлы."""

from __future__ import annotations

import configparser
import os
import re
import tomllib
from pathlib import Path
from typing import Any

from .config import Config, setup_work_home
from .runner import probe, which


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def pip_index() -> str | None:
    """Прочитать index-url; более поздние файлы и PIP_INDEX_URL имеют приоритет."""
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
    """Определить основной индекс uv из uv.toml или UV_DEFAULT_INDEX."""
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


def registry_env() -> dict[str, str]:
    """При SETUP_WORK_HOME настройки и кэши pip/uv/npm сохраняются в общем каталоге."""
    home = setup_work_home()
    if home is None:
        return {}
    return {
        "PIP_CONFIG_FILE": str(home / "registries" / "pip.conf"),
        "UV_CONFIG_FILE": str(home / "registries" / "uv.toml"),
        "NPM_CONFIG_USERCONFIG": str(home / "registries" / "npmrc"),
        "UV_CACHE_DIR": str(home / "cache" / "uv"),
        "npm_config_cache": str(home / "cache" / "npm"),
        "GIT_CONFIG_GLOBAL": str(home / "gitconfig"),
    }


def apply_environment(cfg: Config) -> None:
    """Применить env из corporate.yaml, сохраняя приоритет заданного окружения."""
    os.environ.setdefault("UV_PYTHON_DOWNLOADS", "never")  # зеркало Python не предусмотрено
    for k, v in registry_env().items():
        os.environ.setdefault(k, v)
    try:
        env = cfg.corporate.env
    except Exception:  # ошибки corporate.yaml покажет config validate
        return
    for k, v in env.items():
        os.environ.setdefault(k, os.path.expandvars(os.path.expanduser(v)))


def mask(value: str) -> str:
    return re.sub(r"//[^/@\s]+@", "//***@", value)


def status(cfg: Config) -> dict[str, Any]:
    """Собрать сведения для doctor без изменения настроек."""
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
