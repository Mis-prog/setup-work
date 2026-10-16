"""Software environment detection: tools and versions."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .runner import probe, which

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


def parse_version(text: str) -> tuple[int, ...] | None:
    m = _VERSION_RE.search(text or "")
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split("."))


def version_satisfies(found: str | None, spec: str) -> bool:
    """Minimal comparator for specs like '>=22', '>=3.11'."""
    if not found:
        return False
    v = parse_version(found)
    m = re.fullmatch(r"\s*(>=|>|==|<=|<)?\s*([\d.]+)\s*", spec)
    if v is None or m is None:
        return False
    op, want_s = m.group(1) or ">=", m.group(2)
    want = tuple(int(x) for x in want_s.split("."))
    n = max(len(v), len(want))
    a = v + (0,) * (n - len(v))
    b = want + (0,) * (n - len(want))
    return {
        ">=": a >= b,
        ">": a > b,
        "==": a[: len(want)] == want,
        "<=": a <= b,
        "<": a < b,
    }[op]


@dataclass
class ToolStatus:
    name: str
    installed: bool
    version: str | None = None
    path: str | None = None
    managed: bool = False  # installed by setup-work into data_dir
    satisfies: bool | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# name -> (executable, version argv, minimum)
TOOL_PROBES: dict[str, tuple[str, list[str], str | None]] = {
    "python": ("python3", ["python3", "--version"], ">=3.11"),
    "uv": ("uv", ["uv", "--version"], None),
    "node": ("node", ["node", "--version"], ">=22"),
    "npm": ("npm", ["npm", "--version"], None),
    "git": ("git", ["git", "--version"], None),
    "make": ("make", ["make", "--version"], None),
}


def detect_tool(name: str, cfg: Config | None = None) -> ToolStatus:
    exe, argv, minimum = TOOL_PROBES[name]
    extra = [cfg.bin_dir] if cfg else []
    path = which(exe, extra)
    if name == "python" and cfg is not None:
        # uv-managed interpreters are acceptable even if python3 on PATH is old.
        uv = which("uv", extra)
        if uv:
            res = probe([uv, "python", "find", ">=3.11"], timeout=15)
            if res.ok and res.stdout.strip():
                p = res.stdout.strip().splitlines()[0]
                ver = probe([p, "--version"])
                v = (ver.stdout or ver.stderr).strip()
                return ToolStatus(
                    name,
                    True,
                    _clean(v),
                    p,
                    satisfies=version_satisfies(v, ">=3.11"),
                    note="uv-managed or discovered by uv" if p != path else None,
                )
    if not path:
        return ToolStatus(name, False)
    res = probe([path, *argv[1:]], timeout=15)
    out = (res.stdout or res.stderr).strip()
    if not res.ok:
        return ToolStatus(name, False, path=path, note=out.splitlines()[0][:120] if out else None)
    version = _clean(out)
    managed = bool(cfg and _is_managed(Path(path), cfg))
    sat = version_satisfies(version, minimum) if minimum else None
    if name == "node" and cfg is not None:
        pin = cfg.defaults.tools.get("node")
        if pin and pin.min_version:
            sat = version_satisfies(version, f">={pin.min_version}")
    return ToolStatus(name, True, version, path, managed, sat)


def _clean(text: str) -> str | None:
    first = text.splitlines()[0] if text else ""
    v = _VERSION_RE.search(first) or _VERSION_RE.search(text)
    return v.group(1) if v else (first[:60] or None)


def _is_managed(path: Path, cfg: Config) -> bool:
    try:
        return str(path.resolve()).startswith(str(cfg.data_dir.resolve()))
    except OSError:
        return False


@dataclass
class Environment:
    python_runtime: str
    tools: list[ToolStatus]
    bin_dir_on_path: bool
    bin_dir: str
    proxies: dict[str, str] = field(default_factory=dict)
    shell: str | None = None

    def tool(self, name: str) -> ToolStatus | None:
        return next((t for t in self.tools if t.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tools"] = {t.name: t.to_dict() for t in self.tools}
        return d


def detect_environment(cfg: Config, tools: list[str] | None = None) -> Environment:
    names = tools or list(TOOL_PROBES)
    statuses = [detect_tool(n, cfg) for n in names]
    path_dirs = [Path(p).expanduser() for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    on_path = any(_same(p, cfg.bin_dir) for p in path_dirs)
    proxies = {
        k: _mask_url(v)
        for k, v in os.environ.items()
        if k.lower() in ("http_proxy", "https_proxy", "no_proxy", "all_proxy")
    }
    return Environment(
        python_runtime=sys.version.split()[0],
        tools=statuses,
        bin_dir_on_path=on_path,
        bin_dir=str(cfg.bin_dir),
        proxies=proxies,
        shell=os.environ.get("SHELL"),
    )


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _mask_url(v: str) -> str:
    """Hide credentials in proxy URLs (http://user:pass@host)."""
    return re.sub(r"//[^/@]+@", "//***@", v)
