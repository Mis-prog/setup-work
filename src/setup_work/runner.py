"""Safe execution primitives shared by every module.

* Commands are argv lists, never shell strings (no ``shell=True`` anywhere).
* Every mutating command is displayed before it runs; dry-run prints and skips.
* Secrets are redacted from displayed commands.
* Confirmation is required for destructive or privileged actions unless the
  caller passed ``--yes``; non-interactive sessions without ``--yes`` abort.
* Structured events are appended to ``<state_dir>/logs/events.jsonl``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from .errors import AbortedError, MissingDependencyError, SetupWorkError

console = Console(stderr=False, highlight=False)
err_console = Console(stderr=True, highlight=False)

_SECRET_FLAGS = {"--api-key", "--hf-token", "-hft", "--token", "--password"}
_SECRET_ENV_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)


def redact_argv(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    hide_next = False
    for a in argv:
        if hide_next:
            out.append("***")
            hide_next = False
            continue
        if a in _SECRET_FLAGS:
            out.append(a)
            hide_next = True
            continue
        if "=" in a and a.split("=", 1)[0] in _SECRET_FLAGS:
            out.append(a.split("=", 1)[0] + "=***")
            continue
        out.append(a)
    return out


def format_argv(argv: Sequence[str]) -> str:
    return shlex.join(redact_argv([str(a) for a in argv]))


def redact_env(env: Mapping[str, str]) -> dict[str, str]:
    return {k: ("***" if _SECRET_ENV_RE.search(k) else v) for k, v in env.items()}


def which(name: str, extra_dirs: Sequence[Path] = ()) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for d in extra_dirs:
        p = Path(d) / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


@dataclass
class Result:
    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def probe(
    argv: Sequence[str], timeout: float = 10.0, env: Mapping[str, str] | None = None
) -> Result:
    """Run a read-only command quietly. Never raises for missing executables."""
    argv = [str(a) for a in argv]
    try:
        cp = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
        )
        return Result(argv, cp.returncode, cp.stdout, cp.stderr)
    except FileNotFoundError:
        return Result(argv, 127, "", f"{argv[0]}: not found")
    except PermissionError as e:
        return Result(argv, 126, "", str(e))
    except subprocess.TimeoutExpired:
        return Result(argv, 124, "", f"timed out after {timeout}s")
    except OSError as e:
        return Result(argv, 1, "", str(e))


@dataclass
class Runner:
    """Executes plans. Holds the global --dry-run / --yes / --json switches."""

    dry_run: bool = False
    assume_yes: bool = False
    json_mode: bool = False
    verbose: bool = False
    log_dir: Path | None = None
    interactive: bool = field(default_factory=lambda: sys.stdin.isatty())

    # ----------------------------------------------------------------- output
    def info(self, msg: str) -> None:
        (err_console if self.json_mode else console).print(msg)

    def warn(self, msg: str) -> None:
        err_console.print(f"[yellow]warning:[/] {msg}")

    def show_command(self, argv: Sequence[str], cwd: Path | None = None) -> None:
        prefix = "[dim](dry-run)[/] " if self.dry_run else ""
        where = f" [dim](in {cwd})[/]" if cwd else ""
        (err_console if self.json_mode else console).print(
            f"{prefix}[bold cyan]$[/] {format_argv(argv)}{where}", markup=True, soft_wrap=True
        )

    # ----------------------------------------------------------- confirmation
    def confirm(self, question: str, *, default: bool = False, destructive: bool = False) -> bool:
        if self.assume_yes:
            return True
        if self.dry_run:
            self.info(f"[dim](dry-run) would ask:[/] {question}")
            return True
        if not self.interactive:
            raise AbortedError(
                f"confirmation required: {question}",
                hint="re-run with --yes (make ... YES=1) to approve non-interactively",
            )
        suffix = " [y/N] " if not default else " [Y/n] "
        style = "bold red" if destructive else "bold"
        console.print(f"[{style}]{question}[/]{suffix}", end="")
        try:
            answer = input().strip().lower()
        except EOFError:
            answer = ""
        if not answer:
            return default
        return answer in ("y", "yes")

    def require_confirmation(self, question: str, *, destructive: bool = False) -> None:
        if not self.confirm(question, destructive=destructive):
            raise AbortedError("aborted by user")

    # -------------------------------------------------------------- execution
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        capture: bool = False,
        check: bool = True,
        mutating: bool = True,
    ) -> Result:
        argv = [str(a) for a in argv]
        if not argv:
            raise SetupWorkError("empty command")
        if mutating or self.verbose:
            self.show_command(argv, cwd)
        if mutating and self.dry_run:
            return Result(argv, 0, skipped=True)
        if shutil.which(argv[0]) is None and not Path(argv[0]).is_file():
            raise MissingDependencyError(
                f"{argv[0]} is not installed or not on PATH",
                hint="run `make doctor` to see what is missing",
            )
        full_env = None
        if env is not None:
            full_env = os.environ.copy()
            full_env.update(env)
        self.event("command.start", argv=redact_argv(argv), cwd=str(cwd) if cwd else None)
        try:
            cp = subprocess.run(
                argv,
                cwd=cwd,
                env=full_env,
                timeout=timeout,
                text=True,
                capture_output=capture,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            self.event("command.timeout", argv=redact_argv(argv), timeout=timeout)
            raise SetupWorkError(f"command timed out after {timeout}s: {format_argv(argv)}") from e
        except KeyboardInterrupt:
            self.event("command.interrupted", argv=redact_argv(argv))
            raise
        res = Result(argv, cp.returncode, cp.stdout or "", cp.stderr or "")
        self.event("command.end", argv=redact_argv(argv), returncode=cp.returncode)
        if check and not res.ok:
            detail = (res.stderr or res.stdout).strip().splitlines()[-5:] if capture else []
            raise SetupWorkError(
                f"command failed with exit code {res.returncode}: {format_argv(argv)}"
                + ("\n" + "\n".join(detail) if detail else "")
            )
        return res

    # ---------------------------------------------------------------- logging
    def event(self, event_name: str, /, **fields: Any) -> None:
        if self.log_dir is None or self.dry_run:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            rec = {"ts": _dt.datetime.now(_dt.UTC).isoformat(), "event": event_name, **fields}
            with (self.log_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass


def emit_json(data: Any) -> None:
    """Write machine-readable output to stdout (and nothing else)."""
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    sys.stdout.flush()
