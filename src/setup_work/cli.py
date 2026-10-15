"""setup-work command line interface.

Every option that the Makefile forwards also reads a ``SETUP_WORK_*`` environment
variable, so ``make serve MODEL=x`` passes values through the environment instead
of interpolating them into a shell command line.
"""

from __future__ import annotations

import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from . import __version__, corporate, hub
from . import bootstrap as bs
from . import harness as hz
from .config import Config
from .diagnostics import detect_environment
from .engines import ENGINES, get_engine
from .errors import CheckFailedError, ExitCode, SetupWorkError, UsageError
from .hardware import HardwareInfo, detect
from .inference import ServeOptions, ServerManager, plan_launch
from .memory import gib
from .models import RecommendRequest, recommend, resolve
from .profiles import resolve_profile
from .runner import Runner, console, emit_json, err_console
from .steps import Step, execute

app = typer.Typer(
    name="setup-work",
    help="Bootstrap AI/ML development environments: tools, agent harnesses, model selection, "
    "local inference.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
config_app = typer.Typer(help="Show and validate configuration.", no_args_is_help=True)
model_app = typer.Typer(help="Download and manage model weights.", no_args_is_help=True)
harness_app = typer.Typer(help="Install and manage coding-agent harnesses.", no_args_is_help=True)
server_app = typer.Typer(help="Manage running inference servers.", no_args_is_help=True)
engine_app = typer.Typer(help="Install inference engines (llama.cpp, vLLM).", no_args_is_help=True)
profile_app = typer.Typer(help="Inspect development profiles.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(model_app, name="model")
app.add_typer(harness_app, name="harness")
app.add_typer(server_app, name="server")
app.add_typer(engine_app, name="engine")
app.add_typer(profile_app, name="profile")
corp_app = typer.Typer(
    help="Corporate sources: setup scripts, HDFS, mirrors.", no_args_is_help=True
)
app.add_typer(corp_app, name="corp")

OK = "[green]✔[/]"
BAD = "[red]✘[/]"
WARN = "[yellow]![/]"


class State:
    def __init__(self) -> None:
        self.runner = Runner()
        self._cfg: Config | None = None
        self._hw: HardwareInfo | None = None

    @property
    def cfg(self) -> Config:
        if self._cfg is None:
            self._cfg = Config()
            self.runner.log_dir = self._cfg.state_dir / "logs"
            corporate.apply_environment(self._cfg)
        return self._cfg

    @property
    def hw(self) -> HardwareInfo:
        if self._hw is None:
            cfg = self.cfg
            paths = [cfg.data_dir, hub.cache_dir(cfg), Path.home()]
            self._hw = detect(paths)
        return self._hw

    @property
    def json(self) -> bool:
        return self.runner.json_mode


STATE = State()


def _json_opt() -> Any:
    return typer.Option(
        False, "--json", help="Machine-readable JSON on stdout.", envvar="SETUP_WORK_JSON"
    )


def _set_json(flag: bool) -> bool:
    if flag:
        STATE.runner.json_mode = True
    return STATE.runner.json_mode


@app.callback()
def _root(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        envvar="SETUP_WORK_DRY_RUN",
        help="Print actions without executing them.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        envvar="SETUP_WORK_YES",
        help="Approve confirmations (non-interactive use).",
    ),
    json_out: bool = typer.Option(
        False, "--json", envvar="SETUP_WORK_JSON", help="JSON output where supported."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", envvar="SETUP_WORK_VERBOSE"),
) -> None:
    r = STATE.runner
    r.dry_run, r.assume_yes, r.json_mode, r.verbose = dry_run, yes, json_out, verbose


# =============================================================== basic commands


@app.command()
def version(json_out: bool = _json_opt()) -> None:
    """Print the setup-work version."""
    if _set_json(json_out):
        emit_json({"version": __version__, "python": sys.version.split()[0]})
    else:
        console.print(f"setup-work {__version__}")


def _hw_table(hw: HardwareInfo) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    t.add_row("OS", f"{hw.os.distro or hw.os.system} ({hw.os.system} {hw.os.release})")
    t.add_row(
        "Architecture",
        hw.os.arch + (" · container" if hw.os.container else "") + (" · WSL" if hw.os.wsl else ""),
    )
    t.add_row(
        "CPU",
        f"{hw.cpu.model or 'unknown'} · {hw.cpu.logical_cores} logical / "
        f"{hw.cpu.physical_cores or '?'} physical · {hw.cpu.usable_cores} usable",
    )
    m = hw.memory
    ram = f"{gib(m.total_bytes)} total · {gib(m.available_bytes)} available"
    if m.cgroup_limit_bytes:
        ram += f" · container limit {gib(m.cgroup_limit_bytes)}"
    t.add_row("RAM", ram)
    if hw.accelerator == "apple":
        g = hw.gpus[0]
        t.add_row("GPU", f"{g.name} · unified memory")
        t.add_row(
            "GPU budget",
            f"{gib(hw.metal_working_set_bytes)} Metal working set ({hw.metal_working_set_source})",
        )
    elif hw.gpus:
        names: dict[str, int] = {}
        for g in hw.gpus:
            names[g.name] = names.get(g.name, 0) + 1
        t.add_row("GPUs", ", ".join(f"{n} × {k}" for k, n in names.items()))
        for g in hw.gpus:
            t.add_row(
                f"  GPU {g.index}",
                f"{gib(g.memory_total_bytes)} total · "
                f"{gib(g.memory_free_bytes)} free · CC {g.compute_capability or '?'}"
                + (
                    f" ({g.compute_capability_source})"
                    if g.compute_capability_source == "name-lookup"
                    else ""
                ),
            )
        t.add_row("Total VRAM", f"{gib(hw.total_vram_bytes)} (aggregate of separate devices)")
        t.add_row(
            "Driver / CUDA",
            f"{hw.nvidia_driver or '?'} / CUDA {hw.cuda_driver_version or '?'}"
            + (f" · nvcc {hw.cuda_toolkit_version}" if hw.cuda_toolkit_version else ""),
        )
    else:
        t.add_row("GPU", "none detected (CPU-only)")
    t.add_row("Precisions", ", ".join(hw.precisions) or "unknown")
    for s in hw.storage:
        t.add_row("Disk", f"{s.path}: {gib(s.free_bytes)} free of {gib(s.total_bytes)}")
    if hw.source != "live":
        t.add_row("Source", f"[yellow]{hw.source}[/]")
    for n in hw.notes:
        t.add_row("Note", f"[dim]{escape(n)}[/]")
    return t


@app.command()
def hardware(json_out: bool = _json_opt()) -> None:
    """Detect CPU, RAM, GPUs, VRAM, compute capability and storage."""
    hw = STATE.hw
    if _set_json(json_out):
        emit_json(hw.to_dict())
        return
    console.print(Panel(_hw_table(hw), title="Hardware", expand=False))


def _engine_rows(cfg: Config, hw: HardwareInfo) -> list[tuple[str, str, str]]:
    rows = []
    for name, eng in ENGINES.items():
        binary = eng.find_binary(cfg)
        if name == "vllm" and (hw.os.system != "Linux" or hw.accelerator != "nvidia"):
            status = f"{WARN} unsupported here"
            detail = "needs Linux + NVIDIA GPU; use llama.cpp"
        elif name == "vllm":
            cc = hw.min_compute_capability
            if cc is not None and cc < 7.0:
                status, detail = f"{BAD} unsupported", f"compute capability {cc} < 7.0"
            elif binary:
                status, detail = f"{OK} installed", binary
            else:
                status, detail = f"{WARN} not installed", "make engine-install ENGINE=vllm"
        else:
            backend = {"apple": "Metal", "nvidia": "CUDA"}.get(hw.accelerator, "CPU")
            if binary:
                status, detail = f"{OK} installed", f"{binary} ({backend} backend expected)"
            else:
                status, detail = f"{WARN} not installed", "make engine-install ENGINE=llama.cpp"
        rows.append((name, status, detail))
    return rows


def _corp_table(c: dict[str, Any]) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    t.add_row("HDFS client", c.get("hdfs_cli") or f"{WARN} not found")
    if "kerberos" in c:
        k = c["kerberos"]
        t.add_row(
            "Kerberos",
            f"{OK if k['ok'] else BAD} {k['detail']}" + ("" if k["ok"] else " — run kinit"),
        )
    for sc in c.get("scripts", []):
        t.add_row(f"script {sc['name']}", OK + " done" if sc["done"] else f"{WARN} make corp-setup")
    t.add_row("pip index", c.get("pip_index") or "default (pypi.org)")
    t.add_row("uv index", c.get("uv_index") or "default (pypi.org)")
    if "npm_registry" in c:
        t.add_row("npm registry", c.get("npm_registry") or "?")
    a = c.get("artifacts", {})
    t.add_row(
        "artifacts",
        f"{a.get('models', 0)} model(s), engines: "
        f"{', '.join(a.get('engines', [])) or '-'}, tools: {', '.join(a.get('tools', [])) or '-'}",
    )
    return t


@app.command()
def doctor(
    json_out: bool = _json_opt(),
    strict: bool = typer.Option(False, help="Exit 7 if a core tool is missing."),
) -> None:
    """Diagnose system, tools, hardware and inference readiness. Never installs anything."""
    cfg, hw = STATE.cfg, STATE.hw
    env = detect_environment(cfg)
    problems = cfg.validate_all()
    core = ["python", "uv", "git", "curl"]
    missing_core = [t.name for t in env.tools if t.name in core and not t.installed]
    if _set_json(json_out):
        emit_json(
            {
                "version": __version__,
                "environment": env.to_dict(),
                "hardware": hw.to_dict(),
                "engines": {
                    n: {"status": s, "detail": d}
                    for n, s, d in [
                        (r[0], r[1].split("] ", 1)[-1], r[2]) for r in _engine_rows(cfg, hw)
                    ]
                },
                "config_problems": problems,
                "missing_core_tools": missing_core,
                "corporate": corporate.status(cfg),
            }
        )
    else:
        console.print(Panel.fit("[bold]SETUP-WORK[/] " + __version__, border_style="cyan"))
        sys_t = Table.grid(padding=(0, 2))
        sys_t.add_column(style="bold")
        sys_t.add_column()
        sys_t.add_row("setup-work runs on", f"Python {env.python_runtime}")
        sys_t.add_row("Package managers", ", ".join(env.package_managers) or "none found")
        sys_t.add_row(
            "Privileges",
            "root"
            if env.is_root
            else (
                "user (sudo present, never used silently)" if env.sudo_present else "user (no sudo)"
            ),
        )
        sys_t.add_row(
            "bin dir",
            f"{env.bin_dir} "
            + (OK if env.bin_dir_on_path else f"{WARN} not on PATH: {bs.path_hint(cfg)}"),
        )
        if env.proxies:
            sys_t.add_row("Proxies", ", ".join(f"{k}={v}" for k, v in env.proxies.items()))
        console.print(Panel(sys_t, title="System", expand=False))

        tt = Table(title="Tools", show_edge=False, title_justify="left")
        for c in ("tool", "status", "version", "path"):
            tt.add_column(c)
        for t in env.tools:
            if not t.installed:
                mark = BAD if t.name in core else WARN
                note = "optional: uv replaces pip" if t.name == "pip" else (t.note or "")
                tt.add_row(t.name, f"{mark} missing", "", note)
            else:
                mark = OK if t.satisfies in (None, True) else WARN
                ver = (t.version or "") + ("" if t.satisfies in (None, True) else " (too old)")
                tt.add_row(
                    t.name, f"{mark} ok" + (" (managed)" if t.managed else ""), ver, t.path or ""
                )
        console.print(tt)
        console.print(Panel(_hw_table(hw), title="Hardware", expand=False))
        et = Table(title="Inference engines", show_edge=False, title_justify="left")
        for c in ("engine", "status", "detail"):
            et.add_column(c)
        for row in _engine_rows(cfg, hw):
            et.add_row(*row)
        console.print(et)
        corp = corporate.status(cfg)
        if corp["configured"] or corp.get("pip_index") or corp.get("hdfs_cli"):
            console.print(Panel(_corp_table(corp), title="Corporate", expand=False))
        if problems:
            console.print("[red]Configuration problems:[/]")
            for p in problems:
                console.print(f"  {escape(p)}")
        nxt = []
        if missing_core:
            nxt.append(f"make bootstrap            # install missing: {', '.join(missing_core)}")
        nxt.append("make recommend TASK=coding")
        nxt.append("make harness-list")
        console.print(Panel("\n".join(nxt), title="Next steps", expand=False))
    if problems:
        raise typer.Exit(ExitCode.USAGE)
    if strict and missing_core:
        raise typer.Exit(ExitCode.CHECK_FAILED)


# ================================================================ configuration


@config_app.command("show")
def config_show(json_out: bool = _json_opt()) -> None:
    """Show resolved configuration paths and defaults."""
    data = STATE.cfg.as_dict()
    if _set_json(json_out):
        emit_json(data)
        return
    import yaml

    console.print(f"[bold]built-in config:[/] {data['builtin_dir']}")
    console.print(f"[bold]user overrides:[/]  {data['user_dir']}")
    for k, v in data["sources"].items():
        console.print(f"  {k}: {', '.join(v) or '(none)'}")
    console.print("[bold]resolved paths:[/]")
    for k, v in data["resolved_paths"].items():
        console.print(f"  {k}: {v}")
    console.print("[bold]defaults:[/]")
    console.print(yaml.safe_dump(data["defaults"], sort_keys=False).rstrip(), markup=False)


@config_app.command("validate")
def config_validate(json_out: bool = _json_opt()) -> None:
    """Validate all YAML files (schemas, cross-references)."""
    cfg = STATE.cfg
    problems = cfg.validate_all()
    summary = {}
    if not problems:
        summary = {
            "models": len(cfg.models.models),
            "variants": sum(len(m.variants) for m in cfg.models.models.values()),
            "harnesses": len(cfg.harnesses.harnesses),
            "profiles": len(cfg.profiles.profiles),
        }
        for p in cfg.profiles.profiles:
            try:
                resolve_profile(cfg, p)
            except SetupWorkError as e:
                problems.append(e.message)
    if _set_json(json_out):
        emit_json({"ok": not problems, "problems": problems, "counts": summary})
    elif problems:
        for p in problems:
            err_console.print(f"{BAD} {escape(p)}")
    else:
        console.print(
            f"{OK} configuration valid: " + ", ".join(f"{v} {k}" for k, v in summary.items())
        )
    if problems:
        raise typer.Exit(ExitCode.CHECK_FAILED)


# ==================================================================== bootstrap


def _print_tool_plans(plans: list[bs.ToolPlan]) -> None:
    t = Table(show_edge=False)
    for c in ("tool", "status", "detail"):
        t.add_column(c)
    style = {"satisfied": OK, "install": "[cyan]→[/]", "manual": WARN, "unsupported": BAD}
    for p in plans:
        t.add_row(p.tool, f"{style[p.status]} {p.status}", escape(p.detail))
    STATE.runner.info("")
    (err_console if STATE.json else console).print(t)


@app.command("bootstrap")
def bootstrap_cmd(
    group: str = typer.Argument(
        "all", help="all | python | node | tools", envvar="SETUP_WORK_GROUP"
    ),
    tool: list[str] = typer.Option(None, "--tool", help="Specific tool(s) instead of a group."),
    allow_sudo: bool = typer.Option(
        False,
        "--allow-sudo",
        envvar="SETUP_WORK_ALLOW_SUDO",
        help="Permit sudo package-manager commands (still confirmed).",
    ),
    force: bool = typer.Option(False, "--force", help="Reinstall even if already satisfied."),
    json_out: bool = _json_opt(),
) -> None:
    """Install missing developer tooling, user-space first. Idempotent."""
    cfg = STATE.cfg
    _set_json(json_out)
    if tool:
        tools = list(tool)
    elif group in bs.GROUPS:
        tools = bs.GROUPS[group]
    else:
        raise UsageError(f"unknown group {group!r}", hint="all, python, node or tools")
    if group == "all" and not tool:
        corp_steps = corporate.plan_scripts(cfg)
        if corp_steps:
            STATE.runner.info("[bold]Corporate setup scripts[/]")
            execute(corp_steps, STATE.runner, cfg.data_dir)
            corporate.apply_environment(cfg)
    plans = bs.plan_tools(cfg, tools, allow_sudo=allow_sudo, force=force)
    _run_tool_plans(cfg, plans)


def _run_tool_plans(cfg: Config, plans: list[bs.ToolPlan]) -> None:
    _print_tool_plans(plans)
    if STATE.json and STATE.runner.dry_run:
        emit_json(
            [
                {
                    "tool": p.tool,
                    "status": p.status,
                    "detail": p.detail,
                    "steps": [s.to_dict() for s in p.steps],
                }
                for p in plans
            ]
        )
        return
    todo = [p for p in plans if p.status == "install"]
    for p in plans:
        if p.status == "manual":
            for s in p.steps:
                STATE.runner.info(f"  {WARN} {p.tool}: {escape(s.render())}")
    if todo:
        steps = [s for p in todo for s in p.steps]
        STATE.runner.info(f"\nPlan: {len(steps)} step(s) for {', '.join(p.tool for p in todo)}")
        execute(steps, STATE.runner, cfg.data_dir)
    else:
        STATE.runner.info(f"{OK} nothing to install")
    env = detect_environment(cfg, ["uv"])
    if not env.bin_dir_on_path and any(s.kind == "link" for p in todo for s in p.steps):
        STATE.runner.info(
            f"\n{WARN} {cfg.bin_dir} is not on PATH. setup-work does not edit shell startup "
            f"files; add this line to your shell profile yourself:\n    {bs.path_hint(cfg)}"
        )
    if any(p.status in ("manual", "unsupported") for p in plans):
        STATE.runner.info(f"{WARN} some tools need manual action (see above)")


# ====================================================================== engines


@engine_app.command("install")
def engine_install(
    engine: str = typer.Argument(..., envvar="SETUP_WORK_ENGINE", help="llama.cpp | vllm"),
    source: str = typer.Option(
        "auto", "--source", envvar="SETUP_WORK_SOURCE", help="auto | corporate | upstream"
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Install an inference engine: corporate artifact (HDFS) or pinned upstream release."""
    cfg, hw = STATE.cfg, STATE.hw
    name = get_engine(engine).name
    steps, notes = bs.plan_engine(cfg, name, hw, source=source)
    if _set_json(json_out) and STATE.runner.dry_run:
        emit_json({"engine": name, "notes": notes, "steps": [s.to_dict() for s in steps]})
        return
    for n in notes:
        STATE.runner.warn(n)
    if not STATE.runner.dry_run:
        for i, s in enumerate(steps, 1):
            STATE.runner.info(f"  {i}. {escape(s.render())}")
        STATE.runner.require_confirmation(f"Install {name} ({len(steps)} steps)?")
    execute(steps, STATE.runner, cfg.data_dir)
    if not STATE.runner.dry_run:
        STATE.runner.info(f"{OK} {name} installed: {get_engine(name).find_binary(cfg)}")


@engine_app.command("status")
def engine_status(json_out: bool = _json_opt()) -> None:
    """Show which engines are installed and supported here."""
    rows = _engine_rows(STATE.cfg, STATE.hw)
    if _set_json(json_out):
        emit_json(
            [{"engine": r[0], "status": r[1].split("] ", 1)[-1], "detail": r[2]} for r in rows]
        )
        return
    t = Table(show_edge=False)
    for c in ("engine", "status", "detail"):
        t.add_column(c)
    for r in rows:
        t.add_row(*r)
    console.print(t)


# ======================================================================= models


@app.command("models")
def models_cmd(
    task: str = typer.Option(None, envvar="SETUP_WORK_TASK", help="Filter by workload."),
    json_out: bool = _json_opt(),
) -> None:
    """List the model catalog (no network access)."""
    cfg = STATE.cfg
    rows = []
    for mid, m in sorted(cfg.models.models.items()):
        if task and task not in m.workloads:
            continue
        rows.append((mid, m))
    if _set_json(json_out):
        emit_json([{"id": mid, **m.model_dump()} for mid, m in rows])
        return
    t = Table(title="Model catalog", show_edge=False, title_justify="left")
    for c in ("id", "params", "arch", "context", "workloads", "variants (engine)"):
        t.add_column(c)
    for mid, m in rows:
        params = f"{m.parameters_b:g}B" + (
            f" ({m.active_parameters_b:g}B act.)" if m.active_parameters_b else ""
        )
        vs = ", ".join(f"{v.id}[{'L' if v.engines == ['llama.cpp'] else 'V'}]" for v in m.variants)
        t.add_row(
            mid + (" 🔒" if m.gated else ""),
            params,
            m.architecture,
            str(m.context_length),
            ",".join(m.workloads),
            vs,
        )
    console.print(t)
    console.print(
        "[dim]L = llama.cpp (GGUF), V = vLLM (safetensors); 🔒 = gated on Hugging Face. "
        "Use MODEL=<id>:<variant>.[/]"
    )


@app.command("recommend")
def recommend_cmd(
    task: str = typer.Option(
        "general",
        envvar="SETUP_WORK_TASK",
        help="general | coding | reasoning | agents | embedding",
    ),
    context: int = typer.Option(None, envvar="SETUP_WORK_CONTEXT", help="Context length (tokens)."),
    engine: str = typer.Option(None, envvar="SETUP_WORK_ENGINE", help="llama.cpp | vllm"),
    concurrency: int = typer.Option(
        None, envvar="SETUP_WORK_CONCURRENCY", help="Concurrent sequences."
    ),
    tp: int = typer.Option(None, "--tp", envvar="SETUP_WORK_TP", help="Force tensor parallel."),
    kv_cache_dtype: str = typer.Option(
        None, envvar="SETUP_WORK_KV_CACHE_DTYPE", help="fp16 | fp8 | q8_0"
    ),
    limit: int = typer.Option(None, "--limit", envvar="SETUP_WORK_LIMIT"),
    explain: bool = typer.Option(
        False,
        "--explain",
        envvar="SETUP_WORK_EXPLAIN",
        help="Show rationale and rejected candidates.",
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Recommend models that fit this hardware (no downloads)."""
    cfg, hw = STATE.cfg, STATE.hw
    d = cfg.defaults
    req = RecommendRequest(
        task=task,
        context=context or d.recommend.default_context,
        concurrency=concurrency or d.recommend.default_concurrency,
        engine=engine,
        kv_cache_dtype=kv_cache_dtype or d.inference.default_kv_cache_dtype,
        tensor_parallel=tp,
        gpu_memory_utilization=d.inference.gpu_memory_utilization,
        max_results=limit or d.recommend.max_results,
    )
    if req.kv_cache_dtype not in ("fp16", "bf16", "fp8", "q8_0"):
        raise UsageError(f"invalid KV cache dtype {req.kv_cache_dtype}")
    res = recommend(cfg, hw, req)
    if _set_json(json_out):
        emit_json(res.to_dict())
        return
    console.print(
        f"[bold]Task[/] {task} · [bold]context[/] {req.context} · [bold]concurrency[/] "
        f"{req.concurrency} · [bold]engine[/] {engine or 'any'} · [bold]hardware[/] "
        f"{hw.accelerator}, {hw.gpu_count} GPU(s)"
        + (f" [yellow]({hw.source})[/]" if hw.source != "live" else "")
    )
    if not res.candidates:
        console.print(f"{BAD} no compatible model fits these constraints")
    else:
        t = Table(show_edge=False)
        t.add_column("#")
        t.add_column("model:variant", no_wrap=True)
        t.add_column("engine", no_wrap=True)
        for c in ("quant", "memory", "layout", "max ctx", "confidence", "score"):
            t.add_column(c)
        for i, c in enumerate(res.candidates, 1):
            mem = gib(c.total_bytes) + (f" ×{c.tensor_parallel}" if c.tensor_parallel > 1 else "")
            t.add_row(
                str(i),
                c.ref + (" 🔒" if c.gated else ""),
                c.engine,
                c.quantization,
                mem,
                c.layout + (f" TP{c.tensor_parallel}" if c.tensor_parallel > 1 else ""),
                str(c.max_context or "-"),
                c.confidence + ("" if c.compatibility == "supported" else f" ({c.compatibility})"),
                f"{c.score:.2f}",
            )
        console.print(t)
        top = res.candidates[0]
        console.print(f"\n[bold]Top pick:[/] {top.display_name} ({top.ref}) on {top.engine}")
        for line in top.rationale:
            console.print(f"  • {escape(line)}")
        for w in top.warnings:
            console.print(f"  {WARN} {escape(w)}")
        console.print(f"\n  download: make model-pull MODEL={top.ref}")
        console.print(f"  serve:    {top.serve_hint()}")
        if explain:
            for c in res.candidates[1:]:
                console.print(f"\n[bold]{c.ref}[/] on {c.engine}")
                for line in c.rationale + [f"warning: {w}" for w in c.warnings]:
                    console.print(f"  • {escape(line)}")
    if explain and res.rejections:
        console.print("\n[bold]Rejected[/]")
        for r in res.rejections:
            console.print(f"  {BAD} {r.model_id} [{r.engine}]")
            for why in r.reasons[:4]:
                console.print(f"      {escape(why)}")
    elif res.rejections:
        console.print(
            f"[dim]{len(res.rejections)} model/engine pairs rejected; add --explain "
            "(make recommend EXPLAIN=1) to see why[/]"
        )
    console.print(
        "[dim]Memory figures are estimates; quality scores are editorial, not benchmarks.[/]"
    )


@model_app.command("pull")
def model_pull(
    model: str = typer.Argument(..., envvar="SETUP_WORK_MODEL", help="id:variant or org/repo"),
    revision: str = typer.Option(None, envvar="SETUP_WORK_REVISION", help="Branch, tag or commit."),
    source: str = typer.Option(
        "auto", "--source", envvar="SETUP_WORK_SOURCE", help="auto | corporate | hf"
    ),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip sha256 verification."),
    json_out: bool = _json_opt(),
) -> None:
    """Download model weights: corporate artifact (HDFS/zip) or Hugging Face cache."""
    cfg = STATE.cfg
    _set_json(json_out)
    if source not in ("auto", "corporate", "hf"):
        raise UsageError(f"unknown source {source!r}", hint="auto, corporate or hf")
    if not ("/" in model and ":" not in model):
        spec, variant = resolve(cfg, model)
        if variant is None:
            raise UsageError(
                f"specify which variant of {model} to download",
                hint="variants: " + ", ".join(f"{model}:{v.id}" for v in spec.variants),
            )
        art = hub.model_artifact(cfg, model.split(":")[0], variant)
        if source == "corporate" and art is None:
            raise SetupWorkError(
                f"no corporate artifact configured for {model}",
                hint="map it in corporate.yaml under `models:` (see examples/corp/)",
            )
        if art is not None and source != "hf":
            entry = hub.pull_artifact(cfg, model.split(":")[0], variant, art, STATE.runner)
            if STATE.json:
                emit_json({"ref": model, "source": art.uri, "path": str(entry)})
            elif not STATE.runner.dry_run:
                console.print(f"{OK} {model} available at {entry}")
            return
    plan = hub.plan_pull(cfg, model, revision)
    if STATE.json and STATE.runner.dry_run:
        emit_json(plan.to_dict())
        return
    STATE.runner.info(
        f"{plan.repository}@{(plan.resolved_revision or plan.revision or 'main')[:12]}: "
        f"{len(plan.files)} file(s), {gib(plan.total_bytes)} total, "
        f"{gib(plan.cached_bytes)} cached, "
        f"{gib(plan.needed_bytes)} to download; {gib(plan.free_bytes)} free in {plan.cache_dir}"
    )
    path = hub.pull(cfg, plan, STATE.runner, verify=False if no_verify else None)
    if STATE.json:
        emit_json({**plan.to_dict(), "path": str(path)})
    elif not STATE.runner.dry_run:
        console.print(f"{OK} {model} available at {path}")


@model_app.command("list")
def model_list(json_out: bool = _json_opt()) -> None:
    """List models present in the local cache."""
    cfg = STATE.cfg
    repos = hub.list_cached(cfg)
    arts = hub.list_artifact_models(cfg)
    if _set_json(json_out):
        emit_json(
            {
                "cache_dir": str(hub.cache_dir(cfg)),
                "repos": [r.__dict__ for r in repos],
                "artifact_dir": str(cfg.models_dir),
                "artifacts": arts,
            }
        )
        return
    if arts:
        at = Table(
            title=f"Corporate artifacts ({cfg.models_dir})", show_edge=False, title_justify="left"
        )
        for c in ("model:variant", "size", "source", "installed"):
            at.add_column(c)
        for a in arts:
            at.add_row(
                a["ref"], gib(a["size_bytes"]), a["uri"] or "", (a["installed_at"] or "")[:19]
            )
        console.print(at)
    console.print(f"cache: {hub.cache_dir(cfg)}")
    if not repos and arts:
        return
    if not repos:
        console.print(
            "no models cached. Try: make recommend, then make model-pull MODEL=<id:variant>"
        )
        return
    t = Table(show_edge=False)
    for c in ("repository", "size", "revisions", "catalog refs"):
        t.add_column(c)
    for r in repos:
        t.add_row(r.repo_id, gib(r.size_bytes), ",".join(r.revisions), ", ".join(r.catalog_refs))
    console.print(t)


@model_app.command("info")
def model_info(
    model: str = typer.Argument(..., envvar="SETUP_WORK_MODEL"),
    remote: bool = typer.Option(
        False,
        "--remote",
        envvar="SETUP_WORK_REMOTE",
        help="Query Hugging Face for current file sizes.",
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Show catalog metadata (and optionally remote file sizes)."""
    cfg = STATE.cfg
    spec, variant = resolve(cfg, model)
    data: dict[str, Any] = {"id": model.split(":")[0], **spec.model_dump()}
    for v in data["variants"]:
        var = spec.variant(v["id"])
        v["cached"] = bool(var and hub.local_model(cfg, data["id"], var))
        art = hub.model_artifact(cfg, data["id"], var) if var else None
        v["artifact"] = art.uri if art else None
    if remote and variant:
        data["remote"] = hub.plan_pull(cfg, model).to_dict()
    if _set_json(json_out):
        emit_json(data)
        return
    a = spec.attention
    console.print(
        f"[bold]{spec.display_name}[/] ({data['id']}) · license {spec.license}"
        + (" · gated" if spec.gated else "")
    )
    console.print(
        f"  {spec.architecture}, {spec.parameters_b:g}B params"
        + (f", {spec.active_parameters_b:g}B active" if spec.active_parameters_b else "")
        + f", native context {spec.context_length}"
    )
    console.print(
        f"  attention: {a.num_layers} layers, {a.num_attention_heads} heads, "
        f"{a.num_kv_heads} KV heads × {a.head_dim}"
        + (
            f", {a.sliding_window_layers} sliding layers (window {a.sliding_window})"
            if a.sliding_window_layers
            else ""
        )
    )
    t = Table(show_edge=False)
    for c in ("variant", "format", "quant", "size", "engines", "source", "cached"):
        t.add_column(c)
    for v in data["variants"]:
        t.add_row(
            v["id"],
            v["format"],
            v["quantization"],
            f"{v['size_gb']:g} GB",
            ",".join(v["engines"]),
            v["artifact"] or v["repository"] or "",
            OK if v["cached"] else "",
        )
    console.print(t)
    if "remote" in data:
        r = data["remote"]
        console.print(
            f"remote: {len(r['files'])} files, {gib(r['total_bytes'])} at {r['resolved_revision']}"
        )


@model_app.command("remove")
def model_remove(model: str = typer.Argument(..., envvar="SETUP_WORK_MODEL")) -> None:
    """Delete a cached model repository (asks for confirmation)."""
    freed = hub.remove_artifact_model(STATE.cfg, model, STATE.runner)
    if freed is None:
        freed = hub.remove(STATE.cfg, model, STATE.runner)
    if not STATE.runner.dry_run:
        console.print(f"{OK} freed {gib(freed)}")


# ===================================================================== harnesses


@harness_app.command("list")
def harness_list(json_out: bool = _json_opt()) -> None:
    """List known harness recipes and their installation state."""
    rows = hz.list_harnesses(STATE.cfg)
    if _set_json(json_out):
        emit_json(rows)
        return
    t = Table(show_edge=False)
    for c in ("name", "source", "package / repo", "recipe", "installed", "binaries"):
        t.add_column(c)
    for r in rows:
        t.add_row(
            r["name"],
            r["source"],
            r["package"] or "",
            (r["version"] or "") + ("" if r["pinned"] else " [yellow](unpinned)[/]"),
            f"{OK} {r['installed']}" if r["installed"] else "-",
            ",".join(r["binaries"]),
        )
    console.print(t)
    console.print(f"[dim]custom recipes: {STATE.cfg.user_dir / 'harnesses.yaml'}[/]")


def _harness_apply(name: str, plan: hz.HarnessPlan, action: str) -> None:
    for w in plan.warnings:
        STATE.runner.warn(w)
    if not plan.steps:
        STATE.runner.info(f"{OK} nothing to do for {name}")
        return
    STATE.runner.info(f"[bold]{action} {name}[/] ({plan.spec.source}):")
    if plan.needs_confirmation and not STATE.runner.dry_run:
        for i, s in enumerate(plan.steps, 1):
            STATE.runner.info(f"  {i}. {escape(s.render())}")
        STATE.runner.require_confirmation(f"{action} {name} from {plan.spec.source}?")
    execute(plan.steps, STATE.runner, STATE.cfg.data_dir)
    hz.write_manifest(STATE.cfg, name, plan.spec, STATE.runner)
    if not STATE.runner.dry_run:
        STATE.runner.info(
            f"{OK} {name} {plan.spec.version or plan.spec.ref} ready; "
            f"check with: make harness-doctor NAME={name}"
        )


@harness_app.command("install")
def harness_install(
    name: str = typer.Argument(..., envvar="SETUP_WORK_NAME"),
    version: str = typer.Option(None, envvar="SETUP_WORK_VERSION", help="Override pinned version."),
    ref: str = typer.Option(None, envvar="SETUP_WORK_REF", help="Override git ref."),
    allow_build: bool = typer.Option(False, "--allow-build", envvar="SETUP_WORK_ALLOW_BUILD"),
    json_out: bool = _json_opt(),
) -> None:
    """Install a harness from its declarative recipe."""
    plan = hz.plan_install(STATE.cfg, name, version=version, ref=ref, allow_build=allow_build)
    if _set_json(json_out) and STATE.runner.dry_run:
        emit_json(plan.to_dict())
        return
    _harness_apply(name, plan, "install")


@harness_app.command("update")
def harness_update(
    name: str = typer.Argument(..., envvar="SETUP_WORK_NAME"),
    version: str = typer.Option(None, envvar="SETUP_WORK_VERSION"),
    ref: str = typer.Option(None, envvar="SETUP_WORK_REF"),
    allow_build: bool = typer.Option(False, "--allow-build", envvar="SETUP_WORK_ALLOW_BUILD"),
) -> None:
    """Reinstall a harness at the recipe (or given) version / ref."""
    plan = hz.plan_install(
        STATE.cfg, name, version=version, ref=ref, update=True, allow_build=allow_build
    )
    _harness_apply(name, plan, "update")


@harness_app.command("remove")
def harness_remove(name: str = typer.Argument(..., envvar="SETUP_WORK_NAME")) -> None:
    """Remove a harness and the links setup-work created for it."""
    steps = hz.plan_remove(STATE.cfg, name)
    execute(steps, STATE.runner, STATE.cfg.data_dir)
    if not STATE.runner.dry_run:
        console.print(f"{OK} removed {name}")


@harness_app.command("doctor")
def harness_doctor(
    name: str = typer.Argument(None, envvar="SETUP_WORK_NAME", help="Omit to check all installed."),
    json_out: bool = _json_opt(),
) -> None:
    """Check a harness installation, PATH resolution and required environment variables."""
    cfg = STATE.cfg
    names = [name] if name else [r["name"] for r in hz.list_harnesses(cfg) if r["installed"]]
    if not names:
        console.print("no harnesses installed")
        return
    results = [hz.doctor(cfg, n) for n in names]
    if _set_json(json_out):
        emit_json([r.to_dict() for r in results])
    else:
        for r in results:
            console.print(f"[bold]{r.name}[/] {OK if r.ok else BAD}")
            for c, ok, detail in r.checks:
                console.print(
                    f"  {OK if ok else (WARN if c == 'version pinned' else BAD)} {c}: "
                    f"{escape(detail)}"
                )
    if not all(r.ok for r in results):
        raise typer.Exit(ExitCode.CHECK_FAILED)


# ===================================================================== inference


def _serve_options(**kw: Any) -> ServeOptions:
    extra = kw.pop("extra_args", None) or shlex.split(os.environ.get("SETUP_WORK_ENGINE_ARGS", ""))
    return ServeOptions(extra_args=list(extra), **{k: v for k, v in kw.items() if v is not None})


def _do_serve(model: str, opts: ServeOptions, detach: bool, plan_only: bool) -> None:
    cfg, hw = STATE.cfg, STATE.hw
    plan = plan_launch(cfg, hw, model, opts, check_cache=not plan_only)
    if STATE.json and (plan_only or STATE.runner.dry_run):
        emit_json(plan.to_dict())
        return
    for line in plan.rationale:
        STATE.runner.info(f"  • {escape(line)}")
    for w in plan.warnings:
        STATE.runner.warn(w)
    if plan_only:
        STATE.runner.show_command(plan.argv)
        return
    mgr = ServerManager(cfg, STATE.runner)
    st = mgr.start(plan, detach=detach)
    if st and STATE.json:
        emit_json(st.to_dict())
    elif st and detach:
        console.print(f"\nOpenAI-compatible API: {st.base_url}/v1   model: {st.served_model_name}")
        console.print("test: make serve-test   logs: make serve-logs   stop: make serve-stop")


_SERVE_OPTS = dict(
    engine=typer.Option(None, envvar="SETUP_WORK_ENGINE", help="llama.cpp | vllm"),
    name=typer.Option(
        "default", "--name", envvar="SETUP_WORK_INSTANCE", help="Server instance name."
    ),
    host=typer.Option(None, envvar="SETUP_WORK_HOST", help="Bind address (default 127.0.0.1)."),
    port=typer.Option(None, envvar="SETUP_WORK_PORT"),
    context=typer.Option(None, envvar="SETUP_WORK_CONTEXT"),
    concurrency=typer.Option(None, envvar="SETUP_WORK_CONCURRENCY"),
    tp=typer.Option(None, "--tp", envvar="SETUP_WORK_TP"),
    gpu_util=typer.Option(None, "--gpu-util", envvar="SETUP_WORK_GPU_UTIL"),
    gpu_layers=typer.Option(None, "--gpu-layers", envvar="SETUP_WORK_GPU_LAYERS"),
    threads=typer.Option(None, envvar="SETUP_WORK_THREADS"),
    kv_cache_dtype=typer.Option(None, envvar="SETUP_WORK_KV_CACHE_DTYPE"),
    served_name=typer.Option(None, "--served-name", envvar="SETUP_WORK_SERVED_NAME"),
    revision=typer.Option(None, envvar="SETUP_WORK_REVISION"),
    force=typer.Option(
        False,
        "--force",
        envvar="SETUP_WORK_FORCE",
        help="Launch even if estimated not to fit / unsupported.",
    ),
    detach=typer.Option(True, "--detach/--foreground", envvar="SETUP_WORK_DETACH"),
    plan_only=typer.Option(
        False, "--plan", envvar="SETUP_WORK_PLAN", help="Only print the launch plan."
    ),
)


@app.command("serve")
def serve_cmd(
    model: str = typer.Argument(
        ..., envvar="SETUP_WORK_MODEL", help="id[:variant], org/repo (vLLM) or path to .gguf"
    ),
    engine: str = _SERVE_OPTS["engine"],
    name: str = _SERVE_OPTS["name"],
    host: str = _SERVE_OPTS["host"],
    port: int = _SERVE_OPTS["port"],
    context: int = _SERVE_OPTS["context"],
    concurrency: int = _SERVE_OPTS["concurrency"],
    tp: int = _SERVE_OPTS["tp"],
    gpu_util: float = _SERVE_OPTS["gpu_util"],
    gpu_layers: int = _SERVE_OPTS["gpu_layers"],
    threads: int = _SERVE_OPTS["threads"],
    kv_cache_dtype: str = _SERVE_OPTS["kv_cache_dtype"],
    served_name: str = _SERVE_OPTS["served_name"],
    revision: str = _SERVE_OPTS["revision"],
    force: bool = _SERVE_OPTS["force"],
    detach: bool = _SERVE_OPTS["detach"],
    plan_only: bool = _SERVE_OPTS["plan_only"],
    profile: str = typer.Option(
        None,
        "--profile",
        envvar="SETUP_WORK_PROFILE",
        help="Take engine/host/port defaults from a profile.",
    ),
    extra: list[str] = typer.Argument(None, help="Extra engine args after `--`."),
    json_out: bool = _json_opt(),
) -> None:
    """Start an OpenAI-compatible inference server (localhost by default)."""
    _set_json(json_out)
    if profile:
        inf = resolve_profile(STATE.cfg, profile).inference
        engine = engine or inf.get("engine")
        host = host or inf.get("host")
        port = port or inf.get("port")
        gpu_util = gpu_util or inf.get("gpu_memory_utilization")
    opts = _serve_options(
        engine=engine,
        name=name,
        host=host,
        port=port,
        context=context,
        concurrency=concurrency,
        tensor_parallel=tp,
        gpu_memory_utilization=gpu_util,
        gpu_layers=gpu_layers,
        threads=threads,
        kv_cache_dtype=kv_cache_dtype,
        served_model_name=served_name,
        revision=revision,
        force=force,
        extra_args=extra,
    )
    _do_serve(model, opts, detach, plan_only)


@server_app.command("auto")
def serve_auto(
    task: str = typer.Option("coding", envvar="SETUP_WORK_TASK"),
    engine: str = _SERVE_OPTS["engine"],
    name: str = _SERVE_OPTS["name"],
    host: str = _SERVE_OPTS["host"],
    port: int = _SERVE_OPTS["port"],
    context: int = _SERVE_OPTS["context"],
    concurrency: int = _SERVE_OPTS["concurrency"],
    detach: bool = _SERVE_OPTS["detach"],
    plan_only: bool = _SERVE_OPTS["plan_only"],
    require_cached: bool = typer.Option(
        True, "--cached-only/--any", help="Only consider models already downloaded."
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Serve the best recommended model for TASK that is already downloaded."""
    _set_json(json_out)
    cfg, hw = STATE.cfg, STATE.hw
    d = cfg.defaults
    req = RecommendRequest(
        task=task,
        context=context or d.recommend.default_context,
        concurrency=concurrency or 1,
        engine=engine,
        kv_cache_dtype=d.inference.default_kv_cache_dtype,
        gpu_memory_utilization=d.inference.gpu_memory_utilization,
        max_results=50,
    )
    res = recommend(cfg, hw, req)
    chosen = None
    for c in res.candidates:
        _, var = resolve(cfg, c.ref)
        if not require_cached or (var and hub.local_model(cfg, c.model_id, var)):
            chosen = c
            break
    if chosen is None:
        top = res.candidates[0].ref if res.candidates else None
        raise SetupWorkError(
            f"no downloaded model fits task '{task}'"
            if res.candidates
            else f"no compatible model for task '{task}'",
            hint=f"download the top recommendation first: make model-pull MODEL={top}"
            if top
            else "relax CONTEXT/ENGINE constraints",
        )
    STATE.runner.info(f"selected {chosen.ref} on {chosen.engine} (score {chosen.score:.2f})")
    opts = _serve_options(
        engine=chosen.engine,
        name=name,
        host=host,
        port=port,
        context=req.context,
        concurrency=req.concurrency,
    )
    _do_serve(chosen.ref, opts, detach, plan_only or not require_cached)


@server_app.command("stop")
def serve_stop(
    name: str = _SERVE_OPTS["name"],
    all_servers: bool = typer.Option(False, "--all", help="Stop every recorded server."),
) -> None:
    """Gracefully stop a server started by setup-work (never touches other processes)."""
    mgr = ServerManager(STATE.cfg, STATE.runner)
    names = [s.name for s in mgr.all_states()] if all_servers else [name]
    for n in names:
        if mgr.stop(n):
            console.print(f"{OK} stopped {n}")


@server_app.command("status")
def serve_status(
    name: str = typer.Option(None, "--name", envvar="SETUP_WORK_INSTANCE"),
    json_out: bool = _json_opt(),
) -> None:
    """Show recorded servers, liveness and readiness."""
    rows = ServerManager(STATE.cfg, STATE.runner).status(name)
    if _set_json(json_out):
        emit_json(rows)
        return
    if not rows:
        console.print("no servers recorded")
        return
    t = Table(show_edge=False)
    for c in ("name", "engine", "model", "url", "pid", "state", "uptime"):
        t.add_column(c)
    for r in rows:
        state = (
            (f"{OK} ready" if r["healthy"] else f"{WARN} starting/unhealthy")
            if r["running"]
            else f"{BAD} not running (stale)"
        )
        t.add_row(
            r["name"],
            r["engine"],
            r["served_model_name"],
            r["base_url"] + "/v1",
            str(r["pid"]),
            state,
            f"{r['uptime_s']}s" if r["uptime_s"] is not None else "-",
        )
    console.print(t)


@server_app.command("logs")
def serve_logs(
    name: str = _SERVE_OPTS["name"],
    lines: int = typer.Option(100, "--lines", "-n", envvar="SETUP_WORK_LINES"),
    follow: bool = typer.Option(False, "--follow", "-f", envvar="SETUP_WORK_FOLLOW"),
) -> None:
    """Print (or follow) the server log."""
    mgr = ServerManager(STATE.cfg, STATE.runner)
    for line in mgr.tail(name, lines):
        print(line)
    if follow:
        path = mgr.log_dir / f"serve-{name}.log"
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                fh.seek(0, 2)
                while True:
                    line = fh.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        except FileNotFoundError:
            raise SetupWorkError(f"no log file {path}") from None


@server_app.command("test")
def serve_test(
    name: str = _SERVE_OPTS["name"],
    url: str = typer.Option(None, envvar="SETUP_WORK_URL", help="Test any OpenAI-compatible URL."),
    json_out: bool = _json_opt(),
) -> None:
    """Send a minimal OpenAI-compatible request to the server."""
    res = ServerManager(STATE.cfg, STATE.runner).smoke_test(name, url)
    if _set_json(json_out):
        emit_json(res)
        return
    console.print(
        f"{OK} {res['endpoint']} on {res['url']} (model {res['model']}) in {res['latency_s']}s"
    )
    if res.get("reply") is not None:
        console.print(f"  reply: {escape(str(res['reply'])[:200])!r}")
    if res.get("dimensions"):
        console.print(f"  embedding dimensions: {res['dimensions']}")


# ===================================================================== corporate


@corp_app.command("setup")
def corp_setup(
    script: list[str] = typer.Option(None, "--script", help="Only run these script names."),
) -> None:
    """Run corporate setup scripts (pip.conf, .npmrc, ...). Each is shown and confirmed."""
    cfg = STATE.cfg
    steps = corporate.plan_scripts(cfg, script or None)
    if not steps:
        console.print("no corporate scripts configured (corporate.yaml: scripts)")
        return
    execute(steps, STATE.runner, cfg.data_dir)
    # Re-read pip.conf written by the scripts so later uv calls in this run see it.
    applied = corporate.apply_environment(cfg)
    for k, v in applied.items():
        STATE.runner.info(f"  {k}={corporate.mask(v)}")


@corp_app.command("status")
def corp_status(json_out: bool = _json_opt()) -> None:
    """Show HDFS/Kerberos, mirror and artifact configuration."""
    data = corporate.status(STATE.cfg)
    if _set_json(json_out):
        emit_json(data)
        return
    console.print(Panel(_corp_table(data), title="Corporate", expand=False))


@corp_app.command("env")
def corp_env() -> None:
    """Print `export` lines for the corporate environment (for your shell or CI)."""
    import shlex

    cfg = STATE.cfg
    env = {k: os.path.expandvars(os.path.expanduser(v)) for k, v in cfg.corporate.env.items()}
    if cfg.corporate.bridge_pip_conf:
        env.update(corporate.pip_conf_bridge(corporate.read_pip_settings()))
    for k, v in env.items():
        print(f"export {k}={shlex.quote(v)}")


# ====================================================================== profiles


@profile_app.command("list")
def profile_list(json_out: bool = _json_opt()) -> None:
    """List available profiles."""
    profs = STATE.cfg.profiles.profiles
    if _set_json(json_out):
        emit_json({k: v.model_dump() for k, v in profs.items()})
        return
    for k, v in profs.items():
        ext = f" [dim](extends {', '.join(v.extends)})[/]" if v.extends else ""
        console.print(f"[bold]{k}[/]{ext}: {v.description}")


@profile_app.command("show")
def profile_show(
    name: str = typer.Argument(..., envvar="SETUP_WORK_PROFILE"), json_out: bool = _json_opt()
) -> None:
    """Show a fully resolved profile."""
    rp = resolve_profile(STATE.cfg, name)
    if _set_json(json_out):
        emit_json(rp.to_dict())
        return
    import yaml

    console.print(yaml.safe_dump(rp.to_dict(), sort_keys=False), markup=False)


def _validate_profile(cfg: Config, rp: Any) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    env = None
    for v in rp.validation:
        if v == "tools":
            env = env or detect_environment(cfg)
            for t in rp.tools_required:
                if t in ("build-tools",):
                    continue
                st = env.tool(t)
                out.append((f"tool {t}", bool(st and st.installed), st.version or "" if st else ""))
        elif v == "harnesses":
            for h in rp.harnesses_required:
                out.append((f"harness {h}", hz.read_manifest(cfg, h) is not None, ""))
        elif v == "engines":
            for e in rp.engines:
                if e == "vllm" and (
                    STATE.hw.os.system != "Linux" or STATE.hw.accelerator != "nvidia"
                ):
                    out.append((f"engine {e}", False, "unsupported on this machine"))
                    continue
                b = get_engine(e).find_binary(cfg)
                out.append((f"engine {e}", b is not None, b or "not installed"))
        elif v.startswith("gpu:"):
            want = v.split(":", 1)[1]
            hw = STATE.hw
            ok = hw.accelerator == want if want != "any" else hw.accelerator in ("nvidia", "apple")
            out.append((f"gpu {want}", ok, hw.accelerator))
    return out


@app.command("setup")
def setup_cmd(
    profile: str = typer.Argument("developer", envvar="SETUP_WORK_PROFILE"),
    with_optional: bool = typer.Option(
        False,
        "--with-optional",
        envvar="SETUP_WORK_WITH_OPTIONAL",
        help="Also install optional tools and harnesses.",
    ),
    allow_sudo: bool = typer.Option(False, "--allow-sudo", envvar="SETUP_WORK_ALLOW_SUDO"),
    allow_downloads: bool = typer.Option(
        False,
        "--allow-downloads",
        envvar="SETUP_WORK_ALLOW_DOWNLOADS",
        help="Download the profile's models.pull list.",
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Apply a development profile: tools → engines → harnesses → (models) → validation."""
    cfg, r = STATE.cfg, STATE.runner
    rp = resolve_profile(cfg, profile)
    tools = rp.tools_required + (rp.tools_optional if with_optional else [])
    harnesses = rp.harnesses_required + (rp.harnesses_optional if with_optional else [])
    plans = bs.plan_tools(cfg, tools, allow_sudo=allow_sudo)
    if _set_json(json_out) and r.dry_run:
        emit_json(
            {
                "profile": rp.to_dict(),
                "tools": [
                    {"tool": p.tool, "status": p.status, "steps": [s.to_dict() for s in p.steps]}
                    for p in plans
                ],
                "engines": rp.engines,
                "harnesses": harnesses,
                "models": rp.model_pull if allow_downloads else [],
            }
        )
        return
    r.info(f"[bold]Profile {profile}[/] ({' → '.join(rp.chain)}): {rp.description}")
    r.info("\n[bold]1. Tools[/]")
    _run_tool_plans(cfg, plans)
    r.info("\n[bold]2. Engines[/]")
    for e in rp.engines:
        if get_engine(e).find_binary(cfg):
            r.info(f"  {OK} {e} already installed")
            continue
        try:
            steps, notes = bs.plan_engine(cfg, e, STATE.hw)
        except SetupWorkError as ex:
            r.warn(f"{e}: {ex.message}")
            continue
        for n in notes:
            r.warn(n)
        if r.confirm(f"Install engine {e} ({len(steps)} steps)?"):
            execute(steps, r, cfg.data_dir)
    if not rp.engines:
        r.info("  (none)")
    r.info("\n[bold]3. Harnesses[/]")
    for h in harnesses:
        try:
            plan = hz.plan_install(cfg, h)
        except SetupWorkError as ex:
            r.warn(f"{h}: {ex.message}" + (f" ({ex.hint})" if ex.hint else ""))
            continue
        _harness_apply(h, plan, "install")
    if not harnesses:
        r.info("  (none)")
    r.info("\n[bold]4. Models[/]")
    if rp.model_pull and allow_downloads:
        for ref in rp.model_pull:
            hub.pull(cfg, hub.plan_pull(cfg, ref), r)
    elif rp.model_pull:
        r.info(
            f"  skipped {len(rp.model_pull)} model(s): pass --allow-downloads (ALLOW_DOWNLOADS=1)"
        )
    else:
        pref = rp.model_preferences
        hint = " ".join(f"{k.upper()}={v}" for k, v in pref.items())
        r.info(f"  no downloads configured. Suggested: make recommend {hint}".rstrip())
    r.info("\n[bold]5. Validation[/]")
    if r.dry_run:
        r.info("  (skipped in dry-run)")
        return
    results = _validate_profile(cfg, rp)
    for c, ok, detail in results:
        r.info(f"  {OK if ok else BAD} {c} {escape(detail)}")
    if any(not ok for _, ok, _ in results):
        raise CheckFailedError(
            f"profile {profile}: some validation checks failed",
            hint="see the items marked ✘ above; `make doctor` for details",
        )
    r.info(f"\n{OK} profile {profile} is ready")


# ========================================================================= clean


@app.command("clean")
def clean_cmd(
    downloads: bool = typer.Option(
        True, "--downloads/--no-downloads", help="Remove cached installer archives."
    ),
    stale_state: bool = typer.Option(
        True,
        "--stale-state/--no-stale-state",
        help="Remove state files of servers that are not running.",
    ),
) -> None:
    """Remove setup-work caches (never models, harnesses or engines)."""
    cfg, r = STATE.cfg, STATE.runner
    steps: list[Step] = []
    if downloads:
        for d in (cfg.tools_dir / "downloads", cfg.engines_dir / "llama.cpp" / "downloads"):
            if d.exists():
                steps.append(Step("remove", f"remove installer archives in {d}", dest=d))
    if stale_state:
        mgr = ServerManager(cfg, r)
        for st in mgr.all_states():
            if not mgr.owns(st):
                steps.append(
                    Step(
                        "remove",
                        f"remove stale server state {st.name}",
                        dest=mgr.dir / f"{st.name}.json",
                    )
                )
    if not steps:
        r.info(f"{OK} nothing to clean")
        return
    for s in steps:
        r.info(f"  {escape(s.render())}")
    r.require_confirmation(f"Remove {len(steps)} item(s)?", destructive=True)
    execute(steps, r, cfg.data_dir)


# ========================================================================= entry


def main(argv: list[str] | None = None) -> int:
    STATE.__init__()  # fresh state per invocation (matters when embedded / in tests)
    cmd = typer.main.get_command(app)
    try:
        rv = cmd.main(args=argv, prog_name="setup-work", standalone_mode=False)
        return int(rv) if isinstance(rv, int) else 0
    except SetupWorkError as e:
        err_console.print(f"[bold red]error:[/] {escape(e.message)}")
        if e.hint:
            err_console.print(f"[cyan]hint:[/] {escape(e.hint)}")
        return int(e.exit_code)
    except typer.Exit as e:
        return int(getattr(e, "exit_code", 0) or 0)
    except typer.Abort:
        err_console.print("aborted")
        return int(ExitCode.ABORTED)
    except typer.TyperException as e:
        # Usage errors (bad option values, unknown commands). Typer vendors click.
        show = getattr(e, "show", None)
        if callable(show):
            show()
        else:
            err_console.print(f"error: {e}")
        code = getattr(e, "exit_code", ExitCode.USAGE)
        return int(code) if code is not None else int(ExitCode.USAGE)
    except KeyboardInterrupt:
        err_console.print("\ninterrupted")
        return 130
    except BrokenPipeError:
        return 0


def _entry() -> None:  # pragma: no cover
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _entry()
