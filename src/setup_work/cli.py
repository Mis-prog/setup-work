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
from .errors import ExitCode, NotFoundError, SetupWorkError, UsageError
from .hardware import HardwareInfo, detect
from .inference import ServeOptions, ServerManager, plan_launch
from .memory import gib
from .models import RecommendRequest, recommend, resolve
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
app.add_typer(config_app, name="config")
app.add_typer(model_app, name="model")
app.add_typer(harness_app, name="harness")
app.add_typer(server_app, name="server")
app.add_typer(engine_app, name="engine")

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
            paths = [cfg.data_dir, cfg.models_dir, Path.home()]
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
    t.add_row("pip index", c.get("pip_index") or f"{WARN} not configured (make registries)")
    t.add_row("uv index", c.get("uv_index") or f"{WARN} not configured (make registries)")
    if "npm_registry" in c:
        reg = c.get("npm_registry") or ""
        public = not reg or "registry.npmjs.org" in reg
        t.add_row(
            "npm registry", f"{WARN} {reg or 'not configured'} (make registries)" if public else reg
        )
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
    core = ["python", "uv", "git"]
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
                tt.add_row(t.name, f"{mark} missing", "", t.note or "")
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
        console.print(Panel(_corp_table(corporate.status(cfg)), title="Corporate", expand=False))
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
        }
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
    tool: list[str] = typer.Option(None, "--tool", help="uv, python and/or node (default: all)."),
    force: bool = typer.Option(False, "--force", help="Reinstall even if already satisfied."),
    json_out: bool = _json_opt(),
) -> None:
    """Install missing uv / Node.js from corporate archives; check Python. Idempotent."""
    cfg = STATE.cfg
    _set_json(json_out)
    plans = bs.plan_tools(cfg, list(tool) if tool else bs.BOOTSTRAP_TOOLS, force=force)
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
    json_out: bool = _json_opt(),
) -> None:
    """Install an inference engine from its corporate artifact (HDFS, NFS, HTTPS)."""
    cfg, hw = STATE.cfg, STATE.hw
    name = get_engine(engine).name
    steps, notes = bs.plan_engine(cfg, name, hw)
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
            mid,
            params,
            m.architecture,
            str(m.context_length),
            ",".join(m.workloads),
            vs,
        )
    console.print(t)
    console.print("[dim]L = llama.cpp (GGUF), V = vLLM (safetensors). Use MODEL=<id>:<variant>.[/]")


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
                c.ref,
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
    model: str = typer.Argument(..., envvar="SETUP_WORK_MODEL", help="id:variant"),
    json_out: bool = _json_opt(),
) -> None:
    """Fetch and unpack model weights from a corporate artifact (HDFS, NFS, HTTPS)."""
    cfg = STATE.cfg
    _set_json(json_out)
    spec, variant = resolve(cfg, model)
    if variant is None:
        raise UsageError(
            f"specify which variant of {model} to fetch",
            hint="variants: " + ", ".join(f"{model}:{v.id}" for v in spec.variants),
        )
    model_id = model.split(":")[0]
    art = hub.require_artifact(cfg, model_id, variant)
    entry = hub.pull_artifact(cfg, model_id, variant, art, STATE.runner)
    if STATE.json:
        emit_json({"ref": model, "source": art.uri, "path": str(entry)})
    elif not STATE.runner.dry_run:
        console.print(f"{OK} {model} available at {entry}")


@model_app.command("list")
def model_list(json_out: bool = _json_opt()) -> None:
    """List unpacked models."""
    cfg = STATE.cfg
    arts = hub.list_artifact_models(cfg)
    if _set_json(json_out):
        emit_json({"artifact_dir": str(cfg.models_dir), "artifacts": arts})
        return
    if not arts:
        console.print(
            f"no models in {cfg.models_dir}. "
            "Try: make recommend, then make model-pull MODEL=<id:variant>"
        )
        return
    t = Table(title=f"Models ({cfg.models_dir})", show_edge=False, title_justify="left")
    for c in ("model:variant", "size", "source", "installed"):
        t.add_column(c)
    for a in arts:
        t.add_row(a["ref"], gib(a["size_bytes"]), a["uri"] or "", (a["installed_at"] or "")[:19])
    console.print(t)


@model_app.command("info")
def model_info(
    model: str = typer.Argument(..., envvar="SETUP_WORK_MODEL"),
    json_out: bool = _json_opt(),
) -> None:
    """Show catalog metadata and which variants are available locally."""
    cfg = STATE.cfg
    spec, _ = resolve(cfg, model)
    data: dict[str, Any] = {"id": model.split(":")[0], **spec.model_dump()}
    for v in data["variants"]:
        var = spec.variant(v["id"])
        v["cached"] = bool(var and hub.local_model(cfg, data["id"], var))
        art = hub.model_artifact(cfg, data["id"], var) if var else None
        v["artifact"] = art.uri if art else None
    if _set_json(json_out):
        emit_json(data)
        return
    a = spec.attention
    console.print(f"[bold]{spec.display_name}[/] ({data['id']}) · license {spec.license}")
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


@model_app.command("remove")
def model_remove(model: str = typer.Argument(..., envvar="SETUP_WORK_MODEL")) -> None:
    """Delete an unpacked model (asks for confirmation)."""
    freed = hub.remove_artifact_model(STATE.cfg, model, STATE.runner)
    if freed is None:
        raise NotFoundError(f"{model} is not unpacked locally", hint="see `make model-list`")
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
        ..., envvar="SETUP_WORK_MODEL", help="id[:variant], path to .gguf or a model directory"
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
    force: bool = _SERVE_OPTS["force"],
    detach: bool = _SERVE_OPTS["detach"],
    plan_only: bool = _SERVE_OPTS["plan_only"],
    extra: list[str] = typer.Argument(None, help="Extra engine args after `--`."),
    json_out: bool = _json_opt(),
) -> None:
    """Start an OpenAI-compatible inference server (localhost by default)."""
    _set_json(json_out)
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
