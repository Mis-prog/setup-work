"""Интерфейс командной строки setup-work.

Параметры Makefile передаются через переменные SETUP_WORK_*, без подстановки
пользовательских значений в команды shell."""

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

from . import __version__, corporate, hub, pipeline
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
    help="Подготовка рабочего места AI/ML: инструменты, агенты, подбор моделей, "
    "локальный инференс.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
config_app = typer.Typer(help="Просмотр и проверка конфигурации.", no_args_is_help=True)
model_app = typer.Typer(help="Загрузка и управление весами моделей.", no_args_is_help=True)
harness_app = typer.Typer(help="Установка и обслуживание харнессов агентов.", no_args_is_help=True)
server_app = typer.Typer(help="Управление серверами инференса.", no_args_is_help=True)
engine_app = typer.Typer(help="Установка движков llama.cpp и vLLM.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(model_app, name="model")
app.add_typer(harness_app, name="harness")
app.add_typer(server_app, name="server")
app.add_typer(engine_app, name="engine")

OK = "[green]✔[/]"
BAD = "[red]✘[/]"
WARN = "[yellow]![/]"

COLUMN_TITLES = {
    "model": "модель",
    "name": "имя",
    "engine": "движок",
    "variant": "вариант",
    "format": "формат",
    "quant": "квантование",
    "size": "размер",
    "engines": "движки",
    "source": "источник",
    "cached": "загружено",
    "package / repo": "пакет / репозиторий",
    "recipe": "рецепт",
    "installed": "установлено",
    "binaries": "команды",
    "state": "состояние",
    "uptime": "время работы",
    "fit": "размещение",
    "quality": "качество",
    "confidence": "уверенность",
    "memory": "память",
    "context": "контекст",
    "status": "состояние",
    "detail": "подробности",
    "tool": "инструмент",
    "version": "версия",
    "required": "требуется",
    "workloads": "задачи",
    "score": "оценка",
    "max ctx": "макс. контекст",
}


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
        False, "--json", help="Машинно-читаемый JSON в stdout.", envvar="SETUP_WORK_JSON"
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
        help="Показать действия без выполнения.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        envvar="SETUP_WORK_YES",
        help="Автоматически подтверждать действия.",
    ),
    json_out: bool = typer.Option(
        False, "--json", envvar="SETUP_WORK_JSON", help="Выводить JSON там, где он поддерживается."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", envvar="SETUP_WORK_VERBOSE"),
) -> None:
    r = STATE.runner
    r.dry_run, r.assume_yes, r.json_mode, r.verbose = dry_run, yes, json_out, verbose


# =============================================================== основные команды


@app.command()
def version(json_out: bool = _json_opt()) -> None:
    """Показать версию setup-work."""
    if _set_json(json_out):
        emit_json({"version": __version__, "python": sys.version.split()[0]})
    else:
        console.print(f"setup-work {__version__}")


@app.command("setup")
def setup_cmd(
    uv: bool | None = typer.Option(
        None,
        "--uv/--no-uv",
        envvar="SETUP_WORK_WITH_UV",
        help="Дополнительно установить uv из корпоративного PyPI.",
    ),
    agents: bool | None = typer.Option(
        None,
        "--agents/--no-agents",
        envvar="SETUP_WORK_WITH_AGENTS",
        help="Установить агентов; без настройки — спросить.",
    ),
    harnesses: str | None = typer.Option(
        None,
        "--harnesses",
        envvar="SETUP_WORK_HARNESSES",
        help="Имена через запятую.",
    ),
    local: bool | None = typer.Option(
        None,
        "--local/--no-local",
        envvar="SETUP_WORK_LOCAL_MODEL",
        help="Дополнительно поднять локальную модель при наличии GPU.",
    ),
    updates: bool | None = typer.Option(
        None,
        "--updates/--no-updates",
        envvar="SETUP_WORK_CHECK_UPDATES",
        help="Проверять новые ревизии агентов; перед применением спросить.",
    ),
    model: str | None = typer.Option(None, envvar="SETUP_WORK_MODEL"),
    engine: str | None = typer.Option(None, envvar="SETUP_WORK_ENGINE"),
    context: int | None = typer.Option(None, envvar="SETUP_WORK_CONTEXT", min=256),
    json_out: bool = _json_opt(),
) -> None:
    """Инфраструктура → агенты с GigaChat → дополнительная локальная модель.

    Запускайте make setup, чтобы сначала настроить pip/npm и окружение проекта.
    """
    _set_json(json_out)
    cfg = STATE.cfg
    if engine is not None and engine not in ("llama.cpp", "vllm"):
        raise UsageError("укажите движок llama.cpp или vllm")
    overrides = {
        key: value
        for key, value in {"model": model, "engine": engine, "context": context}.items()
        if value is not None
    }
    cfg.defaults.setup = type(cfg.defaults.setup).model_validate(
        {**cfg.defaults.setup.model_dump(), **overrides}
    )
    names = [name.strip() for name in harnesses.split(",")] if harnesses is not None else None
    result = pipeline.run(
        cfg,
        lambda: STATE.hw,
        STATE.runner,
        agents=agents,
        names=names,
        local=local,
        check_updates=updates,
        uv=uv,
    )
    if STATE.json:
        emit_json(result)


def _hw_table(hw: HardwareInfo) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    t.add_row("ОС", f"{hw.os.distro or hw.os.system} ({hw.os.system} {hw.os.release})")
    t.add_row(
        "Архитектура",
        hw.os.arch + (" · контейнер" if hw.os.container else "") + (" · WSL" if hw.os.wsl else ""),
    )
    t.add_row(
        "CPU",
        f"{hw.cpu.model or 'неизвестно'} · {hw.cpu.logical_cores} логических / "
        f"{hw.cpu.physical_cores or '?'} физических · {hw.cpu.usable_cores} usable",
    )
    m = hw.memory
    ram = f"{gib(m.total_bytes)} всего · {gib(m.available_bytes)} available"
    if m.cgroup_limit_bytes:
        ram += f" · лимит контейнера {gib(m.cgroup_limit_bytes)}"
    t.add_row("RAM", ram)
    if hw.accelerator == "apple":
        g = hw.gpus[0]
        t.add_row("GPU", f"{g.name} · общая память")
        t.add_row(
            "Бюджет GPU",
            f"{gib(hw.metal_working_set_bytes)} объём Metal ({hw.metal_working_set_source})",
        )
    elif hw.gpus:
        names: dict[str, int] = {}
        for g in hw.gpus:
            names[g.name] = names.get(g.name, 0) + 1
        t.add_row("GPUs", ", ".join(f"{n} × {k}" for k, n in names.items()))
        for g in hw.gpus:
            t.add_row(
                f"  GPU {g.index}",
                f"{gib(g.memory_total_bytes)} всего · "
                f"{gib(g.memory_free_bytes)} свободно · CC {g.compute_capability or '?'}"
                + (
                    f" ({g.compute_capability_source})"
                    if g.compute_capability_source == "name-lookup"
                    else ""
                ),
            )
        t.add_row(
            "Всего видеопамяти", f"{gib(hw.total_vram_bytes)} (сумма по отдельным устройствам)"
        )
        t.add_row(
            "Драйвер / CUDA",
            f"{hw.nvidia_driver or '?'} / CUDA {hw.cuda_driver_version or '?'}"
            + (f" · nvcc {hw.cuda_toolkit_version}" if hw.cuda_toolkit_version else ""),
        )
    else:
        t.add_row("GPU", "не обнаружен; используется CPU")
    t.add_row("Типы данных", ", ".join(hw.precisions) or "unknown")
    for s in hw.storage:
        t.add_row("Диск", f"{s.path}: {gib(s.free_bytes)} свободно из {gib(s.total_bytes)}")
    if hw.source != "live":
        t.add_row("Источник", f"[yellow]{hw.source}[/]")
    for n in hw.notes:
        t.add_row("Примечание", f"[dim]{escape(n)}[/]")
    return t


@app.command()
def hardware(json_out: bool = _json_opt()) -> None:
    """Определить процессор, RAM, GPU, видеопамять, вычислительные возможности и диски."""
    hw = STATE.hw
    if _set_json(json_out):
        emit_json(hw.to_dict())
        return
    console.print(Panel(_hw_table(hw), title="Оборудование", expand=False))


def _engine_rows(cfg: Config, hw: HardwareInfo) -> list[tuple[str, str, str]]:
    rows = []
    for name, eng in ENGINES.items():
        binary = eng.find_binary(cfg)
        if name == "vllm" and (hw.os.system != "Linux" or hw.accelerator != "nvidia"):
            status = f"{WARN} не поддерживается на этой машине"
            detail = "нужны Linux и NVIDIA GPU; используйте llama.cpp"
        elif name == "vllm":
            cc = hw.min_compute_capability
            if cc is not None and cc < 7.0:
                status, detail = f"{BAD} unsupported", f"compute capability {cc} < 7.0"
            elif binary:
                status, detail = f"{OK} установлен", binary
            else:
                status, detail = f"{WARN} не установлен", "make engine-install ENGINE=vllm"
        else:
            backend = {"apple": "Metal", "nvidia": "CUDA"}.get(hw.accelerator, "CPU")
            if binary:
                status, detail = (
                    f"{OK} установлен",
                    f"{binary} ({backend} ожидаемый вычислительный модуль)",
                )
            else:
                status, detail = f"{WARN} не установлен", "make engine-install ENGINE=llama.cpp"
        rows.append((name, status, detail))
    return rows


def _corp_table(c: dict[str, Any]) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    t.add_row("клиент HDFS", c.get("hdfs_cli") or f"{WARN} не найден")
    if "kerberos" in c:
        k = c["kerberos"]
        t.add_row(
            "Kerberos",
            f"{OK if k['ok'] else BAD} {k['detail']}" + ("" if k["ok"] else " — выполните kinit"),
        )
    t.add_row(
        "Индекс pip",
        c.get("pip_index")
        or f"{WARN} не настроен; заполните config/bootstrap.env и выполните make registries",
    )
    if "npm_registry" in c:
        reg = c.get("npm_registry") or ""
        public = not reg or "registry.npmjs.org" in reg
        t.add_row(
            "Реестр npm", f"{WARN} {reg or 'не настроен'} (make registries)" if public else reg
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
    strict: bool = typer.Option(
        False, help="Завершить с кодом 7, если отсутствует основной инструмент."
    ),
) -> None:
    """Проверить систему, инструменты, оборудование и готовность к запуску без установки."""
    cfg, hw = STATE.cfg, STATE.hw
    env = detect_environment(cfg)
    problems = cfg.validate_all()
    core = ["python", "git"]
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
        sys_t.add_row("setup-work выполняется на", f"Python {env.python_runtime}")
        sys_t.add_row(
            "каталог команд",
            f"{env.bin_dir} "
            + (OK if env.bin_dir_on_path else f"{WARN} отсутствует в PATH: {bs.path_hint(cfg)}"),
        )
        if env.proxies:
            sys_t.add_row("Proxies", ", ".join(f"{k}={v}" for k, v in env.proxies.items()))
        console.print(Panel(sys_t, title="System", expand=False))

        tt = Table(title="Инструменты", show_edge=False, title_justify="left")
        for c in ("tool", "status", "version", "path"):
            tt.add_column(COLUMN_TITLES.get(c, c))
        for t in env.tools:
            if not t.installed:
                mark = BAD if t.name in core else WARN
                tt.add_row(t.name, f"{mark} отсутствует", "", t.note or "")
            else:
                mark = OK if t.satisfies in (None, True) else WARN
                ver = (t.version or "") + (
                    "" if t.satisfies in (None, True) else " (устаревшая версия)"
                )
                tt.add_row(
                    t.name,
                    f"{mark} готов" + ((" (управляется setup-work)") if t.managed else ""),
                    ver,
                    t.path or "",
                )
        console.print(tt)
        console.print(Panel(_hw_table(hw), title="Оборудование", expand=False))
        et = Table(title="Движки инференса", show_edge=False, title_justify="left")
        for c in ("engine", "status", "detail"):
            et.add_column(COLUMN_TITLES.get(c, c))
        for row in _engine_rows(cfg, hw):
            et.add_row(*row)
        console.print(et)
        console.print(
            Panel(_corp_table(corporate.status(cfg)), title="Корпоративная сеть", expand=False)
        )
        if problems:
            console.print("[red]Ошибки конфигурации:[/]")
            for p in problems:
                console.print(f"  {escape(p)}")
        nxt = []
        if missing_core:
            nxt.append(
                f"make bootstrap            # установить недостающее: {', '.join(missing_core)}"
            )
        nxt.append("make recommend TASK=coding")
        nxt.append("make harness-list")
        console.print(Panel("\n".join(nxt), title="Следующие действия", expand=False))
    if problems:
        raise typer.Exit(ExitCode.USAGE)
    if strict and missing_core:
        raise typer.Exit(ExitCode.CHECK_FAILED)


# ================================================================ конфигурация


@config_app.command("show")
def config_show(json_out: bool = _json_opt()) -> None:
    """Показать используемые файлы настроек, каталоги и значения по умолчанию."""
    data = STATE.cfg.as_dict()
    if _set_json(json_out):
        emit_json(data)
        return
    import yaml

    console.print(f"[bold]встроенные настройки:[/] {data['builtin_dir']}")
    console.print(f"[bold]пользовательские настройки:[/]  {data['user_dir']}")
    for k, v in data["sources"].items():
        console.print(f"  {k}: {', '.join(v) or '(нет)'}")
    console.print("[bold]итоговые каталоги:[/]")
    for k, v in data["resolved_paths"].items():
        console.print(f"  {k}: {v}")
    console.print("[bold]defaults:[/]")
    console.print(yaml.safe_dump(data["defaults"], sort_keys=False).rstrip(), markup=False)


@config_app.command("validate")
def config_validate(json_out: bool = _json_opt()) -> None:
    """Проверить структуру YAML и ссылки между файлами настроек."""
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
            f"{OK} конфигурация корректна: " + ", ".join(f"{v} {k}" for k, v in summary.items())
        )
    if problems:
        raise typer.Exit(ExitCode.CHECK_FAILED)


# ==================================================================== начальная установка


def _print_tool_plans(plans: list[bs.ToolPlan]) -> None:
    t = Table(show_edge=False)
    for c in ("tool", "status", "detail"):
        t.add_column(COLUMN_TITLES.get(c, c))
    style = {"satisfied": OK, "install": "[cyan]→[/]", "manual": WARN, "unsupported": BAD}
    for p in plans:
        t.add_row(p.tool, f"{style[p.status]} {p.status}", escape(p.detail))
    STATE.runner.info("")
    (err_console if STATE.json else console).print(t)


@app.command("bootstrap")
def bootstrap_cmd(
    tool: list[str] = typer.Option(
        None, "--tool", help="python, node или uv; по умолчанию python и node."
    ),
    force: bool = typer.Option(
        False, "--force", help="Переустановить, даже если инструмент уже подходит."
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Установить Node.js/npm из npm registry и проверить Python."""
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
        STATE.runner.info(f"\nPlan: {len(steps)} шагов для {', '.join(p.tool for p in todo)}")
        execute(steps, STATE.runner, cfg.data_dir)
    else:
        STATE.runner.info(f"{OK} установка не требуется")
    env = detect_environment(cfg, ["python", "node"])
    if not env.bin_dir_on_path and any(s.kind == "link" for p in todo for s in p.steps):
        STATE.runner.info(
            f"\n{WARN} {cfg.bin_dir} is not on PATH. setup-work does not edit shell startup "
            f"files; add this line в окружение shell вручную:\n    {bs.path_hint(cfg)}"
        )
    if any(p.status in ("manual", "unsupported") for p in plans):
        STATE.runner.info(f"{WARN} для части инструментов нужны действия вручную; см. выше")


# ====================================================================== движки


@engine_app.command("install")
def engine_install(
    engine: str = typer.Argument(..., envvar="SETUP_WORK_ENGINE", help="llama.cpp | vllm"),
    json_out: bool = _json_opt(),
) -> None:
    """Установить движок из корпоративного артефакта на HDFS, NFS или HTTPS."""
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
        STATE.runner.require_confirmation(f"Установить {name} ({len(steps)} steps)?")
    execute(steps, STATE.runner, cfg.data_dir)
    if not STATE.runner.dry_run:
        STATE.runner.info(f"{OK} {name} installed: {get_engine(name).find_binary(cfg)}")


@engine_app.command("status")
def engine_status(json_out: bool = _json_opt()) -> None:
    """Показать установленные движки и их поддержку на этой машине."""
    rows = _engine_rows(STATE.cfg, STATE.hw)
    if _set_json(json_out):
        emit_json(
            [{"engine": r[0], "status": r[1].split("] ", 1)[-1], "detail": r[2]} for r in rows]
        )
        return
    t = Table(show_edge=False)
    for c in ("engine", "status", "detail"):
        t.add_column(COLUMN_TITLES.get(c, c))
    for r in rows:
        t.add_row(*r)
    console.print(t)


# ======================================================================= модели


@app.command("models")
def models_cmd(
    task: str = typer.Option(None, envvar="SETUP_WORK_TASK", help="Отобрать по задаче."),
    json_out: bool = _json_opt(),
) -> None:
    """Показать каталог моделей без обращения к сети."""
    cfg = STATE.cfg
    rows = []
    for mid, m in sorted(cfg.models.models.items()):
        if task and task not in m.workloads:
            continue
        rows.append((mid, m))
    if _set_json(json_out):
        emit_json([{"id": mid, **m.model_dump()} for mid, m in rows])
        return
    t = Table(title="Каталог моделей", show_edge=False, title_justify="left")
    for c in ("id", "params", "arch", "context", "workloads", "варианты (движок)"):
        t.add_column(COLUMN_TITLES.get(c, c))
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
    console.print(
        "[dim]L = llama.cpp (GGUF), V = vLLM (safetensors). Укажите MODEL=идентификатор:вариант.[/]"
    )


@app.command("recommend")
def recommend_cmd(
    task: str = typer.Option(
        "general",
        envvar="SETUP_WORK_TASK",
        help="general | coding | reasoning | agents | embedding",
    ),
    context: int = typer.Option(
        None, envvar="SETUP_WORK_CONTEXT", help="Длина контекста в токенах."
    ),
    engine: str = typer.Option(None, envvar="SETUP_WORK_ENGINE", help="llama.cpp | vllm"),
    concurrency: int = typer.Option(
        None, envvar="SETUP_WORK_CONCURRENCY", help="Число одновременных последовательностей."
    ),
    tp: int = typer.Option(
        None, "--tp", envvar="SETUP_WORK_TP", help="Явно задать тензорный параллелизм."
    ),
    kv_cache_dtype: str = typer.Option(
        None, envvar="SETUP_WORK_KV_CACHE_DTYPE", help="fp16 | fp8 | q8_0"
    ),
    limit: int = typer.Option(None, "--limit", envvar="SETUP_WORK_LIMIT"),
    explain: bool = typer.Option(
        False,
        "--explain",
        envvar="SETUP_WORK_EXPLAIN",
        help="Показать причины выбора и отклонённые варианты.",
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Подобрать модели под оборудование без загрузок."""
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
        raise UsageError(f"неверный тип KV-кэша {req.kv_cache_dtype}")
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
        console.print(f"{BAD} нет совместимой модели, подходящей под эти ограничения")
    else:
        t = Table(show_edge=False)
        t.add_column("#")
        t.add_column("model:variant", no_wrap=True)
        t.add_column("engine", no_wrap=True)
        for c in ("quant", "memory", "layout", "макс. контекст", "confidence", "score"):
            t.add_column(COLUMN_TITLES.get(c, c))
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
        console.print(
            f"\n[bold]Рекомендуемый вариант:[/] {top.display_name} ({top.ref}) on {top.engine}"
        )
        for line in top.rationale:
            console.print(f"  • {escape(line)}")
        for w in top.warnings:
            console.print(f"  {WARN} {escape(w)}")
        console.print(f"\n  загрузить: make model-pull MODEL={top.ref}")
        console.print(f"  serve:    {top.serve_hint()}")
        if explain:
            for c in res.candidates[1:]:
                console.print(f"\n[bold]{c.ref}[/] через {c.engine}")
                for line in c.rationale + [f"внимание: {w}" for w in c.warnings]:
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
        "[dim]Память указана ориентировочно; оценки "
        "качества не являются результатами бенчмарков.[/]"
    )


@model_app.command("pull")
def model_pull(
    model: str = typer.Argument(..., envvar="SETUP_WORK_MODEL", help="id:variant"),
    json_out: bool = _json_opt(),
) -> None:
    """Загрузить и распаковать веса модели с HDFS, NFS или HTTPS."""
    cfg = STATE.cfg
    _set_json(json_out)
    spec, variant = resolve(cfg, model)
    if variant is None:
        raise UsageError(
            f"укажите вариант модели {model} для загрузки",
            hint="variants: " + ", ".join(f"{model}:{v.id}" for v in spec.variants),
        )
    model_id = model.split(":")[0]
    art = hub.require_artifact(cfg, model_id, variant)
    entry = hub.pull_artifact(cfg, model_id, variant, art, STATE.runner)
    if STATE.json:
        emit_json({"ref": model, "source": art.uri, "path": str(entry)})
    elif not STATE.runner.dry_run:
        console.print(f"{OK} {model} доступна по пути {entry}")


@model_app.command("list")
def model_list(json_out: bool = _json_opt()) -> None:
    """Показать распакованные модели."""
    cfg = STATE.cfg
    arts = hub.list_artifact_models(cfg)
    if _set_json(json_out):
        emit_json({"artifact_dir": str(cfg.models_dir), "artifacts": arts})
        return
    if not arts:
        console.print(
            f"нет моделей в {cfg.models_dir}. "
            "Выполните make recommend, затем make model-pull MODEL=идентификатор:вариант"
        )
        return
    t = Table(title=f"Модели ({cfg.models_dir})", show_edge=False, title_justify="left")
    for c in ("model:variant", "size", "source", "installed"):
        t.add_column(COLUMN_TITLES.get(c, c))
    for a in arts:
        t.add_row(a["ref"], gib(a["size_bytes"]), a["uri"] or "", (a["installed_at"] or "")[:19])
    console.print(t)


@model_app.command("info")
def model_info(
    model: str = typer.Argument(..., envvar="SETUP_WORK_MODEL"),
    json_out: bool = _json_opt(),
) -> None:
    """Показать описание модели и доступные локально варианты."""
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
    console.print(f"[bold]{spec.display_name}[/] ({data['id']}) · лицензия {spec.license}")
    console.print(
        f"  {spec.architecture}, {spec.parameters_b:g}B параметров"
        + (f", {spec.active_parameters_b:g}B активных" if spec.active_parameters_b else "")
        + f", поддерживаемый контекст {spec.context_length}"
    )
    console.print(
        f"  внимание: {a.num_layers} слоёв, {a.num_attention_heads} голов, "
        f"{a.num_kv_heads} KV-голов × {a.head_dim}"
        + (
            f", {a.sliding_window_layers} слоёв со скользящим окном (размер {a.sliding_window})"
            if a.sliding_window_layers
            else ""
        )
    )
    t = Table(show_edge=False)
    for c in ("variant", "format", "quant", "size", "engines", "source", "cached"):
        t.add_column(COLUMN_TITLES.get(c, c))
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
    """Удалить распакованную модель после подтверждения."""
    freed = hub.remove_artifact_model(STATE.cfg, model, STATE.runner)
    if freed is None:
        raise NotFoundError(f"{model} не распакована локально", hint="выполните make model-list")
    if not STATE.runner.dry_run:
        console.print(f"{OK} освобождено {gib(freed)}")


# ===================================================================== харнессы


@harness_app.command("list")
def harness_list(json_out: bool = _json_opt()) -> None:
    """Показать рецепты харнессов и состояние установки."""
    rows = hz.list_harnesses(STATE.cfg)
    if _set_json(json_out):
        emit_json(rows)
        return
    t = Table(show_edge=False)
    for c in ("name", "source", "package / repo", "recipe", "installed", "binaries"):
        t.add_column(COLUMN_TITLES.get(c, c))
    for r in rows:
        t.add_row(
            r["name"],
            r["source"],
            r["package"] or "",
            (r["version"] or "") + ("" if r["pinned"] else " [yellow](версия не закреплена)[/]"),
            f"{OK} {r['installed']}" if r["installed"] else "-",
            ",".join(r["binaries"]),
        )
    console.print(t)
    console.print(f"[dim]пользовательские рецепты: {STATE.cfg.user_dir / 'harnesses.yaml'}[/]")


def _harness_apply(name: str, plan: hz.HarnessPlan, action: str) -> None:
    for w in plan.warnings:
        STATE.runner.warn(w)
    if not plan.steps:
        STATE.runner.info(f"{OK} действия не требуются для {name}")
        return
    STATE.runner.info(f"[bold]{action} {name}[/] ({plan.spec.source}):")
    if plan.needs_confirmation and not STATE.runner.dry_run:
        for i, s in enumerate(plan.steps, 1):
            STATE.runner.info(f"  {i}. {escape(s.render())}")
        STATE.runner.require_confirmation(f"{action} {name} из {plan.spec.source}?")
    execute(plan.steps, STATE.runner, STATE.cfg.data_dir)
    hz.write_manifest(STATE.cfg, name, plan.spec, STATE.runner)
    if not STATE.runner.dry_run:
        STATE.runner.info(
            f"{OK} {name} {plan.spec.version or plan.spec.ref} готов; "
            f"проверьте: make harness-doctor NAME={name}"
        )


@harness_app.command("install")
def harness_install(
    name: str = typer.Argument(..., envvar="SETUP_WORK_NAME"),
    version: str = typer.Option(
        None, envvar="SETUP_WORK_VERSION", help="Переопределить закреплённую версию."
    ),
    ref: str = typer.Option(
        None, envvar="SETUP_WORK_REF", help="Переопределить тег, ветку или коммит Git."
    ),
    allow_build: bool = typer.Option(False, "--allow-build", envvar="SETUP_WORK_ALLOW_BUILD"),
    json_out: bool = _json_opt(),
) -> None:
    """Установить харнесс по его рецепту."""
    plan = hz.plan_install(STATE.cfg, name, version=version, ref=ref, allow_build=allow_build)
    if _set_json(json_out) and STATE.runner.dry_run:
        emit_json(plan.to_dict())
        return
    _harness_apply(name, plan, "Установить")


@harness_app.command("update")
def harness_update(
    name: str = typer.Argument(..., envvar="SETUP_WORK_NAME"),
    version: str = typer.Option(None, envvar="SETUP_WORK_VERSION"),
    ref: str = typer.Option(None, envvar="SETUP_WORK_REF"),
    allow_build: bool = typer.Option(False, "--allow-build", envvar="SETUP_WORK_ALLOW_BUILD"),
) -> None:
    """Переустановить харнесс с версией или ref из рецепта либо параметров."""
    plan = hz.plan_install(
        STATE.cfg, name, version=version, ref=ref, update=True, allow_build=allow_build
    )
    _harness_apply(name, plan, "Обновить")


@harness_app.command("remove")
def harness_remove(name: str = typer.Argument(..., envvar="SETUP_WORK_NAME")) -> None:
    """Удалить харнесс и созданные для него ссылки."""
    steps = hz.plan_remove(STATE.cfg, name)
    execute(steps, STATE.runner, STATE.cfg.data_dir)
    if not STATE.runner.dry_run:
        console.print(f"{OK} удалён {name}")


@harness_app.command("doctor")
def harness_doctor(
    name: str = typer.Argument(
        None,
        envvar="SETUP_WORK_NAME",
        help="Не указывайте для проверки всех установленных харнессов.",
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Проверить установку харнесса, доступность команд в PATH и переменные окружения."""
    cfg = STATE.cfg
    names = [name] if name else [r["name"] for r in hz.list_harnesses(cfg) if r["installed"]]
    if not names:
        console.print("харнессы не установлены; выполните make harness-install NAME=имя")
        return
    results = [hz.doctor(cfg, n) for n in names]
    if _set_json(json_out):
        emit_json([r.to_dict() for r in results])
    else:
        for r in results:
            console.print(f"[bold]{r.name}[/] {OK if r.ok else BAD}")
            for c, ok, detail in r.checks:
                console.print(
                    f"  {OK if ok else (WARN if c == 'версия закреплена' else BAD)} {c}: "
                    f"{escape(detail)}"
                )
    if not all(r.ok for r in results):
        raise typer.Exit(ExitCode.CHECK_FAILED)


# ===================================================================== инференс


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
        console.print(f"\nOpenAI-совместимый API: {st.base_url}/v1   model: {st.served_model_name}")
        console.print(
            "проверка: make serve-test   журнал: make serve-logs   остановка: make serve-stop"
        )


_SERVE_OPTS = dict(
    engine=typer.Option(None, envvar="SETUP_WORK_ENGINE", help="llama.cpp | vllm"),
    name=typer.Option(
        "default", "--name", envvar="SETUP_WORK_INSTANCE", help="Имя экземпляра сервера."
    ),
    host=typer.Option(
        None, envvar="SETUP_WORK_HOST", help="Адрес прослушивания; по умолчанию 127.0.0.1."
    ),
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
        help="Запустить, пропустив отказ по оценке памяти или совместимости.",
    ),
    detach=typer.Option(True, "--detach/--foreground", envvar="SETUP_WORK_DETACH"),
    plan_only=typer.Option(
        False, "--plan", envvar="SETUP_WORK_PLAN", help="Только показать план запуска."
    ),
)


@app.command("serve")
def serve_cmd(
    model: str = typer.Argument(
        ...,
        envvar="SETUP_WORK_MODEL",
        help="идентификатор[:вариант], путь к .gguf или каталогу модели",
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
    extra: list[str] = typer.Argument(None, help="Дополнительные аргументы движка после --."),
    json_out: bool = _json_opt(),
) -> None:
    """Запустить OpenAI-совместимый сервер, по умолчанию на localhost."""
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
        True, "--cached-only/--any", help="Рассматривать только уже загруженные модели."
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Запустить лучшую уже загруженную модель для указанной задачи TASK."""
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
            f"ни одна загруженная модель не подходит для задачи '{task}'"
            if res.candidates
            else f"нет совместимой модели для задачи '{task}'",
            hint=f"сначала загрузите рекомендуемую модель: make model-pull MODEL={top}"
            if top
            else "уменьшите CONTEXT или измените ENGINE",
        )
    STATE.runner.info(f"выбрана {chosen.ref} через {chosen.engine} (оценка {chosen.score:.2f})")
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
    all_servers: bool = typer.Option(
        False, "--all", help="Остановить все зарегистрированные серверы."
    ),
) -> None:
    """Остановить сервер setup-work, не затрагивая чужие процессы."""
    mgr = ServerManager(STATE.cfg, STATE.runner)
    names = [s.name for s in mgr.all_states()] if all_servers else [name]
    for n in names:
        if mgr.stop(n):
            console.print(f"{OK} остановлен {n}")


@server_app.command("status")
def serve_status(
    name: str = typer.Option(None, "--name", envvar="SETUP_WORK_INSTANCE"),
    json_out: bool = _json_opt(),
) -> None:
    """Показать зарегистрированные серверы, состояние процессов и готовность."""
    rows = ServerManager(STATE.cfg, STATE.runner).status(name)
    if _set_json(json_out):
        emit_json(rows)
        return
    if not rows:
        console.print(
            "серверы не зарегистрированы; запустите make serve MODEL=идентификатор:вариант"
        )
        return
    t = Table(show_edge=False)
    for c in ("name", "engine", "model", "url", "pid", "state", "uptime"):
        t.add_column(COLUMN_TITLES.get(c, c))
    for r in rows:
        state = (
            (f"{OK} готов" if r["healthy"] else f"{WARN} запускается или не готов")
            if r["running"]
            else f"{BAD} не работает; состояние устарело"
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
    """Показать журнал сервера или следить за новыми строками."""
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
            raise SetupWorkError(f"файл журнала не найден: {path}") from None


@server_app.command("test")
def serve_test(
    name: str = _SERVE_OPTS["name"],
    url: str = typer.Option(
        None, envvar="SETUP_WORK_URL", help="Проверить указанный OpenAI-совместимый URL."
    ),
    json_out: bool = _json_opt(),
) -> None:
    """Отправить тестовый OpenAI-совместимый запрос серверу."""
    res = ServerManager(STATE.cfg, STATE.runner).smoke_test(name, url)
    if _set_json(json_out):
        emit_json(res)
        return
    console.print(
        f"{OK} {res['endpoint']} по {res['url']} (модель {res['model']}) за {res['latency_s']} с"
    )
    if res.get("reply") is not None:
        console.print(f"  ответ: {escape(str(res['reply'])[:200])!r}")
    if res.get("dimensions"):
        console.print(f"  размерность эмбеддинга: {res['dimensions']}")


# ========================================================================= очистка


@app.command("clean")
def clean_cmd(
    downloads: bool = typer.Option(
        True, "--downloads/--no-downloads", help="Удалить кэш установочных архивов."
    ),
    stale_state: bool = typer.Option(
        True,
        "--stale-state/--no-stale-state",
        help="Удалить файлы состояния остановленных серверов.",
    ),
) -> None:
    """Удалить кэши setup-work; модели, харнессы и движки сохраняются."""
    cfg, r = STATE.cfg, STATE.runner
    steps: list[Step] = []
    if downloads:
        for d in (cfg.tools_dir / "downloads", cfg.engines_dir / "llama.cpp" / "downloads"):
            if d.exists():
                steps.append(Step("remove", f"удалить установочные архивы из {d}", dest=d))
    if stale_state:
        mgr = ServerManager(cfg, r)
        for st in mgr.all_states():
            if not mgr.owns(st):
                steps.append(
                    Step(
                        "remove",
                        f"удалить устаревшее состояние сервера {st.name}",
                        dest=mgr.dir / f"{st.name}.json",
                    )
                )
    if not steps:
        r.info(f"{OK} очистка не требуется")
        return
    for s in steps:
        r.info(f"  {escape(s.render())}")
    r.require_confirmation(f"Удалить {len(steps)} объектов?", destructive=True)
    execute(steps, r, cfg.data_dir)


# ========================================================================= точка входа


def main(argv: list[str] | None = None) -> int:
    STATE.__init__()  # сброс состояния при каждом вызове, в том числе в тестах
    cmd = typer.main.get_command(app)
    try:
        rv = cmd.main(args=argv, prog_name="setup-work", standalone_mode=False)
        return int(rv) if isinstance(rv, int) else 0
    except SetupWorkError as e:
        err_console.print(f"[bold red]Ошибка:[/] {escape(e.message)}")
        if e.hint:
            err_console.print(f"[cyan]что сделать:[/] {escape(e.hint)}")
        return int(e.exit_code)
    except typer.Exit as e:
        return int(getattr(e, "exit_code", 0) or 0)
    except typer.Abort:
        err_console.print("отменено")
        return int(ExitCode.ABORTED)
    except typer.TyperException as e:
        # Ошибки параметров и неизвестных команд обрабатываются Typer/Click.
        show = getattr(e, "show", None)
        if callable(show):
            show()
        else:
            err_console.print(f"ошибка: {e}")
        code = getattr(e, "exit_code", ExitCode.USAGE)
        return int(code) if code is not None else int(ExitCode.USAGE)
    except KeyboardInterrupt:
        err_console.print("\nпрервано")
        return 130
    except BrokenPipeError:
        return 0


def _entry() -> None:  # pragma: no cover
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _entry()
