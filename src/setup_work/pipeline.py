"""Последовательная настройка: инфраструктура → агенты/GigaChat → локальная модель."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from . import artifacts, bootstrap, harness, hub
from .config import Config
from .diagnostics import detect_tool
from .errors import MissingDependencyError, NotFoundError, UsageError
from .hardware import HardwareInfo
from .inference import ServeOptions, ServerManager, plan_launch
from .models import Candidate, RecommendRequest, evaluate_variant, recommend, resolve
from .runner import Runner
from .steps import execute


def infrastructure(cfg: Config, runner: Runner) -> dict[str, Any]:
    """Реестры и окружение проекта готовятся stage-0; здесь Node.js/npm и JAR."""
    pins = cfg.defaults.tools
    node, npm = detect_tool("node", cfg), detect_tool("npm", cfg)
    force = node.version != pins["node"].version or npm.version != pins["npm"].version
    plans = bootstrap.plan_tools(cfg, ["python", "node"], force=False)
    if force:
        plans = [p for p in plans if p.tool != "node"]
        plans.append(bootstrap.ToolPlan("node", "install", bootstrap.plan_node(cfg)))
    if cfg.defaults.setup.uv:
        plans.extend(bootstrap.plan_tools(cfg, ["uv"]))
    for plan in plans:
        runner.info(f"{plan.tool}: {plan.status} {plan.detail}")
        if plan.status in ("manual", "unsupported"):
            raise MissingDependencyError(
                plan.detail, hint="; ".join(s.description for s in plan.steps)
            )
        execute(plan.steps, runner, cfg.data_dir)
    # Установленные инструменты должны быть доступны setup.sh в этом же процессе.
    if not runner.dry_run:
        os.environ["PATH"] = os.pathsep.join([str(cfg.bin_dir), os.environ.get("PATH", "")])
    jars: dict[str, str] = {}
    for name, ref in cfg.corporate.jars.items():
        if ref is None:
            runner.info(f"JAR {name}: источник пока не задан, шаг пропущен")
            jars[name] = "unconfigured"
            continue
        if ref.effective_format != "file":
            raise UsageError(f"jars.{name}: укажите прямой URI файла и format: file")
        artifacts.install(
            cfg,
            ref,
            cfg.tools_dir / "jars" / name,
            runner,
            name=f"jar-{name}",
            confirm_large=False,
        )
        jars[name] = "installed" if not runner.dry_run else "planned"
    return {"status": "ready", "jars": jars}


def select_local(cfg: Config, hw: HardwareInfo) -> Candidate | None:
    """Выбрать первый совместимый GPU-вариант в стабильном порядке ранжирования."""
    settings = cfg.defaults.setup
    request = RecommendRequest(
        task=settings.task,
        context=settings.context,
        concurrency=settings.concurrency,
        engine=settings.engine,
        gpu_memory_utilization=cfg.defaults.inference.gpu_memory_utilization,
        max_results=max(1, len(cfg.models.models) * 2),
    )
    if settings.model:
        model, selected_variant = resolve(cfg, settings.model)
        model_id = settings.model.split(":", 1)[0]
        candidates = []
        for variant in [selected_variant] if selected_variant else model.variants:
            for engine_name in [settings.engine] if settings.engine else variant.engines:
                if engine_name not in variant.engines:
                    continue
                candidate, _ = evaluate_variant(
                    cfg, model_id, model, variant, engine_name, hw, request
                )
                if candidate:
                    candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.total_bytes,
                candidate.ref,
                candidate.engine,
            )
        )
    else:
        candidates = recommend(cfg, hw, request).candidates
    for candidate in candidates:
        if candidate.layout == "cpu" or candidate.compatibility == "unsupported":
            continue
        if settings.model and settings.model not in (candidate.ref, candidate.model_id):
            continue
        _, variant = resolve(cfg, candidate.ref)
        ref = hub.model_artifact(cfg, candidate.model_id, variant)
        engine = cfg.corporate.engines.get(candidate.engine)
        if ref and engine and not any("CHANGE-ME" in uri for uri in (ref.uri, engine.uri)):
            return candidate
    return None


def local_model(cfg: Config, hw: HardwareInfo, runner: Runner) -> dict[str, Any]:
    manager = ServerManager(cfg, runner)
    states = manager.status("local") if cfg.defaults.setup.start_local else []
    running = next((state for state in states if state.get("running")), None)
    if running:
        selected_model = cfg.defaults.setup.model
        if (
            selected_model
            and selected_model not in (running["model_ref"], running["model_ref"].split(":")[0])
        ) or (cfg.defaults.setup.engine and cfg.defaults.setup.engine != running["engine"]):
            raise UsageError("сервер local уже занят другой моделью; остановите его перед setup")
        if not running.get("healthy"):
            raise UsageError(
                "сервер local запущен, но API не отвечает; проверьте serve-logs INSTANCE=local"
            )
        runner.info(f"Локальный сервер уже запущен: {running['model_ref']}")
        return {
            "status": "running",
            "model": running["model_ref"],
            "engine": running["engine"],
            "base_url": running["base_url"],
        }
    candidate = select_local(cfg, hw)
    if candidate is None:
        raise NotFoundError(
            "нет подходящей модели с настроенным источником и движком для доступных GPU",
            hint="проверьте corporate.yaml и make recommend EXPLAIN=1; GigaChat остаётся доступен",
        )
    runner.info(
        f"Локально: {candidate.ref}, {candidate.engine}, GPU: {candidate.tensor_parallel}, "
        f"контекст: {candidate.context}"
    )
    steps, notes = bootstrap.plan_engine(cfg, candidate.engine, hw)
    for note in notes:
        runner.info(note)
    execute(steps, runner, cfg.runtime_dir)
    _, variant = resolve(cfg, candidate.ref)
    hub.pull_artifact(
        cfg,
        candidate.model_id,
        variant,
        hub.require_artifact(cfg, candidate.model_id, variant),
        runner,
    )
    options = ServeOptions(
        engine=candidate.engine,
        context=candidate.context,
        concurrency=candidate.concurrency,
        tensor_parallel=candidate.tensor_parallel,
        gpu_layers=candidate.gpu_layers,
        name="local",
    )
    # В dry-run путь модели ещё может отсутствовать; план запуска всё равно проверяется.
    launch = plan_launch(cfg, hw, candidate.ref, options, check_cache=not runner.dry_run)
    result = {
        "status": "installed",
        "model": candidate.ref,
        "engine": candidate.engine,
        "tensor_parallel": candidate.tensor_parallel,
        "context": candidate.context,
    }
    if cfg.defaults.setup.start_local:
        if runner.dry_run:
            runner.show_command(launch.argv)
        else:
            manager.start(launch, detach=True)
        result["status"] = "planned" if runner.dry_run else "running"
    return result


def _choice(value: bool | None, prompt: str, runner: Runner) -> bool:
    if value is not None:
        return value
    if runner.dry_run:
        runner.info(f"(dry-run) вопрос при запуске: {prompt}")
        return False
    return runner.confirm(prompt)


def run(
    cfg: Config,
    hardware: Callable[[], HardwareInfo],
    runner: Runner,
    *,
    agents: bool | None = None,
    names: list[str] | None = None,
    local: bool | None = None,
    check_updates: bool | None = None,
    uv: bool | None = None,
) -> dict[str, Any]:
    settings = cfg.defaults.setup
    if uv is not None:
        settings.uv = uv
    state: dict[str, Any] = {"backend": "gigachat", "stages": {}}

    def save(stage: str, result: Any) -> None:
        state["stages"][stage] = result
        if runner.dry_run:
            return
        path = cfg.state_dir / "setup-state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    runner.info("1/3 Общая инфраструктура")
    save("infrastructure", infrastructure(cfg, runner))
    runner.info("2/3 Агенты и GigaChat API")
    with_agents = _choice(
        agents if agents is not None else settings.agents,
        "Будете работать с агентами?",
        runner,
    )
    if with_agents:
        selected = names if names is not None else settings.harnesses
        if not selected:
            selected = [
                name
                for name, spec in cfg.harnesses.harnesses.items()
                if spec.source == "git" and spec.install.method == "setup"
            ]
        if not selected:
            raise UsageError("нет агентов: добавьте Git-репозитории с install.method: setup")
        # Сначала проверить весь список, чтобы опечатка не оставила половину установки.
        for name in selected:
            spec = harness.get_spec(cfg, name)
            if spec.source != "git" or spec.install.method != "setup":
                raise UsageError(f"{name}: нужны source: git и install.method: setup")
            if "CHANGE-ME" in f"{spec.repository} {spec.ref}":
                raise UsageError(f"заполните repository и ref для {name} в harnesses.yaml")
        results = []
        for name in dict.fromkeys(selected):
            results.append(
                harness.setup_repository(
                    cfg,
                    name,
                    runner,
                    check_updates=settings.check_updates
                    if check_updates is None
                    else check_updates,
                )
            )
            save("agents", results)
    else:
        save("agents", {"status": "skipped"})
    runner.info("3/3 Дополнительная локальная модель")
    hw = hardware()
    if not hw.gpus or hw.accelerator == "cpu":
        runner.info("Доступных GPU нет. Локальная установка пропущена; режим агентов — GigaChat.")
        save("local", {"status": "no-gpu"})
    elif _choice(
        local if local is not None else settings.local_model,
        "Обнаружены GPU. Дополнительно поднять локальную модель?",
        runner,
    ):
        save("local", local_model(cfg, hw, runner))
    else:
        save("local", {"status": "skipped"})
    return state
