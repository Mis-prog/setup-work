"""Планирование инференса и управление процессом сервера.

План определяет модель, проверяет совместимость и память, выбирает порт и команду.
ServerManager запускает процесс, записывает состояние и ожидает готовности HTTP.
Сигналы отправляются только процессам с совпавшими PID и временем запуска."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, ModelSpec, Variant
from .engines import FitRequest, LaunchConfig, LlamaCppEngine, get_engine, is_loopback
from .errors import (
    CheckFailedError,
    MissingDependencyError,
    NotFoundError,
    SetupWorkError,
    UnsupportedError,
    UsageError,
)
from .hardware import HardwareInfo
from .memory import gib
from .models import RecommendRequest, evaluate_variant, resolve
from .runner import Runner, format_argv, probe, redact_argv

API_KEY_ENV = "SETUP_WORK_API_KEY"


@dataclass
class ServeOptions:
    engine: str | None = None
    name: str = "default"
    host: str | None = None
    port: int | None = None
    context: int | None = None
    concurrency: int = 1
    tensor_parallel: int | None = None
    gpu_memory_utilization: float | None = None
    gpu_layers: int | None = None
    threads: int | None = None
    kv_cache_dtype: str | None = None
    served_model_name: str | None = None
    force: bool = False
    extra_args: list[str] = field(default_factory=list)


@dataclass
class LaunchPlan:
    launch: LaunchConfig
    argv: list[str]
    env: dict[str, str]
    binary: str | None
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    public: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "launch": {k: v for k, v in asdict(self.launch).items() if k != "api_key_file"},
            "argv": redact_argv(self.argv),
            "command": format_argv(self.argv),
            "env": {k: "***" for k in self.env},
            "binary": self.binary,
            "rationale": self.rationale,
            "warnings": self.warnings,
            "public": self.public,
        }


# --------------------------------------------------------------------------- порты


def port_in_use(host: str, port: int) -> bool:
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(fam, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return True
    return False


def find_free_port(host: str, start: int, attempts: int = 50) -> int:
    for p in range(start, start + attempts):
        if not port_in_use(host, p):
            return p
    raise SetupWorkError(
        f"нет свободного порта в диапазоне {start}-{start + attempts - 1} on {host}"
    )


# ------------------------------------------------------------------------ планирование


def _pick_variant(
    cfg: Config,
    model_id: str,
    spec: ModelSpec,
    engine: str | None,
    hw: HardwareInfo,
    ctx: int,
    opts: ServeOptions,
) -> tuple[Variant, str, list[str]]:
    engines = [get_engine(engine).name] if engine else ["vllm", "llama.cpp"]
    req = RecommendRequest(
        task=spec.workloads[0],
        context=ctx,
        concurrency=opts.concurrency,
        kv_cache_dtype=opts.kv_cache_dtype or cfg.defaults.inference.default_kv_cache_dtype,
        tensor_parallel=opts.tensor_parallel,
        gpu_memory_utilization=opts.gpu_memory_utilization
        or cfg.defaults.inference.gpu_memory_utilization,
    )
    best = None
    reasons: list[str] = []
    for e in engines:
        for v in spec.variants:
            if e not in v.engines:
                continue
            cand, why = evaluate_variant(cfg, model_id, spec, v, e, hw, req)
            reasons += [f"[{e}] {w}" for w in why]
            if cand and (best is None or cand.score > best[0].score):
                best = (cand, v, e)
    if best is None:
        raise UnsupportedError(
            f"ни один вариант {model_id} не совместим или не помещается в память "
            f"при контексте {ctx}",
            hint="\n  ".join(["причины:", *reasons[-8:]])
            + "\n  уменьшите CONTEXT, выберите другую MODEL либо выполните make recommend",
        )
    return best[1], best[2], best[0].rationale


def plan_launch(
    cfg: Config, hw: HardwareInfo, model_ref: str, opts: ServeOptions, *, check_cache: bool = True
) -> LaunchPlan:
    from . import hub

    inf = cfg.defaults.inference
    ctx = opts.context or cfg.defaults.recommend.default_context
    warnings: list[str] = []
    rationale: list[str] = []
    kv_dtype = opts.kv_cache_dtype or inf.default_kv_cache_dtype
    spec: ModelSpec | None = None
    variant: Variant | None = None
    model_path: str
    served = opts.served_model_name

    path = Path(model_ref).expanduser()
    if path.exists():
        if path.is_file() and path.suffix == ".gguf":
            engine_name = get_engine(opts.engine or "llama.cpp").name
            if engine_name != "llama.cpp":
                raise UsageError("для файлов GGUF укажите ENGINE=llama.cpp")
        elif path.is_dir():
            engine_name = get_engine(opts.engine or "vllm").name
        else:
            raise UsageError(f"{path} не является файлом .gguf или каталогом модели")
        model_path = str(path.resolve())
        served = served or path.stem
        warnings.append("локальная модель вне каталога: память перед запуском не оценивается")
    elif "/" in model_ref and ":" not in model_ref:
        raise NotFoundError(
            f"{model_ref} не существует; проверьте MODEL",
            hint="укажите MODEL=идентификатор:вариант, локальный файл .gguf или каталог модели",
        )
    else:
        model_id = model_ref.split(":")[0]
        spec, variant = resolve(cfg, model_ref)
        if variant is None:
            variant, engine_name, rationale = _pick_variant(
                cfg, model_id, spec, opts.engine, hw, ctx, opts
            )
            rationale.insert(0, f"автоматически выбран вариант {variant.id} для {engine_name}")
        else:
            engine_name = get_engine(opts.engine).name if opts.engine else variant.engines[0]
            if engine_name not in variant.engines:
                raise UsageError(
                    f"вариант {variant.id} подготовлен для {', '.join(variant.engines)}, "
                    f"а не {engine_name}",
                    hint="выберите вариант для этого движка: "
                    + ", ".join(
                        f"{model_id}:{v.id}" for v in spec.variants if engine_name in v.engines
                    ),
                )
        served = served or model_id
        first = (variant.files or [""])[0].replace("0000?", "00001")
        art = hub.model_artifact(cfg, model_id, variant)
        origin = art.uri if art else variant.origin
        model_path = (
            f"<{origin}/{first} после model-pull>"
            if variant.format == "gguf"
            else f"<{origin} после model-pull>"
        )
        local = hub.local_model(cfg, model_id, variant)
        if local is not None:
            model_path = str(local)
        elif check_cache and local is None:
            raise NotFoundError(
                f"{model_id}:{variant.id} не загружена; serve не скачивает модели автоматически",
                hint=f"make model-pull MODEL={model_id}:{variant.id}",
            )

    engine = get_engine(engine_name)
    launch = LaunchConfig(
        name=opts.name,
        engine=engine.name,
        model_ref=model_ref,
        model_path=model_path,
        served_model_name=served or "model",
        host=opts.host or inf.host,
        port=opts.port or inf.port,
        context=ctx,
        concurrency=opts.concurrency,
        tensor_parallel=opts.tensor_parallel or 1,
        gpu_memory_utilization=opts.gpu_memory_utilization or inf.gpu_memory_utilization,
        gpu_layers=opts.gpu_layers,
        threads=opts.threads,
        kv_cache_dtype=kv_dtype,
        extra_args=list(opts.extra_args),
    )

    if spec is not None and variant is not None:
        compat = engine.compatibility(spec, variant, hw)
        if compat.level == "unsupported":
            msg = f"{model_ref} через {engine.name}: {'; '.join(compat.reasons)}"
            if not opts.force:
                raise UnsupportedError(msg, hint="выберите подходящий вариант через make recommend")
            warnings.append(f"ПРИНУДИТЕЛЬНО: {msg}")
        if compat.level == "experimental":
            warnings.append(f"экспериментальная конфигурация: {'; '.join(compat.reasons)}")
        launch.dtype = compat.dtype
        launch.embedding = "embedding" in spec.workloads
        if ctx > spec.context_length:
            raise UsageError(
                f"контекст {ctx} превышает лимит {spec.display_name}: {spec.context_length}"
            )
        fit = engine.fit(
            spec,
            variant,
            hw,
            FitRequest(
                ctx, opts.concurrency, kv_dtype, opts.tensor_parallel, launch.gpu_memory_utilization
            ),
            cfg,
        )
        if not fit.fits:
            msg = "по оценке недостаточно памяти: " + "; ".join(fit.reasons)
            if not opts.force:
                raise UnsupportedError(
                    msg,
                    hint=(
                        "уменьшите CONTEXT/CONCURRENCY либо выберите "
                        "меньший вариант; FORCE=1 пропускает проверку памяти"
                    ),
                )
            warnings.append(f"ПРИНУДИТЕЛЬНО: {msg}")
        else:
            est = fit.estimate
            if est:
                rationale.append(
                    f"оценка памяти: веса {gib(est.weights_bytes)}, KV {gib(est.kv_bytes)}, "
                    f"накладные расходы {gib(est.overhead_bytes)} ({fit.layout})"
                )
            rationale += fit.reasons
            warnings += fit.warnings
            if engine.name == "vllm":
                if opts.tensor_parallel is None:
                    launch.tensor_parallel = fit.tensor_parallel
                if fit.gpu_memory_utilization and opts.gpu_memory_utilization is None:
                    launch.gpu_memory_utilization = fit.gpu_memory_utilization
            else:
                if opts.gpu_layers is None:
                    launch.gpu_layers = fit.gpu_layers
                launch.tensor_split = fit.tensor_split
    elif engine.name == "llama.cpp" and opts.gpu_layers is None:
        launch.gpu_layers = -1 if hw.accelerator in ("nvidia", "apple") else 0

    public = not is_loopback(launch.host)
    if public:
        warnings.append(
            f"привязка к {launch.host}: сервер будет доступен из сети"
            + ("" if os.environ.get(API_KEY_ENV) else f"без аутентификации (задайте {API_KEY_ENV})")
        )
    binary = engine.find_binary(cfg)
    launch.binary = binary
    if binary is None:
        warnings.append(
            f"{engine.binary_name} не установлен; make engine-install ENGINE={engine.name}"
        )
    argv = engine.build_command(launch, binary or engine.binary_name)
    return LaunchPlan(launch, argv, {}, binary, rationale, warnings, public)


# ------------------------------------------------------------------------- процесс


def _proc_start_time(pid: int) -> str | None:
    res = probe(["ps", "-o", "lstart=", "-p", str(pid)], timeout=5)
    out = res.stdout.strip()
    return out or None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _local_opener(url: str) -> urllib.request.OpenerDirector:
    """Не отправлять запросы к локальному серверу через HTTP(S)_PROXY."""
    host = urllib.request.urlparse(url).hostname or ""
    if is_loopback(host):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def http_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 5.0
) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(API_KEY_ENV)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310 - допускается только HTTP(S)
    if not url.startswith(("http://", "https://")):
        raise UsageError(f"URL должен использовать HTTP(S): {url}")
    try:
        with _local_opener(url).open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body)
            except ValueError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return e.code, body


@dataclass
class ServerState:
    name: str
    pid: int
    pgid: int
    started_at: float
    proc_start: str | None
    engine: str
    model_ref: str
    served_model_name: str
    base_url: str
    health_path: str
    log_file: str
    argv: list[str]
    embedding: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ServerManager:
    def __init__(self, cfg: Config, runner: Runner) -> None:
        self.cfg = cfg
        self.runner = runner
        self.dir = cfg.state_dir / "serve"
        self.log_dir = cfg.state_dir / "logs"

    # ------------------------------------------------------------- файлы состояния
    def _state_path(self, name: str) -> Path:
        if not name.replace("-", "").replace("_", "").isalnum():
            raise UsageError(f"неверное имя сервера {name!r}")
        return self.dir / f"{name}.json"

    def load(self, name: str) -> ServerState | None:
        p = self._state_path(name)
        try:
            return ServerState(**json.loads(p.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (ValueError, TypeError) as e:
            raise SetupWorkError(
                f"повреждён файл состояния {p}: {e}",
                hint=f"проверьте и при необходимости удалите {p}",
            ) from e

    def _save(self, st: ServerState) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._state_path(st.name)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def _clear(self, name: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._state_path(name).unlink()
        with contextlib.suppress(FileNotFoundError):
            (self.dir / f"{name}.key").unlink()

    def all_states(self) -> list[ServerState]:
        if not self.dir.is_dir():
            return []
        out = []
        for p in sorted(self.dir.glob("*.json")):
            st = self.load(p.stem)
            if st:
                out.append(st)
        return out

    def owns(self, st: ServerState) -> bool:
        """Проверить, что PID существует и относится именно к запущенному нами процессу."""
        if not _pid_alive(st.pid):
            return False
        current = _proc_start_time(st.pid)
        return st.proc_start is not None and current == st.proc_start

    # ----------------------------------------------------------------- готовность
    def healthy(self, st: ServerState, timeout: float = 2.0) -> bool:
        try:
            code, _ = http_json(st.base_url + st.health_path, timeout=timeout)
            if code != 200:
                return False
            code, _ = http_json(st.base_url + "/v1/models", timeout=timeout)
            return code == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    # ------------------------------------------------------------------ запуск
    def prepare_api_key(self, plan: LaunchPlan) -> None:
        key = os.environ.get(API_KEY_ENV)
        if not key:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        kf = self.dir / f"{plan.launch.name}.key"
        fd = os.open(kf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(key + "\n")
        plan.launch.api_key_file = str(kf)
        engine = get_engine(plan.launch.engine)
        plan.argv = engine.build_command(plan.launch, plan.binary or engine.name)
        plan.env = engine.build_env(plan.launch)

    def start(self, plan: LaunchPlan, *, detach: bool, wait: bool = True) -> ServerState | None:
        launch = plan.launch
        existing = self.load(launch.name)
        if existing:
            if self.owns(existing):
                raise UsageError(
                    f"server '{launch.name}' уже работает (PID {existing.pid}, "
                    f"{existing.base_url})",
                    hint="выполните make serve-stop либо задайте другие INSTANCE и PORT",
                )
            self._clear(launch.name)
        if not plan.binary:
            raise MissingDependencyError(
                f"{launch.engine} исполняемый файл сервера не найден",
                hint=f"установите его командой `make engine-install ENGINE={launch.engine}`",
            )
        if port_in_use(launch.host, launch.port):
            raise UsageError(
                f"порт {launch.port} на {launch.host} уже используется",
                hint="выберите другой порт, например "
                f"PORT={find_free_port(launch.host, launch.port + 1)}",
            )
        if plan.public:
            self.runner.require_confirmation(
                f"Открыть сервер инференса на {launch.host}:{launch.port}?"
            )
        self.prepare_api_key(plan)
        plan.env = {**get_engine(launch.engine).build_env(launch), **plan.env}
        self.runner.show_command(plan.argv)
        if self.runner.dry_run:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"serve-{launch.name}.log"
        env = os.environ.copy()
        env.update(plan.env)
        if launch.engine == "llama.cpp" and plan.binary:
            env.update(LlamaCppEngine.runtime_env(plan.binary))
        if launch.engine == "vllm" and Path(launch.model_path).is_dir():
            env.setdefault("HF_HUB_OFFLINE", "1")  # веса локальные: не загружать неявно
        with log_path.open("ab") as log:
            log.write(
                f"\n=== запуск setup-work {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"{format_argv(plan.argv)}\n".encode()
            )
            log.flush()
            proc = subprocess.Popen(
                plan.argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        st = ServerState(
            name=launch.name,
            pid=proc.pid,
            pgid=os.getpgid(proc.pid),
            started_at=time.time(),
            proc_start=_proc_start_time(proc.pid),
            engine=launch.engine,
            model_ref=launch.model_ref,
            served_model_name=launch.served_model_name,
            base_url=launch.base_url,
            health_path=get_engine(launch.engine).health_path,
            log_file=str(log_path),
            argv=redact_argv(plan.argv),
            embedding=launch.embedding,
        )
        self._save(st)
        self.runner.event("serve.start", **{k: v for k, v in st.to_dict().items() if k != "argv"})
        self.runner.info(f"запущен PID {proc.pid}; logs: {log_path}")
        if wait:
            self.wait_ready(st, proc)
        if not detach:
            self._foreground(st, proc)
        return st

    def wait_ready(self, st: ServerState, proc: subprocess.Popen[bytes] | None = None) -> None:
        timeout = self.cfg.defaults.inference.ready_timeout_s
        deadline = time.monotonic() + timeout
        self.runner.info(f"ожидание {st.base_url}{st.health_path} (ожидание {timeout} с) ...")
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                tail = self.tail(st.name, 25)
                self._clear(st.name)
                raise SetupWorkError(
                    f"{st.engine} завершился с кодом {proc.returncode} до готовности сервера",
                    hint="последние строки журнала:\n" + "\n".join(tail),
                )
            if self.healthy(st):
                self.runner.info(
                    f"[green]готов[/] {st.base_url}/v1 (модель: {st.served_model_name})"
                )
                self.runner.event("serve.ready", name=st.name, url=st.base_url)
                return
            time.sleep(2)
        raise CheckFailedError(
            f"сервер не стал готов за {timeout}с; процесс ещё работает, PID {st.pid})",
            hint="проверьте make serve-logs; для остановки выполните make serve-stop",
        )

    def _foreground(self, st: ServerState, proc: subprocess.Popen[bytes]) -> None:
        self.runner.info("[dim]вывод журнала; Ctrl-C останавливает сервер[/]")
        try:
            with open(st.log_file, encoding="utf-8", errors="replace") as fh:
                fh.seek(0, os.SEEK_END)
                while proc.poll() is None:
                    line = fh.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.3)
        except KeyboardInterrupt:
            self.runner.info("\nполучено прерывание: остановка сервера")
        finally:
            if proc.poll() is None:
                self.stop(st.name)
            else:
                self._clear(st.name)

    # ------------------------------------------------------------------- остановка
    def stop(self, name: str) -> bool:
        st = self.load(name)
        if st is None:
            self.runner.info(f"нет сервера с именем '{name}' в сохранённом состоянии")
            return False
        if not self.owns(st):
            self.runner.warn(
                f"PID {st.pid} не соответствует процессу setup-work; "
                "устаревшее состояние удаляется без отправки сигналов"
            )
            self._clear(name)
            return False
        self.runner.info(f"остановка '{name}' (PID {st.pid}, группа процессов {st.pgid})")
        if self.runner.dry_run:
            return True
        timeout = self.cfg.defaults.inference.stop_timeout_s
        with contextlib.suppress(ProcessLookupError):
            os.killpg(st.pgid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and _pid_alive(st.pid):
            with contextlib.suppress(ChildProcessError):
                os.waitpid(st.pid, os.WNOHANG)
            time.sleep(0.3)
        if _pid_alive(st.pid) and self.owns(st):
            self.runner.warn(f"процесс работает спустя {timeout} с: отправляется SIGKILL")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(st.pgid, signal.SIGKILL)
        self._clear(name)
        self.runner.event("serve.stop", name=name, pid=st.pid)
        return True

    # ---------------------------------------------------------------- состояние
    def status(self, name: str | None = None) -> list[dict[str, Any]]:
        states = [s for s in self.all_states() if name is None or s.name == name]
        out = []
        for st in states:
            running = self.owns(st)
            out.append(
                {
                    **st.to_dict(),
                    "running": running,
                    "healthy": running and self.healthy(st),
                    "uptime_s": int(time.time() - st.started_at) if running else None,
                }
            )
        return out

    def tail(self, name: str, lines: int = 50) -> list[str]:
        p = self.log_dir / f"serve-{name}.log"
        if not p.exists():
            return []
        with p.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 256 * 1024))
            data = fh.read().decode("utf-8", errors="replace")
        return data.splitlines()[-lines:]

    # ------------------------------------------------------------------ проверка
    def smoke_test(self, name: str, url: str | None = None, timeout: float = 120) -> dict[str, Any]:
        st = self.load(name) if url is None else None
        if url is None:
            if st is None:
                raise NotFoundError(
                    f"нет сервера с именем '{name}'", hint="запустите сервер командой make serve"
                )
            url = st.base_url
        url = url.rstrip("/")
        t0 = time.monotonic()
        code, models = http_json(url + "/v1/models", timeout=10)
        if code != 200 or not isinstance(models, dict) or not models.get("data"):
            raise CheckFailedError(f"GET {url}/v1/models вернул код {code}: {str(models)[:200]}")
        model_id = models["data"][0]["id"]
        embedding = bool(st and st.embedding)
        if embedding:
            code, body = http_json(
                url + "/v1/embeddings",
                {"model": model_id, "input": "setup-work smoke test"},
                timeout=timeout,
            )
            ok = code == 200 and isinstance(body, dict) and body.get("data")
            dims = len(body["data"][0]["embedding"]) if ok else None
            result = {"endpoint": "/v1/embeddings", "dimensions": dims}
        else:
            code, body = http_json(
                url + "/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
                    "max_tokens": 32,
                    "temperature": 0,
                },
                timeout=timeout,
            )
            ok = code == 200 and isinstance(body, dict) and body.get("choices")
            text = None
            if ok:
                msg = body["choices"][0].get("message") or {}
                text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            result = {
                "endpoint": "/v1/chat/completions",
                "reply": text,
                "usage": body.get("usage") if isinstance(body, dict) else None,
            }
        if not ok:
            raise CheckFailedError(f"запрос к {url} завершился с кодом {code}: {str(body)[:300]}")
        return {
            "url": url,
            "model": model_id,
            "ok": True,
            "latency_s": round(time.monotonic() - t0, 3),
            **result,
        }
