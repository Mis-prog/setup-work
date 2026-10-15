"""Inference orchestration: launch planning and process lifecycle.

Planning (``plan_launch``) is pure: it resolves the model, checks engine
compatibility and memory fit, picks a free port and builds the argv. Execution
(``ServerManager``) starts the process in its own session, records a state file,
waits for HTTP readiness, and only ever signals processes it can prove it started
(PID + process start time must match the state file).
"""

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
    revision: str | None = None
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


# --------------------------------------------------------------------------- ports


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
    raise SetupWorkError(f"no free port in {start}-{start + attempts - 1} on {host}")


# ------------------------------------------------------------------------ planning


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
            f"no variant of {model_id} is compatible with and fits on this machine "
            f"at context {ctx}",
            hint="\n  ".join(["reasons:", *reasons[-8:]])
            + "\n  try a smaller CONTEXT, another MODEL, or `make recommend`",
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
                raise UsageError("GGUF files are served with ENGINE=llama.cpp")
        elif path.is_dir():
            engine_name = get_engine(opts.engine or "vllm").name
        else:
            raise UsageError(f"{path} is neither a .gguf file nor a model directory")
        model_path = str(path.resolve())
        served = served or path.stem
        warnings.append("local model outside the catalog: memory fit is not pre-checked")
    elif "/" in model_ref and ":" not in model_ref:
        engine_name = get_engine(opts.engine or "vllm").name
        if engine_name != "vllm":
            raise UsageError(
                "raw Hugging Face repos are served with vLLM; for llama.cpp pass a catalog "
                "ref (model:variant) or a local .gguf path"
            )
        model_path = model_ref
        served = served or model_ref
        warnings.append("repository outside the catalog: memory fit is not pre-checked")
        if check_cache:
            from huggingface_hub import try_to_load_from_cache

            hit = try_to_load_from_cache(
                model_ref, "config.json", cache_dir=str(hub.cache_dir(cfg)), revision=opts.revision
            )
            if not isinstance(hit, str):
                raise NotFoundError(
                    f"{model_ref} is not in the local cache (serve never downloads implicitly)",
                    hint=f"make model-pull MODEL={model_ref}",
                )
            model_path = str(Path(hit).parent)
    else:
        model_id = model_ref.split(":")[0]
        spec, variant = resolve(cfg, model_ref)
        if variant is None:
            variant, engine_name, rationale = _pick_variant(
                cfg, model_id, spec, opts.engine, hw, ctx, opts
            )
            rationale.insert(0, f"auto-selected variant {variant.id} for {engine_name}")
        else:
            engine_name = get_engine(opts.engine).name if opts.engine else variant.engines[0]
            if engine_name not in variant.engines:
                raise UsageError(
                    f"variant {variant.id} is packaged for {', '.join(variant.engines)}, "
                    f"not {engine_name}",
                    hint="pick a variant for that engine: "
                    + ", ".join(
                        f"{model_id}:{v.id}" for v in spec.variants if engine_name in v.engines
                    ),
                )
        served = served or model_id
        first = (variant.files or [""])[0].replace("0000?", "00001")
        art = hub.model_artifact(cfg, model_id, variant)
        origin = art.uri if art else variant.origin
        model_path = (
            f"<{origin}/{first} after model-pull>"
            if variant.format == "gguf"
            else f"<{origin} after model-pull>"
        )
        local = hub.local_model(cfg, model_id, variant)
        if local is not None:
            model_path = str(local)
        elif check_cache and local is None:
            raise NotFoundError(
                f"{model_id}:{variant.id} is not downloaded (serve never downloads implicitly)",
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
        revision=opts.revision if spec is None else None,
        extra_args=list(opts.extra_args),
    )

    if spec is not None and variant is not None:
        compat = engine.compatibility(spec, variant, hw)
        if compat.level == "unsupported":
            msg = f"{model_ref} on {engine.name}: {'; '.join(compat.reasons)}"
            if not opts.force:
                raise UnsupportedError(msg, hint="pick a recommended variant (`make recommend`)")
            warnings.append(f"FORCED: {msg}")
        if compat.level == "experimental":
            warnings.append(f"experimental configuration: {'; '.join(compat.reasons)}")
        launch.dtype = compat.dtype
        launch.embedding = "embedding" in spec.workloads
        if ctx > spec.context_length:
            raise UsageError(
                f"context {ctx} exceeds {spec.display_name}'s native context {spec.context_length}"
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
            msg = "estimated not to fit: " + "; ".join(fit.reasons)
            if not opts.force:
                raise UnsupportedError(
                    msg, hint="lower CONTEXT / CONCURRENCY, choose a smaller variant, or FORCE=1"
                )
            warnings.append(f"FORCED: {msg}")
        else:
            est = fit.estimate
            if est:
                rationale.append(
                    f"estimated memory: weights {gib(est.weights_bytes)}, KV {gib(est.kv_bytes)}, "
                    f"overhead {gib(est.overhead_bytes)} ({fit.layout})"
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
            f"binding to {launch.host}: the endpoint will be reachable from the network"
            + (
                ""
                if os.environ.get(API_KEY_ENV)
                else f" WITHOUT authentication (set {API_KEY_ENV})"
            )
        )
    binary = engine.find_binary(cfg)
    launch.binary = binary
    if binary is None:
        warnings.append(
            f"{engine.binary_name} not installed yet: make engine-install ENGINE={engine.name}"
        )
    argv = engine.build_command(launch, binary or engine.binary_name)
    return LaunchPlan(launch, argv, {}, binary, rationale, warnings, public)


# ------------------------------------------------------------------------- process


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
    """Never route loopback requests through HTTP(S)_PROXY."""
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
    req = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310 - http(s) only
    if not url.startswith(("http://", "https://")):
        raise UsageError(f"refusing non-HTTP URL {url}")
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

    # ------------------------------------------------------------- state files
    def _state_path(self, name: str) -> Path:
        if not name.replace("-", "").replace("_", "").isalnum():
            raise UsageError(f"invalid server name {name!r}")
        return self.dir / f"{name}.json"

    def load(self, name: str) -> ServerState | None:
        p = self._state_path(name)
        try:
            return ServerState(**json.loads(p.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (ValueError, TypeError) as e:
            raise SetupWorkError(
                f"corrupt state file {p}: {e}", hint=f"inspect and delete {p}"
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
        """True only if the PID is alive *and* is the very process we started."""
        if not _pid_alive(st.pid):
            return False
        current = _proc_start_time(st.pid)
        return st.proc_start is not None and current == st.proc_start

    # ----------------------------------------------------------------- health
    def healthy(self, st: ServerState, timeout: float = 2.0) -> bool:
        try:
            code, _ = http_json(st.base_url + st.health_path, timeout=timeout)
            if code != 200:
                return False
            code, _ = http_json(st.base_url + "/v1/models", timeout=timeout)
            return code == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    # ------------------------------------------------------------------ start
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
                    f"server '{launch.name}' is already running (pid {existing.pid}, "
                    f"{existing.base_url})",
                    hint="make serve-stop, or use INSTANCE=<other-name> PORT=<other-port>",
                )
            self._clear(launch.name)
        if not plan.binary:
            raise MissingDependencyError(
                f"{launch.engine} server binary not found",
                hint=f"install it with `make engine-install ENGINE={launch.engine}`",
            )
        if port_in_use(launch.host, launch.port):
            raise UsageError(
                f"port {launch.port} on {launch.host} is already in use",
                hint="choose another port, e.g. "
                f"PORT={find_free_port(launch.host, launch.port + 1)}",
            )
        if plan.public:
            self.runner.require_confirmation(
                f"Expose the inference server on {launch.host}:{launch.port}?"
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
            env.setdefault("HF_HUB_OFFLINE", "1")  # weights are local: never fetch implicitly
        with log_path.open("ab") as log:
            log.write(
                f"\n=== setup-work start {time.strftime('%Y-%m-%dT%H:%M:%S')} "
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
        self.runner.info(f"started pid {proc.pid}; logs: {log_path}")
        if wait:
            self.wait_ready(st, proc)
        if not detach:
            self._foreground(st, proc)
        return st

    def wait_ready(self, st: ServerState, proc: subprocess.Popen[bytes] | None = None) -> None:
        timeout = self.cfg.defaults.inference.ready_timeout_s
        deadline = time.monotonic() + timeout
        self.runner.info(f"waiting for {st.base_url}{st.health_path} (timeout {timeout}s) ...")
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                tail = self.tail(st.name, 25)
                self._clear(st.name)
                raise SetupWorkError(
                    f"{st.engine} exited with code {proc.returncode} before becoming ready",
                    hint="last log lines:\n" + "\n".join(tail),
                )
            if self.healthy(st):
                self.runner.info(
                    f"[green]ready[/] {st.base_url}/v1 (model: {st.served_model_name})"
                )
                self.runner.event("serve.ready", name=st.name, url=st.base_url)
                return
            time.sleep(2)
        raise CheckFailedError(
            f"server did not become ready within {timeout}s (still running, pid {st.pid})",
            hint="check `make serve-logs`; stop it with `make serve-stop`",
        )

    def _foreground(self, st: ServerState, proc: subprocess.Popen[bytes]) -> None:
        self.runner.info("[dim]foreground mode: streaming logs, Ctrl-C stops the server[/]")
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
            self.runner.info("\ninterrupt received: stopping server")
        finally:
            if proc.poll() is None:
                self.stop(st.name)
            else:
                self._clear(st.name)

    # ------------------------------------------------------------------- stop
    def stop(self, name: str) -> bool:
        st = self.load(name)
        if st is None:
            self.runner.info(f"no server named '{name}' is recorded")
            return False
        if not self.owns(st):
            self.runner.warn(
                f"pid {st.pid} is not the process setup-work started (exited or PID reused); "
                "removing stale state without signalling anything"
            )
            self._clear(name)
            return False
        self.runner.info(f"stopping '{name}' (pid {st.pid}, process group {st.pgid})")
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
            self.runner.warn(f"still running after {timeout}s: sending SIGKILL")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(st.pgid, signal.SIGKILL)
        self._clear(name)
        self.runner.event("serve.stop", name=name, pid=st.pid)
        return True

    # ---------------------------------------------------------------- status
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

    # ------------------------------------------------------------------ test
    def smoke_test(self, name: str, url: str | None = None, timeout: float = 120) -> dict[str, Any]:
        st = self.load(name) if url is None else None
        if url is None:
            if st is None:
                raise NotFoundError(f"no server named '{name}'", hint="start one with `make serve`")
            url = st.base_url
        url = url.rstrip("/")
        t0 = time.monotonic()
        code, models = http_json(url + "/v1/models", timeout=10)
        if code != 200 or not isinstance(models, dict) or not models.get("data"):
            raise CheckFailedError(f"GET {url}/v1/models returned {code}: {str(models)[:200]}")
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
            raise CheckFailedError(f"request to {url} failed with {code}: {str(body)[:300]}")
        return {
            "url": url,
            "model": model_id,
            "ok": True,
            "latency_s": round(time.monotonic() - t0, 3),
            **result,
        }
