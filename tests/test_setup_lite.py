from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/setup.sh"
SHELLS = [s for s in ("sh", "dash") if shutil.which(s)]


def configure(tmp_path: Path, agents: str = "", models: str = "", **settings: str) -> Path:
    """Копия скрипта с заполненным блоком НАСТРОЙКИ — так его использует сотрудник."""
    text = SCRIPT.read_text()
    defaults = {
        "PYPI_INDEX": "",
        "NPM_REGISTRY": "",
        "LLAMA_CPP_URI": "",
        "VLLM_URI": "",
        "SERVE_HOST": "127.0.0.1",
        "SERVE_PORT": "8000",
    }
    for key, value in {**defaults, **settings}.items():
        text, n = re.subn(rf"^{key}=.*$", f"{key}='{value}'", text, count=1, flags=re.M)
        assert n == 1, key
    for key, value in (("AGENTS", agents), ("MODELS", models)):
        text, n = re.subn(rf"^{key}='.*?'$", f"{key}='\n{value}\n'", text, flags=re.M | re.S)
        assert n == 1, key
    script = tmp_path / "setup.sh"
    script.write_text(text)
    return script


def run(script: Path, tmp_path: Path, *args: str, shell: str | None = None):
    shell = shell or os.environ.get("SETUP_LITE_SHELL", "sh")
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": os.environ["PATH"],
        "SETUP_WORK_HOME": str(tmp_path / "sw"),
        "TMPDIR": str(tmp_path / "tmp"),
    }
    (tmp_path / "tmp").mkdir(exist_ok=True)
    return subprocess.run(
        [shell, str(script), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "agent-repo"
    repo.mkdir()
    (repo / "setup.sh").write_text(
        'printf "%s %s\\n" "$SETUP_WORK_BACKEND" "$SETUP_WORK_HARNESS_HOME" \\\n'
        '    >> "$SETUP_WORK_HARNESS_CONFIG/runs"\n'
    )
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return repo


def commit(repo: Path, message: str) -> None:
    (repo / "README").write_text(message)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", message], check=True)


@pytest.mark.parametrize("shell", SHELLS)
def test_placeholders_stop_before_any_change(tmp_path, shell):
    result = run(SCRIPT, tmp_path, "--dry-run", shell=shell)
    assert result.returncode == 2
    assert "PYPI_INDEX" in result.stderr and "CHANGE-ME" in result.stderr
    assert not (tmp_path / "sw").exists()
    # Пропущенные флагами шаги заготовки не проверяют: можно пробовать по частям.
    skipped = run(
        SCRIPT, tmp_path, "--dry-run", "--no-registries", "--no-agents", "--no-local", shell=shell
    )
    assert skipped.returncode == 0, skipped.stderr


def test_home_defaults_to_script_directory(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = configure(workdir)
    (tmp_path / "elsewhere").mkdir()
    result = subprocess.run(
        ["sh", str(script), "--no-agents", "--no-local"],
        cwd=tmp_path / "elsewhere",
        env={"HOME": str(tmp_path / "home"), "PATH": os.environ["PATH"], "TMPDIR": str(tmp_path)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    home = workdir.resolve()
    assert f"основной каталог: {home}" in result.stdout
    assert f"export SETUP_WORK_HOME='{home}'" in (workdir / "env.sh").read_text()
    assert not (tmp_path / "elsewhere/env.sh").exists()


@pytest.mark.parametrize("shell", SHELLS)
def test_dry_run_plans_everything_without_writing(tmp_path, shell):
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    (tokens / "npm").write_text("NPM_SECRET\n")
    script = configure(
        tmp_path,
        agents="qwen-code https://git.corp.invalid/scm/ai/qwen-code.git main",
        PYPI_INDEX="https://pypi.corp.invalid/simple",
        PYPI_AUTH_FILE="",
        NPM_REGISTRY="https://npm.corp.invalid/repository/npm",
        NPM_AUTH_FILE=str(tokens / "npm"),
    )
    result = run(script, tmp_path, "--dry-run", shell=shell)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "https://npm.corp.invalid/repository/npm/" in out
    assert "(dry-run) вопрос: Будете работать с агентами?" in out
    assert "(dry-run) вопрос: Установить qwen-code" in out
    assert "qwen-code" in out
    assert "NPM_SECRET" not in out + result.stderr
    assert not (tmp_path / "sw").exists()


def test_registries_are_private_and_token_is_encoded(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    (tokens / "pypi").write_text("a/b:c@d\n")
    (tokens / "npm").write_text("NPM_SECRET\n")
    script = configure(
        tmp_path,
        PYPI_INDEX="https://pypi.corp.invalid/simple",
        PYPI_AUTH_FILE=str(tokens / "pypi"),
        NPM_REGISTRY="https://npm.corp.invalid/npm/",
        NPM_AUTH_FILE=str(tokens / "npm"),
    )
    declined = run(script, tmp_path, "--no-agents")
    assert declined.returncode == 0, declined.stderr
    assert "ответ «нет»" in declined.stdout
    assert not (tmp_path / "sw/registries").exists()
    result = run(script, tmp_path, "--no-agents", "--yes")
    assert result.returncode == 0, result.stderr
    assert "NPM_SECRET" not in result.stdout + result.stderr
    pip_conf = tmp_path / "sw/registries/pip.conf"
    npmrc = tmp_path / "sw/registries/npmrc"
    assert "index-url = https://token:a%2Fb%3Ac%40d@pypi.corp.invalid/simple" in (
        pip_conf.read_text()
    )
    assert "//npm.corp.invalid/npm/:_authToken=NPM_SECRET" in npmrc.read_text()
    assert "audit=false" in npmrc.read_text()
    for path in (pip_conf, npmrc):
        assert path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "sw/env.sh").is_file()

    # Настроено — повторно не спрашивается; новый токен — снова вопрос.
    again = run(script, tmp_path, "--no-agents")
    assert "PyPI: уже настроен" in again.stdout and "npm: уже настроен" in again.stdout
    assert "вопрос" not in again.stdout
    (tokens / "npm").write_text("NEW_SECRET\n")
    rotated = run(script, tmp_path, "--no-agents")
    assert "PyPI: уже настроен" in rotated.stdout
    assert 'вопрос без терминала: "Записать корпоративный npm' in rotated.stdout


def test_registries_configured_by_corporate_scripts_are_kept(tmp_path):
    home = tmp_path / "home"
    (home / ".config/pip").mkdir(parents=True)
    (home / ".config/pip/pip.conf").write_text(
        "[global]\nindex-url = https://u:p@pypi.corp.invalid/simple/\n"
    )
    (home / ".npmrc").write_text("registry=https://npm.corp.invalid/npm\n")
    script = configure(
        tmp_path,
        PYPI_INDEX="https://pypi.corp.invalid/simple",
        PYPI_AUTH_FILE="",
        NPM_REGISTRY="https://npm.corp.invalid/npm/",
        NPM_AUTH_FILE="",
    )
    result = run(script, tmp_path, "--no-agents")
    assert result.returncode == 0, result.stderr
    assert f"PyPI: уже настроен в {home}/.config/pip/pip.conf" in result.stdout
    assert f"npm: уже настроен в {home}/.npmrc" in result.stdout
    assert not (tmp_path / "sw/registries").exists()
    # npm в новых терминалах продолжает читать ~/.npmrc.
    assert f"NPM_CONFIG_USERCONFIG='{home}/.npmrc'" in (tmp_path / "sw/env.sh").read_text()


@pytest.mark.skipif(not shutil.which("git"), reason="нужен git")
def test_agent_install_reuse_and_update(tmp_path):
    repo = make_repo(tmp_path)
    script = configure(tmp_path, agents=f"demo {repo} main")
    runs = tmp_path / "sw/data/harnesses/demo/config/runs"

    first = run(script, tmp_path, "--agents=demo")
    assert first.returncode == 0, first.stdout + first.stderr
    assert runs.read_text().splitlines() == [f"gigachat {tmp_path}/sw/data/harnesses/demo"]

    again = run(script, tmp_path, "--agents=demo")
    assert again.returncode == 0, again.stderr
    assert "актуальна" in again.stdout
    # setup.sh установленного агента выполняется при каждом запуске.
    assert "повторный запуск setup.sh" in again.stdout
    assert len(runs.read_text().splitlines()) == 2

    commit(repo, "v2")
    declined = run(script, tmp_path, "--agents=demo")
    assert declined.returncode == 0, declined.stderr
    assert "на сервере новая ревизия main" in declined.stdout
    assert "новых коммитов: 1" in declined.stdout and " v2 (t, " in declined.stdout
    assert "оставлена установленная версия" in declined.stdout
    assert len(runs.read_text().splitlines()) == 3

    offline = run(script, tmp_path, "--agents=demo", "--no-updates")
    assert offline.returncode == 0, offline.stderr
    assert len(runs.read_text().splitlines()) == 4

    updated = run(script, tmp_path, "--agents=demo", "--yes")
    assert updated.returncode == 0, updated.stderr
    assert len(runs.read_text().splitlines()) == 5
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    marker = tmp_path / "sw/data/harnesses/demo/.setup-lite-commit"
    assert marker.read_text().strip() == head


@pytest.mark.skipif(not shutil.which("git"), reason="нужен git")
def test_broken_agent_can_be_reinstalled(tmp_path):
    repo = make_repo(tmp_path)
    script = configure(tmp_path, agents=f"demo {repo} main")
    root = tmp_path / "sw/data/harnesses/demo"
    runs = root / "config/runs"
    assert run(script, tmp_path, "--agents=demo").returncode == 0
    (root / "config/settings.json").write_text("{}")

    # Исходники пропали: без терминала — оставить как есть, с --yes — переустановить.
    shutil.rmtree(root / "src")
    kept = run(script, tmp_path, "--agents=demo")
    assert kept.returncode == 0, kept.stderr
    assert "установка повреждена" in kept.stdout and "оставлено как есть" in kept.stdout
    fixed = run(script, tmp_path, "--agents=demo", "--yes")
    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    assert "переустановка demo начисто" in fixed.stdout
    assert (root / "src/setup.sh").is_file()
    assert (root / "config/settings.json").is_file()  # настройки агента сохраняются
    assert len(runs.read_text().splitlines()) == 2

    # Явная переустановка без поломки и вопросов.
    forced = run(script, tmp_path, "--reinstall=demo")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert "переустановка demo начисто" in forced.stdout
    assert len(runs.read_text().splitlines()) == 3
    assert run(script, tmp_path, "--reinstall=typo").returncode == 6

    # Изменённые файлы мешают обновлению — предлагается переустановка.
    (root / "src/setup.sh").write_text((root / "src/setup.sh").read_text() + "# local\n")
    commit(repo, "v2")
    blocked = run(script, tmp_path, "--agents=demo", "--yes")
    assert blocked.returncode == 0, blocked.stdout + blocked.stderr
    assert "изменены файлы" in blocked.stdout and "переустановка demo начисто" in blocked.stdout
    assert "# local" not in (root / "src/setup.sh").read_text()


@pytest.mark.skipif(not shutil.which("git"), reason="нужен git")
def test_failed_setup_is_not_marked_installed(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "setup.sh").write_text("exit 3\n")
    commit(repo, "broken")
    script = configure(tmp_path, agents=f"demo {repo} main")
    result = run(script, tmp_path, "--agents=demo")
    assert result.returncode == 1
    assert "завершился с ошибкой" in result.stderr
    assert not (tmp_path / "sw/data/harnesses/demo/.setup-lite-commit").exists()
    # С согласием — одна попытка начисто, затем ошибка.
    retried = run(script, tmp_path, "--agents=demo", "--yes")
    assert retried.returncode == 1
    assert "переустановка demo начисто" in retried.stdout


@pytest.mark.skipif(not shutil.which("git"), reason="нужен git")
def test_failed_rerun_of_installed_agent_is_reported(tmp_path):
    repo = make_repo(tmp_path)
    script = configure(tmp_path, agents=f"demo {repo} main")
    assert run(script, tmp_path, "--agents=demo").returncode == 0
    marker = tmp_path / "sw/data/harnesses/demo/.setup-lite-commit"
    src = tmp_path / "sw/data/harnesses/demo/src/setup.sh"
    src.write_text("exit 3\n")
    result = run(script, tmp_path, "--agents=demo")
    assert result.returncode == 1
    assert "demo: setup.sh завершился с ошибкой" in result.stderr
    assert marker.is_file()  # установленная ревизия остаётся отмеченной


def test_unknown_agent_rejected_before_install(tmp_path):
    script = configure(tmp_path, agents="demo https://git.corp.invalid/demo.git main")
    result = run(script, tmp_path, "--agents=demo,typo")
    assert result.returncode == 6
    assert "typo" in result.stderr
    assert not (tmp_path / "sw/data/harnesses").exists()


def _zip(path: Path, files: dict[str, str | bytes], *, links: dict[str, str] | None = None) -> None:
    """ZIP как `zip -ry`: права Unix и символьные ссылки сохраняются."""
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in files.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o100755 << 16
            archive.writestr(info, body)
        for name, target in (links or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o120777 << 16
            archive.writestr(info, target)


FAKE_LLAMA = '#!/bin/sh\necho "version: 1 (fake)"; echo "LD=$LD_LIBRARY_PATH" >&2\n'


@pytest.mark.skipif(not shutil.which("unzip"), reason="нужен unzip")
def test_llama_and_models_from_local_sources(tmp_path):
    src = tmp_path / "nfs"
    _zip(
        src / "llama.cpp.zip",
        {"llama.cpp/build/bin/llama-server": FAKE_LLAMA, "llama.cpp/build/bin/libggml.so": ""},
    )
    gguf = src / "gguf/Q-Q4.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_text("gguf")
    parts = src / "hf-parts"
    _zip(parts / "Q_Q-1-of-2.zip", {"Q/config.json": "{}"})
    _zip(parts / "Q_Q-2-of-2.zip", {"Q/model.safetensors": "w"})
    script = configure(
        tmp_path,
        models=f"q4 {gguf}\nhf {parts}",
        LLAMA_CPP_URI=str(src / "llama.cpp.zip"),
        LLAMA_CPP_PATH="llama.cpp/build/bin/llama-server",
    )
    result = run(script, tmp_path, "--no-agents", "--local")
    assert result.returncode == 0, result.stdout + result.stderr
    llama = subprocess.run(
        [str(tmp_path / "sw/bin/llama-server"), "--version"], capture_output=True, text=True
    )
    runtime = Path(f"{tmp_path}/tmp/setup-work-{os.getuid()}")
    assert llama.returncode == 0
    assert f"LD={runtime}/engines/llama.cpp/current" in llama.stderr
    models = runtime / "models"
    assert (models / "q4/model").resolve() == gguf.resolve()  # на месте, без копии
    assert "рекомендуемые модели для установленных движков: q4 (llama.cpp)" in result.stdout
    assert "hf: пропущена — нужен vllm" in result.stdout  # vLLM нет — модель HF не предлагается
    assert not (models / "hf").exists()
    assert "llama-server -m" in result.stdout and "--alias q4" in result.stdout
    assert "Запуск OpenAI-совместимого API" in result.stdout  # без терминала запуск — «нет»
    assert not (runtime / "server/server.pid").exists()

    forced = run(script, tmp_path, "--no-agents", "--models=hf")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert (models / "hf/model/config.json").is_file()
    assert (models / "hf/model/model.safetensors").is_file()
    assert "vllm serve" not in forced.stdout  # движка нет — команду запуска не предлагать

    again = run(script, tmp_path, "--no-agents", "--local")
    assert again.returncode == 0, again.stderr
    assert again.stdout.count("уже установлен") == 1  # llama.cpp; q4 подключена на месте


def fake_server(path: Path, mode: str = "ready") -> None:
    """llama-server: --version; иначе /health отвечает 200 через секунду (ready),
    процесс падает (crash) или так и не начинает отвечать (hang)."""
    import sys

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import http.server, sys, time\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('version: 1 (fake)'); sys.exit(0)\n"
        "print('args:', ' '.join(sys.argv[1:]), flush=True)\n"
        f"mode = {mode!r}\n"
        "if mode == 'crash':\n"
        "    print('CUDA out of memory', flush=True); sys.exit(1)\n"
        "time.sleep(1 if mode == 'ready' else 60)\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200 if self.path == '/health' else 404); self.end_headers()\n"
        "    def log_message(self, *a): pass\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()\n"
    )
    path.chmod(0o755)


def free_port() -> str:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


@pytest.mark.parametrize(("sm", "vllm"), [("7.5", False), ("8.6", True)])
def test_vllm_offered_only_from_sm80(tmp_path, monkeypatch, sm, vllm):
    fake = tmp_path / "fakebin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text(
        f'#!/bin/sh\ncase "$*" in *compute_cap*) echo 9.0; echo {sm} ;; '
        '*) echo "GPU 0: NVIDIA (UUID: GPU-x)" ;; esac\n'
    )
    (fake / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    for d in ("llama", "vllm", "hf"):
        (tmp_path / "nfs" / d).mkdir(parents=True)
    (tmp_path / "nfs/hf/config.json").write_text("{}")
    (tmp_path / "nfs/m.gguf").write_text("gguf")
    script = configure(
        tmp_path,
        models=f"m {tmp_path}/nfs/m.gguf\nhf {tmp_path}/nfs/hf",
        LLAMA_CPP_URI=str(tmp_path / "nfs/llama"),
        VLLM_URI=str(tmp_path / "nfs/vllm"),
    )
    result = run(script, tmp_path, "--no-agents", "--dry-run")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "(dry-run) вопрос: Обнаружены GPU" in out
    assert "llama.cpp: используется на месте" in out
    assert ("vllm: " in out) is vllm
    assert ("m (llama.cpp) hf (vllm)" in out) is vllm
    if not vllm:
        assert f"compute capability {sm} (ниже 8.0) — vLLM не предлагается" in out
        assert f"hf: пропущена — нужен vLLM, а GPU sm{sm} его не поддерживает" in out
        served = run(script, tmp_path, "--no-agents", "--serve=hf", "--dry-run")
        assert served.returncode == 4
        assert "через vLLM" in served.stderr


def test_model_served_after_install(tmp_path):
    fake_server(tmp_path / "nfs/llama/bin/llama-server")
    gguf = tmp_path / "nfs/m.gguf"
    gguf.write_text("gguf")
    port = free_port()
    script = configure(
        tmp_path,
        models=f"m {gguf}\nhf {tmp_path}/nfs/hf/ zip-parts",
        LLAMA_CPP_URI=str(tmp_path / "nfs/llama"),
        SERVE_PORT=port,
    )
    dry = run(script, tmp_path, "--no-agents", "--local", "--dry-run", "--models=m")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert (
        f"(dry-run) вопрос: Запустить модель m через llama.cpp на 127.0.0.1:{port}?" in dry.stdout
    )
    assert f"ожидание готовности: http://127.0.0.1:{port}/health" in dry.stdout

    unknown = run(script, tmp_path, "--no-agents", "--serve=typo")
    assert unknown.returncode == 6
    assert "typo" in unknown.stderr

    result = run(script, tmp_path, "--no-agents", "--serve=m")
    assert result.returncode == 0, result.stdout + result.stderr
    server = Path(f"{tmp_path}/tmp/setup-work-{os.getuid()}/server")
    pid, name = (server / "server.pid").read_text().split()
    try:
        assert name == "m"
        # Скрипт отпускает только после ответа /health.
        assert "m готов за" in result.stdout
        health = subprocess.run(
            [
                "curl",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "--noproxy",
                "*",
                f"http://127.0.0.1:{port}/health",
            ],
            capture_output=True,
            text=True,
        )
        assert health.stdout == "200"
        log = (server / "m.log").read_text()
        assert f"--alias m --host 127.0.0.1 --port {port}" in log
        again = run(script, tmp_path, "--no-agents", "--local")
        assert "Сервер уже запущен: m" in again.stdout
    finally:
        os.kill(int(pid), 15)


@pytest.mark.parametrize("mode", ["crash", "hang"])
def test_server_that_never_gets_ready_is_reported(tmp_path, mode):
    fake_server(tmp_path / "nfs/llama/bin/llama-server", mode)
    gguf = tmp_path / "nfs/m.gguf"
    gguf.write_text("gguf")
    script = configure(
        tmp_path,
        models=f"m {gguf}",
        LLAMA_CPP_URI=str(tmp_path / "nfs/llama"),
        SERVE_PORT=free_port(),
        SERVE_TIMEOUT="4",
    )
    result = run(script, tmp_path, "--no-agents", "--serve=m")
    assert result.returncode == 1
    server = Path(f"{tmp_path}/tmp/setup-work-{os.getuid()}/server")
    assert not (server / "server.pid").exists()
    if mode == "crash":
        assert "завершился, не успев запуститься" in result.stderr
        assert "CUDA out of memory" in result.stderr  # хвост лога — сразу в выводе
    else:
        assert "не ответил за 4 с и остановлен" in result.stderr
        busy = subprocess.run(["pgrep", "-f", str(tmp_path / "nfs/llama")], capture_output=True)
        assert busy.returncode == 1  # сервер не остался в фоне


@pytest.mark.skipif(not shutil.which("unzip"), reason="нужен unzip")
def test_models_downloaded_from_hdfs(tmp_path):
    cluster = tmp_path / "cluster"
    _zip(cluster / "parts/M-1-of-2.zip", {"M/config.json": "{}"})
    _zip(cluster / "parts/M-2-of-2.zip", {"M/w.safetensors": "w"})
    (cluster / "gguf").mkdir()
    (cluster / "gguf/M.gguf").write_text("gguf")
    hdfs = tmp_path / "fake-hdfs"
    hdfs.write_text(
        "#!/bin/sh\n"
        f"root={cluster}\n"
        'p() { printf "%s" "$1" | sed "s#^hdfs:///#$root/#"; }\n'
        '[ "$1" = dfs ] || exit 2\n'
        'case "$2" in\n'
        '  -get) cp -R "$(p "$3")" "$4" ;;\n'
        '  -ls) for f in "$(p "$3")"/*; do\n'
        '         echo "-rw-r--r-- 3 u g 1 2026-10-20 10:00 $3${f##*/}"; done ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    hdfs.chmod(0o755)
    script = configure(
        tmp_path,
        models="hf hdfs:///parts/\nq hdfs:///gguf/M.gguf",
        HDFS_CLI=str(hdfs),
        HDFS_KERBEROS="off",
    )
    result = run(script, tmp_path, "--no-agents", "--local", "--models=q,hf")
    assert result.returncode == 0, result.stdout + result.stderr
    models = Path(f"{tmp_path}/tmp/setup-work-{os.getuid()}/models")
    assert (models / "q/model").read_text() == "gguf"
    assert (models / "hf/model/w.safetensors").is_file()
    assert not list((models.parent / "artifacts/downloads").iterdir())


def test_incomplete_zip_parts_rejected(tmp_path):
    parts = tmp_path / "parts"
    _zip(parts / "M-1-of-3.zip", {"a": "1"})
    _zip(parts / "M-3-of-3.zip", {"c": "3"})
    script = configure(tmp_path, models=f"m {parts}")
    result = run(script, tmp_path, "--no-agents", "--models=m")
    assert result.returncode == 2
    assert "отсутствуют части: 2 из 3" in result.stderr


@pytest.mark.skipif(not shutil.which("unzip"), reason="нужен unzip")
def test_vllm_venv_is_relocated(tmp_path):
    """venv собрано в другом месте с другим путём к Python: home, ссылки и shebang исправляются."""
    import sys

    build = tmp_path / "build/vllm-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(build)], check=True)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site = next(build.glob("lib/python*/site-packages"))
    (site / "vllm.py").write_text('__version__ = "0.0-test"\n')
    old = "/nonexistent/build/vllm-venv"
    long_old = "/nonexistent/" + "x" * 150 + "/bin/python"
    files: dict[str, str | bytes] = {
        "vllm-venv/pyvenv.cfg": f"home = /nonexistent/bin\nversion = {version}.0\n",
        "vllm-venv/bin/vllm": f"#!{old}/bin/python\nimport vllm\nprint('vllm', vllm.__version__)\n",
        "vllm-venv/bin/long-tool": (
            f"#!/bin/sh\n'''exec' '{long_old}' \"$0\" \"$@\"\n' '''\nprint('long ok')\n"
        ),
    }
    for f in site.rglob("*"):
        if f.is_file():
            files["vllm-venv/" + str(f.relative_to(build))] = f.read_bytes()
    links = {
        f"vllm-venv/bin/{n}": f"/nonexistent/bin/python{version}" for n in ("python", "python3")
    }
    _zip(tmp_path / "nfs/vllm.zip", files, links=links)
    script = configure(tmp_path, VLLM_URI=str(tmp_path / "nfs/vllm.zip"))
    env_path = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    result = subprocess.run(
        [os.environ.get("SETUP_LITE_SHELL", "sh"), str(script), "--no-agents", "--local"],
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": env_path,
            "SETUP_WORK_HOME": str(tmp_path / "sw"),
            "TMPDIR": str(tmp_path),
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "перепривязан" in result.stdout
    assert "vllm 0.0-test" in result.stdout
    vllm = subprocess.run([str(tmp_path / "sw/bin/vllm")], capture_output=True, text=True)
    assert vllm.stdout.strip() == "vllm 0.0-test", vllm.stderr
    venv = Path(f"{tmp_path}/setup-work-{os.getuid()}/engines/vllm/current")
    long_tool = subprocess.run([str(venv / "bin/long-tool")], capture_output=True, text=True)
    assert long_tool.stdout.strip() == "long ok", long_tool.stderr


def test_local_stage_skipped_without_gpu_unless_forced(tmp_path):
    script = configure(tmp_path, models="m /nonexistent/model.gguf")
    result = run(script, tmp_path, "--no-agents")
    assert result.returncode == 0, result.stderr
    assert "GPU нет" in result.stdout or "Обнаружены GPU" in result.stdout
    dry = run(script, tmp_path, "--no-agents", "--models=m", "--dry-run")
    assert dry.returncode == 6
    assert "/nonexistent/model.gguf" in dry.stderr


def run_tty(script: Path, tmp_path: Path, answers: list[str], *args: str, path: str = ""):
    """Запуск в настоящем терминале (pty): ответы читаются из /dev/tty по одному."""
    import fcntl
    import pty
    import select
    import termios

    master, slave = pty.openpty()
    os.write(master, "".join(a + "\n" for a in answers).encode())
    (tmp_path / "tmp").mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [os.environ.get("SETUP_LITE_SHELL", "sh"), str(script), *args],
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": f"{path}:{os.environ['PATH']}" if path else os.environ["PATH"],
            "SETUP_WORK_HOME": str(tmp_path / "sw"),
            "TMPDIR": str(tmp_path / "tmp"),
        },
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0),
    )
    os.close(slave)
    out = b""
    while True:
        ready, _, _ = select.select([master], [], [], 30)
        if not ready:
            proc.kill()
            break
        try:
            chunk = os.read(master, 65536)
        except OSError:  # Linux: EIO после закрытия pty процессом
            break
        if not chunk:
            break
        out += chunk
    os.close(master)
    return proc.wait(timeout=30), out.decode(errors="replace").replace("\r", "")


@pytest.mark.skipif(not shutil.which("git"), reason="нужен git")
def test_every_step_can_be_declined_in_terminal(tmp_path):
    """Отказ пропускает только свой шаг: дерево решений в одном прогоне."""
    (tmp_path / "keep").mkdir()
    (tmp_path / "skip").mkdir()
    keep, skip = make_repo(tmp_path / "keep"), make_repo(tmp_path / "skip")
    fake = tmp_path / "fakebin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text('#!/bin/sh\necho "GPU 0: NVIDIA A100 (UUID: GPU-x)"\n')
    (fake / "nvidia-smi").chmod(0o755)
    llama = tmp_path / "nfs/llama/bin/llama-server"
    llama.parent.mkdir(parents=True)
    llama.write_text(FAKE_LLAMA)
    llama.chmod(0o755)
    gguf = tmp_path / "nfs/m.gguf"
    gguf.write_text("gguf")
    script = configure(
        tmp_path,
        agents=f"skipme {skip} main\nkeepme {keep} main",
        models=f"m {gguf}",
        LLAMA_CPP_URI=str(tmp_path / "nfs/llama"),
    )
    answers = ["y", "n", "y", "y", "n"]  # агенты, skipme, keepme, GPU, llama.cpp
    code, out = run_tty(script, tmp_path, answers, path=str(fake))
    assert code == 0, out
    for question in (
        "Будете работать с агентами? [y/N]",
        "Установить skipme из",
        "Установить keepme из",
        "Обнаружены GPU. Установить движки и модели? [y/N]",
        "Подключить llama.cpp из",
    ):
        assert question in out
    harnesses = tmp_path / "sw/data/harnesses"
    assert not (harnesses / "skipme").exists()
    assert (harnesses / "keepme/.setup-lite-commit").is_file()
    assert not (tmp_path / "sw/bin/llama-server").exists()
    assert "m: пропущена — нужен llama.cpp" in out  # движок не подключён — GGUF не предлагать
    assert "Подключить модель m" not in out
    assert "llama-server -m" not in out

    # Повторно: про установленное не спрашивается, отказанное предлагается снова.
    # skipme, GPU, llama.cpp (снова предлагается), модель, запуск; keepme установлен — про
    # агентов в целом не спрашивается
    code, out = run_tty(script, tmp_path, ["n", "y", "y", "y", "n"], path=str(fake))
    assert code == 0, out
    assert "Будете работать с агентами?" not in out
    assert "Установить skipme из" in out
    assert "Установить keepme из" not in out
    assert "keepme" in out and "актуальна" in out
    assert "Подключить llama.cpp из" in out
    assert "Подключить модель m на месте" in out
    assert "Запустить модель m через llama.cpp на 127.0.0.1:8000? [y/N]" in out
    models = Path(f"{tmp_path}/tmp/setup-work-{os.getuid()}/models")
    assert (models / "m/model").resolve() == gguf.resolve()
    assert "llama-server -m" in out  # отказ от запуска — показать команду для ручного старта


@pytest.fixture
def pypi_server():
    """PyPI, который пускает только с токеном GOOD-TOKEN (Basic token:GOOD-TOKEN)."""
    import base64
    import http.server
    import threading

    good = "Basic " + base64.b64encode(b"token:GOOD-TOKEN").decode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.headers.get("Authorization") == good else 401)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/simple"
    server.shutdown()


def test_missing_token_is_asked_checked_and_saved(tmp_path, pypi_server):
    token = tmp_path / "tokens/pypi.txt"
    script = configure(tmp_path, PYPI_INDEX=pypi_server, PYPI_AUTH_FILE=str(token))
    # Без терминала спросить нельзя — понятная ошибка до изменений.
    result = run(script, tmp_path, "--no-agents", "--no-local", "--yes")
    assert result.returncode == 6
    assert "токен будет запрошен" in result.stderr

    # PyPI?, неверный токен, верный токен
    code, out = run_tty(script, tmp_path, ["y", "WRONG", "GOOD-TOKEN"], "--no-agents", "--no-local")
    assert code == 0, out
    assert "файл токена" in out and "не найден" in out
    assert "реестр отклонил токен (HTTP 401)" in out
    assert "токен действителен" in out
    assert token.read_text().strip() == "GOOD-TOKEN"
    assert token.stat().st_mode & 0o777 == 0o600
    assert "token:GOOD-TOKEN@127.0.0.1" in (tmp_path / "sw/registries/pip.conf").read_text()

    # Сохранённый токен проверяется без вопросов; отклонённый без терминала — предупреждение.
    again = run(script, tmp_path, "--no-agents", "--no-local")
    assert again.returncode == 0, again.stderr
    assert "PyPI: уже настроен" in again.stdout and "вопрос" not in again.stdout
    token.write_text("STALE\n")
    stale = run(script, tmp_path, "--no-agents", "--no-local", "--yes")
    assert stale.returncode == 0, stale.stderr
    assert "реестр отклонил токен" in stale.stdout and "замените токен" in stale.stdout


def test_interrupted_wait_stops_server(tmp_path):
    import signal
    import time

    fake_server(tmp_path / "nfs/llama/bin/llama-server", "hang")
    gguf = tmp_path / "nfs/m.gguf"
    gguf.write_text("gguf")
    script = configure(
        tmp_path,
        models=f"m {gguf}",
        LLAMA_CPP_URI=str(tmp_path / "nfs/llama"),
        SERVE_PORT=free_port(),
    )
    (tmp_path / "tmp").mkdir()
    proc = subprocess.Popen(
        [os.environ.get("SETUP_LITE_SHELL", "sh"), str(script), "--no-agents", "--serve=m"],
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": os.environ["PATH"],
            "SETUP_WORK_HOME": str(tmp_path / "sw"),
            "TMPDIR": str(tmp_path / "tmp"),
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pidfile = Path(f"{tmp_path}/tmp/setup-work-{os.getuid()}/server/server.pid")
    for _ in range(100):
        if pidfile.exists():
            break
        time.sleep(0.1)
    server_pid = int(pidfile.read_text().split()[0])
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 130
    assert "сервер остановлен" in err
    assert not pidfile.exists()
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)
