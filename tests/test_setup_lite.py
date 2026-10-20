from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/setup-lite.sh"
SHELLS = [s for s in ("sh", "dash") if shutil.which(s)]


def configure(tmp_path: Path, agents: str = "", models: str = "", **settings: str) -> Path:
    """Копия скрипта с заполненным блоком НАСТРОЙКИ — так его использует сотрудник."""
    text = SCRIPT.read_text()
    defaults = {"PYPI_INDEX": "", "NPM_REGISTRY": "", "LLAMA_CPP_URI": "", "VLLM_URI": ""}
    for key, value in {**defaults, **settings}.items():
        text, n = re.subn(rf"^{key}=.*$", f"{key}='{value}'", text, count=1, flags=re.M)
        assert n == 1, key
    for key, value in (("AGENTS", agents), ("MODELS", models)):
        text, n = re.subn(rf"^{key}='.*?'$", f"{key}='\n{value}\n'", text, flags=re.M | re.S)
        assert n == 1, key
    script = tmp_path / "setup-lite.sh"
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
    assert "npm@12.2.0" in out
    assert "(dry-run) вопрос: Будете работать с агентами?" in out
    assert "(dry-run) вопрос: Установить qwen-code" in out
    assert "(dry-run) вопрос: Установить Node.js 24.20.0" in out
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
    declined = run(script, tmp_path, "--no-node", "--no-agents")
    assert declined.returncode == 0, declined.stderr
    assert "ответ «нет»" in declined.stdout
    assert not (tmp_path / "sw/registries").exists()
    result = run(script, tmp_path, "--no-node", "--no-agents", "--yes")
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
    assert len(runs.read_text().splitlines()) == 1

    commit(repo, "v2")
    declined = run(script, tmp_path, "--agents=demo")
    assert declined.returncode == 0, declined.stderr
    assert "оставлена установленная версия" in declined.stdout
    assert len(runs.read_text().splitlines()) == 1

    updated = run(script, tmp_path, "--agents=demo", "--yes")
    assert updated.returncode == 0, updated.stderr
    assert len(runs.read_text().splitlines()) == 2
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    marker = tmp_path / "sw/data/harnesses/demo/.setup-lite-commit"
    assert marker.read_text().strip() == head


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


def test_unknown_agent_rejected_before_install(tmp_path):
    script = configure(tmp_path, agents="demo https://git.corp.invalid/demo.git main")
    result = run(script, tmp_path, "--agents=demo,typo")
    assert result.returncode == 6
    assert "typo" in result.stderr
    assert not (tmp_path / "sw/data/harnesses").exists()


def _npm_package(path: Path, files: dict[str, str]) -> None:
    import tarfile

    staging = path.with_suffix(".d")
    for name, body in files.items():
        target = staging / "package" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        target.chmod(0o755)
    with tarfile.open(path, "w:gz") as tar:
        tar.add(staging / "package", arcname="package")


def test_node_installed_from_cached_registry_packages(tmp_path):
    """Архивы уже в кеше загрузок: сеть не нужна, проверяются распаковка и ссылки."""
    machine = os.uname()
    system = {"Darwin": "darwin", "Linux": "linux"}[machine.sysname]
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}[machine.machine]
    package = (
        "node-bin-darwin-arm64" if f"{system}-{arch}" == "darwin-arm64" else f"node-{system}-{arch}"
    )
    cache = tmp_path / "sw/data/tools/downloads/npm"
    cache.mkdir(parents=True)
    node = (
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then echo v24.20.0; exit; fi\n'
        'case "$1" in *npm-cli.js|*npx-cli.js) [ "$2" = --version ] && echo 12.2.0 ;; esac\n'
    )
    _npm_package(cache / f"{package}-24.20.0.tgz", {"bin/node": node})
    _npm_package(cache / "npm-12.2.0.tgz", {"bin/npm-cli.js": "", "bin/npx-cli.js": ""})
    script = configure(tmp_path, NPM_REGISTRY="https://npm.corp.invalid/npm/", NPM_AUTH_FILE="")
    result = run(script, tmp_path, "--no-agents", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    bin_dir = tmp_path / "sw/bin"
    npm = subprocess.run([str(bin_dir / "npm"), "--version"], capture_output=True, text=True)
    assert npm.stdout.strip() == "12.2.0"
    again = run(script, tmp_path, "--no-agents")
    assert "уже установлены" in again.stdout  # без вопроса


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
    assert (models / "hf/model/config.json").is_file()
    assert (models / "hf/model/model.safetensors").is_file()
    assert "llama-server -m" in result.stdout

    again = run(script, tmp_path, "--no-agents", "--local")
    assert again.returncode == 0, again.stderr
    assert again.stdout.count("уже установлен") == 2


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
    result = run(script, tmp_path, "--no-agents", "--local")
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
    dry = run(script, tmp_path, "--no-agents", "--local", "--dry-run")
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
    answers = ["y", "n", "y", "y", "n", "y"]  # агенты, skipme, keepme, GPU, llama.cpp, модель
    code, out = run_tty(script, tmp_path, answers, path=str(fake))
    assert code == 0, out
    for question in (
        "Будете работать с агентами? [y/N]",
        "Установить skipme из",
        "Установить keepme из",
        "Обнаружены GPU. Установить движки и модели? [y/N]",
        "Подключить llama.cpp из",
        "Подключить модель m на месте",
    ):
        assert question in out
    harnesses = tmp_path / "sw/data/harnesses"
    assert not (harnesses / "skipme").exists()
    assert (harnesses / "keepme/.setup-lite-commit").is_file()
    assert not (tmp_path / "sw/bin/llama-server").exists()
    models = Path(f"{tmp_path}/tmp/setup-work-{os.getuid()}/models")
    assert (models / "m/model").resolve() == gguf.resolve()
    assert "llama-server -m" not in out  # движок не подключён — команду запуска не предлагать

    # Повторно: про установленное не спрашивается, отказанное предлагается снова.
    # агенты, skipme, GPU, llama.cpp (снова предлагается); keepme и модель уже установлены
    code, out = run_tty(script, tmp_path, ["y", "n", "y", "n"], path=str(fake))
    assert code == 0, out
    assert "Установить skipme из" in out
    assert "Установить keepme из" not in out
    assert "keepme" in out and "актуальна" in out
    assert "Подключить llama.cpp из" in out
    assert "Подключить модель m" not in out
    assert "m: " in out and "→" in out
