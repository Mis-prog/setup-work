from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("existing_venv", [True, False])
@pytest.mark.parametrize("section", ["global", "install"])
def test_bootstrap_recovers_without_base_pip_before_checking_index(
    tmp_path, existing_venv, section
):
    """Активная .venv без pip не должна давать ложную ошибку конфигурации PyPI."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/bootstrap.sh", scripts / "bootstrap.sh")
    base = repo / ".venv" if existing_venv else tmp_path / "base-python"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(base)], check=True)
    python = base / "bin/python"
    assert (
        subprocess.run([str(python), "-I", "-c", "import pip"], capture_output=True).returncode != 0
    )
    config = tmp_path / "pip.conf"
    config.write_text(
        f"[{section}]\nindex-url = https://token:TEST_SECRET@pypi.corp.invalid/simple\n"
    )
    result = subprocess.run(
        ["sh", str(scripts / "bootstrap.sh"), "--sync-only"],
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": os.defpath,
            "PYTHON": str(python),
            "SETUP_WORK_USER_CONFIG_DIR": str(tmp_path / "userconf"),
            "PIP_CONFIG_FILE": str(config),
            "PIP_NO_INDEX": "1",  # Только локальный bundled pip; сеть не используется.
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    # В тестовом repo нет проекта для установки. Нужен именно выход на этот шаг,
    # а не ошибка PyPI до создания/восстановления окружения.
    assert result.returncode == 1, result.stdout + result.stderr
    assert "-m pip install" in result.stdout
    assert "не задан корпоративный PyPI" not in result.stderr
    assert "TEST_SECRET" not in result.stdout + result.stderr
    restored = repo / ".venv/bin/python"
    assert (
        subprocess.run([str(restored), "-I", "-c", "import pip"], capture_output=True).returncode
        == 0
    )
    assert not (repo / ".venv/.setup-work-synced").exists()


def test_bootstrap_dry_run_plans_pip_repair_without_writing(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/bootstrap.sh", scripts / "bootstrap.sh")
    python = repo / ".venv/bin/python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(python.parents[1])], check=True
    )
    result = subprocess.run(
        ["sh", str(scripts / "bootstrap.sh"), "--sync-only", "--dry-run"],
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": os.defpath,
            "PYTHON": str(python),
            "PIP_CONFIG_FILE": os.devnull,
            "SETUP_WORK_USER_CONFIG_DIR": str(tmp_path / "conf"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "-m ensurepip --upgrade" in result.stdout
    assert "после восстановления pip проверить" in result.stdout
    assert "-m pip install" in result.stdout
    assert (
        subprocess.run([str(python), "-I", "-c", "import pip"], capture_output=True).returncode != 0
    )
    assert not (repo / ".venv/.setup-work-synced").exists()


def test_missing_ensurepip_is_not_reported_as_missing_index(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/bootstrap.sh", scripts / "bootstrap.sh")
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("""#!/bin/sh
case "$1:$2" in
    -I:-c) exit 1 ;;
    -m:ensurepip) echo 'No module named ensurepip' >&2; exit 1 ;;
esac
exit 99
""")
    python.chmod(0o755)
    result = subprocess.run(
        ["sh", str(scripts / "bootstrap.sh"), "--sync-only"],
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": os.defpath,
            "PYTHON": sys.executable,
            "PIP_CONFIG_FILE": os.devnull,
            "SETUP_WORK_USER_CONFIG_DIR": str(tmp_path / "conf"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "No module named ensurepip" in result.stderr
    assert "не удалось восстановить pip" in result.stderr
    assert "не задан корпоративный PyPI" not in result.stderr
    assert "-m pip install" not in result.stdout


def test_missing_index_still_prevents_installation(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/bootstrap.sh", scripts / "bootstrap.sh")
    result = subprocess.run(
        ["sh", str(scripts / "bootstrap.sh"), "--sync-only"],
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": os.defpath,
            "PYTHON": sys.executable,
            "PIP_CONFIG_FILE": os.devnull,
            "SETUP_WORK_USER_CONFIG_DIR": str(tmp_path / "conf"),
            "PIP_NO_INDEX": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "не задан корпоративный PyPI" in result.stderr
    assert "-m pip install" not in result.stdout
