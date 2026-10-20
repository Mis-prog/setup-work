from __future__ import annotations

import io
import json
import subprocess
import tarfile
import urllib.request

import pytest

from setup_work import bootstrap
from setup_work import node_registry as nr
from setup_work.diagnostics import Environment, ToolStatus
from setup_work.errors import SetupWorkError, UsageError
from setup_work.runner import Runner
from setup_work.steps import execute


def package(name, version, files):
    result = io.BytesIO()
    files = {"package.json": json.dumps({"name": name, "version": version}).encode(), **files}
    with tarfile.open(fileobj=result, mode="w:gz") as archive:
        for path, data in files.items():
            info = tarfile.TarInfo("package/" + path)
            info.size = len(data)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(data))
    return result.getvalue()


class Mirror:
    def __init__(self, cfg):
        self.requests = []
        self.payloads = {}
        for name, version, files in [
            (
                "node-linux-x64",
                cfg.defaults.tools["node"].version,
                {
                    "bin/node": (
                        b'#!/bin/sh\nif [ "$1" = "--version" ]; then echo v24.20.0; '
                        b"else echo 12.2.0; fi\n"
                    )
                },
            ),
            (
                "npm",
                cfg.defaults.tools["npm"].version,
                {"bin/npm-cli.js": b"// npm", "bin/npx-cli.js": b"// npx"},
            ),
        ]:
            payload = package(name, version, files)
            path = f"{name}/-/{name}-{version}.tgz"
            self.payloads[path] = payload
            self.payloads[f"{name}/{version}"] = json.dumps(
                {
                    "name": name,
                    "version": version,
                    "dist": {
                        "tarball": "https://registry.npmjs.org/" + path,
                    },
                }
            ).encode()

    def open(self, request, timeout):
        assert request.full_url.startswith("https://npm.corp/repository/npm/")
        assert request.get_header("Authorization") == "Bearer private-token"
        self.requests.append(request.full_url)
        return io.BytesIO(
            self.payloads[request.full_url.removeprefix("https://npm.corp/repository/npm/")]
        )


@pytest.fixture
def mirror(cfg, monkeypatch):
    mirror = Mirror(cfg)
    monkeypatch.setattr(nr.Registry, "opener", lambda self: mirror)
    return mirror


def test_cold_install_without_node_or_npm(cfg, mirror, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    registry = nr.Registry("https://npm.corp/repository/npm/", "private-token")
    nr.install(cfg, registry, "node-linux-x64", "linux-x64", Runner(interactive=False))
    for command, expected in [("node", "v24.20.0"), ("npm", "12.2.0"), ("npx", "12.2.0")]:
        result = subprocess.run(
            [str(cfg.bin_dir / command), "--version"], capture_output=True, text=True
        )
        assert result.returncode == 0 and result.stdout.strip() == expected
    assert len(mirror.requests) == 4
    # Повторная установка использует архивы закреплённых версий из кэша.
    nr.install(cfg, registry, "node-linux-x64", "linux-x64", Runner(interactive=False))
    assert len(mirror.requests) == 6


def test_corrupt_package_never_activates(cfg, mirror):
    mirror.payloads["node-linux-x64/-/node-linux-x64-24.20.0.tgz"] = b"corrupt"
    with pytest.raises(tarfile.ReadError):
        nr.install(
            cfg,
            nr.Registry("https://npm.corp/repository/npm/", "private-token"),
            "node-linux-x64",
            "linux-x64",
            Runner(interactive=False),
        )
    assert not (cfg.bin_dir / "node").exists()
    assert not list(cfg.tools_dir.rglob("*.part"))


def test_read_registry_token_scope_and_ca(cfg, tmp_path, monkeypatch):
    path = tmp_path / "npmrc"
    path.write_text(
        "registry=https://npm.corp/repository/npm/\n"
        "//other.corp/:_authToken=wrong\n"
        "//npm.corp/:_authToken=general\n"
        "//npm.corp/repository/npm/:_authToken=private-token\n"
        "cafile=/corp/ca.pem\nstrict-ssl=true\n"
    )
    monkeypatch.setenv("NPM_CONFIG_USERCONFIG", str(path))
    for key in ("NPM_CONFIG_REGISTRY", "npm_config_registry", "SETUP_WORK_NPM_REGISTRY"):
        monkeypatch.delenv(key, raising=False)
    registry = nr.read_registry(cfg)
    assert registry.token == "private-token" and registry.cafile == "/corp/ca.pem"
    assert registry.strict_ssl and "private-token" not in repr(registry)


@pytest.mark.parametrize(
    "url", ["https://registry.npmjs.org/", "http://npm.corp/", "https://user:password@npm.corp/"]
)
def test_reject_public_or_insecure_registry(cfg, monkeypatch, url):
    monkeypatch.setenv("SETUP_WORK_NPM_REGISTRY", url)
    with pytest.raises(UsageError):
        nr.read_registry(cfg)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.corp/x",
        "https://npm.corp/other/file",
        "https://npm.corp/repository/npm/../secret",
    ],
)
def test_reject_redirects_outside_registry(url):
    redirect = nr._RegistryRedirect(nr.Registry("https://npm.corp/repository/npm/"))
    request = urllib.request.Request("https://npm.corp/repository/npm/pkg")
    with pytest.raises(SetupWorkError, match="перенаправляет"):
        redirect.redirect_request(request, None, 302, "", {}, url)


@pytest.mark.parametrize(
    "system,arch,name",
    [
        ("Darwin", "arm64", "node-bin-darwin-arm64"),
        ("Darwin", "x86_64", "node-darwin-x64"),
        ("Linux", "arm64", "node-linux-arm64"),
        ("Linux", "x86_64", "node-linux-x64"),
    ],
)
def test_plan_and_dry_run_without_network(cfg, monkeypatch, system, arch, name):
    monkeypatch.setattr(bootstrap, "_platform", lambda: (system, arch))
    monkeypatch.setattr(bootstrap, "_is_musl", lambda: False)
    monkeypatch.setenv("SETUP_WORK_NPM_REGISTRY", "https://npm.corp/repository/npm/")
    monkeypatch.setattr(nr.Registry, "opener", lambda self: pytest.fail("network during dry-run"))
    plan = bootstrap.plan_node(cfg)
    assert name + "@24.20.0" in plan[0].description
    execute(plan, Runner(dry_run=True, interactive=False), cfg.data_dir)
    assert not cfg.tools_dir.exists()


def test_missing_npm_requires_install(cfg, monkeypatch):
    tools = [ToolStatus("node", True, "24.20.0", "/node", satisfies=True), ToolStatus("npm", False)]
    env = Environment(python_runtime="3.12", tools=tools, bin_dir_on_path=True, bin_dir="/bin")
    monkeypatch.setattr(bootstrap, "plan_node", lambda cfg: [])
    assert bootstrap._plan_one(cfg, env, "node", False).status == "install"
