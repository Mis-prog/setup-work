"""Первичная установка Node.js и npm из корпоративного npm registry через Python."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shlex
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .config import Config
from .errors import NotFoundError, SetupWorkError, UsageError
from .runner import Runner, probe
from .steps import ensure_executable, make_link, safe_extract

_VERSION = re.compile(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?\Z")
_METADATA_LIMIT = 4 * 1024**2


def _origin(url: str) -> tuple[str, str | None, int]:
    parsed = urlsplit(url)
    return parsed.scheme, parsed.hostname, parsed.port or 443


@dataclass
class Registry:
    url: str
    token: str = field(default="", repr=False)
    cafile: str | None = None
    strict_ssl: bool = True

    def allows(self, url: str) -> bool:
        try:
            parsed = urlsplit(url)
            origin = _origin(url)
        except ValueError:
            return False
        path = unquote(parsed.path)
        return (
            origin == _origin(self.url)
            and not parsed.username
            and not parsed.password
            and not parsed.fragment
            and path.startswith(unquote(urlsplit(self.url).path))
            and not any(part in (".", "..") for part in path.split("/"))
            and not any(c in url for c in "\r\n\0")
        )

    def opener(self) -> urllib.request.OpenerDirector:
        context = (
            ssl.create_default_context(cafile=self.cafile)
            if self.strict_ssl
            else ssl._create_unverified_context()  # noqa: S323 - явная настройка strict-ssl=false
        )
        return urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context), _RegistryRedirect(self)
        )


class _RegistryRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self.registry.allows(newurl):
            raise SetupWorkError(
                "npm registry перенаправляет загрузку за пределы настроенного реестра",
                hint="исправьте адреса tarball и перенаправления в корпоративном прокси npm",
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def read_registry(cfg: Config) -> Registry:
    path = Path(
        os.environ.get("NPM_CONFIG_USERCONFIG")
        or os.environ.get("npm_config_userconfig")  # noqa: SIM112 - npm поддерживает этот ключ
        or (
            str(cfg.data_dir.parent / "registries/npmrc")
            if os.environ.get("SETUP_WORK_HOME")
            else str(Path.home() / ".npmrc")
        )
    ).expanduser()
    settings: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(("#", ";")) and "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
    url = (
        os.environ.get("SETUP_WORK_NPM_REGISTRY")
        or os.environ.get("NPM_CONFIG_REGISTRY")
        or os.environ.get("npm_config_registry")  # noqa: SIM112 - npm поддерживает этот ключ
        or settings.get("registry", "")
    ).rstrip("/") + "/"
    if url == "/" or "CHANGE-ME" in url:
        raise NotFoundError(
            "не настроен корпоративный npm registry для установки Node.js",
            hint=f"заполните SETUP_WORK_NPM_REGISTRY в {cfg.user_dir / 'bootstrap.env'} "
            "и выполните make registries; затем повторите make bootstrap",
        )
    try:
        parsed = urlsplit(url)
        _origin(url)
    except ValueError as exc:
        raise UsageError("неверный URL npm registry; исправьте SETUP_WORK_NPM_REGISTRY") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.hostname == "npmjs.org"
        or parsed.hostname.endswith(".npmjs.org")
        or any(c in url for c in "\r\n\0")
    ):
        raise UsageError(
            "для Node.js нужен HTTPS-адрес корпоративного npm registry без учётных данных в URL",
            hint="исправьте SETUP_WORK_NPM_REGISTRY в bootstrap.env и выполните make registries",
        )
    token = ""
    # Учитывать только токен с областью действия, покрывающей выбранный реестр.
    for key in sorted(settings, key=len):
        if key.startswith("//") and key.endswith(":_authToken"):
            scope = "https:" + key.removesuffix(":_authToken")
            if _origin(scope) == _origin(url) and parsed.path.startswith(urlsplit(scope).path):
                token = settings[key]
    auth_file = os.environ.get("SETUP_WORK_NPM_AUTH_FILE")
    if auth_file:
        try:
            token = Path(os.path.expandvars(auth_file)).expanduser().read_text().strip()
        except OSError as exc:
            raise SetupWorkError(
                "не читается SETUP_WORK_NPM_AUTH_FILE; проверьте путь и права файла токена"
            ) from exc
    token = os.path.expandvars(token)
    if any(c in token for c in "\r\n\0") or "${" in token:
        raise UsageError("неверный токен npm; проверьте файл токена и переменные окружения")
    cafile = os.environ.get("NPM_CONFIG_CAFILE") or settings.get("cafile")
    strict = os.environ.get("NPM_CONFIG_STRICT_SSL", settings.get("strict-ssl", "true"))
    registry = Registry(
        url,
        token,
        os.path.expandvars(os.path.expanduser(cafile)) if cafile else None,
        strict.lower() != "false",
    )
    if not registry.allows(url):
        raise UsageError("неверный путь npm registry; уберите переходы . и .. из URL")
    return registry


def _request(registry: Registry, opener, url: str):
    if not registry.allows(url):
        raise SetupWorkError("загрузка Node.js разрешена только из настроенного npm registry")
    headers = {"User-Agent": "setup-work", "Accept": "application/json"}
    if registry.token:
        headers["Authorization"] = "Bearer " + registry.token
    try:
        return opener.open(urllib.request.Request(url, headers=headers), timeout=120)  # noqa: S310 - URL проверен
    except urllib.error.HTTPError as exc:
        hint = (
            "проверьте токен npm и повторите make registries"
            if exc.code in (401, 403)
            else "проверьте наличие указанной версии пакета в корпоративном npm registry"
        )
        raise SetupWorkError(f"npm registry вернул HTTP {exc.code} для {url}", hint=hint) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise SetupWorkError(
            "не удалось обратиться к корпоративному npm registry",
            hint="проверьте адрес, прокси и SETUP_WORK_NPM_CAFILE в bootstrap.env",
        ) from exc


def _integrity(value: str) -> tuple[str, bytes]:
    for algorithm in ("sha512", "sha256"):
        for token in value.split():
            if token.startswith(algorithm + "-"):
                try:
                    digest = base64.b64decode(token.split("-", 1)[1], validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise SetupWorkError("неверное dist.integrity в ответе npm registry") from exc
                if len(digest) == hashlib.new(algorithm).digest_size:
                    return algorithm, digest
    raise SetupWorkError("пакет npm не содержит проверяемый dist.integrity SHA512 или SHA256")


def _digest(path: Path, algorithm: str) -> bytes:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.digest()


def _fetch_package(registry: Registry, opener, name: str, version: str, cache: Path) -> Path:
    with _request(registry, opener, f"{registry.url}{name}/{version}") as response:
        raw = response.read(_METADATA_LIMIT + 1)
    try:
        if len(raw) > _METADATA_LIMIT:
            raise ValueError("слишком большой ответ")
        meta = json.loads(raw)
        if meta["name"] != name or meta["version"] != version:
            raise ValueError("версия или имя не совпадают")
        dist = meta["dist"]
        algorithm, expected = _integrity(dist["integrity"])
        url = dist["tarball"]
        parsed = urlsplit(url)
        # Прокси npm иногда возвращает публичный tarball; запрашивать его через прокси.
        if parsed.hostname == "registry.npmjs.org":
            relative = f"{name}/-/{name}-{version}.tgz"
            if parsed.path != "/" + relative or parsed.query or parsed.fragment:
                raise ValueError("неожиданный адрес tarball")
            url = registry.url + relative
        if not registry.allows(url):
            raise ValueError("tarball находится вне корпоративного реестра")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise SetupWorkError(
            f"неверные метаданные {name}@{version} в npm registry",
            hint="проверьте имя, версию, dist.integrity и адрес dist.tarball в зеркале npm",
        ) from exc
    archive = cache / f"{name}-{version}.tgz"
    if archive.is_file() and _digest(archive, algorithm) == expected:
        return archive
    cache.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(".part")
    try:
        with _request(registry, opener, url) as response, part.open("wb") as stream:
            shutil.copyfileobj(response, stream, 1024**2)
        if _digest(part, algorithm) != expected:
            raise SetupWorkError(f"не совпадает dist.integrity пакета {name}@{version}")
        part.replace(archive)
    finally:
        part.unlink(missing_ok=True)
    return archive


def _verify(node: Path, npm_cli: Path, node_version: str, npm_version: str) -> None:
    for argv, expected in (
        ([str(node), "--version"], "v" + node_version),
        ([str(node), str(npm_cli), "--version"], npm_version),
    ):
        result = probe(argv, timeout=60)
        if not result.ok or result.stdout.strip() != expected:
            raise SetupWorkError(
                "распакованные Node.js/npm не прошли проверку запуска",
                hint="проверьте ОС, архитектуру, glibc и совместимость версий в defaults.yaml",
            )


def install(cfg: Config, registry: Registry, package: str, target: str, runner: Runner) -> None:
    if runner.dry_run:
        return
    node_version = cfg.defaults.tools["node"].version
    npm_version = cfg.defaults.tools["npm"].version
    root = cfg.tools_dir / "node" / node_version / ("npm-" + npm_version) / target
    cache = cfg.tools_dir / "downloads" / "npm"
    try:
        opener = registry.opener()
    except (OSError, ssl.SSLError) as exc:
        raise SetupWorkError(
            "не удалось настроить TLS npm registry; проверьте SETUP_WORK_NPM_CAFILE"
        ) from exc
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".node-", dir=root.parent) as work:
        staging = Path(work) / "runtime"
        for name, version, folder in ((package, node_version, "node"), ("npm", npm_version, "npm")):
            runner.info(f"npm registry: {name}@{version}")
            archive = _fetch_package(registry, opener, name, version, cache)
            safe_extract(archive, staging / folder)
            try:
                manifest = json.loads((staging / folder / "package/package.json").read_text())
            except (OSError, ValueError) as exc:
                raise SetupWorkError(f"не читается package.json в архиве {name}@{version}") from exc
            if (
                not isinstance(manifest, dict)
                or manifest.get("name") != name
                or manifest.get("version") != version
            ):
                raise SetupWorkError(f"неверный package.json внутри архива {name}@{version}")
        node = staging / "node/package/bin/node"
        npm_cli = staging / "npm/package/bin/npm-cli.js"
        npx_cli = staging / "npm/package/bin/npx-cli.js"
        if not all(path.is_file() for path in (node, npm_cli, npx_cli)):
            raise SetupWorkError("в пакетах отсутствуют bin/node, npm-cli.js или npx-cli.js")
        ensure_executable(node)
        _verify(node, npm_cli, node_version, npm_version)
        (staging / "launchers").mkdir()
        for command in ("npm", "npx"):
            launcher = staging / "launchers" / command
            launcher.write_text(
                "#!/bin/sh\n# Создано setup-work: npm использует установленный рядом Node.js.\n"
                f"exec {shlex.quote(str(root / 'node/package/bin/node'))} "
                f'{shlex.quote(str(root / f"npm/package/bin/{command}-cli.js"))} "$@"\n'
            )
            launcher.chmod(0o755)
        if root.exists():
            backup = Path(work) / "previous"
            root.rename(backup)
            try:
                staging.rename(root)
            except OSError:
                backup.rename(root)
                raise
        else:
            staging.rename(root)
    for command, relative in (
        ("node", "node/package/bin/node"),
        ("npm", "launchers/npm"),
        ("npx", "launchers/npx"),
    ):
        make_link(root / relative, cfg.bin_dir / command, cfg.data_dir)
    runner.info(f"Node.js {node_version}, npm {npm_version}: {cfg.bin_dir}")


def versions(cfg: Config) -> tuple[str, str]:
    node = cfg.defaults.tools.get("node")
    npm = cfg.defaults.tools.get("npm")
    if not node or not npm or not all(_VERSION.fullmatch(pin.version) for pin in (node, npm)):
        raise UsageError(
            "задайте точные tools.node.version и tools.npm.version в config/defaults.yaml"
        )
    return node.version, npm.version
