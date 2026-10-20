#!/bin/sh
# Начальный этап setup-work. POSIX sh; нужен заранее установленный python3.
# Настройки: bootstrap.env в SETUP_WORK_USER_CONFIG_DIR, SETUP_WORK_HOME/config
# либо ~/.config/setup-work. Для config/ репозитория явно задайте
# SETUP_WORK_USER_CONFIG_DIR="$PWD/config".
# Этапы: реестры pip/npm → venv → pip install через корпоративный PyPI.
# Из публичного интернета ничего не загружается; sudo и правки профиля не нужны.
# Запуск: sh scripts/bootstrap.sh [--dry-run] [--init] [--registries-only] [--sync-only]

set -eu

say() { printf '%s\n' "$*"; }
die() { say "ошибка: $1" >&2; exit "${2:-1}"; }
expand_home() {
    case "$1" in
        "~/"*) say "$HOME/${1#"~/"}" ;;
        '$HOME/'*) say "$HOME/${1#'$HOME/'}" ;;
        *) say "$1" ;;
    esac
}

# ---- каталоги настроек и данных ---------------------------------------------------
SW_HOME="${SETUP_WORK_HOME:+$(expand_home "$SETUP_WORK_HOME")}"
if [ -n "$SW_HOME" ]; then
    # Переменные направляют настройки и кэши pip/npm на постоянный том.
    export SETUP_WORK_HOME="$SW_HOME"
    export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-$SW_HOME/registries/pip.conf}"
    export UV_CONFIG_FILE="${UV_CONFIG_FILE:-$SW_HOME/registries/uv.toml}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$SW_HOME/cache/uv}"
    export NPM_CONFIG_USERCONFIG="${NPM_CONFIG_USERCONFIG:-$SW_HOME/registries/npmrc}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SW_HOME/cache/pip}"
    export npm_config_cache="${npm_config_cache:-$SW_HOME/cache/npm}"
    export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-$SW_HOME/gitconfig}"
fi
CONFIG_DIR="${SETUP_WORK_USER_CONFIG_DIR:-${SW_HOME:+$SW_HOME/config}}"
CONFIG_DIR="${CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/setup-work}"

# ---- настройки KEY=VALUE; содержимое не исполняется как shell -----------------
load_env_file() {
    [ -f "$1" ] || return 0
    say "чтение настроек из $1"
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        key=${line%%=*}
        value=${line#*=}
        case "$key" in
            *[!A-Za-z0-9_]*|[0-9]*|'') say "  пропущена неверная строка; используйте KEY=VALUE: $line"; continue ;;
        esac
        case "$key" in
            *TOKEN*|*SECRET*|*PASSWORD*|*API_KEY*) say "  секретный ключ нельзя хранить в bootstrap.env; используйте файл токена: $key"; continue ;;
        esac
        # Убрать обрамляющие кавычки либо комментарий после некавыченного значения.
        case "$value" in
            \"*\") value=${value#\"}; value=${value%\"} ;;
            \'*\') value=${value#\'}; value=${value%\'} ;;
            *)
                value=${value%%[[:space:]]#*}
                while :; do case "$value" in *[[:space:]]) value=${value%?} ;; *) break ;; esac; done ;;
        esac
        eval "already=\${$key+set}"
        if [ "$already" = set ]; then continue; fi  # окружение пользователя имеет приоритет
        export "$key=$value"
        say "  $key задана"
    done < "$1"
}
load_env_file "$CONFIG_DIR/bootstrap.env"

DRY_RUN="${SETUP_WORK_DRY_RUN:-}"
INIT=
REGISTRIES_ONLY=
SYNC_ONLY=
SETUP=
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --init) INIT=1 ;;
        --registries-only) REGISTRIES_ONLY=1 ;;
        --sync-only) SYNC_ONLY=1 ;;
        --setup) SETUP=1 ;;
        *) die "неизвестный аргумент: $arg (допустимы --dry-run, --init, --registries-only, --sync-only, --setup)" 2 ;;
    esac
done
case "$DRY_RUN" in 1|true|yes) DRY_RUN=1 ;; *) DRY_RUN= ;; esac

run() {
    say "  \$ $*"
    [ -n "$DRY_RUN" ] && return 0
    "$@"
}

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

# ---- init: шаблоны настроек и env.sh при заданном SETUP_WORK_HOME -----------------------
if [ -n "$INIT" ]; then
    mkdir -p "$CONFIG_DIR"
    for f in bootstrap.env corporate.yaml harnesses.yaml; do
        if [ -e "$CONFIG_DIR/$f" ]; then say "  сохранён $CONFIG_DIR/$f (уже существует)"
        else cp "$ROOT/examples/corp/$f" "$CONFIG_DIR/$f" && say "  создан $CONFIG_DIR/$f"; fi
    done
    if [ -n "$SW_HOME" ]; then
        py=$(command -v "${PYTHON:-python3}" || true)
        [ -n "$py" ] || die "для создания env.sh нужен python3" 3
        # shellcheck disable=SC2016 # переменные PATH раскрываются при подключении env.sh
        "$py" -I -c '
import shlex, sys
home, config = sys.argv[1:]
print("# Создано make init. Подключайте этот файл в каждом новом терминале.")
print("export SETUP_WORK_HOME=" + shlex.quote(home))
print("export SETUP_WORK_USER_CONFIG_DIR=" + shlex.quote(config))
paths = {
    "PIP_CONFIG_FILE": "registries/pip.conf", "UV_CONFIG_FILE": "registries/uv.toml",
    "NPM_CONFIG_USERCONFIG": "registries/npmrc", "PIP_CACHE_DIR": "cache/pip",
    "UV_CACHE_DIR": "cache/uv", "npm_config_cache": "cache/npm",
    "GIT_CONFIG_GLOBAL": "gitconfig",
}
for key, value in paths.items():
    print("export " + key + "=" + shlex.quote(home + "/" + value))
print("export UV_PYTHON_DOWNLOADS=never")
print("""case ":$PATH:" in
    *":$SETUP_WORK_HOME/bin:"*) ;;
    *) export PATH="$SETUP_WORK_HOME/bin:$PATH" ;;
esac
for sw_agent_bin in "$SETUP_WORK_HOME"/data/harnesses/*/bin; do
    [ -d "$sw_agent_bin" ] || continue
    case ":$PATH:" in
        *":$sw_agent_bin:"*) ;;
        *) export PATH="$sw_agent_bin:$PATH" ;;
    esac
done
unset sw_agent_bin""")
' "$SW_HOME" "$CONFIG_DIR" > "$SW_HOME/env.sh"
        say "  создан $SW_HOME/env.sh"
        say "Заполните $CONFIG_DIR/*, затем:  . $SW_HOME/env.sh && make setup"
    else
        say "Заполните $CONFIG_DIR/*, затем: make setup"
    fi
    exit 0
fi

# ---- 1. реестры → pip.conf, .npmrc ---------------------------------
# Учётные данные читаются из файлов, не выводятся и не передаются в аргументах.
#   SETUP_WORK_PYPI_INDEX         https://host/path/simple
#   SETUP_WORK_PYPI_USER          логин в URL индекса, по умолчанию token
#   SETUP_WORK_PYPI_AUTH_FILE     путь к файлу с паролем или токеном индекса
#   SETUP_WORK_PYPI_TRUSTED_HOST  хост без проверки TLS для pip
#   SETUP_WORK_NPM_REGISTRY       https://host/path/
#   SETUP_WORK_NPM_AUTH_FILE      путь к файлу с npm _authToken
#   SETUP_WORK_NPM_CAFILE         путь к доверенному CA; предпочтительнее отключения TLS
#   SETUP_WORK_NPM_STRICT_SSL     false отключает проверку TLS для npm
#   SETUP_WORK_NPM_OPTIONS        дополнительные ключи .npmrc, например "audit=false fetch-retries=5"
read_secret() {
    f=$(expand_home "$1")
    [ -r "$f" ] || die "не читается файл $2: $f; проверьте AUTH_FILE в $CONFIG_DIR/bootstrap.env и права доступа" 6
    tr -d ' \t\r\n' < "$f"
}

setup_pypi() {
    [ -n "${SETUP_WORK_PYPI_INDEX:-}" ] || return 0
    case "$SETUP_WORK_PYPI_INDEX" in
        *CHANGE-ME*) die "заполните SETUP_WORK_PYPI_INDEX в $CONFIG_DIR/bootstrap.env: замените CHANGE-ME адресом корпоративного PyPI" 2 ;;
    esac
    conf="${PIP_CONFIG_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/pip/pip.conf}"
    auth="${SETUP_WORK_PYPI_AUTH_FILE:+$(expand_home "$SETUP_WORK_PYPI_AUTH_FILE")}"
    say "настройка PyPI $SETUP_WORK_PYPI_INDEX в $conf${auth:+ (учётные данные из $auth)}"
    [ -n "$DRY_RUN" ] && return 0
    if [ -n "$auth" ] && [ ! -r "$auth" ]; then die "не читается файл токена PyPI: $auth; проверьте SETUP_WORK_PYPI_AUTH_FILE в $CONFIG_DIR/bootstrap.env" 6; fi
    py=$(command -v "${PYTHON:-python3}" || true)
    [ -n "$py" ] || die "требуется заранее установленный python3; установите Python 3.11 или новее" 3
    # Остальные настройки pip.conf сохраняются. Токена нет в argv.
    "$py" -I -c '
import configparser, json, os, sys
from urllib.parse import quote, urlsplit, urlunsplit
conf, index, user, auth, trusted = sys.argv[1:6]
if auth:
    with open(auth) as fh:
        token = "".join(fh.read().split())
    u = urlsplit(index)
    host = u.hostname + (":%d" % u.port if u.port else "")
    netloc = quote(user, safe="") + ":" + quote(token, safe="") + "@" + host
    index = urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))

def write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".setup-work.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)

cp = configparser.RawConfigParser()
cp.read(conf)
if not cp.has_section("global"):
    cp.add_section("global")
cp.set("global", "index-url", index)
if trusted:
    cp.set("global", "trusted-host", trusted)
if os.environ.get("PIP_CERT"):
    cp.set("global", "cert", os.environ["PIP_CERT"])
import io
buf = io.StringIO()
cp.write(buf)
write(conf, buf.getvalue())

uv_conf = os.environ.get("UV_CONFIG_FILE") or os.path.join(
    os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "uv", "uv.toml"
)
mark = "# managed by setup-work"
if os.path.exists(uv_conf):
    with open(uv_conf) as fh:
        if not fh.read().startswith(mark):
            sys.exit("ошибка: %s содержит пользовательские настройки; перенесите их перед make registries" % uv_conf)
lines = [mark, "python-downloads = \"never\""]
if trusted:
    lines.append("allow-insecure-host = [%s]" % json.dumps(trusted))
lines += ["", "[[index]]", "name = \"corp\"", "url = %s" % json.dumps(index), "default = true", ""]
write(uv_conf, "\n".join(lines))

' "$conf" "$SETUP_WORK_PYPI_INDEX" "${SETUP_WORK_PYPI_USER:-token}" "$auth" \
        "${SETUP_WORK_PYPI_TRUSTED_HOST:-}" || die "не удалось записать $conf; проверьте права на каталог" 1
}

setup_npm() {
    [ -n "${SETUP_WORK_NPM_REGISTRY:-}" ] || return 0
    case "$SETUP_WORK_NPM_REGISTRY" in
        *CHANGE-ME*) die "заполните SETUP_WORK_NPM_REGISTRY в $CONFIG_DIR/bootstrap.env: замените CHANGE-ME адресом корпоративного npm" 2 ;;
    esac
    rc="${NPM_CONFIG_USERCONFIG:-$HOME/.npmrc}"
    reg="$SETUP_WORK_NPM_REGISTRY"
    case "$reg" in */) ;; *) reg="$reg/" ;; esac
    key="//${reg#*://}:_authToken"
    auth="${SETUP_WORK_NPM_AUTH_FILE:+$(expand_home "$SETUP_WORK_NPM_AUTH_FILE")}"
    cafile="${SETUP_WORK_NPM_CAFILE:+$(expand_home "$SETUP_WORK_NPM_CAFILE")}"
    strict="${SETUP_WORK_NPM_STRICT_SSL:-}"
    opts="${SETUP_WORK_NPM_OPTIONS:-}"
    managed="registry= $key= cafile= strict-ssl="
    for o in $opts; do
        case "$o" in
            [A-Za-z]*=?*) case "${o%%=*}" in *[!A-Za-z0-9_-]*) die "неверный элемент SETUP_WORK_NPM_OPTIONS в $CONFIG_DIR/bootstrap.env: $o" 2 ;; esac ;;
            *) die "неверный элемент SETUP_WORK_NPM_OPTIONS в $CONFIG_DIR/bootstrap.env: $o (требуется ключ=значение)" 2 ;;
        esac
        managed="$managed ${o%%=*}="
    done
    say "настройка реестра npm $reg в $rc${auth:+ (токен из $auth)}"
    [ -n "$DRY_RUN" ] && return 0
    tok=
    [ -n "$auth" ] && tok=$(read_secret "$auth" "npm auth")
    mkdir -p "$(dirname -- "$rc")"
    tmp="$rc.setup-work.tmp"
    # Заменить только управляемые ключи .npmrc, остальные сохранить.
    ( umask 077
      {
        if [ -f "$rc" ]; then
            awk -v m="$managed" 'BEGIN { n = split(m, ks, " ") }
                { for (i = 1; i <= n; i++) if (index($0, ks[i]) == 1) next; print }' "$rc"
        fi
        say "registry=$reg"
        if [ -n "$tok" ]; then say "$key=$tok"; fi
        if [ -n "$cafile" ]; then say "cafile=$cafile"; fi
        if [ -n "$strict" ]; then say "strict-ssl=$strict"; fi
        for o in $opts; do say "$o"; done
      } > "$tmp" ) && mv "$tmp" "$rc" || die "не удалось записать $rc; проверьте права на каталог" 1
}

if [ -z "$SYNC_ONLY" ]; then
    setup_pypi
    setup_npm
fi
if [ -n "$REGISTRIES_ONLY" ]; then
    say "корпоративные реестры настроены"
    exit 0
fi

# ---- 2. окружение проекта -----------------------------------------------------
py="${PYTHON:-python3}"
command -v "$py" >/dev/null 2>&1 || die "требуется Python 3.11 или новее с модулями venv и pip" 3

say "создание окружения проекта: venv + pip из настроенного PyPI"
if [ -z "$DRY_RUN" ]; then
    "$py" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
        || die "требуется Python 3.11 или новее; задайте PYTHON=путь/к/python3" 3
fi
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    run "$py" -m venv "$ROOT/.venv" \
        || die "не удалось создать .venv; установите модуль venv для выбранного Python" 3
fi
venv_py="$ROOT/.venv/bin/python"
# Выбранный python3 может уже указывать на активную .venv без pip.
# Проверять индекс до восстановления pip нельзя: это даёт ложную ошибку PyPI.
if [ ! -x "$venv_py" ] || ! "$venv_py" -I -c 'import pip' >/dev/null 2>&1; then
    run "$venv_py" -m ensurepip --upgrade \
        || die "не удалось восстановить pip в .venv: проверьте ошибку ensurepip выше; новый Python не устанавливается" 3
fi
if [ -z "$DRY_RUN" ]; then
    "$venv_py" -m pip --version \
        || die "pip в .venv не запускается: проверьте PIP_CONFIG_FILE и ошибку выше" 3
fi

# Без явно настроенного индекса не обращаться к публичному PyPI.
# Настройки проверяет тот же pip, который будет устанавливать зависимости.
if [ -n "$DRY_RUN" ] && [ -z "$SYNC_ONLY" ] && [ -n "${SETUP_WORK_PYPI_INDEX:-}" ]; then
    : # setup_pypi выше показал план записи индекса.
elif [ -z "${PIP_INDEX_URL:-}" ]; then
    if [ -n "$DRY_RUN" ] && { [ ! -x "$venv_py" ] || ! "$venv_py" -I -c 'import pip' >/dev/null 2>&1; }; then
        say "  (dry-run) после восстановления pip проверить корпоративный индекс PyPI"
    else
        # config get выбирает один слой (user/site) и пропускает PIP_CONFIG_FILE.
        # config list объединяет слои; значения с токенами остаются только в pipe.
        "$venv_py" -m pip config list | "$venv_py" -I -c '
import ast, sys
values = {}
for line in sys.stdin:
    key, sep, raw = line.partition("=")
    if sep and key in ("global.index-url", "install.index-url"):
        try:
            values[key] = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            sys.exit(1)
index = values.get("install.index-url", values.get("global.index-url"))
sys.exit(0 if isinstance(index, str) and index.strip() else 1)
' || die "не задан корпоративный PyPI: заполните SETUP_WORK_PYPI_INDEX в $CONFIG_DIR/bootstrap.env и выполните make registries" 2
    fi
fi
run "$venv_py" -m pip install --disable-pip-version-check -e "$ROOT[dev]" \
    || die "pip install завершился с ошибкой; проверьте Python >= 3.11, доступ к PyPI и настройки pip" 1
if [ -z "$DRY_RUN" ]; then
    touch "$ROOT/.venv/.setup-work-synced"
fi
say "начальная установка завершена"
if [ -n "$SETUP" ]; then
    export SETUP_WORK_DRY_RUN="$DRY_RUN"
    export UV_PYTHON_DOWNLOADS=never
    if [ -n "${PIP_CERT:-}" ]; then
        export SSL_CERT_FILE="${SSL_CERT_FILE:-$PIP_CERT}"
    fi
    if [ -n "$DRY_RUN" ] && [ ! -x "$ROOT/.venv/bin/python" ]; then
        say "  далее: инфраструктура (Node.js/npm, JAR) → агенты/setup.sh + GigaChat → проверка GPU и локальная модель"
    else
        exec "$ROOT/.venv/bin/python" -m setup_work setup
    fi
fi
