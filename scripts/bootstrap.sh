#!/bin/sh
# Начальный этап setup-work. POSIX sh; нужен заранее установленный python3.
# Настройки: bootstrap.env в SETUP_WORK_USER_CONFIG_DIR, SETUP_WORK_HOME/config
# либо ~/.config/setup-work. Для config/ репозитория явно задайте
# SETUP_WORK_USER_CONFIG_DIR="$PWD/config".
# Этапы: реестры pip/npm/uv → установка uv → uv sync через корпоративный PyPI.
# Из публичного интернета ничего не загружается; sudo и правки профиля не нужны.
# Запуск: sh scripts/bootstrap.sh [--dry-run] [--init] [--registries-only] [--sync-only]

set -eu

UV_VERSION="0.12.13"
# SHA256 официальных архивов uv-<target>.tar.gz версии 0.12.13.
SHA_aarch64_apple_darwin="7e6ddb9316acc00f2296c82ff4d99977870ee34b2f0ddcae9444d714db9364ed"
SHA_x86_64_apple_darwin="5e287ef61cb6a9b61b3a83fef124fd143e400468a7dac794230147a810e17119"
SHA_aarch64_unknown_linux_gnu="2eaa5d94f5db7b3a1a092156b9420459e42ab0217d917fe74a876309cef9b5e9"
SHA_x86_64_unknown_linux_gnu="745765a3b6e360ad76743599ae5c42e9278c7edf8bbff9fc76d05bf2623a04dd"

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
    # Переменные направляют настройки и кэши pip/uv/npm на постоянный том.
    export SETUP_WORK_HOME="$SW_HOME"
    export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-$SW_HOME/registries/pip.conf}"
    export UV_CONFIG_FILE="${UV_CONFIG_FILE:-$SW_HOME/registries/uv.toml}"
    export NPM_CONFIG_USERCONFIG="${NPM_CONFIG_USERCONFIG:-$SW_HOME/registries/npmrc}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$SW_HOME/cache/uv}"
    export npm_config_cache="${npm_config_cache:-$SW_HOME/cache/npm}"
    export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-$SW_HOME/gitconfig}"
fi
CONFIG_DIR="${SETUP_WORK_USER_CONFIG_DIR:-${SW_HOME:+$SW_HOME/config}}"
CONFIG_DIR="${CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/setup-work}"
BIN_DIR="${SETUP_WORK_BIN_DIR:-${SW_HOME:+$SW_HOME/bin}}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
DATA_DIR="${SW_HOME:+$SW_HOME/data}"
DATA_DIR="${DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/setup-work}"


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

UV_VERSION="${SETUP_WORK_UV_VERSION:-$UV_VERSION}"
DRY_RUN="${SETUP_WORK_DRY_RUN:-}"
INIT=
REGISTRIES_ONLY=
SYNC_ONLY=
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --init) INIT=1 ;;
        --registries-only) REGISTRIES_ONLY=1 ;;
        --sync-only) SYNC_ONLY=1 ;;
        *) die "неизвестный аргумент: $arg (допустимы --dry-run, --init, --registries-only, --sync-only)" 2 ;;
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
        { say "# Создано командой \`make init\`. В каждом новом shell или контейнере:  . $SW_HOME/env.sh"
          say "export SETUP_WORK_HOME=\"$SW_HOME\""
          say "export PATH=\"\$SETUP_WORK_HOME/bin:\$PATH\""
          say "export PIP_CONFIG_FILE=\"\$SETUP_WORK_HOME/registries/pip.conf\""
          say "export UV_CONFIG_FILE=\"\$SETUP_WORK_HOME/registries/uv.toml\""
          say "export NPM_CONFIG_USERCONFIG=\"\$SETUP_WORK_HOME/registries/npmrc\""
          say "export UV_CACHE_DIR=\"\$SETUP_WORK_HOME/cache/uv\""
          say "export npm_config_cache=\"\$SETUP_WORK_HOME/cache/npm\""
          say "export UV_PYTHON_DOWNLOADS=never"
          say "export GIT_CONFIG_GLOBAL=\"\$SETUP_WORK_HOME/gitconfig\"   # git config --global и логин Bitbucket"
        } > "$SW_HOME/env.sh"
        say "  создан $SW_HOME/env.sh"
        say "Заполните $CONFIG_DIR/*, затем:  . $SW_HOME/env.sh && make bootstrap"
    else
        say "Заполните $CONFIG_DIR/*, затем: make bootstrap"
    fi
    exit 0
fi

# ---- 1. реестры → pip.conf, uv.toml, .npmrc ---------------------------------
# Учётные данные читаются из файлов, не выводятся и не передаются в аргументах.
#   SETUP_WORK_PYPI_INDEX         https://host/path/simple
#   SETUP_WORK_PYPI_USER          логин в URL индекса, по умолчанию token
#   SETUP_WORK_PYPI_AUTH_FILE     путь к файлу с паролем или токеном индекса
#   SETUP_WORK_PYPI_TRUSTED_HOST  хост без проверки TLS для pip/uv
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

UV_TOML="${UV_CONFIG_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml}"

setup_pypi() {
    [ -n "${SETUP_WORK_PYPI_INDEX:-}" ] || return 0
    case "$SETUP_WORK_PYPI_INDEX" in
        *CHANGE-ME*) die "заполните SETUP_WORK_PYPI_INDEX в $CONFIG_DIR/bootstrap.env: замените CHANGE-ME адресом корпоративного PyPI" 2 ;;
    esac
    conf="${PIP_CONFIG_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/pip/pip.conf}"
    auth="${SETUP_WORK_PYPI_AUTH_FILE:+$(expand_home "$SETUP_WORK_PYPI_AUTH_FILE")}"
    say "настройка PyPI $SETUP_WORK_PYPI_INDEX в $conf и $UV_TOML${auth:+ (учётные данные из $auth)}"
    [ -n "$DRY_RUN" ] && return 0
    if [ -n "$auth" ] && [ ! -r "$auth" ]; then die "не читается файл токена PyPI: $auth; проверьте SETUP_WORK_PYPI_AUTH_FILE в $CONFIG_DIR/bootstrap.env" 6; fi
    py=$(command -v python3 || true)
    [ -n "$py" ] || die "требуется заранее установленный python3; установите Python 3.11 или новее" 3
    # Остальные настройки pip.conf сохраняются; uv.toml ведётся целиком. Токена нет в argv.
    "$py" -I -c '
import configparser, json, os, sys
from urllib.parse import quote, urlsplit, urlunsplit
conf, uv_toml, index, user, auth, trusted = sys.argv[1:7]
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
import io
buf = io.StringIO()
cp.write(buf)
write(conf, buf.getvalue())

MARK = "# managed by setup-work"
if os.path.exists(uv_toml) and not open(uv_toml).read().startswith(MARK):
    sys.exit("ошибка: %s создан не setup-work; сохраните его настройки и переместите файл перед make registries" % uv_toml)
lines = [MARK + " (make registries): редактируйте исходный bootstrap.env", "python-downloads = \"never\""]
if trusted:
    lines.append("allow-insecure-host = [%s]" % json.dumps(trusted))
lines += ["", "[[index]]", "name = \"corp\"", "url = %s" % json.dumps(index), "default = true", ""]
write(uv_toml, "\n".join(lines))
' "$conf" "$UV_TOML" "$SETUP_WORK_PYPI_INDEX" "${SETUP_WORK_PYPI_USER:-token}" "$auth" \
        "${SETUP_WORK_PYPI_TRUSTED_HOST:-}" || die "не удалось записать $conf / $UV_TOML; проверьте права на каталоги" 1
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

# ---- 2. uv ---------------------------------------------------------------------
find_uv() {
    if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
    if [ -x "$BIN_DIR/uv" ]; then say "$BIN_DIR/uv"; return 0; fi
    return 1
}

uv_target() {
    os=$(uname -s); arch=$(uname -m)
    case "$arch" in
        x86_64|amd64) arch=x86_64 ;;
        arm64|aarch64) arch=aarch64 ;;
        *) die "неподдерживаемая архитектура: $arch" 4 ;;
    esac
    case "$os" in
        Darwin) say "${arch}-apple-darwin" ;;
        Linux)
            if ldd --version 2>&1 | grep -qi musl; then
                die "для musl libc нет закреплённой сборки uv; установите подходящий uv вручную" 4
            fi
            say "${arch}-unknown-linux-gnu" ;;
        *) die "неподдерживаемая ОС: $os" 4 ;;
    esac
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else die "для проверки SHA256 установите sha256sum либо shasum" 3
    fi
}

fetch() {  # URL и путь назначения
    if command -v curl >/dev/null 2>&1; then
        run curl --proto '=https' --tlsv1.2 -fL --retry 3 --connect-timeout 20 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        run wget --https-only -q -O "$2" "$1"
    else
        die "для HTTPS установите curl либо wget; другой вариант — путь, hdfs:// или SETUP_WORK_UV_FROM_PIP=1" 3
    fi
}

link_uv() {  # каталог с uv и uvx
    run mkdir -p "$BIN_DIR"
    for b in uv uvx; do
        [ -e "$1/$b" ] || [ -n "$DRY_RUN" ] || continue
        link="$BIN_DIR/$b"
        if [ -e "$link" ] && [ ! -L "$link" ]; then
            die "$link существует и не является символьной ссылкой; перезапись отменена"
        fi
        run ln -sfn "$1/$b" "$link"
    done
}

install_uv_archive() {
    target=$(uv_target)
    var="SHA_$(say "$target" | tr '-' '_')"
    eval "expected=\${$var:-}"
    expected="${SETUP_WORK_UV_SHA256:-$expected}"
    [ -n "$expected" ] || die "нет закреплённой контрольной суммы для $target" 4
    dest_dir="$DATA_DIR/tools/uv/$UV_VERSION"
    archive="$DATA_DIR/tools/downloads/uv-${UV_VERSION}-${target}.tar.gz"
    run mkdir -p "$DATA_DIR/tools/downloads" "$dest_dir"
    if [ -f "$archive" ] && [ "$(sha256_of "$archive")" = "$expected" ]; then
        say "  используется кэш $archive"
    else
        src="${SETUP_WORK_UV_ARCHIVE:-}"
        case "$src" in
            https://*) fetch "$src" "$archive.part" ;;
            hdfs:*|viewfs:*)
                command -v "${SETUP_WORK_HDFS_CLI:-hdfs}" >/dev/null 2>&1 || die "клиент hdfs не найден; установите его или задайте SETUP_WORK_HDFS_CLI в bootstrap.env" 3
                if command -v klist >/dev/null 2>&1 && ! klist -s; then
                    die "нет действующего билета Kerberos; выполните kinit и повторите команду" 1
                fi
                rm -f "$archive.part"
                run "${SETUP_WORK_HDFS_CLI:-hdfs}" dfs -get "$src" "$archive.part" ;;
            /*) run cp "$src" "$archive.part" ;;
            *) die "исправьте SETUP_WORK_UV_ARCHIVE в bootstrap.env: нужен абсолютный путь, hdfs:// или https://" 2 ;;
        esac
        if [ -z "$DRY_RUN" ]; then
            got=$(sha256_of "$archive.part")
            if [ "$got" != "$expected" ]; then
                rm -f "$archive.part"
                die "не совпадает SHA256 для архива uv (ожидается $expected, получено $got); для собственной сборки задайте SETUP_WORK_UV_SHA256 в bootstrap.env"
            fi
            mv "$archive.part" "$archive"
        fi
    fi
    run tar -xzf "$archive" -C "$dest_dir"
    link_uv "$dest_dir/uv-$target"
}

install_uv_pip() {
    py=$(command -v python3 || true)
    [ -n "$py" ] || die "для SETUP_WORK_UV_FROM_PIP=1 требуется python3 с модулем pip" 3
    target="$DATA_DIR/tools/uv-pip/$UV_VERSION"
    say "установка uv==$UV_VERSION из корпоративного PyPI в $target"
    run "$py" -m pip install --disable-pip-version-check --no-deps --target "$target" "uv==$UV_VERSION"
    link_uv "$target/bin"
}

if UV=$(find_uv); then
    say "uv: $UV ($("$UV" --version 2>/dev/null || say неизвестно))"
elif [ -n "$SYNC_ONLY" ]; then
    die "uv не найден; выполните make bootstrap или sh scripts/bootstrap.sh" 3
else
    say "uv не найден: установка uv $UV_VERSION в $BIN_DIR"
    use_pip=
    case "${SETUP_WORK_UV_FROM_PIP:-}" in 1|true|yes) use_pip=1 ;; esac
    if [ -n "${SETUP_WORK_UV_ARCHIVE:-}" ]; then
        install_uv_archive
    elif [ -n "$use_pip" ]; then
        install_uv_pip
    else
        die "не задан источник uv: заполните SETUP_WORK_UV_ARCHIVE либо SETUP_WORK_UV_FROM_PIP=1 в $CONFIG_DIR/bootstrap.env" 3
    fi
    UV="$BIN_DIR/uv"
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) say ""
           say "внимание: $BIN_DIR отсутствует в PATH; setup-work не изменяет профиль shell"
           if [ -n "$SW_HOME" ]; then say "      выполните: . $SW_HOME/env.sh"
           else say "      добавьте в окружение shell:  export PATH=\"$BIN_DIR:\$PATH\""; fi ;;
    esac
fi

# ---- 3. окружение проекта -----------------------------------------------------
# uv sync перепривязывает зависимости uv.lock к корпоративному индексу из uv.toml.
# Без настроенного индекса останавливаемся, чтобы не обращаться к pypi.org.
UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"   # зеркало Python не предусмотрено
export UV_PYTHON_DOWNLOADS
if [ -z "${UV_DEFAULT_INDEX:-}${UV_INDEX_URL:-}" ] && ! grep -qs '^\[\[index\]\]' "$UV_TOML"; then
    die "не задан корпоративный PyPI: заполните SETUP_WORK_PYPI_INDEX в $CONFIG_DIR/bootstrap.env и выполните make registries" 2
fi
say "создание окружения проекта: uv sync обновит привязку uv.lock к корпоративному индексу"
if [ -n "$DRY_RUN" ]; then
    say "  \$ $UV sync   (в $ROOT)"
else
    (cd "$ROOT" && "$UV" sync --quiet) \
        || die "uv sync завершился с ошибкой; проверьте установленный Python >= 3.11 и адрес индекса в $UV_TOML; Python автоматически не скачивается" 1
    mkdir -p "$ROOT/.venv" && touch "$ROOT/.venv/.setup-work-synced"
fi
say "начальная установка завершена"
