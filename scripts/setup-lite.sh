#!/bin/sh
# setup-lite: облегчённая самодостаточная настройка рабочего места.
# Один файл POSIX sh: корпоративные pip/npm → Node.js/npm из npm registry →
# агенты из внутренних Git-репозиториев через их setup.sh →
# при наличии GPU: llama.cpp, venv vLLM и модели из HDFS, NFS или HTTPS.
# Не нужны Python, репозиторий setup-work, sudo и правки профиля shell.
# Нужны: sh, git, bash (для setup.sh агентов), curl, tar, awk, od;
# для локальной модели — unzip и клиент hdfs (если источники в HDFS).
# Автоподбор модели под GPU, запуск сервера, JAR и метрики не входят — для них есть make setup.
#
# Заполните блок НАСТРОЙКИ (CHANGE-ME — заготовки), затем:
#   sh setup-lite.sh --dry-run
#   sh setup-lite.sh
#   . "$SETUP_WORK_HOME/env.sh"
# Параметры: --dry-run --yes --agents=a,b --no-agents --no-node --no-updates
#            --local --no-local --models=a,b --help

set -eu

# ==== НАСТРОЙКИ ===============================================================
# Постоянный каталог: реестры, Node.js, агенты, env.sh. Переменная окружения важнее.
SW_HOME="${SETUP_WORK_HOME:-$HOME/nfs/setup-work}"

# Корпоративный PyPI для pip. Пустой PYPI_INDEX — шаг пропускается.
# В AUTH_FILE — путь к файлу с токеном, не сам токен; пусто — без авторизации.
PYPI_INDEX='https://CHANGE-ME-repo-host/repository/pypi-group/simple'
PYPI_USER='token'
PYPI_AUTH_FILE='~/nfs/tokens/pypi-token.txt'
PYPI_TRUSTED_HOST=''
PIP_CERT_FILE=''

# Корпоративный npm. Пустой NPM_REGISTRY — пропускаются npm и Node.js.
NPM_REGISTRY='https://CHANGE-ME-repo-host/repository/npm-group/'
NPM_AUTH_FILE='~/nfs/tokens/npm-token.txt'
NPM_CAFILE=''
NPM_STRICT_SSL='true'
NPM_OPTIONS='audit=false fetch-retries=5 save-exact=true'

# Закреплённые версии Node.js и npm из npm registry.
NODE_VERSION='24.20.0'
NPM_VERSION='12.2.0'

# Агенты по одному на строку: имя  репозиторий  ref  [скрипт, по умолчанию setup.sh].
# ref — ветка, тег или коммит. Порядок строк — порядок установки.
AGENTS='
qwen-code  https://CHANGE-ME-bitbucket-host/scm/CHANGE-ME/qwen-code.git     CHANGE-ME
opencode   https://CHANGE-ME-bitbucket-host/scm/CHANGE-ME/opencode.git      CHANGE-ME
pi         https://CHANGE-ME-bitbucket-host/scm/CHANGE-ME/pi-mono.git       CHANGE-ME
hermes     https://CHANGE-ME-bitbucket-host/scm/CHANGE-ME/hermes-agent.git  CHANGE-ME
deepseek   https://CHANGE-ME-bitbucket-host/scm/CHANGE-ME/deepseek-agent.git CHANGE-ME
skillhub   https://CHANGE-ME-bitbucket-host/scm/CHANGE-ME/skillhub.git      CHANGE-ME
'

# ---- Локальная модель (только при наличии GPU или с --local) ----
# Источник: hdfs:///…, локальный/NFS-путь или https://…. Пустой URI — шаг пропускается.
# Движки и модели по умолчанию — во временном каталоге контейнера (RUNTIME_DIR).
HDFS_CLI='hdfs'          # или полный путь: /opt/hadoop/bin/hdfs
HDFS_KERBEROS='auto'     # auto | required | off; нужен действующий билет (kinit)
ENGINES_DIR=''           # пусто — $RUNTIME_DIR/engines
MODELS_DIR=''            # пусто — $RUNTIME_DIR/models

# llama.cpp: ZIP готовой сборки (llama-server и .so рядом) или уже распакованный каталог на NFS.
LLAMA_CPP_URI='hdfs:///CHANGE-ME/engines/llama.cpp.zip'
LLAMA_CPP_PATH=''        # путь к llama-server внутри; пусто — найти автоматически

# vLLM: ZIP с venv (`zip -ry`), собранным на той же версии Python, что стоит на машине.
VLLM_URI='hdfs:///CHANGE-ME/engines/vllm-venv.zip'
VLLM_PATH=''             # каталог venv внутри (с pyvenv.cfg); пусто — найти автоматически

# Модели по одному на строку: имя  источник  [формат].
# Форматы: file — один файл (GGUF); zip — один архив; zip-parts — каталог с <имя>-1-of-N.zip;
# dir — готовый каталог. Без формата: *.gguf → file, *.zip → zip, каталог HDFS с / в конце →
# zip-parts. Локальные файл и каталог используются на месте, без копирования.
MODELS='
qwen3.8-27b-awq  hdfs:///CHANGE-ME/models/hf/qwen3.8-27b-awq/           zip-parts
qwen3.6-27b-q4   hdfs:///CHANGE-ME/models/gguf/Qwen3.6-27B-Q4_K_M.gguf  file
'

# Прокси, если нужен:
# export HTTPS_PROXY=http://CHANGE-ME-proxy-host:3128
# export HTTP_PROXY=http://CHANGE-ME-proxy-host:3128
# export NO_PROXY=localhost,127.0.0.1,CHANGE-ME-repo-host,CHANGE-ME-bitbucket-host
# ==== КОНЕЦ НАСТРОЕК ==========================================================

say() { printf '%s\n' "$*"; }
die() { say "ошибка: $1" >&2; exit "${2:-1}"; }
need() { command -v "$1" >/dev/null 2>&1 || die "не найдена команда $1; установите её заранее" 3; }
expand_home() {
    case "$1" in
        "~/"*) say "$HOME/${1#"~/"}" ;;
        '$HOME/'*) say "$HOME/${1#'$HOME/'}" ;;
        *) say "$1" ;;
    esac
}
usage() {
    say "Использование: sh setup-lite.sh [параметры]"
    say "  --dry-run     показать действия без загрузок и изменений"
    say "  --yes         ответить «да» на все вопросы"
    say "  --no-registries  не записывать pip.conf и npmrc"
    say "  --no-node     пропустить Node.js/npm"
    say "  --agents=a,b  установить эти агенты в указанном порядке, без вопросов о них"
    say "  --no-agents   пропустить агентов"
    say "  --no-updates  не проверять новые ревизии установленных агентов"
    say "  --local       установить движки и модели даже без GPU, без вопросов о них"
    say "  --no-local    пропустить локальную модель"
    say "  --models=a,b  загрузить только эти модели из MODELS, без вопросов о них"
    say "На любой вопрос можно ответить «нет»: пропускается только этот шаг."
    say "Без терминала ответ по умолчанию «нет» — задайте ответы флагами или --yes."
    say "Настройки — в блоке НАСТРОЙКИ в начале файла."
}

DRY_RUN=
YES=
WITH_REGISTRIES=
WITH_AGENTS=
WITH_NODE=
CHECK_UPDATES=1
SELECTED=
WITH_LOCAL=
SELECTED_MODELS=
LOCAL_FLAG=
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --yes|-y) YES=1 ;;
        --agents=*) SELECTED=$(say "${arg#--agents=}" | tr ',' ' '); WITH_AGENTS=1 ;;
        --no-agents) WITH_AGENTS=0 ;;
        --no-registries) WITH_REGISTRIES=0 ;;
        --no-node) WITH_NODE=0 ;;
        --no-updates) CHECK_UPDATES= ;;
        --local) WITH_LOCAL=1; LOCAL_FLAG=1 ;;
        --no-local) WITH_LOCAL=0 ;;
        --models=*) SELECTED_MODELS=$(say "${arg#--models=}" | tr ',' ' '); WITH_LOCAL=${WITH_LOCAL:-1} ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "неизвестный аргумент: $arg" 2 ;;
    esac
done

INTERACTIVE=
if [ -t 0 ] && { : </dev/tty; } 2>/dev/null; then INTERACTIVE=1; fi

confirm() {
    if [ -n "$YES" ]; then return 0; fi
    if [ -z "$INTERACTIVE" ]; then
        say "  вопрос без терминала: \"$1\" — ответ «нет» (подтвердите заранее через --yes)"
        return 1
    fi
    printf '%s [y/N] ' "$1" >/dev/tty
    read -r answer </dev/tty || answer=
    case "$answer" in [yYдД]*) return 0 ;; *) return 1 ;; esac
}

# Вопрос перед шагом: «нет» пропускает только этот шаг. $2 непусто — ответ уже дан флагом.
# В dry-run вопрос показывается, а план продолжается так, будто ответ «да».
ask() {
    if [ -n "${2:-}" ] || [ -n "$YES" ]; then return 0; fi
    if [ -n "$DRY_RUN" ]; then say "  (dry-run) вопрос: $1"; return 0; fi
    confirm "$1"
}

run() {
    say "  \$ $*"
    [ -n "$DRY_RUN" ] && return 0
    "$@"
}

LITE_TMP=
cleanup() { if [ -n "$LITE_TMP" ]; then rm -rf "$LITE_TMP"; fi; }
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---- каталоги и окружение для дочерних процессов --------------------------------
SW_HOME=$(expand_home "$SW_HOME")
case "$SW_HOME" in /*) ;; *) die "SETUP_WORK_HOME должен быть абсолютным путём: $SW_HOME" 2 ;; esac
export SETUP_WORK_HOME="$SW_HOME"
BIN_DIR="$SW_HOME/bin"
DATA_DIR="$SW_HOME/data"
REG_DIR="$SW_HOME/registries"
RUNTIME_DIR="${SETUP_WORK_RUNTIME_DIR:-${TMPDIR:-/tmp}/setup-work-$(id -u)}"
ENGINES_DIR=$(expand_home "${ENGINES_DIR:-$RUNTIME_DIR/engines}")
MODELS_DIR=$(expand_home "${MODELS_DIR:-$RUNTIME_DIR/models}")
DOWNLOADS_DIR="$RUNTIME_DIR/artifacts/downloads"
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-$REG_DIR/pip.conf}"
export NPM_CONFIG_USERCONFIG="${NPM_CONFIG_USERCONFIG:-$REG_DIR/npmrc}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SW_HOME/cache/pip}"
export npm_config_cache="${npm_config_cache:-$SW_HOME/cache/npm}"
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-$SW_HOME/gitconfig}"
export UV_PYTHON_DOWNLOADS=never
if [ -n "$PIP_CERT_FILE" ]; then export SSL_CERT_FILE="${SSL_CERT_FILE:-$(expand_home "$PIP_CERT_FILE")}"; fi
export PATH="$BIN_DIR:$PATH"

read_secret() {
    f=$(expand_home "$1")
    [ -r "$f" ] || die "не читается файл $2: $f; проверьте путь в блоке НАСТРОЙКИ и права доступа" 6
    tr -d ' \t\r\n' < "$f"
}

# Процентное кодирование для логина и токена в URL индекса.
urlencode() {
    printf '%s' "$1" | LC_ALL=C od -An -v -tx1 | LC_ALL=C awk '
        BEGIN { for (i = 0; i < 256; i++) code[sprintf("%02x", i)] = i }
        { for (i = 1; i <= NF; i++) {
            c = code[$i]
            if ((c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122) ||
                c == 45 || c == 46 || c == 95 || c == 126) printf "%c", c
            else printf "%%%s", toupper($i)
        } }'
}

# Файл с правами 600; содержимое — stdin.
write_private() {
    mkdir -p "$(dirname -- "$1")"
    if ! ( umask 077 && cat > "$1.setup-lite.tmp" && mv "$1.setup-lite.tmp" "$1" ); then
        die "не удалось записать $1; проверьте права на каталог" 1
    fi
}

# ---- 1. реестры ---------------------------------------------------------------
setup_pypi() {
    [ -n "$PYPI_INDEX" ] || { say "PyPI: не задан, шаг пропущен"; return 0; }
    case "$PYPI_INDEX" in
        *CHANGE-ME*) die "заполните PYPI_INDEX в блоке НАСТРОЙКИ: замените CHANGE-ME адресом корпоративного PyPI" 2 ;;
        https://*|http://*) ;;
        *) die "PYPI_INDEX должен начинаться с https://: $PYPI_INDEX" 2 ;;
    esac
    auth="${PYPI_AUTH_FILE:+$(expand_home "$PYPI_AUTH_FILE")}"
    if [ "$WITH_REGISTRIES" = 0 ]; then say "PyPI: пропущен (--no-registries)"; return 0; fi
    if ! ask "Записать корпоративный PyPI $PYPI_INDEX в $PIP_CONFIG_FILE?"; then
        say "PyPI: пропущен"
        return 0
    fi
    say "PyPI: $PYPI_INDEX → $PIP_CONFIG_FILE${auth:+ (токен из $auth)}"
    [ -n "$DRY_RUN" ] && return 0
    index=$PYPI_INDEX
    if [ -n "$auth" ]; then
        tok=$(read_secret "$auth" "токена PyPI")
        scheme=${index%%://*}
        rest=${index#*://}
        index="$scheme://$(urlencode "$PYPI_USER"):$(urlencode "$tok")@${rest#*@}"
    fi
    {
        say "# Создано setup-lite.sh; файл перезаписывается при следующем запуске."
        say "[global]"
        say "index-url = $index"
        if [ -n "$PYPI_TRUSTED_HOST" ]; then say "trusted-host = $PYPI_TRUSTED_HOST"; fi
        if [ -n "$PIP_CERT_FILE" ]; then say "cert = $(expand_home "$PIP_CERT_FILE")"; fi
    } | write_private "$PIP_CONFIG_FILE"
}

NPM_REG=
setup_npm() {
    [ -n "$NPM_REGISTRY" ] || { say "npm: не задан, шаг пропущен"; return 0; }
    case "$NPM_REGISTRY" in
        *CHANGE-ME*) die "заполните NPM_REGISTRY в блоке НАСТРОЙКИ: замените CHANGE-ME адресом корпоративного npm" 2 ;;
        https://*) ;;
        *) die "NPM_REGISTRY должен начинаться с https://: $NPM_REGISTRY" 2 ;;
    esac
    NPM_REG=$NPM_REGISTRY
    case "$NPM_REG" in */) ;; *) NPM_REG="$NPM_REG/" ;; esac
    for o in $NPM_OPTIONS; do
        case "$o" in
            [A-Za-z]*=?*) ;;
            *) die "неверный элемент NPM_OPTIONS: $o (требуется ключ=значение)" 2 ;;
        esac
    done
    auth="${NPM_AUTH_FILE:+$(expand_home "$NPM_AUTH_FILE")}"
    # NPM_REG уже задан: Node.js можно установить и без записи npmrc.
    if [ "$WITH_REGISTRIES" = 0 ]; then say "npm: пропущен (--no-registries)"; return 0; fi
    if ! ask "Записать корпоративный npm $NPM_REG в $NPM_CONFIG_USERCONFIG?"; then
        say "npm: пропущен"
        return 0
    fi
    say "npm: $NPM_REG → $NPM_CONFIG_USERCONFIG${auth:+ (токен из $auth)}"
    [ -n "$DRY_RUN" ] && return 0
    tok=
    if [ -n "$auth" ]; then tok=$(read_secret "$auth" "токена npm"); fi
    {
        say "# Создано setup-lite.sh; файл перезаписывается при следующем запуске."
        say "registry=$NPM_REG"
        if [ -n "$tok" ]; then say "//${NPM_REG#*://}:_authToken=$tok"; fi
        if [ -n "$NPM_CAFILE" ]; then say "cafile=$(expand_home "$NPM_CAFILE")"; fi
        if [ -n "$NPM_STRICT_SSL" ]; then say "strict-ssl=$NPM_STRICT_SSL"; fi
        for o in $NPM_OPTIONS; do say "$o"; done
    } | write_private "$NPM_CONFIG_USERCONFIG"
}

# ---- 2. Node.js и npm из npm registry ---------------------------------------------
node_target() {
    case "$(uname -s)" in Darwin) s=darwin ;; Linux) s=linux ;; *) s= ;; esac
    case "$(uname -m)" in x86_64|amd64) a=x64 ;; arm64|aarch64) a=arm64 ;; *) a= ;; esac
    if [ -z "$s" ] || [ -z "$a" ]; then die "нет сборки Node.js для $(uname -s)/$(uname -m)" 4; fi
    if [ "$s" = linux ] && ldd --version 2>&1 | grep -qi musl; then
        die "Node.js из npm registry не поддерживает musl (Alpine); используйте glibc-образ" 4
    fi
    say "$s-$a"
}

# Загрузка tarball из настроенного реестра; токен передаётся curl через stdin, не в argv.
fetch_npm() {
    name=$1 version=$2 out=$3
    if [ -s "$out" ]; then return 0; fi
    url="$NPM_REG$name/-/$name-$version.tgz"
    say "  загрузка $name@$version"
    mkdir -p "$(dirname -- "$out")"
    tok=
    auth="${NPM_AUTH_FILE:+$(expand_home "$NPM_AUTH_FILE")}"
    if [ -n "$auth" ]; then tok=$(read_secret "$auth" "токена npm"); fi
    case "$tok" in *'"'*|*'\'*) die "токен npm содержит недопустимые символы" 2 ;; esac
    set -- -fsSL --retry 3 --proto '=https' -o "$out.part"
    if [ -n "$NPM_CAFILE" ]; then set -- "$@" --cacert "$(expand_home "$NPM_CAFILE")"; fi
    if [ "$NPM_STRICT_SSL" = false ]; then set -- "$@" -k; fi
    if [ -n "$tok" ]; then
        printf 'header = "Authorization: Bearer %s"\n' "$tok" | curl -K - "$@" -- "$url"
    else
        curl "$@" -- "$url" </dev/null
    fi || {
        rm -f "$out.part"
        die "не удалось загрузить $url; проверьте токен, наличие версии в реестре, прокси и NPM_CAFILE" 1
    }
    mv "$out.part" "$out"
}

install_node() {
    if [ "$WITH_NODE" = 0 ]; then say "Node.js: пропущен (--no-node)"; return 0; fi
    if [ -z "$NPM_REG" ]; then say "Node.js: npm registry не задан, шаг пропущен"; return 0; fi
    case "$NODE_VERSION$NPM_VERSION" in *[!0-9.]*|'') die "задайте точные NODE_VERSION и NPM_VERSION" 2 ;; esac
    have_node=$("$BIN_DIR/node" --version 2>/dev/null || true)
    have_npm=$("$BIN_DIR/npm" --version 2>/dev/null || true)
    if [ "$have_node" = "v$NODE_VERSION" ] && [ "$have_npm" = "$NPM_VERSION" ]; then
        say "Node.js $NODE_VERSION и npm $NPM_VERSION уже установлены"
        return 0
    fi
    target=$(node_target)
    pkg="node-$target"
    if [ "$target" = darwin-arm64 ]; then pkg=node-bin-darwin-arm64; fi
    root="$DATA_DIR/tools/node/$NODE_VERSION/npm-$NPM_VERSION/$target"
    if ! ask "Установить Node.js $NODE_VERSION и npm $NPM_VERSION из $NPM_REG?"; then
        say "Node.js: пропущен"
        return 0
    fi
    say "Node.js: $pkg@$NODE_VERSION и npm@$NPM_VERSION из $NPM_REG → $root"
    [ -n "$DRY_RUN" ] && return 0
    need curl
    need tar
    cache="$DATA_DIR/tools/downloads/npm"
    fetch_npm "$pkg" "$NODE_VERSION" "$cache/$pkg-$NODE_VERSION.tgz"
    fetch_npm npm "$NPM_VERSION" "$cache/npm-$NPM_VERSION.tgz"
    mkdir -p "$(dirname -- "$root")"
    LITE_TMP=$(mktemp -d "$(dirname -- "$root")/.node-XXXXXX")
    stage="$LITE_TMP/runtime"
    mkdir -p "$stage/node" "$stage/npm" "$stage/launchers"
    if ! tar -xzf "$cache/$pkg-$NODE_VERSION.tgz" -C "$stage/node" \
        || ! tar -xzf "$cache/npm-$NPM_VERSION.tgz" -C "$stage/npm"; then
        die "не удалось распаковать архивы Node.js/npm; удалите $cache и повторите" 1
    fi
    for f in node/package/bin/node npm/package/bin/npm-cli.js npm/package/bin/npx-cli.js; do
        [ -f "$stage/$f" ] || die "в пакетах отсутствует $f" 1
    done
    chmod 755 "$stage/node/package/bin/node"
    got_node=$("$stage/node/package/bin/node" --version 2>/dev/null || true)
    got_npm=$("$stage/node/package/bin/node" "$stage/npm/package/bin/npm-cli.js" --version 2>/dev/null || true)
    if [ "$got_node" != "v$NODE_VERSION" ] || [ "$got_npm" != "$NPM_VERSION" ]; then
        die "распакованные Node.js/npm не запускаются; проверьте ОС, архитектуру и glibc" 1
    fi
    for cmd in npm npx; do
        {
            say "#!/bin/sh"
            say "# Создано setup-lite.sh: npm использует установленный рядом Node.js."
            say "exec '$root/node/package/bin/node' '$root/npm/package/bin/$cmd-cli.js' \"\$@\""
        } > "$stage/launchers/$cmd"
        chmod 755 "$stage/launchers/$cmd"
    done
    if [ -e "$root" ]; then mv "$root" "$LITE_TMP/previous"; fi
    mv "$stage" "$root"
    mkdir -p "$BIN_DIR"
    ln -sf "$root/node/package/bin/node" "$BIN_DIR/node"
    ln -sf "$root/launchers/npm" "$BIN_DIR/npm"
    ln -sf "$root/launchers/npx" "$BIN_DIR/npx"
    rm -rf "$LITE_TMP"
    LITE_TMP=
    say "  Node.js $NODE_VERSION, npm $NPM_VERSION: $BIN_DIR"
}

# ---- 3. агенты ----------------------------------------------------------------
# Git без собственных запросов пароля; вход выполняет git_login один раз на хост.
gitq() { GIT_TERMINAL_PROMPT=0 GIT_ASKPASS='' SSH_ASKPASS='' git "$@"; }

LOGGED_HOSTS=' '
FAILED_HOSTS=' '
# 1 — вход не удался или отменён (пустой логин, Ctrl+D): агенты этого хоста пропускаются.
git_login() {
    case "$1" in https://*) ;; *) return 0 ;; esac
    [ -n "$INTERACTIVE" ] || return 0
    host=${1#https://}
    host=${host%%/*}
    host=${host##*@}
    case "$LOGGED_HOSTS" in *" $host "*) return 0 ;; esac
    case "$FAILED_HOSTS" in *" $host "*) say "  вход на $host не выполнен ранее — пропущен"; return 1 ;; esac
    # Сокет кеша — на локальном томе, даже если SETUP_WORK_HOME на NFS.
    mkdir -p "$RUNTIME_DIR/git-credentials"
    chmod 700 "$RUNTIME_DIR" "$RUNTIME_DIR/git-credentials"
    helper="cache --timeout=28800 --socket=$RUNTIME_DIR/git-credentials/socket"
    key="credential.https://$host.helper"
    if ! git config --global --get-all "$key" 2>/dev/null | grep -qxF -- "$helper"; then
        git config --global --add "$key" "$helper"
    fi
    say "  Git: вход на $host. Введите логин и пароль/токен, если Git их запросит;"
    say "  пароль не отображается и кешируется в памяти на 8 часов. Отказ — пустой логин."
    if ! GIT_TERMINAL_PROMPT=1 git ls-remote -- "$1" HEAD >/dev/null; then
        say "  нет доступа к $host (вход отменён или неверные данные): агенты с этого хоста пропущены"
        FAILED_HOSTS="$FAILED_HOSTS$host "
        return 1
    fi
    LOGGED_HOSTS="$LOGGED_HOSTS$host "
}

# Точная ревизия ветки/тега; для аннотированного тега — коммит.
remote_rev() {
    if say "$2" | grep -Eqx '[0-9a-f]{7,40}'; then say "$2"; return 0; fi
    refs=$(gitq ls-remote -- "$1" "refs/heads/$2" "refs/tags/$2" "refs/tags/$2^{}") \
        || die "не удалось прочитать ревизии $1" 1
    rev=$(say "$refs" | awk -v r="$2" '
        $2 == "refs/tags/" r "^{}" { peeled = $1 }
        $2 == "refs/heads/" r { head = $1 }
        $2 == "refs/tags/" r { tag = $1 }
        END { print (peeled != "" ? peeled : (head != "" ? head : tag)) }')
    [ -n "$rev" ] || die "ревизия '$2' не найдена в $1; исправьте ref в AGENTS" 6
    say "$rev"
}

setup_agent() {
    name=$1 repo=$2 ref=$3 script=${4:-setup.sh}
    root="$DATA_DIR/harnesses/$name"
    src="$root/src"
    mark="$root/.setup-lite-commit"
    say "агент $name: $repo ($ref)"
    if [ -n "$DRY_RUN" ]; then
        if [ ! -f "$mark" ]; then ask "Установить $name из $repo ($ref)?" "$AGENTS_EXPLICIT"; fi
        say "  проверить ревизию $ref; обновление установленного — после подтверждения"
        say "  \$ git clone/fetch $repo $ref → $src"
        say "  \$ bash $script (в $src, SETUP_WORK_BACKEND=gigachat)"
        return 0
    fi
    current=$(cat "$mark" 2>/dev/null || true)
    if [ -n "$current" ] && [ -z "$CHECK_UPDATES" ]; then
        say "  установлен ${current%"${current#????????????}"}; проверка обновлений выключена"
        return 0
    fi
    if [ -z "$current" ] && ! ask "Установить $name из $repo ($ref)?" "$AGENTS_EXPLICIT"; then
        say "  $name: пропущен"
        return 0
    fi
    git_login "$repo" || return 0
    rev=$(remote_rev "$repo" "$ref")
    head=$(gitq -C "$src" rev-parse HEAD 2>/dev/null || true)
    if [ -n "$current" ]; then
        if [ "$head" = "$current" ]; then
            case "$current" in
                "$rev"*) say "  установленная ревизия актуальна"; return 0 ;;
            esac
        fi
        say "  установлено $current, доступно $rev"
        if ! confirm "Обновить $name и повторить $script?"; then
            say "  оставлена установленная версия"
            return 0
        fi
    fi
    if [ -d "$src/.git" ]; then
        if [ -n "$(gitq -C "$src" status --porcelain --untracked-files=no)" ]; then
            die "в $src есть изменения; сохраните их перед установкой $name" 2
        fi
    else
        mkdir -p "$root"
        gitq clone --no-checkout --filter=blob:none -- "$repo" "$src" \
            || die "не удалось клонировать $repo" 1
    fi
    # Ветку/тег загружать по имени: не все серверы отдают коммит по SHA.
    gitq -C "$src" fetch --tags "$repo" -- "$ref" || die "не удалось загрузить $ref из $repo" 1
    gitq -C "$src" -c advice.detachedHead=false checkout --detach FETCH_HEAD \
        || die "не удалось переключить $src на $ref" 1
    [ -f "$src/$script" ] || die "в $repo нет $script" 6
    mkdir -p "$root/config" "$root/bin"
    rm -f "$mark"
    say "  \$ bash $script"
    # setup.sh получает терминал, чтобы пройти собственную настройку.
    if [ -n "$INTERACTIVE" ]; then exec 3</dev/tty; else exec 3</dev/null; fi
    ( cd "$src" && SETUP_WORK_BACKEND=gigachat \
        SETUP_WORK_HARNESS_HOME="$root" \
        SETUP_WORK_HARNESS_CONFIG="$root/config" \
        SETUP_WORK_HARNESS_BIN="$root/bin" \
        SETUP_WORK_METRICS_ENABLED=0 \
        SETUP_WORK_METRICS_AGENT="$name" \
        PATH="$root/bin:$PATH" \
        bash "./$script" <&3 ) || die "$name: $script завершился с ошибкой; установка не отмечена успешной" 1
    exec 3<&-
    gitq -C "$src" rev-parse HEAD > "$mark"
    say "  $name установлен ($(cat "$mark"))"
}

setup_agents() {
    list=$(say "$AGENTS" | sed -e 's/#.*//' -e '/^[[:space:]]*$/d')
    if [ -z "$list" ]; then say "агенты: список AGENTS пуст, шаг пропущен"; return 0; fi
    if [ "$WITH_AGENTS" = 0 ]; then say "агенты: пропущены"; return 0; fi
    AGENTS_EXPLICIT=$WITH_AGENTS
    if ! ask "Будете работать с агентами?" "$WITH_AGENTS"; then
        say "агенты: пропущены"
        return 0
    fi
    if [ -z "$SELECTED" ]; then SELECTED=$(say "$list" | awk '{ print $1 }'); fi
    # Сначала проверить весь список, чтобы опечатка не оставила половину установки.
    chosen=
    for name in $SELECTED; do
        line=$(say "$list" | awk -v n="$name" '$1 == n { print; exit }')
        [ -n "$line" ] || die "агент $name не найден в AGENTS" 6
        set -f
        # shellcheck disable=SC2086 # строка намеренно разбивается на поля
        set -- $line
        set +f
        case "$name" in *[!A-Za-z0-9._-]*) die "недопустимое имя агента: $name" 2 ;; esac
        if [ $# -lt 3 ] || [ $# -gt 4 ]; then
            die "строка AGENTS для $name: нужно имя, репозиторий, ref и необязательный скрипт" 2
        fi
        case "$2 $3" in *CHANGE-ME*) die "заполните репозиторий и ref для $name в AGENTS" 2 ;; esac
        case "${4:-setup.sh}" in /*|*..*) die "скрипт $name должен быть путём внутри репозитория" 2 ;; esac
        case " $chosen " in *" $name "*) ;; *) chosen="$chosen $name" ;; esac
    done
    if [ -z "$DRY_RUN" ]; then need git; need bash; fi
    for name in $chosen; do
        line=$(say "$list" | awk -v n="$name" '$1 == n { print; exit }')
        set -f
        # shellcheck disable=SC2086 # поля строки; glob внутри функции нужен
        set -- $line
        set +f
        setup_agent "$@"
    done
}

# ---- 4. локальная модель: llama.cpp, vLLM и модели -----------------------------------
NL='
'
TAB=$(printf '\t')
ARTIFACT_MARK=.setup-lite-artifact

uri_scheme() {
    case "$1" in
        hdfs:*|viewfs:*) say hdfs ;;
        https://*) say https ;;
        *) say local ;;
    esac
}
local_path() { expand_home "${1#file://}"; }

HDFS_READY=
hdfs_ready() {
    if [ -n "$HDFS_READY" ]; then return 0; fi
    need "$HDFS_CLI"
    case "$HDFS_KERBEROS" in
        off) ;;
        auto|required)
            if command -v klist >/dev/null 2>&1; then
                klist -s || die "нет действующего билета Kerberos для HDFS; выполните kinit и повторите" 3
            elif [ "$HDFS_KERBEROS" = required ]; then
                die "klist не найден, а HDFS_KERBEROS=required; установите клиент Kerberos" 3
            fi ;;
        *) die "HDFS_KERBEROS: допустимы auto, required или off" 2 ;;
    esac
    HDFS_READY=1
}

# Загрузить удалённый файл или каталог HDFS в $2 (через $2.part).
fetch_remote() {
    rm -rf "$2.part"
    mkdir -p "$(dirname -- "$2")"
    case "$(uri_scheme "$1")" in
        hdfs)
            hdfs_ready
            # В FsShell нет разделителя --: путь не должен выглядеть как параметр.
            case "$1" in -*) die "путь HDFS похож на параметр команды: $1" 2 ;; esac
            "$HDFS_CLI" dfs -get "$1" "$2.part" \
                || die "не удалось загрузить $1 из HDFS; проверьте путь (hdfs dfs -ls), права и билет Kerberos" 1 ;;
        https)
            need curl
            rc=0
            if [ -n "${SSL_CERT_FILE:-}" ]; then
                curl -fSL --retry 3 --proto '=https' --cacert "$SSL_CERT_FILE" -o "$2.part" -- "$1" \
                    </dev/null || rc=$?
            else
                curl -fSL --retry 3 --proto '=https' -o "$2.part" -- "$1" </dev/null || rc=$?
            fi
            if [ "$rc" -ne 0 ]; then
                rm -f "$2.part"
                die "не удалось загрузить $1; проверьте адрес, прокси и SSL_CERT_FILE" 1
            fi ;;
        *) die "внутренняя ошибка: $1 не удалённый источник" 1 ;;
    esac
    rm -rf "$2"
    mv "$2.part" "$2"
}

# Распаковать ZIP; Info-ZIP отбрасывает абсолютные пути и .., код 1 — предупреждения.
unzip_into() {
    rc=0
    unzip -q -o "$1" -d "$2" || rc=$?
    if [ "$rc" -gt 1 ]; then die "не удалось распаковать $1: архив повреждён или не ZIP" 1; fi
}

# Полный комплект <имя>-1-of-N.zip из каталога, в числовом порядке.
list_parts() {
    case "$(uri_scheme "$1")" in
        hdfs)
            hdfs_ready
            names=$("$HDFS_CLI" dfs -ls "$1" | awk '/^-/ { n = $NF; sub(".*/", "", n); print n }') \
                || die "не удалось прочитать каталог частей $1; проверьте путь и права HDFS" 6 ;;
        local)
            d=$(local_path "$1")
            [ -d "$d" ] || die "не найден каталог частей $d" 6
            names=$(for f in "$d"/*; do if [ -f "$f" ]; then say "${f##*/}"; fi; done) ;;
        *) die "для zip-parts укажите каталог HDFS или локальный каталог: $1" 2 ;;
    esac
    out=$(say "$names" | awk -v base="${1%/}" '
        {
            if (err != "" || !match($0, /-[0-9]+-of-[0-9]+\.[zZ][iI][pP]$/)) next
            prefix = substr($0, 1, RSTART - 1)
            split(substr($0, RSTART + 1), a, "-of-|[.]")
            i = a[1] + 0; n = a[2] + 0; key = prefix SUBSEP n
            if (!(key in seen)) { seen[key] = 1; groups++; total = n; k = key }
            if (i < 1 || i > n || (key SUBSEP i) in part) err = "неверный или повторяющийся номер части: " $0
            part[key SUBSEP i] = $0
        }
        END {
            if (err != "") { print "!" err; exit }
            if (groups != 1) { print "!нужен один комплект <имя>-1-of-N.zip, найдено комплектов: " groups + 0; exit }
            for (i = 1; i <= total; i++) if (!((k SUBSEP i) in part)) miss = miss " " i
            if (miss != "") { print "!отсутствуют части:" miss " из " total; exit }
            for (i = 1; i <= total; i++) print base "/" part[k SUBSEP i]
        }')
    case "$out" in "!"*) die "$1: ${out#!}" 2 ;; esac
    say "$out"
}

unpack_parts() {
    p_name=$1 p_uri=$2 p_dir=$3
    p_list=$(list_parts "$p_uri")
    p_files=
    p_downloaded=
    p_index=0
    o_ifs=$IFS
    set -f
    IFS=$NL
    for p in $p_list; do
        IFS=$o_ifs
        p_index=$((p_index + 1))
        if [ "$(uri_scheme "$p")" = local ]; then
            f=$(local_path "$p")
        else
            f="$DOWNLOADS_DIR/$p_name-$p_index.zip"
            say "  загрузка части $p_index: $p"
            fetch_remote "$p" "$f"
            p_downloaded="$p_downloaded$f$NL"
        fi
        p_files="$p_files$f$NL"
        IFS=$NL
    done
    # Части бывают отдельными ZIP или последовательными кусками одного ZIP.
    independent=1
    for f in $p_files; do
        unzip -l "$f" >/dev/null 2>&1 || { independent=; break; }
    done
    if [ -n "$independent" ]; then
        for f in $p_files; do
            say "  распаковка ${f##*/}"
            unzip_into "$f" "$p_dir"
        done
    else
        joined="$DOWNLOADS_DIR/$p_name--joined.zip"
        say "  сборка частей в один ZIP"
        for f in $p_files; do cat "$f"; done > "$joined"
        unzip_into "$joined" "$p_dir"
        rm -f "$joined"
    fi
    for f in $p_downloaded; do rm -f "$f"; done
    IFS=$o_ifs
    set +f
}

artifact_current() { [ -f "$1/$ARTIFACT_MARK" ] && [ "$(cat "$1/$ARTIFACT_MARK")" = "$2 $3" ]; }

# Установить источник в $4/data; маркер с источником позволяет не загружать повторно.
INSTALLED_FRESH=
install_artifact() {
    a_name=$1 a_uri=$2 a_fmt=$3 a_dest=$4
    INSTALLED_FRESH=
    a_stamp="$a_uri $a_fmt"
    if artifact_current "$a_dest" "$a_uri" "$a_fmt"; then
        say "  $a_name: уже установлен в $a_dest"
        return 0
    fi
    say "  $a_name: $a_uri ($a_fmt) → $a_dest"
    [ -n "$DRY_RUN" ] && return 0
    a_parent=$(dirname -- "$a_dest")
    a_stage="$a_parent/.${a_dest##*/}.unpacking"
    mkdir -p "$a_parent" "$DOWNLOADS_DIR"
    rm -rf "$a_stage"
    mkdir -p "$a_stage/data"
    LITE_TMP=$a_stage
    a_scheme=$(uri_scheme "$a_uri")
    a_base=${a_uri%/}
    a_base=${a_base##*/}
    case "$a_fmt" in
        zip)
            need unzip
            if [ "$a_scheme" = local ]; then
                a_file=$(local_path "$a_uri")
                [ -f "$a_file" ] || die "не найден архив $a_file" 6
            else
                a_file="$DOWNLOADS_DIR/$a_name.zip"
                fetch_remote "$a_uri" "$a_file"
            fi
            say "  распаковка $a_base"
            unzip_into "$a_file" "$a_stage/data"
            if [ "$a_scheme" != local ]; then rm -f "$a_file"; fi ;;
        zip-parts)
            need unzip
            unpack_parts "$a_name" "$a_uri" "$a_stage/data" ;;
        file|dir)
            if [ "$a_scheme" = local ]; then
                a_src=$(local_path "$a_uri")
                [ -e "$a_src" ] || die "не найден $a_src" 6
                if [ "$a_fmt" = dir ]; then
                    cp -RP "$a_src/." "$a_stage/data/"
                else
                    cp -P "$a_src" "$a_stage/data/$a_base"
                fi
            elif [ "$a_fmt" = dir ]; then
                fetch_remote "$a_uri" "$a_stage/data"
            else
                fetch_remote "$a_uri" "$a_stage/data/$a_base"
            fi ;;
        *) die "$a_name: неизвестный формат $a_fmt" 2 ;;
    esac
    say "$a_stamp" > "$a_stage/$ARTIFACT_MARK"
    a_old="$a_parent/.${a_dest##*/}.old"
    rm -rf "$a_old"
    if [ -e "$a_dest" ]; then mv "$a_dest" "$a_old"; fi
    mv "$a_stage" "$a_dest"
    LITE_TMP=
    rm -rf "$a_old"
    INSTALLED_FRESH=1
}

# Ближайший к корню путь по имени; при равной глубине — по алфавиту.
find_shallow() {
    find "$1/" "$2" "$3" 2>/dev/null | sed 's#//*#/#g' \
        | awk -F/ -v t="$TAB" '{ print NF t $0 }' | sort -t "$TAB" -k1,1n -k2 | head -n 1 | cut -f2-
}

relink() { rm -f "$2"; ln -s "$1" "$2"; }

LLAMA_READY=
setup_llama() {
    if [ -z "$LLAMA_CPP_URI" ]; then say "llama.cpp: LLAMA_CPP_URI не задан, шаг пропущен"; return 0; fi
    dest="$ENGINES_DIR/llama.cpp"
    if [ "$(uri_scheme "$LLAMA_CPP_URI")" = local ] && [ -d "$(local_path "$LLAMA_CPP_URI")" ]; then
        if [ ! -L "$dest/current" ] && ! ask "Подключить llama.cpp из $LLAMA_CPP_URI?" "$LOCAL_FLAG"; then
            say "  llama.cpp: пропущен"
            return 0
        fi
        root=$(local_path "$LLAMA_CPP_URI")
        say "  llama.cpp: используется на месте $root"
        [ -n "$DRY_RUN" ] && return 0
        mkdir -p "$dest"
    else
        if ! artifact_current "$dest" "$LLAMA_CPP_URI" zip \
            && ! ask "Установить llama.cpp из $LLAMA_CPP_URI?" "$LOCAL_FLAG"; then
            say "  llama.cpp: пропущен"
            return 0
        fi
        install_artifact llama.cpp "$LLAMA_CPP_URI" zip "$dest"
        [ -n "$DRY_RUN" ] && return 0
        root="$dest/data"
    fi
    if [ -n "$LLAMA_CPP_PATH" ]; then
        binary="$root/$LLAMA_CPP_PATH"
    else
        binary=$(find_shallow "$root" -name llama-server)
    fi
    if [ -z "$binary" ] || [ ! -f "$binary" ]; then
        die "llama-server не найден в $root; задайте LLAMA_CPP_PATH — путь к нему внутри архива" 6
    fi
    bdir=$(dirname -- "$binary")
    for f in "$bdir"/llama-*; do
        if [ -f "$f" ] && [ ! -x "$f" ]; then chmod +x "$f" 2>/dev/null || true; fi
    done
    relink "$bdir" "$dest/current"
    libs=
    for f in "$bdir"/*.so*; do if [ -e "$f" ]; then libs=1; break; fi; done
    mkdir -p "$BIN_DIR"
    # Обёртки в bin: библиотеки рядом с llama-server находятся через LD_LIBRARY_PATH.
    for f in "$bdir"/llama-*; do
        if [ ! -f "$f" ] || [ ! -x "$f" ]; then continue; fi
        {
            say "#!/bin/sh"
            say "# Создано setup-lite.sh: llama.cpp из $dest/current."
            if [ -n "$libs" ]; then
                # shellcheck disable=SC2016 # раскрывается при запуске обёртки
                say "export LD_LIBRARY_PATH=$(shq "$dest/current")"'"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
            fi
            say "exec $(shq "$dest/current/${f##*/}") \"\$@\""
        } > "$BIN_DIR/${f##*/}"
        chmod 755 "$BIN_DIR/${f##*/}"
    done
    "$BIN_DIR/llama-server" --version >/dev/null 2>&1 \
        || die "llama-server не запускается; проверьте драйвер, CUDA и glibc: $BIN_DIR/llama-server --version" 1
    say "  llama.cpp: $BIN_DIR/llama-server"
    LLAMA_READY=1
}

# Перенесённое venv: home в pyvenv.cfg, ссылки bin/python* и строки запуска скриптов.
repair_venv() {
    v=$1
    pv="$v/pyvenv.cfg"
    ver=$(sed -n 's/^[[:space:]]*version\(_info\)\{0,1\}[[:space:]]*=[[:space:]]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\2/p' "$pv" | head -n 1)
    [ -n "$ver" ] || die "не удалось определить версию Python по $pv" 1
    if ! "$v/bin/python" -c 'import sys' >/dev/null 2>&1; then
        home=$(sed -n 's/^[[:space:]]*home[[:space:]]*=[[:space:]]*//p' "$pv" | head -n 1)
        interp=
        for c in "$home/python$ver" "$home/python3" "$(command -v "python$ver" || true)" "$(command -v python3 || true)"; do
            if [ -z "$c" ] || [ ! -f "$c" ]; then continue; fi
            got=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
            if [ "$got" = "$ver" ]; then
                interp=$("$c" -c 'import os, sys; print(os.path.realpath(sys.executable))')
                break
            fi
        done
        [ -n "$interp" ] || die "venv vLLM собрано с Python $ver (home=$home), а на машине его нет; пересоберите архив под установленный Python" 4
        awk -v h="$(dirname -- "$interp")" '{ if ($0 ~ /^[ \t]*home[ \t]*=/) print "home = " h; else print }' \
            "$pv" > "$pv.tmp"
        mv "$pv.tmp" "$pv"
        for n in python python3 "python$ver"; do relink "$interp" "$v/bin/$n"; done
        "$v/bin/python" -c 'import sys' >/dev/null 2>&1 \
            || die "$v/bin/python не запускается после перепривязки к $interp" 1
        say "  интерпретатор venv перепривязан к $interp"
    fi
    new="#!$v/bin/python"
    exec_line="'''exec' $(shq "$v/bin/python") \"\$0\" \"\$@\""
    fixed=0
    for f in "$v"/bin/*; do
        if [ ! -f "$f" ] || [ -L "$f" ]; then continue; fi
        if [ $(($(wc -c < "$f"))) -gt 1048576 ]; then continue; fi
        IFS= read -r first < "$f" || continue
        case "$first" in
            "$new") continue ;;
            '#!'*/bin/python*) { say "$new"; tail -n +2 "$f"; } > "$f.tmp" ;;
            '#!/bin/sh')
                # Длинные пути pip записывает двухстрочной формой через /bin/sh.
                second=$(sed -n 2p "$f")
                case "$second" in
                    "$exec_line") continue ;;
                    "'''exec' "*/bin/python*) { say "$first"; say "$exec_line"; tail -n +3 "$f"; } > "$f.tmp" ;;
                    *) continue ;;
                esac ;;
            *) continue ;;
        esac
        cat "$f.tmp" > "$f"
        rm -f "$f.tmp"
        fixed=$((fixed + 1))
    done
    if [ "$fixed" -gt 0 ]; then say "  исправлено строк запуска в $v/bin: $fixed"; fi
}

VLLM_READY=
setup_vllm() {
    if [ -z "$VLLM_URI" ]; then say "vLLM: VLLM_URI не задан, шаг пропущен"; return 0; fi
    dest="$ENGINES_DIR/vllm"
    fmt=zip
    if [ "$(uri_scheme "$VLLM_URI")" = local ] && [ -d "$(local_path "$VLLM_URI")" ]; then fmt=dir; fi
    if ! artifact_current "$dest" "$VLLM_URI" "$fmt" \
        && ! ask "Установить vLLM из $VLLM_URI?" "$LOCAL_FLAG"; then
        say "  vLLM: пропущен"
        return 0
    fi
    # Каталог копируется: исправление путей не должно менять общее venv на NFS.
    install_artifact vllm "$VLLM_URI" "$fmt" "$dest"
    fresh=$INSTALLED_FRESH
    [ -n "$DRY_RUN" ] && return 0
    if [ -n "$VLLM_PATH" ]; then
        venv="$dest/data/$VLLM_PATH"
    else
        marker=$(find_shallow "$dest/data" -name pyvenv.cfg)
        venv=$(dirname -- "${marker:-$dest/data/x}")
    fi
    [ -f "$venv/pyvenv.cfg" ] || die "в $venv нет pyvenv.cfg; задайте VLLM_PATH — каталог venv внутри архива" 6
    repair_venv "$venv"
    relink "$venv" "$dest/current"
    if [ -f "$venv/bin/vllm" ]; then
        mkdir -p "$BIN_DIR"
        {
            say "#!/bin/sh"
            say "# Создано setup-lite.sh: vLLM из $dest/current."
            say "exec $(shq "$dest/current/bin/vllm") \"\$@\""
        } > "$BIN_DIR/vllm"
        chmod 755 "$BIN_DIR/vllm"
    fi
    if [ -n "$fresh" ]; then
        say "  проверка import vllm (может занять несколько минут)"
        "$dest/current/bin/python" -c 'import vllm; print("  vllm", vllm.__version__)' \
            || die "vLLM не импортируется; проверьте драйвер, CUDA и ошибку выше" 1
    fi
    say "  vLLM: $dest/current"
    VLLM_READY=1
}

model_format() {
    if [ -n "$2" ]; then say "$2"; return 0; fi
    if [ "$(uri_scheme "$1")" = local ]; then
        p=$(local_path "$1")
        if [ -d "$p" ]; then
            for f in "$p"/*-1-of-*.zip "$p"/*-1-of-*.ZIP; do
                if [ -f "$f" ]; then say zip-parts; return 0; fi
            done
            say dir
            return 0
        fi
    fi
    case "$1" in
        *.gguf|*.GGUF) say file ;;
        *.zip|*.ZIP) say zip ;;
        */) say zip-parts ;;
        *) say "" ;;
    esac
}

# Точка входа модели: GGUF-файл (кроме mmproj) или каталог с config.json.
model_entry() {
    if [ -f "$1" ]; then say "$1"; return 0; fi
    g=$(find "$1/" -name '*.gguf' ! -name 'mmproj*' 2>/dev/null | sed 's#//*#/#g' \
        | awk -F/ -v t="$TAB" '{ print NF t $0 }' | sort -t "$TAB" -k1,1n -k2 | head -n 1 | cut -f2-)
    if [ -n "$g" ]; then say "$g"; return 0; fi
    c=$(find_shallow "$1" -name config.json)
    if [ -n "$c" ]; then dirname -- "$c"; return 0; fi
    say "$1"
}

MODEL_HINTS=
setup_model() {
    m_name=$1 m_uri=$2
    m_fmt=$(model_format "$m_uri" "${3:-}")
    dest="$MODELS_DIR/$m_name"
    m_explicit=$LOCAL_FLAG
    case " $SELECTED_MODELS " in *" $m_name "*) m_explicit=1 ;; esac
    if [ "$(uri_scheme "$m_uri")" = local ] && { [ "$m_fmt" = file ] || [ "$m_fmt" = dir ]; }; then
        root=$(local_path "$m_uri")
        [ -e "$root" ] || die "модель $m_name: не найден $root" 6
        if [ ! -L "$dest/model" ] && ! ask "Подключить модель $m_name на месте ($root)?" "$m_explicit"; then
            say "  $m_name: пропущена"
            return 0
        fi
        say "  $m_name: используется на месте $root"
        [ -n "$DRY_RUN" ] && return 0
        mkdir -p "$dest"
    else
        if ! artifact_current "$dest" "$m_uri" "$m_fmt" \
            && ! ask "Загрузить модель $m_name ($m_fmt) из $m_uri?" "$m_explicit"; then
            say "  $m_name: пропущена"
            return 0
        fi
        install_artifact "$m_name" "$m_uri" "$m_fmt" "$dest"
        [ -n "$DRY_RUN" ] && return 0
        root="$dest/data"
    fi
    entry=$(model_entry "$root")
    relink "$entry" "$dest/model"
    say "  $m_name: $dest/model → $entry"
    case "$entry" in
        *.gguf|*.GGUF)
            if [ -n "$LLAMA_READY" ]; then
                MODEL_HINTS="${MODEL_HINTS}  llama-server -m $(shq "$dest/model") --host 127.0.0.1 --port 8000$NL"
            fi ;;
        *)
            if [ -n "$VLLM_READY" ]; then
                MODEL_HINTS="${MODEL_HINTS}  vllm serve $(shq "$dest/model") --host 127.0.0.1 --port 8000$NL"
            fi ;;
    esac
}

has_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; }

setup_local() {
    models=$(say "$MODELS" | sed -e 's/#.*//' -e '/^[[:space:]]*$/d')
    if [ -z "$LLAMA_CPP_URI$VLLM_URI$models" ]; then say "локальная модель: источники не заданы, шаг пропущен"; return 0; fi
    if [ "$WITH_LOCAL" = 0 ]; then say "локальная модель: пропущена"; return 0; fi
    if [ -z "$WITH_LOCAL" ]; then
        if ! has_gpu; then
            say "Доступных GPU нет — локальная модель пропущена (установить всё равно: --local)"
            return 0
        fi
        if ! ask "Обнаружены GPU. Установить движки и модели?"; then
            say "локальная модель: пропущена"
            return 0
        fi
    fi
    for v in "$LLAMA_CPP_URI" "$VLLM_URI"; do
        case "$v" in *CHANGE-ME*) die "заполните LLAMA_CPP_URI и VLLM_URI в блоке НАСТРОЙКИ или оставьте ненужный пустым" 2 ;; esac
    done
    names=${SELECTED_MODELS:-$(say "$models" | awk '{ print $1 }')}
    chosen=
    # Сначала проверить весь список, чтобы опечатка не оставила половину загрузок.
    for name in $names; do
        line=$(say "$models" | awk -v n="$name" '$1 == n { print; exit }')
        [ -n "$line" ] || die "модель $name не найдена в MODELS" 6
        set -f
        # shellcheck disable=SC2086 # строка намеренно разбивается на поля
        set -- $line
        set +f
        case "$name" in *[!A-Za-z0-9._-]*) die "недопустимое имя модели: $name" 2 ;; esac
        if [ $# -lt 2 ] || [ $# -gt 3 ]; then die "строка MODELS для $name: нужно имя, источник и необязательный формат" 2; fi
        case "$2" in *CHANGE-ME*) die "заполните источник модели $name в MODELS" 2 ;; esac
        case "$(model_format "$2" "${3:-}")" in
            file|zip|zip-parts|dir) ;;
            '') die "модель $name: укажите формат (file, zip, zip-parts или dir) для $2" 2 ;;
            *) die "модель $name: неизвестный формат ${3:-}" 2 ;;
        esac
        case " $chosen " in *" $name "*) ;; *) chosen="$chosen $name" ;; esac
    done
    say "  движки: $ENGINES_DIR; модели: $MODELS_DIR"
    setup_llama
    setup_vllm
    for name in $chosen; do
        line=$(say "$models" | awk -v n="$name" '$1 == n { print; exit }')
        set -f
        # shellcheck disable=SC2086 # поля строки; glob внутри функции нужен
        set -- $line
        set +f
        setup_model "$@"
    done
    if [ -n "$MODEL_HINTS" ]; then
        say "Запуск OpenAI-совместимого API (в отдельном терминале после . env.sh):"
        printf '%s' "$MODEL_HINTS"
    fi
}

# ---- env.sh для новых терминалов ------------------------------------------------
shq() { printf "'%s'" "$(say "$1" | sed "s/'/'\\\\''/g")"; }

write_env() {
    say "env.sh: $SW_HOME/env.sh"
    [ -n "$DRY_RUN" ] && return 0
    mkdir -p "$SW_HOME"
    {
        say "# Создано setup-lite.sh. Подключайте этот файл в каждом новом терминале."
        say "export SETUP_WORK_HOME=$(shq "$SW_HOME")"
        say "export PIP_CONFIG_FILE=$(shq "$PIP_CONFIG_FILE")"
        say "export NPM_CONFIG_USERCONFIG=$(shq "$NPM_CONFIG_USERCONFIG")"
        say "export PIP_CACHE_DIR=$(shq "$PIP_CACHE_DIR")"
        say "export npm_config_cache=$(shq "$npm_config_cache")"
        say "export GIT_CONFIG_GLOBAL=$(shq "$GIT_CONFIG_GLOBAL")"
        say "export UV_PYTHON_DOWNLOADS=never"
        # shellcheck disable=SC2016 # переменные раскрываются при подключении env.sh
        say 'case ":$PATH:" in
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
unset sw_agent_bin'
    } > "$SW_HOME/env.sh.tmp" && mv "$SW_HOME/env.sh.tmp" "$SW_HOME/env.sh"
}

# ---- выполнение -------------------------------------------------------------------
if [ -n "$DRY_RUN" ]; then say "(dry-run) изменения не выполняются"; fi
say "1/4 Корпоративные реестры"
setup_pypi
setup_npm
say "2/4 Node.js и npm"
install_node
say "3/4 Агенты"
setup_agents
say "4/4 Локальная модель"
setup_local
write_env
say "готово. В новом терминале: . $(shq "$SW_HOME/env.sh")"
