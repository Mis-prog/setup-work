#!/bin/sh
# setup-work stage-0 bootstrap. POSIX sh, no Python required.
#
#  0. Optionally load team settings ($SETUP_WORK_TEAM_CONFIG_DIR/bootstrap.env),
#     run corporate pre-scripts ($SETUP_WORK_PRE_SCRIPTS), each confirmed, and
#     write the corporate package registries into pip.conf / .npmrc
#     (SETUP_WORK_PYPI_* / SETUP_WORK_NPM_*, see section 0c).
#  1. If uv is missing, obtain it from a corporate source (one is required):
#       SETUP_WORK_UV_ARCHIVE   absolute path, hdfs:// or internal https:// URL of the
#                               official uv tarball, sha256-verified against the pin
#       SETUP_WORK_UV_FROM_PIP=1  `python3 -m pip install --user uv==<pin>` (corporate PyPI)
#     Nothing is downloaded from public hosts.
#  2. Create the project environment with `uv sync --frozen`. Index settings from
#     pip.conf are passed to uv (uv does not read pip.conf itself).
#
# Never uses sudo, never pipes remote scripts into a shell, never edits shell
# startup files. Honours HTTP(S)_PROXY / NO_PROXY through curl/wget.
#
# Usage: sh scripts/bootstrap.sh [--dry-run] [--registries-only]
#        SETUP_WORK_DRY_RUN=1 sh scripts/bootstrap.sh

set -eu

UV_VERSION="0.12.13"
# sha256 of the official uv-<target>.tar.gz release files (uv 0.12.13)
SHA_aarch64_apple_darwin="7e6ddb9316acc00f2296c82ff4d99977870ee34b2f0ddcae9444d714db9364ed"
SHA_x86_64_apple_darwin="5e287ef61cb6a9b61b3a83fef124fd143e400468a7dac794230147a810e17119"
SHA_aarch64_unknown_linux_gnu="2eaa5d94f5db7b3a1a092156b9420459e42ab0217d917fe74a876309cef9b5e9"
SHA_x86_64_unknown_linux_gnu="745765a3b6e360ad76743599ae5c42e9278c7edf8bbff9fc76d05bf2623a04dd"

say() { printf '%s\n' "$*"; }
die() { say "error: $1" >&2; exit "${2:-1}"; }

# ---- 0a. team settings: KEY=VALUE lines only, never executed as shell ---------
load_env_file() {
    [ -f "$1" ] || return 0
    say "loading team settings from $1"
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        key=${line%%=*}
        value=${line#*=}
        case "$key" in
            *[!A-Za-z0-9_]*|[0-9]*|'') say "  skipping invalid line: $line"; continue ;;
        esac
        case "$key" in
            *TOKEN*|*SECRET*|*PASSWORD*|*API_KEY*) say "  refusing secret-like key $key"; continue ;;
        esac
        # strip optional surrounding quotes
        case "$value" in \"*\") value=${value#\"}; value=${value%\"} ;; \'*\') value=${value#\'}; value=${value%\'} ;; esac
        eval "already=\${$key+set}"
        if [ "$already" = set ]; then continue; fi  # the user's environment wins
        export "$key=$value"
        say "  $key set"
    done < "$1"
}
if [ -n "${SETUP_WORK_TEAM_CONFIG_DIR:-}" ]; then
    load_env_file "$SETUP_WORK_TEAM_CONFIG_DIR/bootstrap.env"
fi

UV_VERSION="${SETUP_WORK_UV_VERSION:-$UV_VERSION}"
BIN_DIR="${SETUP_WORK_BIN_DIR:-$HOME/.local/bin}"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/setup-work"
DRY_RUN="${SETUP_WORK_DRY_RUN:-}"
ASSUME_YES="${SETUP_WORK_YES:-}"
REGISTRIES_ONLY=
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --registries-only) REGISTRIES_ONLY=1 ;;
        *) die "unknown argument: $arg (expected --dry-run, --registries-only)" 2 ;;
    esac
done
case "$DRY_RUN" in 1|true|yes) DRY_RUN=1 ;; *) DRY_RUN= ;; esac
case "$ASSUME_YES" in 1|true|yes) ASSUME_YES=1 ;; *) ASSUME_YES= ;; esac

run() {
    say "  \$ $*"
    [ -n "$DRY_RUN" ] && return 0
    "$@"
}

confirm() {
    [ -n "$ASSUME_YES" ] && return 0
    [ -n "$DRY_RUN" ] && { say "  (dry-run) would ask: $1"; return 0; }
    [ -t 0 ] || die "confirmation required: $1 (re-run with SETUP_WORK_YES=1)" 5
    printf '%s [y/N] ' "$1"
    read -r answer || answer=
    case "$answer" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

# ---- 0b. corporate pre-scripts (e.g. the scripts that write pip.conf/.npmrc) ---
if [ -n "${SETUP_WORK_PRE_SCRIPTS:-}" ]; then
    old_ifs=$IFS; IFS=:
    for script in $SETUP_WORK_PRE_SCRIPTS; do
        IFS=$old_ifs
        case "$script" in /*) ;; *) die "pre-script must be an absolute path: $script" 2 ;; esac
        [ -f "$script" ] || die "pre-script not found: $script" 6
        if confirm "Run corporate setup script $script ?"; then
            run sh "$script"
        else
            say "  skipped $script"
        fi
    done
    IFS=$old_ifs
fi

# ---- 0c. corporate package registries -> pip.conf / .npmrc ------------------------
# Set in bootstrap.env (or the environment). Credentials are read from files and are
# never printed or passed on a command line.
#   SETUP_WORK_PYPI_INDEX         https://host/path/simple
#   SETUP_WORK_PYPI_USER          user name in the index URL (default: token)
#   SETUP_WORK_PYPI_AUTH_FILE     file with the password/token for the index
#   SETUP_WORK_PYPI_TRUSTED_HOST  host pip/uv may reach without TLS verification
#   SETUP_WORK_NPM_REGISTRY       https://host/path/
#   SETUP_WORK_NPM_AUTH_FILE      file with the npm _authToken
#   SETUP_WORK_NPM_CAFILE         CA bundle for the registry (preferred over the next one)
#   SETUP_WORK_NPM_STRICT_SSL     false disables TLS verification for npm
expand_home() {
    case "$1" in
        "~/"*) say "$HOME/${1#"~/"}" ;;
        '$HOME/'*) say "$HOME/${1#'$HOME/'}" ;;
        *) say "$1" ;;
    esac
}

read_secret() {
    f=$(expand_home "$1")
    [ -r "$f" ] || die "cannot read $2 file: $f" 6
    tr -d ' \t\r\n' < "$f"
}

setup_pypi() {
    [ -n "${SETUP_WORK_PYPI_INDEX:-}" ] || return 0
    conf="${PIP_CONFIG_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/pip/pip.conf}"
    user="${SETUP_WORK_PYPI_USER:-token}"
    auth="${SETUP_WORK_PYPI_AUTH_FILE:+$(expand_home "$SETUP_WORK_PYPI_AUTH_FILE")}"
    trusted="${SETUP_WORK_PYPI_TRUSTED_HOST:-}"
    say "configuring pip index $SETUP_WORK_PYPI_INDEX in $conf${auth:+ (credentials from $auth)}"
    [ -n "$DRY_RUN" ] && return 0
    if [ -n "$auth" ] && [ ! -r "$auth" ]; then die "cannot read the PyPI auth file: $auth" 6; fi
    mkdir -p "$(dirname -- "$conf")"
    py=$(command -v python3 || true)
    if [ -n "$py" ]; then
        # configparser keeps the other settings in pip.conf; the token stays off argv.
        "$py" -I -c '
import configparser, os, sys
from urllib.parse import quote, urlsplit, urlunsplit
conf, index, user, auth, trusted = sys.argv[1:6]
if auth:
    with open(auth) as fh:
        token = "".join(fh.read().split())
    u = urlsplit(index)
    host = u.hostname + (":%d" % u.port if u.port else "")
    netloc = quote(user, safe="") + ":" + quote(token, safe="") + "@" + host
    index = urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))
cp = configparser.RawConfigParser()
cp.read(conf)
if not cp.has_section("global"):
    cp.add_section("global")
cp.set("global", "index-url", index)
if trusted:
    cp.set("global", "trusted-host", trusted)
tmp = conf + ".setup-work.tmp"
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as fh:
    cp.write(fh)
os.replace(tmp, conf)
' "$conf" "$SETUP_WORK_PYPI_INDEX" "$user" "$auth" "$trusted" \
            || die "failed to write $conf" 1
    elif [ -f "$conf" ]; then
        die "python3 is needed to update the existing $conf; edit it by hand or remove it" 3
    else
        idx="$SETUP_WORK_PYPI_INDEX"
        if [ -n "$auth" ]; then
            tok=$(read_secret "$auth" "PyPI auth")
            idx="${idx%%://*}://$user:$tok@${idx#*://}"
        fi
        ( umask 077
          { say "[global]"; say "index-url = $idx"
            if [ -n "$trusted" ]; then say "trusted-host = $trusted"; fi; } > "$conf" ) \
            || die "failed to write $conf" 1
    fi
}

setup_npm() {
    [ -n "${SETUP_WORK_NPM_REGISTRY:-}" ] || return 0
    rc="${NPM_CONFIG_USERCONFIG:-$HOME/.npmrc}"
    reg="$SETUP_WORK_NPM_REGISTRY"
    case "$reg" in */) ;; *) reg="$reg/" ;; esac
    key="//${reg#*://}:_authToken"
    auth="${SETUP_WORK_NPM_AUTH_FILE:+$(expand_home "$SETUP_WORK_NPM_AUTH_FILE")}"
    cafile="${SETUP_WORK_NPM_CAFILE:+$(expand_home "$SETUP_WORK_NPM_CAFILE")}"
    strict="${SETUP_WORK_NPM_STRICT_SSL:-}"
    say "configuring npm registry $reg in $rc${auth:+ (token from $auth)}"
    [ -n "$DRY_RUN" ] && return 0
    tok=
    [ -n "$auth" ] && tok=$(read_secret "$auth" "npm auth")
    tmp="$rc.setup-work.tmp"
    # Replace only the keys managed here; everything else in .npmrc is kept.
    ( umask 077
      {
        if [ -f "$rc" ]; then
            awk -v k="$key=" 'index($0, "registry=") != 1 && index($0, k) != 1 &&
                index($0, "cafile=") != 1 && index($0, "strict-ssl=") != 1' "$rc"
        fi
        say "registry=$reg"
        if [ -n "$tok" ]; then say "$key=$tok"; fi
        if [ -n "$cafile" ]; then say "cafile=$cafile"; fi
        if [ -n "$strict" ]; then say "strict-ssl=$strict"; fi
      } > "$tmp" ) && mv "$tmp" "$rc" || die "failed to write $rc" 1
}

setup_pypi
setup_npm
if [ -n "$REGISTRIES_ONLY" ]; then
    say "registries configured"
    exit 0
fi

# ---- pip.conf -> uv (uv ignores pip.conf) --------------------------------------
pip_index_from_conf() {
    for f in "${PIP_CONFIG_FILE:-}" "${XDG_CONFIG_HOME:-$HOME/.config}/pip/pip.conf" \
             "$HOME/.pip/pip.conf" /etc/pip.conf; do
        [ -n "$f" ] && [ -f "$f" ] || continue
        v=$(sed -n 's/^[[:space:]]*index-url[[:space:]]*=[[:space:]]*//p' "$f" | head -n 1)
        if [ -n "$v" ]; then say "$v"; return 0; fi
    done
    return 1
}
if [ -z "${UV_DEFAULT_INDEX:-}" ] && [ -z "${UV_INDEX_URL:-}" ]; then
    if [ -n "${PIP_INDEX_URL:-}" ]; then
        export UV_DEFAULT_INDEX="$PIP_INDEX_URL"
    elif idx=$(pip_index_from_conf); then
        export UV_DEFAULT_INDEX="$idx"
    fi
    [ -n "${UV_DEFAULT_INDEX:-}" ] && say "uv will use the package index from pip settings"
fi

# ---- 1. uv ---------------------------------------------------------------------
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
        *) die "unsupported architecture: $arch" 4 ;;
    esac
    case "$os" in
        Darwin) say "${arch}-apple-darwin" ;;
        Linux)
            if ldd --version 2>&1 | grep -qi musl; then
                die "musl libc is not covered by the pinned uv build; install uv manually" 4
            fi
            say "${arch}-unknown-linux-gnu" ;;
        *) die "unsupported OS: $os" 4 ;;
    esac
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else die "need sha256sum or shasum to verify downloads" 3
    fi
}

fetch() {  # url dest
    if command -v curl >/dev/null 2>&1; then
        run curl --proto '=https' --tlsv1.2 -fL --retry 3 --connect-timeout 20 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        run wget --https-only -q -O "$2" "$1"
    else
        die "need curl or wget for an https:// SETUP_WORK_UV_ARCHIVE (or use a path, hdfs:// or SETUP_WORK_UV_FROM_PIP=1)" 3
    fi
}

link_uv() {  # dir containing uv and uvx
    run mkdir -p "$BIN_DIR"
    for b in uv uvx; do
        [ -e "$1/$b" ] || [ -n "$DRY_RUN" ] || continue
        link="$BIN_DIR/$b"
        if [ -e "$link" ] && [ ! -L "$link" ]; then
            die "$link exists and is not a symlink; refusing to overwrite"
        fi
        run ln -sfn "$1/$b" "$link"
    done
}

install_uv_archive() {
    target=$(uv_target)
    var="SHA_$(say "$target" | tr '-' '_')"
    eval "expected=\${$var:-}"
    expected="${SETUP_WORK_UV_SHA256:-$expected}"
    [ -n "$expected" ] || die "no pinned checksum for $target" 4
    dest_dir="$DATA_DIR/tools/uv/$UV_VERSION"
    archive="$DATA_DIR/tools/downloads/uv-${UV_VERSION}-${target}.tar.gz"
    run mkdir -p "$DATA_DIR/tools/downloads" "$dest_dir"
    if [ -f "$archive" ] && [ "$(sha256_of "$archive")" = "$expected" ]; then
        say "  using cached $archive"
    else
        src="${SETUP_WORK_UV_ARCHIVE:-}"
        case "$src" in
            https://*) fetch "$src" "$archive.part" ;;
            hdfs:*|viewfs:*)
                command -v "${SETUP_WORK_HDFS_CLI:-hdfs}" >/dev/null 2>&1 || die "hdfs client not found" 3
                if command -v klist >/dev/null 2>&1 && ! klist -s; then
                    die "no valid Kerberos ticket: run kinit first" 1
                fi
                rm -f "$archive.part"
                run "${SETUP_WORK_HDFS_CLI:-hdfs}" dfs -get "$src" "$archive.part" ;;
            /*) run cp "$src" "$archive.part" ;;
            *) die "SETUP_WORK_UV_ARCHIVE must be an absolute path, hdfs:// or https:// URI" 2 ;;
        esac
        if [ -z "$DRY_RUN" ]; then
            got=$(sha256_of "$archive.part")
            if [ "$got" != "$expected" ]; then
                rm -f "$archive.part"
                die "checksum mismatch for uv archive (expected $expected, got $got); set SETUP_WORK_UV_SHA256 for a custom build"
            fi
            mv "$archive.part" "$archive"
        fi
    fi
    run tar -xzf "$archive" -C "$dest_dir"
    link_uv "$dest_dir/uv-$target"
}

install_uv_pip() {
    py=$(command -v python3 || true)
    [ -n "$py" ] || die "SETUP_WORK_UV_FROM_PIP=1 needs python3 with pip" 3
    say "installing uv==$UV_VERSION from the configured Python package index (pip --user)"
    run "$py" -m pip install --user --disable-pip-version-check "uv==$UV_VERSION"
    if [ -z "$DRY_RUN" ]; then
        uvbin=$("$py" -c 'import uv; print(uv.find_uv_bin())') || die "uv installed but not found"
        link_uv "$(dirname "$uvbin")"
    fi
}

if UV=$(find_uv); then
    say "uv: $UV ($("$UV" --version 2>/dev/null || say unknown))"
else
    say "uv not found: installing uv $UV_VERSION into $BIN_DIR"
    use_pip=
    case "${SETUP_WORK_UV_FROM_PIP:-}" in 1|true|yes) use_pip=1 ;; esac
    if [ -n "${SETUP_WORK_UV_ARCHIVE:-}" ]; then
        install_uv_archive
    elif [ -n "$use_pip" ]; then
        install_uv_pip
    else
        die "no corporate source for uv: set SETUP_WORK_UV_ARCHIVE (path, hdfs://, https://) or SETUP_WORK_UV_FROM_PIP=1 in bootstrap.env" 3
    fi
    UV="$BIN_DIR/uv"
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) say ""
           say "note: $BIN_DIR is not on PATH. setup-work does not edit shell startup files."
           say "      add this to your shell profile:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
    esac
fi

# ---- 2. project environment -----------------------------------------------------
# uv.lock records pypi.org URLs. With a corporate index, re-resolve against it
# (locked versions are kept as preferences) instead of using the recorded URLs.
# There is no Python mirror: uv must use an interpreter already on the machine.
UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
export UV_PYTHON_DOWNLOADS
SYNC_FLAGS="--frozen"
if [ -n "${UV_DEFAULT_INDEX:-}${UV_INDEX_URL:-}" ]; then
    SYNC_FLAGS=""
    say "custom package index detected: uv.lock will be re-resolved against it"
fi
say "creating project environment (uv sync $SYNC_FLAGS)"
if [ -n "$DRY_RUN" ]; then
    say "  \$ $UV sync $SYNC_FLAGS   (in $ROOT)"
else
    # shellcheck disable=SC2086
    (cd "$ROOT" && "$UV" sync $SYNC_FLAGS --quiet) \
        || die "uv sync failed; check that Python >= 3.11 is installed (uv does not download it) and the package index (UV_DEFAULT_INDEX / pip.conf)" 1
    mkdir -p "$ROOT/.venv" && touch "$ROOT/.venv/.setup-work-synced"
fi
say "stage-0 bootstrap complete"
