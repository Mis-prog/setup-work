#!/bin/sh
# setup-work stage-0 bootstrap. POSIX sh, no Python required.
#
#  0. Optionally load team settings ($SETUP_WORK_TEAM_CONFIG_DIR/bootstrap.env)
#     and run corporate pre-scripts ($SETUP_WORK_PRE_SCRIPTS), each confirmed.
#  1. If uv is missing, obtain it from (first match):
#       SETUP_WORK_UV_ARCHIVE   local path or hdfs:// URI of the official uv tarball
#       SETUP_WORK_UV_FROM_PIP=1  `python3 -m pip install --user uv==<pin>` (corporate PyPI)
#       GitHub release (default), sha256-verified against the pinned digest.
#  2. Create the project environment with `uv sync --frozen`. Index settings from
#     pip.conf are passed to uv (uv does not read pip.conf itself).
#
# Never uses sudo, never pipes remote scripts into a shell, never edits shell
# startup files. Honours HTTP(S)_PROXY / NO_PROXY through curl/wget.
#
# Usage: sh scripts/bootstrap.sh [--dry-run]
#        SETUP_WORK_DRY_RUN=1 sh scripts/bootstrap.sh

set -eu

UV_VERSION="0.12.13"
# sha256 of uv-<target>.tar.gz from https://github.com/astral-sh/uv/releases/tag/0.12.13
SHA_aarch64_apple_darwin="7e6ddb9316acc00f2296c82ff4d99977870ee34b2f0ddcae9444d714db9364ed"
SHA_x86_64_apple_darwin="5e287ef61cb6a9b61b3a83fef124fd143e400468a7dac794230147a810e17119"
SHA_aarch64_unknown_linux_gnu="2eaa5d94f5db7b3a1a092156b9420459e42ab0217d917fe74a876309cef9b5e9"
SHA_x86_64_unknown_linux_gnu="745765a3b6e360ad76743599ae5c42e9278c7edf8bbff9fc76d05bf2623a04dd"

say() { printf '%s\n' "$*"; }
die() { say "error: $*" >&2; exit "${2:-1}"; }

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
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
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
        die "need curl or wget to download uv (or set SETUP_WORK_UV_ARCHIVE / SETUP_WORK_UV_FROM_PIP=1)" 3
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
            '') fetch "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${target}.tar.gz" "$archive.part" ;;
            hdfs:*|viewfs:*)
                command -v "${SETUP_WORK_HDFS_CLI:-hdfs}" >/dev/null 2>&1 || die "hdfs client not found" 3
                if command -v klist >/dev/null 2>&1 && ! klist -s; then
                    die "no valid Kerberos ticket: run kinit first" 1
                fi
                rm -f "$archive.part"
                run "${SETUP_WORK_HDFS_CLI:-hdfs}" dfs -get "$src" "$archive.part" ;;
            /*) run cp "$src" "$archive.part" ;;
            *) die "SETUP_WORK_UV_ARCHIVE must be an absolute path or hdfs:// URI" 2 ;;
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
    if [ -n "$use_pip" ] && [ -z "${SETUP_WORK_UV_ARCHIVE:-}" ]; then
        install_uv_pip
    else
        install_uv_archive
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
        || die "uv sync failed; check the package index (UV_DEFAULT_INDEX / pip.conf) and proxy settings" 1
    mkdir -p "$ROOT/.venv" && touch "$ROOT/.venv/.setup-work-synced"
fi
say "stage-0 bootstrap complete"
