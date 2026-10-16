#!/usr/bin/env python3
"""Join a multi-part zip (.zip, .z01, .z02, ..., .z11, ...) into one file.

Parts are found by a common name prefix, so names may differ after it (e.g. a hash):
Qwen3.8-Flash-3f9a1c.zip, Qwen3.8-Flash-77b2e0.z01, Qwen3.8-Flash-c41d08.z02, ...

Parts are ordered numerically (.z02 before .z11, .z99 before .z100); name.zip goes
first unless --zip-last is given. The plan is shown and nothing is written until it is
confirmed. Stdlib only, streams data, so it works on large files.

Usage:
    python3 scripts/join_zip.py /data/Qwen3.8-Flash   # prefix -> /data/Qwen3.8-Flash.joined.zip
    python3 scripts/join_zip.py /data/name.zip        # parts named name.zNN
    python3 scripts/join_zip.py /data/dir             # the dir holds exactly one set of parts
    make zip-join ZIP=/data/Qwen3.8-Flash [OUT=...] [ZIP_LAST=1] [DRY_RUN=1] [YES=1] [FORCE=1]

The output goes next to the parts as <prefix>.joined.zip unless -o/OUT is given.

--yes/--dry-run/--force default to SETUP_WORK_YES/_DRY_RUN/_FORCE (set by make).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

PART_RE = re.compile(r"\.z(\d+)$", re.IGNORECASE)


def find_parts(target: Path, zip_last: bool) -> tuple[list[Path], str]:
    """Return the parts in join order and a base name for the output file.

    target is a directory (every part in it), a .zip file (parts named <stem>.zNN) or a
    name prefix such as /data/Qwen3.8-Flash (every .zip/.zNN starting with it, whatever
    follows: Qwen3.8-Flash-3f9a1c.zip, Qwen3.8-Flash-77b2e0.z01, ...).
    """
    if target.is_dir():
        folder, prefix = target, ""
    elif target.suffix.lower() == ".zip":
        if not target.is_file():
            sys.exit(f"error: {target} not found")
        folder, prefix = target.parent, target.stem + "."
    else:
        folder, prefix = target.parent, target.name
    if not folder.is_dir():
        sys.exit(f"error: folder {folder} not found")

    heads: list[Path] = []
    numbered: dict[int, list[Path]] = {}
    for p in sorted(folder.iterdir()):
        if not (p.is_file() and p.name.startswith(prefix)):
            continue
        if p.name.lower().endswith(".joined.zip"):
            continue
        if p.suffix.lower() == ".zip":
            heads.append(p)
        elif m := PART_RE.search(p.name):
            numbered.setdefault(int(m.group(1)), []).append(p)

    where = f"{folder}/{prefix}*" if prefix else str(folder)
    if len(heads) != 1:
        found = ", ".join(p.name for p in heads) or "none"
        sys.exit(f"error: expected one .zip in {where}, found: {found} (narrow the prefix)")
    dupes = {n: ps for n, ps in numbered.items() if len(ps) > 1}
    if dupes:
        lines = "; ".join(f"z{n:02d}: " + ", ".join(p.name for p in ps) for n, ps in dupes.items())
        sys.exit(f"error: several files claim the same part number: {lines}")
    missing = sorted(set(range(1, max(numbered, default=0) + 1)) - set(numbered))
    if missing:
        sys.exit("error: missing parts: " + ", ".join(f".z{n:02d}" for n in missing))

    head = heads[0]
    rest = [numbered[n][0] for n in sorted(numbered)]
    base = prefix.rstrip("-_. ") or head.stem
    return ([*rest, head] if zip_last else [head, *rest]), base


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        sys.exit(f"error: confirmation required: {question} (re-run with --yes or YES=1)")
    try:
        answer = input(f"{question} [y/N] ")
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def join(parts: list[Path], out: Path) -> int:
    total = 0
    tmp = out.with_name(out.name + ".part")
    with tmp.open("wb") as dst:
        for p in parts:
            size = p.stat().st_size
            print(f"  + {p.name} ({size / 2**20:.1f} MiB)")
            with p.open("rb") as src:
                shutil.copyfileobj(src, dst, length=16 * 2**20)
            total += size
    tmp.replace(out)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("target", type=Path, help="name prefix, the .zip part or a directory")
    ap.add_argument("-o", "--output", type=Path, help="output file (default: <prefix>.joined.zip)")
    ap.add_argument("--zip-last", action="store_true", help="put name.zip after the .zNN parts")
    ap.add_argument("--force", action="store_true", help="overwrite the output file")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--dry-run", action="store_true", help="only show the plan")
    args = ap.parse_args()
    args.yes = args.yes or env_flag("SETUP_WORK_YES")
    args.dry_run = args.dry_run or env_flag("SETUP_WORK_DRY_RUN")
    args.force = args.force or env_flag("SETUP_WORK_FORCE")

    parts, base = find_parts(args.target, args.zip_last)
    out = args.output or parts[0].with_name(base + ".joined.zip")
    if out.resolve() in {p.resolve() for p in parts}:
        sys.exit(f"error: output {out} would overwrite an input part")
    if out.exists() and not args.force:
        sys.exit(f"error: {out} exists (use --force to overwrite)")

    total = sum(p.stat().st_size for p in parts)
    print(f"{len(parts)} part(s), {total / 2**30:.2f} GiB, in this order:")
    for i, p in enumerate(parts, 1):
        print(f"  {i:3d}. {p.name} ({p.stat().st_size / 2**20:.1f} MiB)")
    print(f"output: {out}" + (" (overwrite)" if out.exists() else ""))
    free = shutil.disk_usage(out.resolve().parent).free
    if free < total:
        sys.exit(f"error: not enough space: need {total / 2**30:.2f} GiB, {free / 2**30:.2f} free")
    if args.dry_run:
        print("dry-run: nothing written")
        return
    if not args.yes and not confirm(f"Join {len(parts)} part(s) into {out}?"):
        sys.exit("aborted, nothing written")

    total = join(parts, out)
    print(f"done: {total / 2**30:.2f} GiB")
    if not zipfile.is_zipfile(out):
        sys.exit("warning: result is not recognised as a zip archive; check the part order")


if __name__ == "__main__":
    main()
