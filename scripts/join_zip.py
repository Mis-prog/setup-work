#!/usr/bin/env python3
"""Объединение частей ZIP (.zip, .z01, .z02 и далее) в один файл.

Части ищутся по общему префиксу имени и сортируются по числовому суффиксу.
По умолчанию .zip идёт первым; --zip-last перемещает его в конец.
До подтверждения выводится только план. Данные копируются потоком.

Примеры:
    python3 scripts/join_zip.py /data/Qwen3.8-Flash
    python3 scripts/join_zip.py /data/name.zip
    make zip-join ZIP=/data/Qwen3.8-Flash [OUT=...] [ZIP_LAST=1]

Без OUT результат записывается рядом с частями как <префикс>.joined.zip.
Параметры --yes/--dry-run/--force учитывают SETUP_WORK_YES/_DRY_RUN/_FORCE."""

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
    """Найти части в порядке объединения и базовое имя результата.

    Допускаются каталог, файл .zip или общий префикс имён частей."""
    if target.is_dir():
        folder, prefix = target, ""
    elif target.suffix.lower() == ".zip":
        if not target.is_file():
            sys.exit(f"ошибка: {target} не найден; проверьте ZIP")
        folder, prefix = target.parent, target.stem + "."
    else:
        folder, prefix = target.parent, target.name
    if not folder.is_dir():
        sys.exit(f"ошибка: каталог {folder} не найден; проверьте ZIP")

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
        found = ", ".join(p.name for p in heads) or "нет"
        sys.exit(f"ошибка: в {where} нужен один .zip; найдено: {found}; уточните префикс ZIP")
    dupes = {n: ps for n, ps in numbered.items() if len(ps) > 1}
    if dupes:
        lines = "; ".join(f"z{n:02d}: " + ", ".join(p.name for p in ps) for n, ps in dupes.items())
        sys.exit(f"ошибка: несколько файлов с одинаковым номером части: {lines}")
    missing = sorted(set(range(1, max(numbered, default=0) + 1)) - set(numbered))
    if missing:
        sys.exit("ошибка: отсутствуют части: " + ", ".join(f".z{n:02d}" for n in missing))

    head = heads[0]
    rest = [numbered[n][0] for n in sorted(numbered)]
    base = prefix.rstrip("-_. ") or head.stem
    return ([*rest, head] if zip_last else [head, *rest]), base


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        sys.exit(f"ошибка: требуется подтверждение: {question}; повторите с YES=1")
    try:
        answer = input(f"{question} [д/Н] ")
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes", "д", "да")


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
    ap.add_argument("target", type=Path, help="префикс имени, часть .zip либо каталог")
    ap.add_argument(
        "-o", "--output", type=Path, help="файл результата; по умолчанию рядом с частями"
    )
    ap.add_argument("--zip-last", action="store_true", help="поместить .zip после частей .zNN")
    ap.add_argument("--force", action="store_true", help="перезаписать файл результата")
    ap.add_argument("-y", "--yes", action="store_true", help="не запрашивать подтверждение")
    ap.add_argument("--dry-run", action="store_true", help="только показать план")
    args = ap.parse_args()
    args.yes = args.yes or env_flag("SETUP_WORK_YES")
    args.dry_run = args.dry_run or env_flag("SETUP_WORK_DRY_RUN")
    args.force = args.force or env_flag("SETUP_WORK_FORCE")

    parts, base = find_parts(args.target, args.zip_last)
    out = args.output or parts[0].with_name(base + ".joined.zip")
    if out.resolve() in {p.resolve() for p in parts}:
        sys.exit(f"ошибка: результат {out} перезапишет исходную часть; измените OUT")
    if out.exists() and not args.force:
        sys.exit(f"ошибка: {out} уже существует; для перезаписи укажите FORCE=1")

    total = sum(p.stat().st_size for p in parts)
    print(f"{len(parts)} частей, {total / 2**30:.2f} GiB, порядок:")
    for i, p in enumerate(parts, 1):
        print(f"  {i:3d}. {p.name} ({p.stat().st_size / 2**20:.1f} MiB)")
    print(f"результат: {out}" + (" (перезапись)" if out.exists() else ""))
    free = shutil.disk_usage(out.resolve().parent).free
    if free < total:
        sys.exit(f"ошибка: требуется {total / 2**30:.2f} GiB, свободно {free / 2**30:.2f} GiB")
    if args.dry_run:
        print("dry-run: файлы не изменены")
        return
    if not args.yes and not confirm(f"Объединить {len(parts)} частей в {out}?"):
        sys.exit("отменено, файлы не изменены")

    total = join(parts, out)
    print(f"готово: {total / 2**30:.2f} GiB")
    if not zipfile.is_zipfile(out):
        sys.exit("внимание: результат не распознан как ZIP; проверьте порядок частей")


if __name__ == "__main__":
    main()
