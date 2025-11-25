#!/usr/bin/env python3
"""
Simple script: create 'clean', 'rendered', and 'boxes' subfolders inside
a directory and move files whose filenames contain those keywords
(case-insensitive) into the corresponding folder.

Usage:
    python split_move.py /path/to/dir
"""
import sys
import shutil
from pathlib import Path

KEYWORDS = ["clean", "rendered", "boxes"]

def unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suf = dest.stem, dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suf}"
        if not candidate.exists():
            return candidate
        i += 1

def main(root_dir: str):
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        print(f"Error: not a directory: {root}")
        return

    # create subfolders
    subdirs = {k: root / k for k in KEYWORDS}
    for d in subdirs.values():
        d.mkdir(exist_ok=True)

    # iterate top-level files only
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        name_l = entry.name.lower()
        matched = None
        for kw in KEYWORDS:
            if kw in name_l:
                matched = kw
                break
        if matched:
            dest = subdirs[matched] / entry.name
            dest = unique_path(dest)
            shutil.move(str(entry), str(dest))
            print(f"Moved: {entry.name} -> {matched}/{dest.name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_move.py /path/to/dir")
        sys.exit(1)
    main(sys.argv[1])