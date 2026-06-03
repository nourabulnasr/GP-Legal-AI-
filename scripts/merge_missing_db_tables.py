# -*- coding: utf-8 -*-
"""Apply missing tables from a reference SQLite DB onto a target DB (schema only)."""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def tables(conn: sqlite3.Connection) -> set[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return {r[0] for r in cur.fetchall()}


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python merge_missing_db_tables.py <target_db> <reference_db>")
        return 1
    target_path = Path(sys.argv[1])
    ref_path = Path(sys.argv[2])
    if not target_path.is_file() or not ref_path.is_file():
        print("Both database paths must exist.")
        return 1

    backup = target_path.with_suffix(
        target_path.suffix + f".bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    shutil.copy2(target_path, backup)
    print(f"Backup: {backup}")

    tgt = sqlite3.connect(str(target_path))
    ref = sqlite3.connect(str(ref_path))
    missing = sorted(tables(ref) - tables(tgt))
    if not missing:
        print("No missing tables — target already has all reference tables.")
        tgt.close()
        ref.close()
        return 0

    print(f"Creating {len(missing)} missing table(s): {', '.join(missing)}")
    cur_ref = ref.cursor()
    cur_tgt = tgt.cursor()
    for name in missing:
        cur_ref.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        row = cur_ref.fetchone()
        if not row or not row[0]:
            print(f"  SKIP {name}: no DDL in reference")
            continue
        ddl = row[0]
        cur_tgt.execute(ddl)
        print(f"  OK {name}")
    tgt.commit()
    tgt.close()
    ref.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
