# -*- coding: utf-8 -*-
"""Compare two SQLite databases: tables, columns, row counts."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def schema_info(path: str) -> Dict[str, dict]:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    info: Dict[str, dict] = {}
    for name, ddl in cur.fetchall():
        cur.execute(f'PRAGMA table_info("{name}")')
        cols = [(r[1], r[2], r[3], r[5]) for r in cur.fetchall()]  # name, type, notnull, pk
        cur.execute(f'SELECT COUNT(*) FROM "{name}"')
        cnt = cur.fetchone()[0]
        info[name] = {"ddl": ddl or "", "cols": cols, "rows": cnt}
    conn.close()
    return info


def col_names(cols: List[Tuple]) -> List[str]:
    return [c[0] for c in cols]


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python compare_db_schemas.py <old_db> <new_db>")
        return 1
    old_path, new_path = sys.argv[1], sys.argv[2]
    for p in (old_path, new_path):
        if not Path(p).is_file():
            print(f"Missing file: {p}")
            return 1

    old = schema_info(old_path)
    new = schema_info(new_path)
    old_tables = set(old)
    new_tables = set(new)

    only_new = sorted(new_tables - old_tables)
    only_old = sorted(old_tables - new_tables)
    common = sorted(old_tables & new_tables)

    print(f"OLD: {old_path} ({len(old)} tables)")
    print(f"NEW: {new_path} ({len(new)} tables)")
    print()

    if only_new:
        print("=== Tables in NEW only (missing from OLD) ===")
        for t in only_new:
            d = new[t]
            print(f"  {t}: {d['rows']} rows, cols={col_names(d['cols'])}")
    else:
        print("=== No tables missing from OLD (all NEW tables exist in OLD) ===")

    print()
    if only_old:
        print("=== Tables in OLD only (not in NEW) ===")
        for t in only_old:
            d = old[t]
            print(f"  {t}: {d['rows']} rows")

    print()
    print("=== Column diffs in common tables ===")
    col_diffs = False
    for t in common:
        oc = set(col_names(old[t]["cols"]))
        nc = set(col_names(new[t]["cols"]))
        add = sorted(nc - oc)
        rem = sorted(oc - nc)
        if add or rem:
            col_diffs = True
            print(f"  {t}:")
            if add:
                print(f"    + in NEW: {add}")
            if rem:
                print(f"    - only in OLD: {rem}")
    if not col_diffs:
        print("  (none — same column names on shared tables)")

    print()
    print("=== Row counts (common tables) ===")
    for t in common:
        print(f"  {t}: OLD={old[t]['rows']} NEW={new[t]['rows']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
