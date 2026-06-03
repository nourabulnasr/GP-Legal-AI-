# -*- coding: utf-8 -*-
"""Inspect SQLite DB row details for merge planning."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Windows console UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def inspect(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [r[0] for r in cur.fetchall()]
    print(f"\n=== {path} ===")
    for t in tables:
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        n = cur.fetchone()[0]
        print(f"{t}: {n} rows")
    print("\nusers:")
    for r in cur.execute("SELECT id, email, role, email_verified FROM users ORDER BY id"):
        print(f"  {dict(r)}")
    print("\nanalyses (id, user_id, filename):")
    for r in cur.execute("SELECT id, user_id, filename FROM analyses ORDER BY id"):
        print(f"  {dict(r)}")
    conn.close()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if Path(p).is_file():
            inspect(p)
