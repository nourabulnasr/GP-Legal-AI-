# -*- coding: utf-8 -*-
"""Compare user emails and analysis keys between two DBs."""
from __future__ import annotations

import sqlite3
import sys


def main() -> None:
    tgt, src = sys.argv[1], sys.argv[2]
    tc = sqlite3.connect(tgt)
    sc = sqlite3.connect(src)
    te = {r[1]: r[0] for r in tc.execute("SELECT id, email FROM users")}
    se = {r[0]: r[1] for r in sc.execute("SELECT id, email FROM users")}
    overlap = set(te) & set(se.values())
    only_src = set(se.values()) - set(te)
    only_tgt = set(te) - set(se.values())
    print("overlap emails:", sorted(overlap))
    print("only in source:", len(only_src), sorted(only_src)[:5], "...")
    print("only in target:", sorted(only_tgt))
    print("\nsource user ids:", sorted(se.keys()))
    print("target user ids:", sorted(te.values()))
    ta = list(tc.execute("SELECT id, user_id, filename, sha256 FROM analyses"))
    sa = list(sc.execute("SELECT id, user_id, filename, sha256 FROM analyses"))
    print(f"\ntarget analyses: {len(ta)} source analyses: {len(sa)}")
    tc.close()
    sc.close()


if __name__ == "__main__":
    main()
