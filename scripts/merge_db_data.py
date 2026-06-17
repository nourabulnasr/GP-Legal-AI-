# -*- coding: utf-8 -*-
"""
Merge row data from a source SQLite DB into a target DB.
- Preserves all target rows
- Adds source users (by email), remaps foreign keys
- Skips duplicate unique keys; reassigns conflicting primary keys on insert
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Insert order respects FK dependencies
TABLE_ORDER: List[str] = [
    "users",
    "analyses",
    "legato_profiles",
    "legato_deal_threads",
    "legato_deal_messages",
    "legato_shares",
    "legato_timeline_events",
    "legato_signatures",
    "social_posts",
    "social_post_likes",
    "social_post_comments",
    "social_post_shares",
    "network_invites",
    "skill_endorsements",
    "profile_recommendations",
    "profile_user_documents",
    "verification_codes",
    "password_reset_tokens",
]

# column -> id_map key for FK remapping
FK_MAP: Dict[str, Dict[str, str]] = {
    "analyses": {"user_id": "users"},
    "legato_profiles": {"user_id": "users"},
    "legato_deal_threads": {"analysis_id": "analyses", "user_id": "users"},
    "legato_deal_messages": {"thread_id": "legato_deal_threads", "author_id": "users"},
    "legato_shares": {"analysis_id": "analyses", "user_id": "users"},
    "legato_timeline_events": {"analysis_id": "analyses", "user_id": "users"},
    "legato_signatures": {"analysis_id": "analyses", "user_id": "users"},
    "social_posts": {"author_id": "users"},
    "social_post_likes": {"post_id": "social_posts", "user_id": "users"},
    "social_post_comments": {"post_id": "social_posts", "author_id": "users"},
    "social_post_shares": {"post_id": "social_posts", "user_id": "users"},
    "network_invites": {"requester_id": "users", "addressee_id": "users"},
    "skill_endorsements": {"endorser_id": "users", "recipient_id": "users"},
    "profile_recommendations": {"author_id": "users", "recipient_id": "users"},
    "profile_user_documents": {"user_id": "users"},
    "verification_codes": {"user_id": "users"},
    "password_reset_tokens": {"user_id": "users"},
}

UNIQUE_KEYS: Dict[str, List[List[str]]] = {
    "users": [["email"]],
    "legato_profiles": [["user_id"]],
    "social_post_likes": [["post_id", "user_id"]],
    "network_invites": [["requester_id", "addressee_id"]],
    "skill_endorsements": [["endorser_id", "recipient_id", "skill"]],
    "legato_shares": [["token"]],
}


def _row_hash(row: Dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    return [r[1] for r in cur.fetchall()]


def _pk_col(conn: sqlite3.Connection, table: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    for r in cur.fetchall():
        if r[5]:
            return r[1]
    return None


def _fetch_rows(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    cols = _columns(conn, table)
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM "{table}"')
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _exists_unique(
    conn: sqlite3.Connection,
    table: str,
    row: Dict[str, Any],
    keys: List[List[str]],
) -> bool:
    cur = conn.cursor()
    for key_cols in keys:
        if not all(k in row for k in key_cols):
            continue
        where = " AND ".join(f'"{k}" = ?' for k in key_cols)
        vals = tuple(row[k] for k in key_cols)
        cur.execute(f'SELECT 1 FROM "{table}" WHERE {where} LIMIT 1', vals)
        if cur.fetchone():
            return True
    return False


def _merge_users(
    tgt: sqlite3.Connection,
    src: sqlite3.Connection,
    id_maps: Dict[str, Dict[int, int]],
) -> Tuple[int, int]:
    id_maps["users"] = {}
    tgt_users = {r[1]: r[0] for r in tgt.execute("SELECT id, email FROM users")}
    added = skipped = 0
    cols = [c for c in _columns(src, "users") if c != "id"]
    for row in _fetch_rows(src, "users"):
        src_id = row["id"]
        email = row["email"]
        if email in tgt_users:
            id_maps["users"][src_id] = tgt_users[email]
            skipped += 1
            continue
        insert = {k: row[k] for k in cols}
        placeholders = ", ".join("?" for _ in insert)
        col_names = ", ".join(f'"{k}"' for k in insert)
        cur = tgt.cursor()
        cur.execute(
            f'INSERT INTO users ({col_names}) VALUES ({placeholders})',
            tuple(insert.values()),
        )
        new_id = cur.lastrowid
        id_maps["users"][src_id] = new_id
        tgt_users[email] = new_id
        added += 1
    tgt.commit()
    return added, skipped


def _remap_row(table: str, row: Dict[str, Any], id_maps: Dict[str, Dict[int, int]]) -> Optional[Dict[str, Any]]:
    out = dict(row)
    for col, map_key in FK_MAP.get(table, {}).items():
        if col not in out or out[col] is None:
            continue
        old = int(out[col])
        mapped = id_maps.get(map_key, {}).get(old)
        if mapped is None:
            return None
        out[col] = mapped
    return out


def _merge_table(
    tgt: sqlite3.Connection,
    src: sqlite3.Connection,
    table: str,
    id_maps: Dict[str, Dict[int, int]],
) -> Tuple[int, int, int]:
    if table == "users":
        return 0, 0, 0

    pk = _pk_col(tgt, table)
    cols = _columns(tgt, table)
    src_rows = _fetch_rows(src, table)
    added = skipped = errors = 0
    id_maps.setdefault(table, {})

    # Existing PKs and content hashes in target
    existing_pks: Set[Any] = set()
    if pk:
        existing_pks = {r[0] for r in tgt.execute(f'SELECT "{pk}" FROM "{table}"')}

    for row in src_rows:
        src_pk = row.get(pk) if pk else None
        remapped = _remap_row(table, row, id_maps)
        if remapped is None:
            skipped += 1
            continue

        if _exists_unique(tgt, table, remapped, UNIQUE_KEYS.get(table, [])):
            skipped += 1
            if pk and src_pk is not None and pk in remapped:
                # map for FK even if duplicate
                cur = tgt.cursor()
                keys = UNIQUE_KEYS[table][0]
                where = " AND ".join(f'"{k}" = ?' for k in keys)
                vals = tuple(remapped[k] for k in keys)
                cur.execute(f'SELECT "{pk}" FROM "{table}" WHERE {where} LIMIT 1', vals)
                found = cur.fetchone()
                if found and src_pk is not None:
                    id_maps[table][int(src_pk)] = int(found[0])
            continue

        insert_cols = [c for c in cols if c in remapped]
        insert_vals = [remapped[c] for c in insert_cols]

        # Reassign PK if collision
        if pk and pk in insert_cols and remapped[pk] in existing_pks:
            insert_cols = [c for c in insert_cols if c != pk]
            insert_vals = [remapped[c] for c in insert_cols]

        try:
            placeholders = ", ".join("?" for _ in insert_cols)
            col_sql = ", ".join(f'"{c}"' for c in insert_cols)
            cur = tgt.cursor()
            cur.execute(
                f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})',
                tuple(insert_vals),
            )
            new_pk = cur.lastrowid if pk and pk not in insert_cols else remapped.get(pk)
            if pk and src_pk is not None:
                if pk not in insert_cols:
                    new_pk = cur.lastrowid
                id_maps[table][int(src_pk)] = int(new_pk)
                existing_pks.add(new_pk)
            added += 1
        except sqlite3.IntegrityError:
            errors += 1
            skipped += 1

    tgt.commit()
    return added, skipped, errors


def merge_data(target_path: str, source_path: str, *, dry_run: bool = False) -> dict:
    if dry_run:
        tgt = sqlite3.connect(":memory:")
        src = sqlite3.connect(source_path)
        # clone schema from target file
        real = sqlite3.connect(target_path)
        real.backup(tgt)
        real.close()
    else:
        backup = Path(target_path).with_suffix(
            Path(target_path).suffix + f".bak-data-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(target_path, backup)
        print(f"Backup: {backup}")
        tgt = sqlite3.connect(target_path)
        src = sqlite3.connect(source_path)

    tgt.execute("PRAGMA foreign_keys = OFF")
    id_maps: Dict[str, Dict[int, int]] = {}
    stats: Dict[str, dict] = {}

    u_add, u_skip = _merge_users(tgt, src, id_maps)
    stats["users"] = {"added": u_add, "skipped": u_skip}

    for table in TABLE_ORDER:
        if table == "users":
            continue
        # skip empty source tables quickly
        n = src.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if n == 0:
            stats[table] = {"added": 0, "skipped": 0, "errors": 0}
            continue
        a, s, e = _merge_table(tgt, src, table, id_maps)
        stats[table] = {"added": a, "skipped": s, "errors": e}
        print(f"  {table}: +{a} skipped={s} err={e}")

    tgt.execute("PRAGMA foreign_keys = ON")
    tgt.commit()
    tgt.close()
    src.close()
    return stats


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python merge_db_data.py <target_db> <source_db> [--dry-run]")
        return 1
    target, source = sys.argv[1], sys.argv[2]
    dry = "--dry-run" in sys.argv
    if not Path(target).is_file() or not Path(source).is_file():
        print("Both database files must exist.")
        return 1
    print(f"Merging data: {source} -> {target}" + (" (dry-run)" if dry else ""))
    stats = merge_data(target, source, dry_run=dry)
    total_add = sum(v.get("added", 0) for v in stats.values())
    print(f"Done. Total rows added: {total_add}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
