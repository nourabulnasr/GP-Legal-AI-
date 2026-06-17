import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.law_paths import (  # noqa: E402
    ARTICLES_CANONICAL,
    CHUNKS_JSONL,
    LEGACY_ARTICLES_JSON,
    LEGACY_CHUNKS_JSONL,
    migrate_legacy_labor_law_artifacts,
)


def main():
    migrate_legacy_labor_law_artifacts()

    articles_path = ARTICLES_CANONICAL if ARTICLES_CANONICAL.is_file() else LEGACY_ARTICLES_JSON
    chunks_path = CHUNKS_JSONL if CHUNKS_JSONL.is_file() else LEGACY_CHUNKS_JSONL

    problems = []
    if not articles_path.is_file():
        problems.append(f"Missing: {ARTICLES_CANONICAL} (legacy: {LEGACY_ARTICLES_JSON})")
    if not chunks_path.is_file():
        problems.append(f"Missing: {CHUNKS_JSONL} (legacy: {LEGACY_CHUNKS_JSONL})")

    if problems:
        print("❌ Files missing:")
        for p in problems:
            print("-", p)
        raise SystemExit(1)

    data = json.loads(articles_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) == 0:
        raise SystemExit("❌ articles.json must be a non-empty list")

    n = 0
    bad = 0
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
                if "text" not in obj:
                    bad += 1
            except Exception:
                bad += 1

    print("✅ Validation OK")
    print(f"- Articles: {articles_path}")
    print(f"- Chunks:   {chunks_path}")
    print(f"- Articles count: {len(data)}")
    print(f"- Chunks lines:   {n}")
    print(f"- Bad chunk lines:{bad}")

    if bad > 0:
        raise SystemExit("❌ Some chunk lines are invalid JSON or missing 'text'.")


if __name__ == "__main__":
    main()
