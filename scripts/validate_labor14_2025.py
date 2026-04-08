import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTICLES = ROOT / "laws" / "processed" / "labor14_2025_articles.json"
CHUNKS = ROOT / "chunks" / "labor14_2025_chunks.jsonl"

def main():
    problems = []

    if not ARTICLES.exists():
        problems.append(f"Missing: {ARTICLES}")
    if not CHUNKS.exists():
        problems.append(f"Missing: {CHUNKS}")

    if problems:
        print("❌ Files missing:")
        for p in problems:
            print("-", p)
        raise SystemExit(1)

    # Check articles
    data = json.loads(ARTICLES.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) == 0:
        raise SystemExit("❌ articles.json must be a non-empty list")

    # Check chunks jsonl
    n = 0
    bad = 0
    with CHUNKS.open("r", encoding="utf-8") as f:
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
    print(f"- Articles count: {len(data)}")
    print(f"- Chunks lines:   {n}")
    print(f"- Bad chunk lines:{bad}")

    if bad > 0:
        raise SystemExit("❌ Some chunk lines are invalid JSON or missing 'text'.")

if __name__ == "__main__":
    main()
