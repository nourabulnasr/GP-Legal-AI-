from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_DIR = REPO_ROOT / "rules"
LAWS_CLEANED = REPO_ROOT / "laws" / "processed" / "labor14_2025_articles.cleaned.json"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    law: str
    article: Optional[str]
    description: str
    rationale: str
    severity: str
    suggestion: Optional[str]


def _read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8-sig").strip() or None
    except Exception:
        return None


def load_rules() -> Dict[str, RuleSpec]:
    rules: Dict[str, RuleSpec] = {}
    for p in sorted(RULES_DIR.glob("*.y*ml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8-sig")) or []
        if isinstance(data, dict):
            data = [data]
        for r in data:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("id") or r.get("rule_id") or "").strip()
            if not rid:
                continue
            law = str(r.get("law") or "").strip()
            article = r.get("article")
            article_str = str(article).strip() if article is not None else None
            desc = str(r.get("description") or "").strip()
            rat = str(r.get("rationale") or "").strip()
            sev = str(r.get("severity") or "info").strip().lower()
            sugg: Optional[str] = None
            suggestion_ref = r.get("suggestion_ref")
            if suggestion_ref:
                sp = Path(str(suggestion_ref))
                if not sp.is_absolute():
                    sp = REPO_ROOT / sp
                sugg = _read_text(sp)
            rules[rid] = RuleSpec(
                rule_id=rid,
                law=law,
                article=article_str,
                description=desc,
                rationale=rat,
                severity=sev,
                suggestion=sugg,
            )
    return rules


def load_law_articles_cleaned(path: Path) -> Dict[Tuple[str, str], str]:
    """Return map[(law_name, article_number)] -> article_text."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    out: Dict[Tuple[str, str], str] = {}
    if isinstance(raw, list):
        for x in raw:
            if not isinstance(x, dict):
                continue
            law = str(x.get("law") or "").strip()
            art = str(x.get("article") or "").strip()
            text = str(x.get("text") or "").strip()
            if law and art and text:
                out[(law, art)] = text
    return out


def iter_jsonl(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                if isinstance(x, dict):
                    yield x


def _law_articles_for_rule(
    rule: RuleSpec,
    law_articles: Dict[Tuple[str, str], str],
    max_chars: int = 1500,
) -> List[Dict[str, Any]]:
    if not rule.article or not rule.law:
        return []
    txt = law_articles.get((rule.law, rule.article))
    if not txt:
        return []
    return [
        {
            "text": txt[:max_chars],
            "metadata": {"article": rule.article, "law": rule.law},
        }
    ]


def _tokenize_query_ar(s: str) -> List[str]:
    # Very lightweight tokenization; good enough for keyword overlap fallback.
    s = (s or "").replace("\u200f", " ").replace("\u200e", " ")
    for ch in "،؛:.!?()[]{}\"'“”«»/\\|*_-–—\n\r\t":
        s = s.replace(ch, " ")
    toks = [t.strip() for t in s.split() if t.strip()]
    # Drop extremely short tokens (noise) but keep numbers.
    out: List[str] = []
    for t in toks:
        if t.isdigit():
            out.append(t)
        elif len(t) >= 3:
            out.append(t)
    return out[:32]


def _fallback_retrieve_law_articles(
    rule: RuleSpec,
    law_articles: Dict[Tuple[str, str], str],
    *,
    max_chars: int = 1500,
    max_hits: int = 1,
) -> List[Dict[str, Any]]:
    """
    If a rule doesn't specify an article (or it's missing), pick the most relevant
    article text from the cleaned law corpus by simple keyword overlap.
    """
    if not rule.law:
        return []
    q = f"{rule.description}\n{rule.rationale}".strip()
    q_tokens = _tokenize_query_ar(q)
    if not q_tokens:
        return []

    scored: List[Tuple[int, str, str]] = []  # (score, article, text)
    for (law, art), txt in law_articles.items():
        if law != rule.law:
            continue
        t = txt or ""
        score = 0
        # Count token occurrences (cheap).
        for tok in q_tokens:
            if tok and tok in t:
                score += 1
        if score:
            scored.append((score, art, t))

    scored.sort(key=lambda x: (-x[0], int(x[1]) if x[1].isdigit() else 10**9))
    out: List[Dict[str, Any]] = []
    for score, art, t in scored[:max_hits]:
        out.append({"text": t[:max_chars], "metadata": {"article": art, "law": rule.law}})
    return out


def build_instruction(
    *,
    rule_id: str,
    description: str,
    matched_text: str,
    law_articles: List[Dict[str, Any]],
    language: str = "ar",
) -> str:
    """Mirror the production prompt shape in app/local_llm.build_explanation_prompt()."""
    # Keep aligned with app/local_llm.py (Arabic system prompt).
    sys_prompt = (
        "أنت مساعد قانوني. مهمتك فقط شرح سبب مخالفة البند للقانون واستنتاج تصحيح بناءً على النصوص المقدمة.\n"
        "- استخدم فقط المواد القانونية المقدمة أدناه. لا تخترع أرقام مواد أو نصوصاً.\n"
        '- إذا لم يكن السياق كافياً، قل: "لا يوجد نص قانوني كافٍ في المستند."\n'
        "- اذكر رقم المادة عند الاقتباس.\n"
        "- قدم تصحيحاً مقترحاً للبند بناءً على النص القانوني فقط."
        if language == "ar"
        else
        "You are a legal assistant. Your task is only to explain why the clause violates the law and suggest a correction based strictly on the provided texts.\n"
        "- Use ONLY the law articles provided below. Do not invent article numbers or text.\n"
        '- If the context is insufficient, say: "Insufficient legal text in the provided document."\n'
        "- Cite article numbers when quoting.\n"
        "- Provide a suggested correction for the clause based only on the law text."
    )

    blocks: List[str] = []
    for a in (law_articles or [])[:6]:
        meta = a.get("metadata") or {}
        art = str(meta.get("article") or "").strip()
        law = str(meta.get("law") or "").strip()
        text = str(a.get("text") or "").strip()
        if not text:
            continue
        if language == "ar":
            blocks.append(f"المادة {art} - {law}:\n{text}")
        else:
            blocks.append(f"Article {art} - {law}:\n{text}")
    law_str = "\n\n---\n\n".join(blocks) if blocks else ("لم تُقدّم مواد قانونية." if language == "ar" else "No law text was provided.")

    if language == "ar":
        return (
            f"{sys_prompt}\n\n"
            f"المخالفة: {rule_id}\n"
            f"الوصف: {description}\n\n"
            f"نص العقد المعني:\n{matched_text}\n\n"
            f"النصوص القانونية المقدمة:\n{law_str}\n\n"
            "الشرح والتصحيح المقترح (بناءً على النصوص أعلاه فقط):"
        )
    return (
        f"{sys_prompt}\n\n"
        f"Violation: {rule_id}\n"
        f"Description: {description}\n\n"
        f"Relevant clause text:\n{matched_text}\n\n"
        f"Provided law texts:\n{law_str}\n\n"
        "Explanation and suggested correction (based only on the texts above):"
    )


def build_output_ar(rule: RuleSpec, matched_text: str, law_articles: List[Dict[str, Any]]) -> str:
    """Create a strong, structured target answer."""
    # Quote a short excerpt from the first article.
    cite = ""
    if law_articles:
        meta = law_articles[0].get("metadata") or {}
        art = str(meta.get("article") or "").strip()
        law = str(meta.get("law") or "").strip()
        txt = str(law_articles[0].get("text") or "").strip().replace("\n", " ")
        excerpt = (txt[:260] + "…") if len(txt) > 260 else txt
        if art and law and excerpt:
            cite = f"وفقًا للمادة {art} من {law}: «{excerpt}»\n\n"

    why = rule.rationale or rule.description
    # Keep the explanation grounded without inventing facts.
    explain = (
        f"{cite}"
        f"سبب المشكلة:\n{why}\n\n"
        f"كيف يرتبط ذلك بالنص:\nالنص المعروض يقول: «{matched_text.strip()}». هذا يستدعي تطبيق القاعدة المذكورة أعلاه، ولذلك يلزم تعديل الصياغة لتتوافق مع النص القانوني.\n\n"
    )
    if rule.suggestion:
        fix = f"التصحيح المقترح:\n{rule.suggestion.strip()}"
    else:
        # Safe generic correction aligned with rule description.
        fix = f"التصحيح المقترح:\nأعد صياغة البند بحيث يلتزم بما ورد في المادة المذكورة أعلاه وبما يحقق: {rule.description}."
    return (explain + fix).strip()


def build_examples(
    *,
    rows: Iterable[Dict[str, Any]],
    rules: Dict[str, RuleSpec],
    law_articles: Dict[Tuple[str, str], str],
    language: str = "ar",
    max_per_rule: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Convert labeled clause chunks into explain-clause instruction/output pairs.

    We treat each (text, label) as an explain request for that rule id, and attach
    the rule's referenced law article text (when available) as "provided law texts".
    """
    # Only include rules that have meaningful corrective/violation behavior.
    include_sev = {"error", "high", "warning"}
    per_rule_counts: Dict[str, int] = {}
    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []

    for r in rows:
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        labels = r.get("labels") or []
        if not isinstance(labels, list) or not labels:
            continue
        for lbl in labels:
            rid = str(lbl or "").strip()
            if not rid:
                continue
            spec = rules.get(rid)
            if not spec:
                continue
            if spec.severity.lower() not in include_sev:
                continue
            if max_per_rule is not None and per_rule_counts.get(rid, 0) >= max_per_rule:
                continue

            key = (rid, text)
            if key in seen:
                continue
            seen.add(key)

            las = _law_articles_for_rule(spec, law_articles)
            if not las:
                las = _fallback_retrieve_law_articles(spec, law_articles, max_hits=1)
            instruction = build_instruction(
                rule_id=rid,
                description=spec.description or "طلب شرح لهذه المخالفة.",
                matched_text=text[:1500],
                law_articles=las,
                language=language,
            )
            if language == "ar":
                output = build_output_ar(spec, text[:500], las) if las else 'لا يوجد نص قانوني كافٍ في المستند.'
            else:
                # Minimal English target (not primary in this repo).
                output = "Insufficient legal text in the provided document." if not las else (
                    "Based on the provided article text, explain the issue and suggest a correction aligned with the law."
                )

            out.append(
                {
                    "instruction": instruction,
                    "output": output,
                    "meta": {
                        "rule_id": rid,
                        "contract_id": r.get("contract_id"),
                        "chunk_id": r.get("chunk_id"),
                        "source": r.get("source"),
                        "file_name": r.get("file_name"),
                        "severity": spec.severity,
                        "law": spec.law,
                        "article": spec.article,
                    },
                }
            )
            per_rule_counts[rid] = per_rule_counts.get(rid, 0) + 1

    return out


def split_train_val(items: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rnd = random.Random(seed)
    items2 = list(items)
    rnd.shuffle(items2)
    n_val = int(round(len(items2) * val_ratio))
    val = items2[:n_val]
    train = items2[n_val:]
    return train, val


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="*", default=[
        str(REPO_ROOT / "data" / "dataset" / "silver_dataset.jsonl"),
        str(REPO_ROOT / "data" / "dataset" / "augmented.jsonl"),
        str(REPO_ROOT / "data" / "dataset" / "train_all.jsonl"),
    ])
    ap.add_argument("--out-train", default=str(REPO_ROOT / "data" / "dataset" / "explain_clause_train.jsonl"))
    ap.add_argument("--out-val", default=str(REPO_ROOT / "data" / "dataset" / "explain_clause_val.jsonl"))
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-per-rule", type=int, default=120)
    args = ap.parse_args()

    rules = load_rules()
    law_map = load_law_articles_cleaned(LAWS_CLEANED)
    rows = list(iter_jsonl(Path(p) for p in args.inputs))
    examples = build_examples(
        rows=rows,
        rules=rules,
        law_articles=law_map,
        language="ar",
        max_per_rule=args.max_per_rule if args.max_per_rule > 0 else None,
    )

    train, val = split_train_val(examples, args.val_ratio, args.seed)
    write_jsonl(Path(args.out_train), train)
    write_jsonl(Path(args.out_val), val)

    print(f"Loaded rows: {len(rows)}")
    print(f"Explain-clause examples: {len(examples)}")
    print(f"Train: {len(train)}  Val: {len(val)}")
    print(f"Wrote: {args.out_train}")
    print(f"Wrote: {args.out_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

