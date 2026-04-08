# ML Model, Violations, and LLM Integration

## How are rules relevant?

Rules are **YAML-based regex patterns** in `rules/*.yaml`. Each rule has:
- `id` / `rule_id`
- `match` (regex patterns)
- `severity`, `description`, `article`, `law`

Relevance is determined by:
1. **Scope** – labor, cross_border, etc.
2. **Contract type** – employment, service, etc.
3. **Pattern match** – regex matches text → rule is relevant and a violation is reported.

## How are weak labels identified by the RuleEngine?

**Weak (silver) labels** come from the RuleEngine when it runs on contract text:

1. **`build_silver_labels()`** in `train_ml_predictor.py`:
   - Loads contract text from `data/contracts_raw/` (PDF, DOCX, TXT).
   - Normalizes text with `normalize_for_rules()`.
   - Calls `rule_engine.check_text(text_norm, law_scope)` for each scope (labor, cross_border).
   - Each hit has `rule_id`. The set of rule_ids = labels for that text.
   - Labels = {rule_id for each hit in rule_engine.check_text(...)}.

2. **Training data**:
   - X = normalized contract texts
   - Y = multi-label set of rule_ids (labels = rules that fired)
   - The ML learns: "texts like this tend to trigger rules X, Y, Z."

3. **Why "weak"?** – Labels are rule-based, not human-labeled. Rules can miss or over-match, so labels are noisy.

## Is the ML model working as a classifier?

**Yes.** The ML model is a **multi-label classifier** that predicts *rule relevance scores* (not violations directly). It has been extended to act as a **hybrid detector** (see below).

### How it works

1. **Training** (`app/train_ml_predictor.py`):
   - Uses weak labels from the RuleEngine (silver labels from YAML regex rules).
   - Train data: contract texts → rule_ids that fired on that text.
   - Model: TF-IDF vectorizer + OneVsRestClassifier(LogisticRegression).
   - Outputs: scores per rule_id (probability that rule is relevant for given text).

2. **Inference** (`app/ml_predictor.py`):
   - `predict_rule_scores_full(text)` → list of `{rule_id, score, threshold, passed_threshold}`.
   - `predict_rule_ids(text, top_k=15)` → shortlist of rule IDs to run.

3. **Pipeline** (in `main.py`):
   - ML shortlists which rules to run.
   - RuleEngine then runs those rules on the text.
   - **Violations are detected by the RuleEngine** (regex/pattern matching), not by ML.

## Can ML detect violations instead of rules?

Yes, but it requires **retraining and a different architecture**:

1. **Binary classifier per rule**: Train "this clause violates rule X: yes/no" instead of "rule X might be relevant."
2. **Sequence / span model**: Use NER or span-extraction to find violation spans in text (e.g. transformers).
3. **End-to-end violation detector**: Train on labeled violations (manual or from rules) to predict violation presence and spans.

Current setup uses ML only for *rule selection*; detection stays rule-based. To move fully to ML-based detection, you’d need:
- Labeled violation data (per clause or span),
- A model trained for violation classification or extraction,
- Integration into the analysis pipeline.

## What if you integrate an LLM?

Possible roles for an LLM:

1. **Reasoning layer** (hybrid):
   - Rules/ML find candidate violations.
   - LLM explains why it’s a violation and suggests fixes.
   - Keeps deterministic detection, adds interpretability.

2. **Detection layer**:
   - LLM classifies clauses as violating/not violating.
   - Pros: flexible, can handle nuance.
   - Cons: slower, cost, less deterministic, needs careful prompting.

3. **RAG for context** (already partially used):
   - LLM uses retrieved law chunks as context to answer questions.
   - Improves reasoning over contract + law.

Recommended direction: keep rules/ML for detection, add LLM for explanations and suggestions (hybrid).
