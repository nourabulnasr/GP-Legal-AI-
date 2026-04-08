# How Does the ML Identify Violations on Unseen Contracts?

Short answer: **The ML does not “read” the law or the rule logic.** It learns from **what the rule engine did** on many clauses, then **generalizes patterns** so it can guess on new text that looks similar.

---

## 1. Where does the ML get the idea of “violation”?

**From the rule engine’s *behavior* on training data (silver labels).**

- We take a lot of **clauses** (from contracts we already have).
- For each clause we run the **RuleEngine** (your static YAML rules + regex).
- Whatever **rules fire** on that clause become the **labels** for that clause:
  - **Unified (binary) model:** label = “has violation” (1) if *any* rule fired, “no violation” (0) otherwise.
  - **Rule-relevance (multi-label) model:** label = set of **rule_ids** that fired (e.g. `LABOR25_SALARY`, `LABOR25_PROBATION_LIMIT`).

So the ML is **not** given a written definition of “violation” or the law text. It is only given:

- **Input:** clause text (and optionally law chunks in law-aware mode).
- **Output:** “this clause made these rules fire” (or “at least one rule fired”).

The **definition** of “violation” is still the **static rules**. The ML learns: *“When the rules said ‘violation’ (or ‘this rule applies’), the text usually looked like *this*.”*

---

## 2. How does it “identify” or “highlight” a violation on an unseen contract?

By **pattern matching in a learned space**, not by applying the rules step-by-step.

- At **training time**, the model sees many **(text, label)** pairs:
  - Text is turned into **features** (e.g. character n-grams, TF‑IDF).
  - Label is “has violation” or “which rule_ids fired.”
- The model learns **statistical patterns**: e.g. “these combinations of character/word patterns often go with ‘violation’ or with rule X.”

On an **unseen contract**:

- We split it into clauses (same way as in training).
- For each clause we compute the **same kind of features** (same vectorization).
- The model outputs:
  - **Binary:** a score 0–1 (“how much does this clause look like clauses that had a violation in training?”).
  - **Multi-label:** a score per rule_id (“how much does this clause look like clauses where this rule fired?”).

So “detection” = **this new clause looks similar (in feature space) to clauses that the rule engine previously marked as violations (or as triggering certain rules).** The ML is **generalizing from examples**, not executing the rules.

---

## 3. Rule engine = static rules. ML = learned imitation of their behavior.

| | Rule engine | ML |
|---|-------------|-----|
| **What it uses** | Fixed regex + logic (YAML) | Weights learned from (text, label) pairs |
| **“Violation” means** | This text matched this pattern → this rule fires | In training, “violation” = “at least one rule fired” (or “these rule_ids fired”) |
| **On new text** | Runs the same regex/logic | Asks: “Does this text *look like* text that had violations / these rules in the training set?” |
| **Generalization** | Only catches what the patterns literally match | Can flag similar *phrasings* that never appeared in training (if the pattern is in the learned space) |

So:

- **Rules:** define *what* counts as a violation and *which* rule (e.g. salary missing, probation > 3 months). They are static and interpretable.
- **ML:** learns *“text that looks like this → rule engine tended to say violation / these rule_ids.”* It can **highlight** possible violations on unseen contracts by **similarity to past examples**, without running regex.

---

## 4. Why it works on unseen contracts (and when it doesn’t)

**Why it can work:**

- Contracts share **recurring topics and wording** (salary, working hours, probation, leave, etc.).
- The model learns **features** (n-grams, etc.) that correlate with “rule fired” or “rule X fired.”
- So a **new** clause that talks about salary in a similar way (even if the exact sentence is new) can get a high “violation” or “LABOR25_SALARY” score.

**When it can fail:**

- **Very different style/vocabulary** (e.g. contracts from another country or in a different register): the “look” of the text may be outside what the model saw.
- **New violation type** that never appeared in training (no rule ever fired for it): the model has no examples to learn from.
- **Noisy silver labels:** if the rule engine often fires by mistake or misses real violations, the ML learns that noise too (“garbage in, garbage out”).

---

## 5. One-sentence summary

**The ML identifies violations on unseen contracts by learning from the rule engine’s *output* on many clauses (“this text → these rules fired”), then predicting “violation” or “which rules” on new text by **similarity** to those training examples — it does not run the static rules itself and does not “read” the law as logic.**
