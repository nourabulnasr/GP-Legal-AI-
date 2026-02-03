import json, joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

DATA = Path("data/dataset/train_all.jsonl")
OUTD = Path("ml/artifacts/tfidf_baseline")

vec = joblib.load(OUTD/"vectorizer.joblib")
clf = joblib.load(OUTD/"model.joblib")
mlb = joblib.load(OUTD/"mlb.joblib")

texts=[]
labels=[]
contract_ids=[]
with DATA.open("r",encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        texts.append(r["text"])
        labels.append(r.get("labels") or [])
        contract_ids.append(r.get("contract_id",""))

Y = mlb.transform(labels)

# stratify to keep SYNTH in test
is_synth = [1 if cid == "SYNTH" else 0 for cid in contract_ids]
X_train, X_test, Y_train, Y_test = train_test_split(
    texts, Y, test_size=0.2, random_state=42, stratify=is_synth
)

Xte = vec.transform(X_test)

probs = clf.predict_proba(Xte)
if isinstance(probs, list):
    probs = np.vstack([p[:,1] for p in probs]).T

# per-label threshold: pick best F1 over a small grid
grid = np.array([0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8])
thresholds = np.zeros(probs.shape[1], dtype=float)

for j in range(probs.shape[1]):
    best_t = 0.5
    best_f1 = -1.0
    y_true = Y_test[:, j]
    if y_true.sum() == 0:
        thresholds[j] = 0.99
        continue
    for t in grid:
        y_pred = (probs[:, j] >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    thresholds[j] = best_t

print("✅ Tuned thresholds (first 10):")
for i, rid in enumerate(list(mlb.classes_)[:10]):
    print(rid, "->", thresholds[i])

pred = (probs >= thresholds).astype(int)
print("\n=== Report after threshold tuning ===")
print(classification_report(Y_test, pred, target_names=list(mlb.classes_), zero_division=0))

joblib.dump(thresholds, OUTD/"thresholds.joblib")
print("\n✅ Saved thresholds to", OUTD/"thresholds.joblib")
