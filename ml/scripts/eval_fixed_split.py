import json, joblib
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report

DATA = Path("data/dataset/train_all.jsonl")
OUTD = Path("ml/artifacts/tfidf_baseline")

vec = joblib.load(OUTD/"vectorizer.joblib")
clf = joblib.load(OUTD/"model.joblib")
mlb = joblib.load(OUTD/"mlb.joblib")
thresholds = joblib.load(OUTD/"thresholds.joblib")

rows=[]
with DATA.open("r",encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

# split: put 20% of SYNTH into test, and 20% of REAL into test
synth = [r for r in rows if r.get("contract_id")=="SYNTH"]
real  = [r for r in rows if r.get("contract_id")!="SYNTH"]

def split_20(lst):
    n = max(1, int(0.2*len(lst)))
    return lst[n:], lst[:n]  # train, test

train_s, test_s = split_20(synth)
train_r, test_r = split_20(real)

train = train_s + train_r
test  = test_s + test_r

X_test = [r["text"] for r in test]
Y_test = mlb.transform([r.get("labels") or [] for r in test])

Xte = vec.transform(X_test)
probs = clf.predict_proba(Xte)
if isinstance(probs, list):
    probs = np.vstack([p[:,1] for p in probs]).T

pred = (probs >= thresholds).astype(int)

print("Test size:", len(test), "SYNTH in test:", sum(1 for r in test if r.get("contract_id")=="SYNTH"))
print(classification_report(Y_test, pred, target_names=list(mlb.classes_), zero_division=0))
