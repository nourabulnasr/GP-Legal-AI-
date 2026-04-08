import json
from pathlib import Path
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score

DATA = Path("data/dataset/train_all.jsonl")

rows = []
with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

# split: put 20% of SYNTH into test, and 20% of REAL into test
synth = [r for r in rows if r.get("contract_id") == "SYNTH"]
real  = [r for r in rows if r.get("contract_id") != "SYNTH"]

# ✅ IMPORTANT: shuffle before taking first 20% as test
random.seed(42)
random.shuffle(synth)
random.shuffle(real)

def split_20(lst):
    n = max(1, int(0.2 * len(lst)))
    return lst[n:], lst[:n]  # train, test

train_s, test_s = split_20(synth)
train_r, test_r = split_20(real)

train = train_s + train_r
test  = test_s + test_r

X_train = [r["text"] for r in train]
X_test  = [r["text"] for r in test]

y_train_raw = [r.get("labels") or [] for r in train]
y_test_raw  = [r.get("labels") or [] for r in test]

# ✅ sanity check: any label in test not present in train?
train_labels = set(lb for labs in y_train_raw for lb in labs)
test_labels  = set(lb for labs in y_test_raw  for lb in labs)
missing_in_train = sorted(test_labels - train_labels)

print("Test size:", len(test), "SYNTH in test:", sum(1 for r in test if r.get("contract_id") == "SYNTH"))
print("Unique labels in train:", len(train_labels), "| in test:", len(test_labels))

if missing_in_train:
    print("⚠️ WARNING: Labels in TEST but NOT in TRAIN (evaluation is not valid for these):")
    for m in missing_in_train:
        print("  -", m)
else:
    print("✅ OK: All test labels exist in train.")

# train model
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(y_train_raw)
Y_test  = mlb.transform(y_test_raw)

vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=200000
)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = OneVsRestClassifier(
    LogisticRegression(max_iter=400, class_weight="balanced")
)
clf.fit(Xtr, Y_train)

# proba
probs = clf.predict_proba(Xte)
if isinstance(probs, list):
    probs = np.vstack([p[:, 1] for p in probs]).T

# fixed threshold = 0.5 (fair eval)
pred = (probs >= 0.5).astype(int)

print("Labels:", len(mlb.classes_))
print(classification_report(Y_test, pred, target_names=list(mlb.classes_), zero_division=0))

micro = f1_score(Y_test, pred, average="micro", zero_division=0)
macro = f1_score(Y_test, pred, average="macro", zero_division=0)
print("F1 micro:", round(micro, 3), "F1 macro:", round(macro, 3))