import json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA = Path("data/dataset/train_all.jsonl")
OUTD = Path("ml/artifacts/tfidf_baseline")
OUTD.mkdir(parents=True, exist_ok=True)

texts=[]
labels=[]
contract_ids=[]
with DATA.open("r",encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        texts.append(r["text"])
        labels.append(r.get("labels") or [])
        contract_ids.append(r.get("contract_id",""))

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

# ✅ stratify to ensure SYNTH samples appear in test
is_synth = [1 if cid == "SYNTH" else 0 for cid in contract_ids]

X_train, X_test, Y_train, Y_test = train_test_split(
    texts,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=is_synth
)

# Char ngrams ممتازة للعربي + OCR noise + paraphrasing
vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    min_df=2,
    max_features=200000
)

Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = OneVsRestClassifier(
    LogisticRegression(max_iter=400, class_weight="balanced")
)
clf.fit(Xtr, Y_train)

pred = clf.predict(Xte)

print("Labels:", len(mlb.classes_))
print(classification_report(Y_test, pred, target_names=list(mlb.classes_), zero_division=0))

joblib.dump(vec, OUTD/"vectorizer.joblib")
joblib.dump(clf, OUTD/"model.joblib")
joblib.dump(mlb, OUTD/"mlb.joblib")
print("✅ Saved artifacts to", OUTD)