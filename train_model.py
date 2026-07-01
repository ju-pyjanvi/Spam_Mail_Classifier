"""
Spam Mail Detector - Model Training Script
----------------------------------------------
Loads the SMS Spam Collection dataset, cleans the text, converts it to
TF-IDF features, trains a few classifiers, picks the best one, evaluates
it, and saves the trained model + vectorizer to disk so the FastAPI
backend can load them instantly.

Run:  python train_model.py
"""

import re
import string
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42

# make sure stopwords are available (no-op if already downloaded)
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
# The classic UCI SMS Spam Collection ships as v1 (label), v2 (text)
# plus 3 unnamed junk columns -> encoding is latin-1, not utf-8.
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
df = df.dropna(subset=["text"]).drop_duplicates().reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# ---------------------------------------------------------------
# 2. TEXT PREPROCESSING
# ---------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # urls
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"\d+", " ", text)                        # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)
df["char_count"] = df["text"].apply(len)

print("\nSample cleaned text:")
print(df[["text", "clean_text"]].head(3).to_string())

# ---------------------------------------------------------------
# 3. QUICK EDA
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label", palette=["#8B5E3C", "#E97EA8"])
plt.title("Ham vs Spam Count")
plt.savefig("eda_class_balance.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="char_count", hue="label", bins=40, kde=True,
             palette=["#8B5E3C", "#E97EA8"])
plt.title("Message Length Distribution (chars)")
plt.xlim(0, 400)
plt.savefig("eda_message_length.png", dpi=120, bbox_inches="tight")
plt.close()

print("Saved EDA plots: eda_class_balance.png, eda_message_length.png")

# ---------------------------------------------------------------
# 4. FEATURE EXTRACTION (TF-IDF) + LABELS
# ---------------------------------------------------------------
X_text = df["clean_text"]
y = (df["label"] == "spam").astype(int)  # spam=1, ham=0

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------------------------------------------------------------
# 5. TRAIN MULTIPLE MODELS AND COMPARE
# ---------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results[name] = {
        "model": model, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "preds": preds,
    }
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

# ---------------------------------------------------------------
# 6. PICK THE BEST MODEL (by F1 - spam detection cares about balance)
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f})")

best_preds = results[best_name]["preds"]
print("\nClassification Report:\n",
      classification_report(y_test, best_preds, target_names=["ham", "spam"]))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 7. SAVE MODEL, VECTORIZER + METRICS FOR THE BACKEND
# ---------------------------------------------------------------
joblib.dump(best_model, "spam_model.pkl")
joblib.dump(vectorizer, "spam_vectorizer.pkl")

metrics_summary = {
    "best_model_name": best_name,
    "all_results": {
        name: {k: v for k, v in r.items() if k not in ("model", "preds")}
        for name, r in results.items()
    },
}
joblib.dump(metrics_summary, "spam_metrics.pkl")

print("\nSaved: spam_model.pkl, spam_vectorizer.pkl, spam_metrics.pkl")
print("Done! Now run: uvicorn backend:app --reload   (then open frontend/index.html)")
