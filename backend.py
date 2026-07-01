
import re
import string
import math

import joblib
import nltk
from nltk.corpus import stopwords

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

app = FastAPI(title="Spam Mail Detector API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("spam_vectorizer.pkl")
metrics = joblib.load("spam_metrics.pkl")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))



class MessageIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    label: str              # "spam" or "ham"
    is_spam: bool
    confidence: float       # 0-1, confidence in the predicted label
    spam_probability: float # 0-1, raw probability/score of being spam
    cleaned_text: str
    top_signal_words: list[str]  # words in the message with the highest TF-IDF weight



@app.get("/api/health")
def health():
    return {"status": "ok", "model": metrics["best_model_name"]}


@app.get("/api/metrics")
def get_metrics():
    """Return training metrics so the frontend can show model performance."""
    return metrics


@app.post("/api/predict", response_model=PredictionOut)
def predict(message: MessageIn):
    raw = message.text
    cleaned = clean_text(raw)

    vec = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])  # 1 = spam, 0 = ham

    # LinearSVC has no predict_proba -> use decision_function + sigmoid
    if hasattr(model, "predict_proba"):
        spam_prob = float(model.predict_proba(vec)[0][1])
    else:
        score = float(model.decision_function(vec)[0])
        spam_prob = sigmoid(score)

    confidence = spam_prob if pred == 1 else (1 - spam_prob)


    feature_names = vectorizer.get_feature_names_out()
    row = vec.toarray()[0]
    top_indices = row.argsort()[::-1][:5]
    top_words = [feature_names[i] for i in top_indices if row[i] > 0]

    return PredictionOut(
        label="spam" if pred == 1 else "ham",
        is_spam=bool(pred),
        confidence=round(confidence, 4),
        spam_probability=round(spam_prob, 4),
        cleaned_text=cleaned,
        top_signal_words=top_words,
    )
