# ✉️ Spam Mail Detector

Classifies SMS/email messages as spam or ham, with a brown & pink HTML/CSS/JS
frontend backed by a FastAPI model server.

## Files
- `spam.csv` — SMS Spam Collection dataset (5,572 messages, from UCI/Kaggle)
- `train_model.py` — cleans text (lowercase, strip urls/emails/numbers/punctuation,
  remove stopwords), builds TF-IDF features, trains Naive Bayes / Logistic Regression /
  Linear SVM, picks the best by F1 score, evaluates it, saves model + vectorizer
- `backend.py` — FastAPI server exposing:
  - `POST /api/predict` — send `{"text": "..."}`, get back label, confidence,
    spam probability, cleaned text, and top TF-IDF signal words
  - `GET /api/metrics` — training metrics for all 3 models
  - Also serves `frontend/` as static files at `/`
- `frontend/index.html`, `frontend/style.css`, `frontend/script.js` — the brown & pink UI
- `requirements.txt`
- Generated after training: `spam_model.pkl`, `spam_vectorizer.pkl`, `spam_metrics.pkl`,
  plus EDA plots (`eda_class_balance.png`, `eda_message_length.png`, `confusion_matrix.png`)

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates the .pkl files the backend needs)
python train_model.py

# 3. Start the backend (also serves the frontend)
uvicorn backend:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser — that's it, frontend and backend
in one place. (You can also open `frontend/index.html` directly as a file; it talks to
the backend at `http://localhost:8000` via CORS, which is already enabled.)

## Model results (on this dataset)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 96.7% | 99.0% | 74.8% | 85.2% |
| Logistic Regression | 95.6% | 98.9% | 65.6% | 78.9% |
| **Linear SVM (best)** | **98.2%** | **98.3%** | **87.0%** | **92.3%** |

Linear SVM is auto-selected because it gets the best F1 (best balance of catching
spam without misflagging real messages — recall was the weak point for the other two).

## Notes
- `LinearSVC` has no `predict_proba`, so the backend converts its decision score to a
  0–1 confidence with a sigmoid — this is an approximation, not a calibrated probability,
  but works well for the UI's confidence meter.
- If you retrain, just restart `uvicorn` to pick up the new `.pkl` files.
