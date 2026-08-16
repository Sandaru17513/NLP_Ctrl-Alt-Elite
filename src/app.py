"""
Spam classifier demo — cit-24-01-0014
Loads both the Logistic Regression and LSTM models and serves a single page
where a message can be classified by both at once.

Run from the src/ folder:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import re
import sys
import pickle

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from joblib import load as joblib_load

# TensorFlow/Keras is only needed for the LSTM model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import the real preprocessing pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'cit-24-01-0014'))
from preprocessing import preprocess_text


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # .../src
MODELS_DIR = os.path.join(BASE_DIR, "..", "models", "cit-24-01-0014")  # .../models/cit-24-01-0014
LR_DIR = os.path.join(MODELS_DIR, "logistic regression")
LSTM_DIR = os.path.join(MODELS_DIR, "lstm")

LABELS = {0: "Ham", 1: "Spam"}

# TODO: confirm this against the training notebook (cit-24-01-0014_lstm.ipynb).
# It must match the maxlen used when the LSTM was trained, or predictions
# will be wrong/inconsistent.
LSTM_MAX_LEN = 100

app = Flask(__name__, static_folder=BASE_DIR)

# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------
lr_model = joblib_load(os.path.join(LR_DIR, "logistic_regression_model.pkl"))
tfidf_vectorizer = joblib_load(os.path.join(LR_DIR, "tfidf_vectorizer.pkl"))

lstm_model = load_model(os.path.join(LSTM_DIR, "spam_lstm_model.keras"), compile=False)

with open(os.path.join(LSTM_DIR, "tokenizer.pkl"), "rb") as f:
    lstm_tokenizer = pickle.load(f)

lstm_threshold = float(np.load(os.path.join(LSTM_DIR, "best_threshold.npy")))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Use the real preprocessing pipeline from cit-24-01-0014."""
    return preprocess_text(text)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    raw_text = (data.get("text") or "").strip()

    if not raw_text:
        return jsonify({"error": "Please enter a message."}), 400

    cleaned = clean_text(raw_text)

    # --- Logistic Regression ---
    lr_vec = tfidf_vectorizer.transform([cleaned])
    lr_pred = int(lr_model.predict(lr_vec)[0])
    lr_proba = lr_model.predict_proba(lr_vec)[0]
    lr_confidence = float(lr_proba[lr_pred])

    # --- LSTM ---
    seq = lstm_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=LSTM_MAX_LEN, padding="post", truncating="post")
    lstm_raw = float(lstm_model.predict(padded, verbose=0)[0][0])
    lstm_pred = int(lstm_raw >= lstm_threshold)
    lstm_confidence = lstm_raw if lstm_pred == 1 else 1 - lstm_raw

    return jsonify({
        "logistic_regression": {
            "label": LABELS[lr_pred],
            "confidence": round(lr_confidence * 100, 2),
        },
        "lstm": {
            "label": LABELS[lstm_pred],
            "confidence": round(lstm_confidence * 100, 2),
            "threshold": round(lstm_threshold, 3),
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
