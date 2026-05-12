"""
AI Supply Chain Risk Predictor — Flask API
==========================================
Endpoints:
  GET  /              → serves the dashboard HTML
  GET  /api/health    → health check
  POST /api/predict   → predict risk from input features
  POST /api/train     → retrain the model (async-friendly)
  GET  /api/stats     → dataset + model statistics
  GET  /api/history   → recent predictions (in-memory log)
  GET  /api/trends    → synthetic trend data for charts
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime, timedelta
from collections import deque

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── path setup so we can import from model/ ──────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
FRONT_DIR = os.path.join(BASE_DIR, "..", "frontend")
sys.path.insert(0, MODEL_DIR)

from train_model import (
    generate_dataset, train, load_model, predict_risk, FEATURE_COLS
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(FRONT_DIR, "static"),
    template_folder=os.path.join(FRONT_DIR),
)
CORS(app)  # allow the standalone HTML frontend to call the API

# ── State ─────────────────────────────────────────────────────────────────────
prediction_history: deque = deque(maxlen=50)   # last 50 predictions
_pipeline = None
_meta     = None


def get_model():
    """Lazy-load model; auto-train if not found."""
    global _pipeline, _meta
    if _pipeline is None:
        model_path = os.path.join(MODEL_DIR, "risk_model.joblib")
        if not os.path.exists(model_path):
            print("[API] No model found — training now …")
            df = generate_dataset(5000)
            _pipeline, _meta = train(df)
        else:
            _pipeline, _meta = load_model()
            print("[API] Model loaded from disk.")
    return _pipeline, _meta


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONT_DIR, "index.html")


@app.route("/api/health")
def health():
    pipeline, meta = get_model()
    return jsonify({
        "status":   "ok",
        "model_accuracy": meta.get("accuracy"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # ── Validate ──────────────────────────────────────────────────────────────
    errors = []
    bounds = {
        "supplier_reliability": (0, 100),
        "demand_level":         (0, 100),
        "transport_delay":      (0, 30),
        "weather_impact":       (0, 10),
        "inventory_level":      (0, 100),
    }
    features = {}
    for field, (lo, hi) in bounds.items():
        val = data.get(field)
        if val is None:
            errors.append(f"Missing field: {field}")
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number")
            continue
        if not lo <= val <= hi:
            errors.append(f"{field} must be between {lo} and {hi}")
            continue
        features[field] = val

    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    # ── Predict ───────────────────────────────────────────────────────────────
    pipeline, meta = get_model()
    result = predict_risk(features, pipeline, meta)

    # log it
    entry = {
        "id":          str(uuid.uuid4())[:8],
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "features":    features,
        "risk_label":  result["risk_label"],
        "confidence":  result["confidence"],
    }
    prediction_history.appendleft(entry)

    return jsonify({
        "success":   True,
        "result":    result,
        "meta":      {
            "model_accuracy": meta.get("accuracy"),
            "cv_mean":        meta.get("cv_mean"),
        },
    })


@app.route("/api/train", methods=["POST"])
def retrain():
    """Retrain the model with a fresh synthetic dataset."""
    global _pipeline, _meta
    t0 = time.time()
    df = generate_dataset(5000)
    _pipeline, _meta = train(df)
    elapsed = round(time.time() - t0, 2)
    return jsonify({
        "success":  True,
        "message":  "Model retrained successfully",
        "accuracy": _meta["accuracy"],
        "elapsed_seconds": elapsed,
    })


@app.route("/api/stats")
def stats():
    pipeline, meta = get_model()
    return jsonify({
        "model": {
            "accuracy":            meta["accuracy"],
            "cv_mean":             meta["cv_mean"],
            "cv_std":              meta["cv_std"],
            "n_train_samples":     meta["n_train_samples"],
            "feature_importances": meta["feature_importances"],
        },
        "predictions_logged": len(prediction_history),
        "risk_distribution":   _risk_distribution(),
    })


@app.route("/api/history")
def history():
    return jsonify({"history": list(prediction_history)})


@app.route("/api/trends")
def trends():
    """
    Return 30-day synthetic trend data for all five metrics.
    In production this would come from a real time-series database.
    """
    np.random.seed(int(time.time()) % 1000)  # vary each call slightly
    days = 30
    dates = [(datetime.utcnow() - timedelta(days=days - i)).strftime("%b %d")
             for i in range(days)]

    def smooth(arr, w=4):
        return np.convolve(arr, np.ones(w) / w, mode="same").round(1).tolist()

    data = {
        "dates":                dates,
        "supplier_reliability": smooth(np.random.normal(72, 8, days).clip(20, 100)),
        "demand_level":         smooth(np.random.normal(55, 15, days).clip(0, 100)),
        "transport_delay":      smooth(np.random.exponential(6, days).clip(0, 30)),
        "weather_impact":       smooth(np.random.exponential(2.5, days).clip(0, 10)),
        "inventory_level":      smooth(np.random.normal(58, 12, days).clip(0, 100)),
        "risk_score":           smooth(np.random.normal(45, 18, days).clip(0, 100)),
    }

    # Supplier comparison (bar chart)
    suppliers = ["Supplier A", "Supplier B", "Supplier C", "Supplier D", "Supplier E"]
    data["supplier_comparison"] = {
        s: {
            "reliability": round(float(np.random.uniform(40, 95)), 1),
            "on_time_rate": round(float(np.random.uniform(55, 98)), 1),
            "defect_rate":  round(float(np.random.uniform(0.5, 8)), 2),
        }
        for s in suppliers
    }

    return jsonify(data)


# ── Risk distribution helper ──────────────────────────────────────────────────

def _risk_distribution():
    if not prediction_history:
        return {"Low": 40, "Medium": 35, "High": 25}   # demo defaults
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for p in prediction_history:
        counts[p["risk_label"]] = counts.get(p["risk_label"], 0) + 1
    return counts


# ── Boot ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AI Supply Chain Risk Predictor — API")
    print("=" * 55)
    get_model()                              # warm-up
    app.run(host="0.0.0.0", port=5000, debug=False)
