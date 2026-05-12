"""
AI Supply Chain Risk Predictor - ML Model
==========================================
Generates synthetic data, trains a Random Forest classifier,
and saves the model + scaler for the prediction API.

Features:
  - supplier_reliability: 0–100 (how reliable the supplier is historically)
  - demand_level:         0–100 (current demand intensity)
  - transport_delay:      0–30  (days of transport delay)
  - weather_impact:       0–10  (severity of weather disruption)
  - inventory_level:      0–100 (current inventory as % of target)

Target (risk_level):
  0 = Low Risk
  1 = Medium Risk
  2 = High Risk
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import json
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Create a realistic synthetic supply-chain dataset.
    Risk is a non-linear combination of all features so the model
    must actually learn interactions — not just a trivial threshold.
    """
    supplier_reliability = np.random.beta(5, 2, n_samples) * 100        # skewed high
    demand_level         = np.random.normal(50, 20, n_samples).clip(0, 100)
    transport_delay      = np.random.exponential(5, n_samples).clip(0, 30)
    weather_impact       = np.random.exponential(2, n_samples).clip(0, 10)
    inventory_level      = np.random.normal(60, 20, n_samples).clip(0, 100)

    # --- composite risk score (domain-expert heuristic) ---
    risk_score = (
          (100 - supplier_reliability) * 0.30   # low reliability → higher risk
        + demand_level                 * 0.25   # high demand → strain
        + transport_delay              * 1.50   # every extra day hurts
        + weather_impact               * 2.50   # weather is highly disruptive
        + (100 - inventory_level)      * 0.20   # low inventory → vulnerability
    )

    # add realism noise
    risk_score += np.random.normal(0, 5, n_samples)

    # categorise: Low <40 · Medium 40–70 · High >70  (percentile-based)
    low_thresh  = np.percentile(risk_score, 40)
    high_thresh = np.percentile(risk_score, 70)

    risk_level = np.where(
        risk_score < low_thresh, 0,
        np.where(risk_score < high_thresh, 1, 2)
    )

    df = pd.DataFrame({
        "supplier_reliability": supplier_reliability.round(2),
        "demand_level":         demand_level.round(2),
        "transport_delay":      transport_delay.round(2),
        "weather_impact":       weather_impact.round(2),
        "inventory_level":      inventory_level.round(2),
        "risk_score":           risk_score.round(2),
        "risk_level":           risk_level,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "supplier_reliability",
    "demand_level",
    "transport_delay",
    "weather_impact",
    "inventory_level",
]
TARGET_COL = "risk_level"
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


def train(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Pipeline: scaler + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    print(f"\n{'='*55}")
    print(f"  Model Accuracy  : {acc*100:.2f}%")
    print(f"  CV Mean±Std     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

    # Feature importance
    rf = pipeline.named_steps["clf"]
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    print("Feature Importances:", importances)

    # Persist artefacts
    model_path   = os.path.join(SAVE_DIR, "risk_model.joblib")
    meta_path    = os.path.join(SAVE_DIR, "model_meta.json")

    joblib.dump(pipeline, model_path)

    meta = {
        "accuracy":           round(acc, 4),
        "cv_mean":            round(float(cv_scores.mean()), 4),
        "cv_std":             round(float(cv_scores.std()), 4),
        "feature_cols":       FEATURE_COLS,
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "risk_labels":        RISK_LABELS,
        "n_train_samples":    len(X_train),
        "n_test_samples":     len(X_test),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model  → {model_path}")
    print(f"Saved meta   → {meta_path}")
    return pipeline, meta


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICTION HELPER (used by the API)
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    model_path = os.path.join(SAVE_DIR, "risk_model.joblib")
    meta_path  = os.path.join(SAVE_DIR, "model_meta.json")
    pipeline   = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return pipeline, meta


def predict_risk(features: dict, pipeline, meta: dict) -> dict:
    """
    features: dict with keys matching FEATURE_COLS
    Returns a rich prediction result dict.
    """
    row = pd.DataFrame([[features[c] for c in meta["feature_cols"]]],
                       columns=meta["feature_cols"])

    label_idx   = int(pipeline.predict(row)[0])
    proba       = pipeline.predict_proba(row)[0]
    confidence  = round(float(proba[label_idx]) * 100, 1)
    risk_label  = meta["risk_labels"][str(label_idx)]

    # Build per-class probabilities
    class_probs = {
        meta["risk_labels"][str(i)]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
    }

    # Simple rule-based explanation
    explanations = _build_explanation(features, risk_label)

    return {
        "risk_level":    label_idx,
        "risk_label":    risk_label,
        "confidence":    confidence,
        "class_probs":   class_probs,
        "explanations":  explanations,
        "feature_importance": meta["feature_importances"],
    }


def _build_explanation(features: dict, risk_label: str) -> list[str]:
    msgs = []
    if features["supplier_reliability"] < 40:
        msgs.append("⚠️ Supplier reliability is critically low — consider backup suppliers.")
    elif features["supplier_reliability"] < 65:
        msgs.append("🔶 Moderate supplier reliability — monitor closely.")

    if features["transport_delay"] > 15:
        msgs.append(f"🚛 Transport delay of {features['transport_delay']} days is severely impacting the chain.")
    elif features["transport_delay"] > 7:
        msgs.append(f"🚛 Transport delay of {features['transport_delay']} days is above normal thresholds.")

    if features["weather_impact"] > 7:
        msgs.append("🌩️ Extreme weather conditions are disrupting logistics routes.")
    elif features["weather_impact"] > 4:
        msgs.append("🌧️ Moderate weather disruption detected.")

    if features["demand_level"] > 80:
        msgs.append("📈 Demand spike detected — inventory buffers may be insufficient.")

    if features["inventory_level"] < 25:
        msgs.append("📦 Critically low inventory — risk of stockout is imminent.")
    elif features["inventory_level"] < 45:
        msgs.append("📦 Inventory is below safe operating levels.")

    if not msgs:
        msgs.append("✅ All supply chain parameters are within acceptable ranges.")

    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic dataset …")
    df = generate_dataset(n_samples=5000)
    df.to_csv(os.path.join(SAVE_DIR, "supply_chain_data.csv"), index=False)
    print(f"Dataset saved ({len(df)} rows).")
    print("\nTraining model …")
    train(df)
    print("\nDone! ✓")
