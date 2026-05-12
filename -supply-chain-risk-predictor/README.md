# Supply Chain Risk Predictor

An ML-powered web application that predicts supply chain disruption risk using Gradient Boosting, served through a Flask REST API with an animated dark-theme dashboard.

---

## What It Does

- Accepts supplier and shipment data via a web form or CSV upload
- Runs it through a trained Gradient Boosting classifier
- Returns a risk score (0–100) with the top contributing risk factors
- Visualises results with animated Chart.js charts on a glassmorphism dashboard

---

## System Architecture

```
Browser (Frontend)
      │
      │  HTTP / JSON
      ▼
Flask REST API  (/api/predict, /api/predict/batch, /api/health)
      │
      │  Python
      ▼
ML Model Layer
  ├── sklearn Preprocessing Pipeline
  ├── GradientBoostingClassifier
  └── SHAP Explainer  →  top risk factors per prediction
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML Model | scikit-learn `GradientBoostingClassifier` |
| Explainability | SHAP |
| Data Processing | pandas, NumPy |
| Backend | Flask 3.x |
| Frontend | HTML, CSS, Vanilla JS |
| Charts | Chart.js |
| Serialisation | joblib |

---

## Dataset

Trained on the **DataCo Smart Supply Chain Dataset** (Kaggle) — 180k+ order and shipment records — augmented with synthetic features for geopolitical risk and port congestion.

**Input features used:**

| Feature | Description |
|---------|-------------|
| `supplier_reliability_score` | Historical on-time delivery rate |
| `lead_time_days` | Expected days from order to delivery |
| `lead_time_variance` | Std. deviation of past lead times |
| `country_risk_index` | Geopolitical + logistics risk index |
| `shipping_mode` | Air / Sea / Road / Rail |
| `order_quantity` | Units ordered |
| `port_congestion_flag` | Active congestion advisory (0 or 1) |
| `days_for_shipping_real` | Historical average shipping days |
| `benefit_per_order` | Revenue value of the order |
| `category_name` | Product category |

**Target:** `low_risk` (disruption probability < 30%) or `high_risk` (≥ 30%)

---

## Project Structure

```
supply-chain-risk-predictor/
├── backend/
│   ├── app.py              # Flask app and route registration
│   ├── routes/
│   │   └── predict.py      # Prediction endpoints
│   └── services/
│       └── predictor.py    # Model loading and inference
├── model/
│   ├── train.py            # Training script
│   ├── model.pkl           # Saved model
│   └── preprocessor.pkl    # Saved preprocessing pipeline
├── frontend/
│   ├── templates/
│   │   └── index.html      # Main dashboard
│   └── static/
│       ├── css/styles.css
│       └── js/app.js
├── requirements.txt
└── README.md
```

---

## Getting Started

**1. Clone and set up environment**

```bash
git clone https://github.com/<your-username>/supply-chain-risk-predictor.git
cd supply-chain-risk-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the model** (optional — pre-trained artefacts included)

```bash
python model/train.py
```

**3. Run the app**

```bash
python backend/app.py
```

Open `http://localhost:5000` in your browser.

---

## API

**Single prediction**

```bash
POST /api/predict
Content-Type: application/json

{
  "supplier_reliability_score": 0.65,
  "lead_time_days": 21,
  "country_risk_index": 0.72,
  "shipping_mode": "Sea",
  ...
}
```

Response:
```json
{
  "risk_score": 0.78,
  "risk_label": "high_risk",
  "top_factors": [
    {"feature": "country_risk_index", "impact": "+0.31"},
    {"feature": "lead_time_variance", "impact": "+0.22"}
  ]
}
```

**Batch prediction:** `POST /api/predict/batch` with a CSV file upload.

**Health check:** `GET /api/health`

## Areas of Improvement

This is a base-level project built for learning and resume purposes. Planned improvements include:

- Add a database (PostgreSQL) to store prediction history
- Improve model accuracy with more real-world training data
- Add user authentication for multi-user support
- Deploy to cloud (Render / Railway / AWS)
- Add automated model retraining when data drifts
- Write proper unit tests for all endpoints
- Add input validation and better error messages on the frontend
---

## License

MIT — see [LICENSE](LICENSE) for details.

---

