# 🛍️ Will Repeat? — Repeat-Purchase Prediction API

[![CI](https://github.com/kashishniranjan-ai/churn-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/kashishniranjan-ai/churn-ai/actions/workflows/ci.yml)

An ML service that answers one business question: **"which of these customers
will place another order next quarter?"**

It is trained on the real **UCI Online Retail** dataset (a UK gifts wholesaler,
2010–2011), served through a FastAPI REST API, explained with model-derived
drivers, and wrapped in a Streamlit dashboard.

> **Honest headline metric:** ROC-AUC **0.740** on a held-out 20% of customers
> (CV 0.753 ± 0.029). Because 56% of customers repeat anyway, the useful number
> for a real campaign is **precision at the top of the list**: contacting the
> top 10% of customers by score reaches buyers **1.72× more efficiently than
> random** (97% of them do repeat). Full metric table [below](#-how-good-is-the-model).

---

## 🧭 What it does

| Piece | File | Role |
|---|---|---|
| Feature engineering | `src/features.py` | Turns raw invoices into 12 leakage-safe, per-customer features |
| Training | `src/train_model.py` | CV model selection, threshold freezing, artifact serialization |
| API | `app/main.py` + `app/schemas.py` | Validation, inference, explanations, batch scoring |
| Dashboard | `frontend/dashboard.py` | UI over the API — predicts nothing on its own |
| Artifacts | `model/` | One pipeline + metrics + feature metadata (committed, so deploys need no retraining) |

**The prediction target:** a customer who placed at least one order before
`2011-09-01` is labelled `1` if they ordered again in
`[2011-09-01, 2011-12-01)` — a full quarter, so "repeat" is never defined on a
partial observation window.

**No leakage by construction:** every input feature is computed from
transactions strictly *before* the cutoff, including the returns rate. The
label is only ever read from the window *after* it. Splitting is by customer,
so the same person never appears in both train and test.

---

## 🚀 Quick start (local)

Requires Python 3.11+ (developed and verified on 3.13.2).

```bash
# 1. Install
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

# 2. Train — optional! model/ already contains working artifacts.
python src/download_data.py    # fetches the UCI xlsx (~23 MB)
python src/clean.py            # cleans + builds data/processed/retail.db
python src/train_model.py      # trains and writes model/*
#   (or all of the above at once: python run_pipeline.py)

# 3. Run the API
uvicorn app.main:app --reload --port 8000

# 4. Run the dashboard (separate terminal)
streamlit run frontend/dashboard.py
```

- API: <http://localhost:8000> · interactive docs: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>

The API refuses to start if `model/customer_prediction_pipeline.joblib` is
missing, and fails at startup if the Pydantic schema's feature order drifts from
`model/feature_metadata.json` — a mismatch would silently scramble predictions,
so the service crashes instead of serving wrong scores.

---

## 🔗 API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info + links |
| `GET` | `/health` | Liveness: loaded model, version, headline metrics |
| `GET` | `/metrics` | Full training report — CV scores, test metrics, baselines, lift |
| `GET` | `/features` | Importances, **signed** effects, training distributions |
| `POST` | `/predict` | One customer → probability, verdict, drivers |
| `POST` | `/predict/batch` | Up to 500 customers in one vectorised pass |
| `GET` | `/docs` | Swagger UI · `/redoc` for ReDoc |

### Input

All 12 fields are aggregate behaviour for one customer, measured at the scoring
date. Example — an engaged buyer who has been quiet for ~5 weeks:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "tenure_days": 250, "recency_days": 40, "months_active": 5,
        "max_gap_days": 62, "frequency": 6, "monetary": 1850.0,
        "avg_order_value": 308.33, "total_items": 720,
        "avg_items_per_order": 120.0, "distinct_products": 45,
        "avg_unit_price": 2.90, "returns_rate": 0.05
      }'
```

<details>
<summary><b>Response</b> (actual output, not a mock-up)</summary>

```json
{
  "customer_features": { "tenure_days": 250, "recency_days": 40, "...": "..." },
  "prediction": "Will Repeat",
  "will_repeat": true,
  "repeat_probability": 0.8217,
  "confidence": 0.8217,
  "engagement_tier": "Highly Likely",
  "key_drivers": [
    "Months with purchase activity is higher than the training average (5.00 vs 2.46), which raises the repeat-purchase estimate.",
    "Days since last purchase is lower than the training average (40.00 vs 91.92), which raises the repeat-purchase estimate.",
    "Return / cancellation rate is lower than the training average (0.05 vs 0.15), which lowers the repeat-purchase estimate."
  ],
  "threshold": 0.3,
  "timestamp": "2026-09-29T08:36:02.217607+00:00"
}
```

</details>

Explanations are **computed from the served artifacts**, never from a hard-coded
list: magnitude comes from the model's stored importances, direction from its
fitted coefficients, and "typical" from the stored training distributions. If a
feature were unimportant, it could not show up as a driver.

### The 12 features

| Field | Meaning |
|---|---|
| `tenure_days` | Days from first order to the scoring cutoff |
| `recency_days` | Days from last order to the cutoff |
| `months_active` | Distinct calendar months containing ≥1 order |
| `max_gap_days` | Longest silent stretch inside their history |
| `frequency` | Number of distinct orders |
| `monetary` | Total spend (GBP) |
| `avg_order_value` | Spend ÷ orders |
| `total_items` | Units purchased in total |
| `avg_items_per_order` | Basket size in units |
| `distinct_products` | Distinct product codes bought |
| `avg_unit_price` | Mean price paid per unit |
| `returns_rate` | Cancellations ÷ (cancellations + orders), on [0, 1] |

Batch endpoint: `POST /predict/batch` with `{"customers": [ … ]}` — same fields,
max 500 per call, returns `{"count": N, "predictions": [...]}`.

More examples, expected responses and failure cases:
**[docs/API_TESTING.md](docs/API_TESTING.md)** · importable collection:
`docs/postman_collection.json`.

---

## 📊 How good is the model?

**Setup:** 3,314 customers · 12 features · 56.3% repeat · stratified 80/20 split
by customer · 5-fold stratified CV for selection · decision threshold `0.30`
frozen on out-of-fold predictions (the test set was touched once, at the end).

**Model selection — CV ROC-AUC on the training split only:**

| Candidate | CV ROC-AUC |
|---|---|
| **LogisticRegression** ← selected | **0.7534** |
| ExtraTreesClassifier | 0.7454 |
| GradientBoostingClassifier | 0.7438 |
| RandomForestClassifier | 0.7408 |

**Held-out test set (663 customers):**

| Metric | Model | Trivial baseline |
|---|---|---|
| ROC-AUC | **0.740** | 0.500 |
| Accuracy | 62.4% | 56.3% — "always say will repeat" |
| Recall | 88.5% | 100% — same trivial rule |
| F1 | 0.726 | **0.720 — same trivial rule** |

> **Read this honestly.** F1 of 0.726 looks decent until you notice that
> "predict *everyone* repeats" scores 0.720. With a 56% positive rate, F1 and
> accuracy are nearly useless as headlines — which is exactly why a previous
> version of this project bragged about numbers that meant nothing.
>
> What actually creates value in a retention campaign with limited budget is
> **ranking**. Measured on the same held-out set:

| Contact the top… | Customers | Precision | Random | **Lift** | Repeaters reached |
|---|---|---|---|---|---|
| 10% | 66 | **97.0%** | 56.3% | **1.72×** | 17.2% |
| 20% | 133 | 86.5% | 56.3% | 1.54× | 30.8% |
| 30% | 199 | 80.9% | 56.3% | 1.44× | 43.2% |

Email the top decile and 97% of them buy again, versus 56% if you picked names
at random. Reproduce any of this with `GET /metrics`.

---

## ⚠️ Limitations — read before trusting this

1. **One historical snapshot** (Dec 2010 – Nov 2011, UK gift wholesale). Scores
   are a *relative ranking*, not calibrated 2026 probabilities. Recalibrate on
   current data before real spend decisions.
2. **Random split, not a temporal holdout.** Customers are split randomly, which
   is weaker than predicting a genuinely later period; reported AUC may be
   optimistic. A next step is a rolling-origin backtest.
3. **One quarter horizon, one cutoff.** "Repeat" means Sept–Nov 2011 specifically.
   A customer who returns in month 5 is labelled negative — a censoring choice,
   not a fact about them.
4. **`returns_rate` has a *positive* coefficient** (+0.04): customers who cancel
   more actually repeat more here, because cancellation volume tracks order
   volume. That is correlation, not causation — do not build a "reduce returns"
   retention policy on it.
5. **A linear model won.** Logistic regression beat three ensembles, so
   interactions are under-modelled; richer features (categories, seasonality,
   geography) could flip the winner.
6. **Propensity ≠ uplift.** The model finds people likely to buy — many of whom
   would have bought anyway. Measuring true campaign lift needs a holdout group.
7. **Cost-blind.** All errors are weighted equally; no margin or send-cost is in
   the objective.
8. **Coverage:** transactions with no `CustomerID` are dropped during cleaning —
   **135,080 of 541,909 raw rows (24.9%)** — so a quarter of all line items, and
   any anonymous buyer, is invisible to the model.

---

## 🧪 Tests

```bash
python -m pytest tests/ -v        # 31 tests, ~5s
```

Covered: artifact loading, **feature-order validation at startup**, `/health`
and `/metrics` payloads, single + batch prediction, threshold boundary behaviour
(probability just under/over the frozen threshold), probability↔verdict
consistency, rejection of missing and out-of-range fields (`422`), batch size
limits, acceptance of unknown extra keys so older clients stay compatible,
drivers grounded in real features, and **inference on real customers taken from
`retail.db`**.

### CI

`.github/workflows/ci.yml` runs on every push to `main`, on every pull request,
and on demand:

| Step | What it actually proves |
|---|---|
| Artifacts load | `model/*` is committed, non-empty and loadable — and scikit-learn's `InconsistentVersionWarning` is escalated to an error, so artifact/pin drift fails the build instead of quietly degrading explanations |
| App boots | Real lifespan startup through the production code path: `/health` says `healthy`, `/metrics` reports ROC-AUC > 0.5, `/features` order matches `FEATURE_ORDER` |
| Test suite | `python -m pytest tests/ -v` — **30 passed, 1 skipped** |
| API contract | Newman runs `docs/postman_collection.json` against a live server — `continue-on-error` for now, see below |

CI never downloads the 23 MB dataset: `data/processed/` is gitignored, so the one
DB-backed test (`test_predicts_real_customers`) skips itself and the count drops
from the local 31 to 30 + 1 skipped. That gap is deliberate, but it is also how a
regression could hide — if the skip count ever changes, the suite changed.

The Newman job is informational rather than blocking because the collection has
never been executed end to end in the authoring environment (Node isn't
installed). Delete its `continue-on-error` line once it has been seen green and
it becomes the contract gate.

---

## 🐳 Docker

```bash
docker compose up --build
#   API       http://localhost:8000/docs
#   Dashboard http://localhost:8501   (reaches the API as http://api:8000)
```

Or single images — `api` is the default (last) build stage:

```bash
docker build -t churn-api .                        # FastAPI + model artifacts
docker build --target dashboard -t churn-ui .      # Streamlit
docker run -p 8000:8000 -e PORT=8000 churn-api
```

Notes: the image runs as a non-root user, installs only `requirements.txt`, and
copies `app/` + `model/` — inference needs neither the training code nor the
data, so the image stays small. **Retraining does not update a running
container**: rebuild the image after `src/train_model.py`.

---

## ☁️ Deployment

### Render (recommended — `render.yaml` is committed)

1. Push this repo to GitHub.
2. Render → **New → Blueprint** → select the repo. Render reads `render.yaml`.
3. Set `ALLOWED_ORIGINS` to wherever the dashboard will run (a comma-separated
   list; `http://localhost:8501` is fine for local-only use).
4. Deploy. `healthCheckPath: /health` gates the rollout.

It uses the **native Python runtime** on the free plan (`pip install -r
requirements.txt`, then `uvicorn … --port $PORT`). The trained artifacts are in
git under `model/`, so **the build never downloads data or retrains** — deploys
are fast and reproducible. To use the Docker image instead (paid plans), swap
`runtime: python` for the commented `env: docker` block in `render.yaml`.

Free-tier behaviour worth knowing: instances **spin down when idle**, so the
first request after a quiet period pays a cold start (~1–2 s — the pipeline load
is the bulk of it), and any local filesystem writes are discarded on redeploy
(this service writes nothing, which is why that's safe).

### Railway / Fly.io

Point them at the repo. The default build stage is the API, `PORT` is read from
the environment, and the `HEALTHCHECK` in the Dockerfile probes `/health`.

### Google Cloud Run

```bash
docker build -t gcr.io/PROJECT_ID/repeat-api .
docker push gcr.io/PROJECT_ID/repeat-api
gcloud run deploy repeat-api \
  --image gcr.io/PROJECT_ID/repeat-api \
  --platform managed --region asia-south1 \
  --allow-unauthenticated --port 8000
```

### Verify any deployment

```bash
curl -s $URL/health | python -m json.tool   # expect "model_loaded": true
curl -s $URL/metrics  | python -m json.tool # expect the metrics in this README
```

---

## 🔧 Configuration

Copy `.env.example` to `.env`; **every value has a working default**, so the app
runs with no env file at all.

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `MODEL_DIR` | API | `./model` next to the code | Where the three artifacts live |
| `ALLOWED_ORIGINS` | API | `http://localhost:8501,http://127.0.0.1:8501` | CORS browser origins |
| `PORT` | API / dashboard | `8000` / `8501` | Listen port (injected by Render) |
| `API_URL` | dashboard | `http://localhost:8000` | Where the UI finds the API |

`MODEL_DIR` is deliberately **unset** in `render.yaml`: `app/main.py` derives it
from its own file location, which survives a change of working directory.

There are no secrets in this service, which is also why there is no
authentication — see the roadmap if you deploy it publicly.

---

## 📁 Layout

```
churn-ai-app/
├── src/
│   ├── download_data.py     # UCI xlsx -> data/raw
│   ├── clean.py             # cleaning -> data/processed/clean_retail.csv + retail.db
│   ├── create_db.py         # SQLite load
│   ├── analysis.py          # cohort / RFM / market-basket exploration
│   ├── features.py          # ⭐ leakage-safe per-customer feature builder
│   └── train_model.py       # ⭐ CV selection, threshold freezing, serialization
├── app/
│   ├── main.py              # ⭐ FastAPI service
│   └── schemas.py           # ⭐ Pydantic models; defines FEATURE_ORDER
├── frontend/dashboard.py    # Streamlit UI (API-only, no local inference)
├── model/
│   ├── customer_prediction_pipeline.joblib   # one artifact: scaler + classifier
│   ├── training_metrics.json                 # metrics, baselines, lift
│   └── feature_metadata.json                 # order, stats, importances, effects
├── tests/test_api.py        # 31 tests
├── sql/                     # rfm_analysis.sql, cohort_retention.sql
├── docs/
│   ├── insights.md          # data-analysis findings
│   ├── API_TESTING.md       # endpoint walkthrough + expected responses
│   └── postman_collection.json
├── .github/workflows/ci.yml # CI: artifact gates + boot check + pytest (+ Newman)
├── Dockerfile               # targets: api (default) | dashboard
├── docker-compose.yml
├── render.yaml
└── run_pipeline.py          # download → clean → db → analysis → train
```

---

## 🩺 Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Startup error naming a feature | `FEATURE_ORDER` in `app/schemas.py` ≠ `feature_metadata.json`. Retraining reordered features, or the schema drifted. Re-run training or revert the schema. |
| `/health` says `"status": "degraded"`, `model_loaded: false` | The pipeline never loaded — `model/customer_prediction_pipeline.joblib` missing → `python src/train_model.py`. (In practice the app refuses to boot without it, so this mostly appears in tests where startup was skipped.) |
| `404` from `/metrics` or `/features` | `training_metrics.json` / `feature_metadata.json` not next to `MODEL_DIR`. `/predict` still works — the model alone is enough to score. |
| `InconsistentVersionWarning` on boot | Artifact written by a different scikit-learn minor. `requirements.txt` pins `scikit-learn==1.8.0`; match it or retrain. |
| `422` on a valid-looking payload | A required field is absent or outside its bounds — the response body lists which. Check names and ranges against `/docs`. |
| Sent a `customer_id` / extra key and it "vanished" | Intentional: unknown keys are ignored so clients can add fields without breaking. Only the 12 known features reach the model. |
| Every test "skipped" | Artifacts or `data/processed/retail.db` absent — training is a prerequisite for the real-data tests. |
| Dashboard shows "API offline" | `API_URL` wrong. Inside `docker compose` it must be `http://api:8000`, not `localhost`. |
| CORS error in a browser | Add the dashboard's origin to `ALLOWED_ORIGINS`. The Streamlit server talks to the API server-side and is unaffected — this only matters for browser-based clients. |

---

## 🗺️ What I'd do next

1. **Temporal backtest** — score earlier months, evaluate on later ones; the
   honest successor to a random split.
2. **Calibration** — `CalibratedClassifierCV`, so "82% likely" means 82% of such
   customers repeat.
3. **Uplift testing** — randomised holdout to measure whether contact *changes*
   behaviour rather than just identifying buyers.
4. **Richer features** — product categories, order cadence, seasonality, country;
   then re-run the model selection that currently favours logistic regression.
5. **Ops** — API auth + rate limiting, prediction logging, drift monitoring
   against `feature_stats` in `feature_metadata.json`. CI runs the suite on
   every push; what is left is promoting the Newman contract job to a blocking
   check and testing more than the single Python version (3.13) these artifacts
   were verified against, since the README only claims a 3.11+ floor.

---

## 📄 License

MIT. Data: UCI Online Retail II data set (available under its original terms).


