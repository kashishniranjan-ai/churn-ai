# API testing guide

Manual walkthrough for the Repeat-Purchase API: every endpoint, the exact
commands, the responses they should produce, and the failure modes worth
checking by hand.

- **Base URL (local):** `http://localhost:8000`
- **Interactive docs:** `/docs` (Swagger) · `/redoc` · raw schema `/openapi.json`
- **Automated suite:** `python -m pytest tests/ -v` → 31 tests, run on every
  push/PR by [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) (which
  reports 30 passed + 1 skipped, because the dataset is not in the repo)

Start the service first:

```bash
uvicorn app.main:app --reload --port 8000
```

An importable Postman collection with executable assertions for all of this:
**[`postman_collection.json`](postman_collection.json)** — import it, then set
the `base_url` collection variable if you are not testing localhost.

---

## 1. `GET /health` — is it up, and what did it load?

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Expect `200` with `status: "healthy"`, `model_loaded: true`,
`feature_count: 12`, `model_type`, `version`, and a small `training_metrics`
block — all read from the loaded artifacts at request time.

`"status": "degraded"` with `"model_loaded": false` means the pipeline never
loaded; check that `MODEL_DIR` points at a directory containing
`customer_prediction_pipeline.joblib`.

> The app deliberately **refuses to start** when that file is missing, so a
> deployed service that boots is a service that can predict.

## 2. `GET /metrics` — the honest scorecard

```bash
curl -s http://localhost:8000/metrics | python -m json.tool
```

`200` with the full training report: every candidate's CV score, `test_metrics`,
`baselines` (`majority_class_accuracy`, `always_predict_repeat_f1`) and
`campaign_metrics` (precision + lift at the top 10/20/30%). Returns `404` if
`training_metrics.json` is absent.

Use it to check the README's claims against whatever deployment you are looking
at — these numbers come from the artifact, not from prose.

## 3. `GET /features` — what drives a score

```bash
curl -s http://localhost:8000/features | python -m json.tool
```

`200` with the stored `feature_order`, `feature_stats` (mean/std/median/min/max
per feature), `feature_importances`, and **`feature_effects` — signed**, so the
sign tells you whether a higher value of that feature raises or lowers the
repeat probability. `effect_method` is `"linear_coefficient"` for the current
logistic model. `404` if `feature_metadata.json` is absent.

## 4. `POST /predict` — one customer

Engaged buyer, quiet for ~5 weeks → expect a high score:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_days":250,"recency_days":40,"frequency":6,"monetary":1850.0,
       "avg_order_value":308.33,"total_items":720,"avg_items_per_order":120.0,
       "distinct_products":45,"avg_unit_price":2.9,"months_active":5,
       "max_gap_days":62,"returns_rate":0.05}'
```

Actual response:

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

Swap in a one-off browser —

```json
{"tenure_days":20,"recency_days":18,"frequency":1,"monetary":95.0,
 "avg_order_value":95.0,"total_items":12,"avg_items_per_order":12.0,
 "distinct_products":3,"avg_unit_price":7.9,"months_active":1,
 "max_gap_days":0,"returns_rate":0.0}
```

— and the verdict flips to `"Will Not Repeat"` with a low probability.

Invariant to re-check on every response: `will_repeat == (repeat_probability >=
threshold)`, `confidence == repeat_probability`, `prediction` derived from
`will_repeat`, `key_drivers` non-empty, and every driver naming one of the 12
real features with the training average it actually stored.

## 5. `POST /predict/batch` — many customers, one vectorised pass

```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"customers":[
        {"tenure_days":250,"recency_days":40,"frequency":6,"monetary":1850.0,"avg_order_value":308.33,"total_items":720,"avg_items_per_order":120.0,"distinct_products":45,"avg_unit_price":2.9,"months_active":5,"max_gap_days":62,"returns_rate":0.05},
        {"tenure_days":20,"recency_days":18,"frequency":1,"monetary":95.0,"avg_order_value":95.0,"total_items":12,"avg_items_per_order":12.0,"distinct_products":3,"avg_unit_price":7.9,"months_active":1,"max_gap_days":0,"returns_rate":0.0}
      ]}'
```

`200` → `{"count": 2, "predictions": [...]}`, in request order, each entry
identical in shape to `/predict`. Between **1 and 500** customers per call;
outside that range the request is rejected before any inference runs. The engaged
customer must outscore the one-off browser — if it doesn't, something is broken.

For bigger lists, use the dashboard's **Batch upload** tab, which splits a CSV
into compliant chunks.

---

## Validation rules worth probing by hand

All 12 fields are required; `null` is rejected; wrong types are rejected.

| Field | Type | Bounds |
|---|---|---|
| `tenure_days`, `recency_days`, `max_gap_days` | int | 0 … 36 500 |
| `frequency` | int | 1 … 100 000 |
| `months_active` | int | 0 … 1 200 |
| `total_items` | int | 0 … 10 000 000 |
| `distinct_products` | int | 0 … 1 000 000 |
| `monetary`, `avg_order_value` | float | 0 … 100 000 000 |
| `avg_items_per_order` | float | 0 … 1 000 000 |
| `avg_unit_price` | float | 0 … 100 000 |
| `returns_rate` | float | 0 … 1 |

```bash
# missing fields -> 422
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{"tenure_days":250}'

# out of range -> 422 (returns_rate above 1)
curl -s -w "\nHTTP %{http_code}\n" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_days":250,"recency_days":40,"frequency":6,"monetary":1850.0,"avg_order_value":308.33,"total_items":720,"avg_items_per_order":120.0,"distinct_products":45,"avg_unit_price":2.9,"months_active":5,"max_gap_days":62,"returns_rate":1.5}'

# empty batch -> 422
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" -d '{"customers":[]}'

# unknown extra key -> 200 (ignored on purpose: forward-compatible clients)
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_days":250,"recency_days":40,"frequency":6,"monetary":1850.0,"avg_order_value":308.33,"total_items":720,"avg_items_per_order":120.0,"distinct_products":45,"avg_unit_price":2.9,"months_active":5,"max_gap_days":62,"returns_rate":0.05,"customer_id":"C123"}'
```

A `422` body is FastAPI's validation detail and names the offending field:

```json
{"detail": [{"loc": ["body", "returns_rate"], "msg": "Input should be less than or equal to 1"}]}
```

### Status codes

| Code | Where | Meaning |
|---|---|---|
| 200 | all | Success |
| 404 | `/metrics`, `/features` | The corresponding report JSON is missing |
| 422 | `/predict*` | Missing field, wrong type, out-of-range value, empty or oversized batch |
| 500 | `/predict*` | Inference raised; the message is surfaced in `detail` |
| 503 | `/predict*` | Reached before the model loaded (only possible if startup was bypassed) |

---

## PowerShell equivalent

```powershell
$base = "http://localhost:8000"
Invoke-RestMethod "$base/health" | ConvertTo-Json -Depth 5

$body = @{
  tenure_days = 250; recency_days = 40; frequency = 6; monetary = 1850.0
  avg_order_value = 308.33; total_items = 720; avg_items_per_order = 120.0
  distinct_products = 45; avg_unit_price = 2.9; months_active = 5
  max_gap_days = 62; returns_rate = 0.05
} | ConvertTo-Json

Invoke-RestMethod "$base/predict" -Method Post -ContentType "application/json" -Body $body
```

---

## Using the Postman collection

1. **Import** → `docs/postman_collection.json`.
2. Set the collection variable `base_url` (default `http://localhost:8000`).
3. Run requests individually, or **Run all** in the Collection Runner. Each
   request carries a test script asserting the contract documented here, so a
   green run means the served model, frozen threshold, tiers and validation
   rules all still behave as written.

Requests: health · metrics · features · predict (engaged) · predict (one-off) ·
batch ranking · missing field → 422 · out-of-range → 422 · empty batch → 422 ·
unknown extra key → 200.

---

## Known gaps

- **No auth, no rate limiting.** `/predict` is open to anyone who can reach it;
  add a key or a gateway before sending real traffic.
- **Metrics describe a 2011 snapshot.** A high score for a present-day customer
  is an extrapolation, and the probabilities are not calibrated.
- **`422` bodies use FastAPI's default shape.** Fine to read; if your client
  branches on `detail[].loc`, pin that assumption.
- **CI gates the pytest suite, not this collection.** `ci.yml` blocks on the
  unit/API tests plus the artifact and boot checks, but the Newman run of
  `postman_collection.json` is `continue-on-error` until it has been observed
  green — so contract regressions are *reported* here without yet *failing* a
  build. Remove that line to make this document enforceable.
- **CI covers one interpreter.** Only Python 3.13 is exercised, matching the
  Docker image; the 3.11+ floor in the README is an inference from the pins, not
  a tested claim.


