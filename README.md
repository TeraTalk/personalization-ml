# Teratalk ML service (local FastAPI)

Provides:

- `POST /v1/bandit/select-word` — disjoint LinUCB over candidate word IDs
- `POST /v1/bandit/reward` — optional online reward update
- `POST /v1/speech-level/predict` — pass/fail via **logistic regression** on severity + context; level transitions still use the same rule windows as Node. Weights in `app/models/speech_threshold_lr.json` (override path with `SPEECH_THRESHOLD_MODEL_PATH`).
- `POST /v1/hint/template` — template-based hint text

## Setup

```bash
cd teratalk-ml-service
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8090
```

Optional shared secret (must match Node `ML_INTERNAL_API_KEY`):

```bash
set ML_INTERNAL_API_KEY=dev-secret
```

## Node backend

In `teratalk-backend-node/.env`:

```
ML_SERVICE_URL=http://127.0.0.1:8090
ML_INTERNAL_API_KEY=
ML_WORDS_ENABLED=1
ML_SPEECH_LEVEL_ENABLED=1
ML_HINTS_ENABLED=1
```

If `ML_SERVICE_URL` is unset, the backend skips all ML calls.

## Speech threshold model (personalized pass cutoff)

The bundled `intercept` and severity coefficient in `speech_threshold_lr.json` are set so that, with zero contribution from age / problem-sounds / history features (their coefficients are 0 in the default model), pass/fail boundaries match the static cutoffs: beginner 0.5 / intermediate 0.35 / advanced 0.2. Non-zero context features then shift the boundary via the same linear model. `clip_low` / `clip_high` bound the reported personalized threshold.

Retrain from labeled rows (export from your DB into CSV):

```bash
pip install -r requirements-train.txt
python scripts/train_speech_threshold.py --csv data/attempts.csv --out app/models/speech_threshold_lr.json
```

CSV columns: `severity`, `is_pass`, `speech_level`, plus optional `age`, `num_problem_sounds`, `hist_pass_rate`, `hist_mean_severity`.

## Terminal logs

Each request is logged to stdout with the `teratalk_ml` logger, for example:

- `Starting Teratalk ML service ...` on startup
- `GET /health`
- `POST /v1/bandit/select-word` (candidate count, then `chosen_word_id`)
- `POST /v1/bandit/reward` (reward update)
- `POST /v1/speech-level/predict` (inputs, then `after` / `is_pass`)
- `POST /v1/hint/template` (word / tone, then `template_id`)

Format: `HH:MM:SS | ML | LEVEL | message`
