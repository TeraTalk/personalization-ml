# Teratalk ML service (local FastAPI)

Provides:

- `POST /v1/bandit/select-word` — disjoint LinUCB over candidate word IDs
- `POST /v1/bandit/reward` — optional online reward update
- `POST /v1/speech-level/predict` — speech level after + pass (mirrors Node `speech_level_policy` v0)
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

## Terminal logs

Each request is logged to stdout with the `teratalk_ml` logger, for example:

- `Starting Teratalk ML service ...` on startup
- `GET /health`
- `POST /v1/bandit/select-word` (candidate count, then `chosen_word_id`)
- `POST /v1/bandit/reward` (reward update)
- `POST /v1/speech-level/predict` (inputs, then `after` / `is_pass`)
- `POST /v1/hint/template` (word / tone, then `template_id`)

Format: `HH:MM:SS | ML | LEVEL | message`
