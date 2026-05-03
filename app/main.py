import logging
import os
import sys
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.bandit_linucb import get_selector
from app.hint_templates import choose_template_id, render_hint
from app.speech_policy import normalize_speech_level, predict_speech_level_after

MODEL_VERSION = "local-v0.1.0"
API_KEY = os.environ.get("ML_INTERNAL_API_KEY", "")

_log = logging.getLogger("teratalk_ml")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s | ML | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
    )
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)

app = FastAPI(title="Teratalk ML service", version=MODEL_VERSION)


@app.on_event("startup")
async def _startup_log() -> None:
    _log.info(
        "Starting Teratalk ML service version=%s internal_auth=%s",
        MODEL_VERSION,
        "enabled" if API_KEY else "disabled (empty ML_INTERNAL_API_KEY)",
    )


def _auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


class SelectWordRequest(BaseModel):
    user_id: str
    candidate_word_ids: list[str] = Field(min_length=1)
    requested_level: float = 2
    num_problem_sounds: int = 0
    source: str = "global_fallback"


class SelectWordResponse(BaseModel):
    chosen_word_id: str
    model_version: str


class SpeechLevelRequest(BaseModel):
    speech_level_before: str
    history_with_current: list[bool]
    severity: float | None = None


class SpeechLevelResponse(BaseModel):
    speech_level_after: str
    is_pass: bool
    severity_threshold_used: float
    model_version: str


class HintRequest(BaseModel):
    expected_word: str
    expected_sound: str = ""
    hint_tone: str | None = None
    attempt: int = 1
    severity: float | None = None


class HintResponse(BaseModel):
    template_id: str
    hint_text: str
    model_version: str


class BanditRewardRequest(BaseModel):
    user_id: str
    word_id: str
    reward: float = Field(ge=0.0, le=1.0)
    requested_level: float = 2
    num_problem_sounds: int = 0
    source: str = "global_fallback"


@app.get("/health")
def health() -> dict[str, str]:
    _log.info("GET /health")
    return {"status": "ok", "model_version": MODEL_VERSION, "message": "its live"}


@app.post("/v1/bandit/select-word", response_model=SelectWordResponse)
def select_word(
    body: SelectWordRequest,
    authorization: str | None = Header(default=None),
) -> SelectWordResponse:
    _auth(authorization)
    n = len(body.candidate_word_ids)
    _log.info(
        "POST /v1/bandit/select-word | user_id=%s | candidates=%d | level=%s | problem_sounds=%d | source=%s",
        body.user_id,
        n,
        body.requested_level,
        body.num_problem_sounds,
        body.source,
    )
    sel = get_selector().select_arm(
        body.user_id,
        body.candidate_word_ids,
        body.requested_level,
        body.num_problem_sounds,
        body.source,
    )
    _log.info("POST /v1/bandit/select-word | done | chosen_word_id=%s", sel)
    return SelectWordResponse(chosen_word_id=sel, model_version=MODEL_VERSION)


@app.post("/v1/bandit/reward")
def bandit_reward(
    body: BanditRewardRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _auth(authorization)
    _log.info(
        "POST /v1/bandit/reward | user_id=%s | word_id=%s | reward=%.3f | level=%s | source=%s",
        body.user_id,
        body.word_id,
        body.reward,
        body.requested_level,
        body.source,
    )
    get_selector().record_reward(
        body.user_id,
        body.word_id,
        body.reward,
        {
            "requested_level": body.requested_level,
            "num_problem_sounds": body.num_problem_sounds,
            "source": body.source,
        },
    )
    _log.info("POST /v1/bandit/reward | done | LinUCB arm updated")
    return {"ok": True, "model_version": MODEL_VERSION}


@app.post("/v1/speech-level/predict", response_model=SpeechLevelResponse)
def speech_level(
    body: SpeechLevelRequest,
    authorization: str | None = Header(default=None),
) -> SpeechLevelResponse:
    _auth(authorization)
    _log.info(
        "POST /v1/speech-level/predict | before=%s | history_len=%d | severity=%s",
        body.speech_level_before,
        len(body.history_with_current),
        body.severity,
    )
    before = normalize_speech_level(body.speech_level_before)
    after, is_pass, thr = predict_speech_level_after(
        before,
        body.history_with_current,
        body.severity,
    )
    _log.info(
        "POST /v1/speech-level/predict | done | after=%s | is_pass=%s | threshold=%.2f",
        after,
        is_pass,
        thr,
    )
    return SpeechLevelResponse(
        speech_level_after=after,
        is_pass=is_pass,
        severity_threshold_used=thr,
        model_version=MODEL_VERSION,
    )


@app.post("/v1/hint/template", response_model=HintResponse)
def hint_template(
    body: HintRequest,
    authorization: str | None = Header(default=None),
) -> HintResponse:
    _auth(authorization)
    _log.info(
        "POST /v1/hint/template | word=%s | tone=%s | attempt=%s | severity=%s",
        body.expected_word,
        body.hint_tone,
        body.attempt,
        body.severity,
    )
    tid = choose_template_id(body.hint_tone, body.attempt, body.severity)
    text = render_hint(tid, body.expected_word, body.expected_sound)
    preview = text if len(text) <= 120 else text[:117] + "..."
    _log.info(
        "POST /v1/hint/template | done | template_id=%s | hint_len=%d", tid, len(text)
    )
    _log.info("POST /v1/hint/template | hint_text=%s", preview)
    return HintResponse(template_id=tid, hint_text=text, model_version=MODEL_VERSION)
