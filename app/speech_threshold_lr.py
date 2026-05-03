"""Logistic regression for personalized pass threshold (severity upper bound)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app.speech_policy import SpeechLevel, get_severity_threshold, normalize_speech_level

_log = logging.getLogger("teratalk_ml")


@dataclass(frozen=True)
class SpeechThresholdLRModel:
    intercept: float
    coef: np.ndarray
    clip_low: dict[str, float]
    clip_high: dict[str, float]

    def feature_vector(
        self,
        severity: float,
        level: SpeechLevel,
        age: float | None,
        num_problem_sounds: int,
        hist_pass_rate: float | None,
        hist_mean_severity: float | None,
        n_history: int,
    ) -> np.ndarray:
        if age is None or age != age:
            age_norm = 0.0
        else:
            age_norm = float(np.clip((float(age) - 6.0) / 6.0, -1.0, 1.0))

        ps_norm = min(max(float(num_problem_sounds), 0.0) / 5.0, 1.0)

        if n_history <= 0 or hist_pass_rate is None or hist_pass_rate != hist_pass_rate:
            pr = 0.5
        else:
            pr = float(np.clip(hist_pass_rate, 0.0, 1.0))

        if n_history <= 0 or hist_mean_severity is None or hist_mean_severity != hist_mean_severity:
            msev = 0.5
        else:
            msev = float(np.clip(hist_mean_severity, 0.0, 1.0))

        sev = float(np.clip(severity, 0.0, 1.0))
        inter = 1.0 if level == "intermediate" else 0.0
        adv = 1.0 if level == "advanced" else 0.0

        return np.array(
            [sev, age_norm, ps_norm, pr, msev, inter, adv],
            dtype=np.float64,
        )

    def linear_score(self, x: np.ndarray) -> float:
        return float(self.intercept + float(self.coef @ x))

    def personalized_threshold(self, level: SpeechLevel, x_no_severity: np.ndarray) -> float:
        """Solve intercept + w0*T + w_rest·x_rest = 0 for T (pass if severity <= T)."""
        base = float(get_severity_threshold(level))
        w0 = float(self.coef[0])
        if abs(w0) < 1e-9:
            return base
        rest = self.coef[1:] * x_no_severity
        z_rest = float(self.intercept + float(np.sum(rest)))
        t_raw = -z_rest / w0
        lo = self.clip_low.get(level, 0.1)
        hi = self.clip_high.get(level, 0.9)
        if w0 > 0:
            _log.warning("speech_threshold_lr: coef[severity] should be negative; using base threshold")
            return base
        return float(np.clip(t_raw, lo, hi))

    def predict_pass(
        self,
        severity: float | None,
        level: SpeechLevel,
        age: float | None,
        num_problem_sounds: int,
        hist_pass_rate: float | None,
        hist_mean_severity: float | None,
        n_history: int,
    ) -> tuple[bool, float]:
        level_n = normalize_speech_level(level)
        base_thr = float(get_severity_threshold(level_n))
        if severity is None or severity != severity:
            _log.info(
                "[speech_threshold_lr] used | severity=missing | level=%s | "
                "is_pass=False | personalized_thr=%.3f (static fallback) | n_hist=%d",
                level_n,
                base_thr,
                n_history,
            )
            return False, base_thr
        sev = float(np.clip(severity, 0.0, 1.0))
        x = self.feature_vector(
            sev,
            level_n,
            age,
            num_problem_sounds,
            hist_pass_rate,
            hist_mean_severity,
            n_history,
        )
        z = self.linear_score(x)
        is_pass = z >= 0.0
        thr = self.personalized_threshold(level_n, x[1:])
        _log.info(
            "[speech_threshold_lr] used | level=%s | severity=%.3f | linear_z=%.4f | "
            "is_pass=%s | personalized_thr=%.3f | static_baseline_thr=%.3f | "
            "age=%s | n_ps=%d | n_hist=%d | hist_pass_rate=%s | hist_mean_sev=%s",
            level_n,
            sev,
            z,
            is_pass,
            thr,
            base_thr,
            age if age is not None and age == age else "null",
            num_problem_sounds,
            n_history,
            f"{hist_pass_rate:.3f}" if hist_pass_rate is not None and hist_pass_rate == hist_pass_rate else "neutral",
            f"{hist_mean_severity:.3f}"
            if hist_mean_severity is not None and hist_mean_severity == hist_mean_severity
            else "neutral",
        )
        return is_pass, thr


_model: SpeechThresholdLRModel | None = None


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "speech_threshold_lr.json"


def load_speech_threshold_model() -> SpeechThresholdLRModel:
    global _model
    if _model is not None:
        return _model
    path = os.environ.get("SPEECH_THRESHOLD_MODEL_PATH", "").strip()
    p = Path(path) if path else _default_model_path()
    if not p.is_file():
        raise FileNotFoundError(f"Speech threshold model not found: {p}")
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)
    coef = np.array(raw["coef"], dtype=np.float64)
    if coef.shape != (7,):
        raise ValueError(f"Expected 7 coefficients, got {coef.shape}")
    _model = SpeechThresholdLRModel(
        intercept=float(raw["intercept"]),
        coef=coef,
        clip_low=dict(raw["clip_low"]),
        clip_high=dict(raw["clip_high"]),
    )
    _log.info("Loaded speech threshold LR model from %s", p)
    return _model


def get_speech_threshold_model() -> SpeechThresholdLRModel:
    return load_speech_threshold_model()
