"""Mirrors Node speech_level_policy.ts — same thresholds and transitions."""

from typing import Literal

SpeechLevel = Literal["beginner", "intermediate", "advanced"]

SPEECH_LEVEL_THRESHOLDS: dict[SpeechLevel, float] = {
    "beginner": 0.65,
    "intermediate": 0.4,
    "advanced": 0.2,
}

SPEECH_LEVEL_TRANSITIONS = {
    "promotion": {
        "beginner": {"to": "intermediate", "requiredPasses": 3, "window": 6},
        "intermediate": {"to": "advanced", "requiredPasses": 4, "window": 8},
    },
    "demotion": {
        "advanced": {"to": "intermediate", "requiredFails": 3, "window": 6},
        "intermediate": {"to": "beginner", "requiredFails": 3, "window": 6},
    },
}


def normalize_speech_level(value: str | None) -> SpeechLevel:
    if not value or not isinstance(value, str):
        return "beginner"
    n = value.strip().lower()
    if n == "intermediate":
        return "intermediate"
    if n == "advanced":
        return "advanced"
    return "beginner"


def get_severity_threshold(level: SpeechLevel) -> float:
    return SPEECH_LEVEL_THRESHOLDS[level]


def is_pass_for_level(level: SpeechLevel, severity: float) -> bool:
    return severity <= get_severity_threshold(level)


def next_speech_level_from_history(level: SpeechLevel, history: list[bool]) -> SpeechLevel:
    promotion_rule = None
    if level == "beginner":
        promotion_rule = SPEECH_LEVEL_TRANSITIONS["promotion"]["beginner"]
    elif level == "intermediate":
        promotion_rule = SPEECH_LEVEL_TRANSITIONS["promotion"]["intermediate"]

    if promotion_rule:
        scoped = history[: promotion_rule["window"]]
        if sum(1 for x in scoped if x) >= promotion_rule["requiredPasses"]:
            return promotion_rule["to"]  # type: ignore[return-value]

    demotion_rule = None
    if level == "advanced":
        demotion_rule = SPEECH_LEVEL_TRANSITIONS["demotion"]["advanced"]
    elif level == "intermediate":
        demotion_rule = SPEECH_LEVEL_TRANSITIONS["demotion"]["intermediate"]

    if demotion_rule:
        scoped = history[: demotion_rule["window"]]
        if sum(1 for x in scoped if not x) >= demotion_rule["requiredFails"]:
            return demotion_rule["to"]  # type: ignore[return-value]

    return level


def predict_speech_level_after(
    speech_level_before: SpeechLevel,
    history_with_current: list[bool],
    severity: float | None,
) -> tuple[SpeechLevel, bool, float]:
    thr = get_severity_threshold(speech_level_before)
    if severity is not None and severity == severity:
        bounded = max(0.0, min(1.0, float(severity)))
        is_pass = is_pass_for_level(speech_level_before, bounded)
    else:
        is_pass = False
    after = next_speech_level_from_history(speech_level_before, history_with_current)
    return after, is_pass, thr
