"""Template-based hints (no LLM)."""

TEMPLATES: dict[str, str] = {
    "retry_soft": 'Try once more: "{word}"',
    "retry_slow": 'Say it slowly: "{word}" — listen for the {sound} sound.',
    "retry_focus": 'Focus on the {sound} sound in "{word}".',
}


def choose_template_id(hint_tone: str | None, attempt: int, severity: float | None) -> str:
    t = (hint_tone or "light").lower()
    if severity is None or severity != severity:
        sev = 0.5
    else:
        sev = max(0.0, min(1.0, float(severity)))
    if t == "direct" or sev > 0.7:
        return "retry_focus"
    if t == "supportive" or attempt >= 2 or sev > 0.35:
        return "retry_slow"
    return "retry_soft"


def render_hint(template_id: str, word: str, sound: str) -> str:
    body = TEMPLATES.get(template_id, TEMPLATES["retry_soft"])
    return body.format(word=word, sound=sound or "target")
