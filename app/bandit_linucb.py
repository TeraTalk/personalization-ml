"""Disjoint LinUCB per (user_id, word_id), in-memory for local dev."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np

FEATURE_DIM = 8
ALPHA = 1.0


def _context_vector(
    requested_level: float,
    num_problem_sounds: int,
    source: str,
    user_id: str,
) -> np.ndarray:
    src_exact = 1.0 if source == "exact" else 0.0
    src_nearby = 1.0 if source == "nearby" else 0.0
    src_fb = 1.0 if source == "global_fallback" else 0.0
    h = int(hashlib.sha256(user_id.encode()).hexdigest()[:8], 16)
    return np.array(
        [
            1.0,
            float(requested_level) / 10.0,
            min(float(num_problem_sounds) / 5.0, 1.0),
            src_exact,
            src_nearby,
            src_fb,
            (h % 7) / 7.0,
            ((h // 7) % 5) / 5.0,
        ],
        dtype=np.float64,
    )


@dataclass
class ArmState:
    A: np.ndarray = field(default_factory=lambda: np.eye(FEATURE_DIM, dtype=np.float64))
    b: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM, dtype=np.float64))

    def ucb_score(self, x: np.ndarray, alpha: float) -> float:
        inv_A = np.linalg.inv(self.A)
        theta = inv_A @ self.b
        exploit = float(theta @ x)
        explore = alpha * float(np.sqrt(max(float(x @ inv_A @ x), 0.0)))
        return exploit + explore

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x


class LinUCBWordSelector:
    def __init__(self) -> None:
        self._arms: dict[tuple[str, str], ArmState] = {}

    def select_arm(
        self,
        user_id: str,
        candidate_ids: list[str],
        requested_level: float,
        num_problem_sounds: int,
        source: str,
    ) -> str:
        if not candidate_ids:
            raise ValueError("candidate_ids empty")
        if len(candidate_ids) == 1:
            return candidate_ids[0]

        x = _context_vector(requested_level, num_problem_sounds, source, user_id)
        scores: list[tuple[str, float]] = []
        for wid in candidate_ids:
            key = (user_id, wid)
            if key not in self._arms:
                self._arms[key] = ArmState()
            scores.append((wid, self._arms[key].ucb_score(x, ALPHA)))

        max_s = max(s for _, s in scores)
        top = [w for w, s in scores if abs(s - max_s) < 1e-9]
        return random.choice(top)

    def record_reward(self, user_id: str, word_id: str, reward: float, context: dict) -> None:
        x = _context_vector(
            float(context.get("requested_level", 2)),
            int(context.get("num_problem_sounds", 0)),
            str(context.get("source", "global_fallback")),
            user_id,
        )
        key = (user_id, word_id)
        if key not in self._arms:
            self._arms[key] = ArmState()
        self._arms[key].update(x, reward)


_selector = LinUCBWordSelector()


def get_selector() -> LinUCBWordSelector:
    return _selector
