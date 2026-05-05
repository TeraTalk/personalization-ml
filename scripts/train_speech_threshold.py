"""
Fit logistic regression for personalized speech pass threshold, export JSON for runtime.

CSV columns (header row required):
  severity, is_pass, speech_level, age, num_problem_sounds, hist_pass_rate, hist_mean_severity

speech_level: beginner | intermediate | advanced
is_pass: 0 or 1
Optional columns default to neutral features when missing.

Usage:
  pip install scikit-learn pandas
  python scripts/train_speech_threshold.py --csv data/attempts.csv --out app/models/speech_threshold_lr.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

CLIP_LOW = {"beginner": 0.25, "intermediate": 0.2, "advanced": 0.05}
CLIP_HIGH = {"beginner": 0.65, "intermediate": 0.5, "advanced": 0.35}


def build_X(df: pd.DataFrame) -> np.ndarray:
    age = df["age"].fillna(6.0).astype(float)
    age_norm = ((age - 6.0) / 6.0).clip(-1, 1)
    ps = df["num_problem_sounds"].fillna(0).astype(float)
    ps_norm = (ps / 5.0).clip(0, 1)
    pr = df["hist_pass_rate"].fillna(0.5).astype(float).clip(0, 1)
    msev = df["hist_mean_severity"].fillna(0.5).astype(float).clip(0, 1)
    sev = df["severity"].astype(float).clip(0, 1)
    level = df["speech_level"].str.strip().str.lower()
    inter = (level == "intermediate").astype(float)
    adv = (level == "advanced").astype(float)
    return np.column_stack([sev, age_norm, ps_norm, pr, msev, inter, adv])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for col in ("severity", "is_pass", "speech_level"):
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")
    if "age" not in df.columns:
        df["age"] = np.nan
    if "num_problem_sounds" not in df.columns:
        df["num_problem_sounds"] = 0
    if "hist_pass_rate" not in df.columns:
        df["hist_pass_rate"] = np.nan
    if "hist_mean_severity" not in df.columns:
        df["hist_mean_severity"] = np.nan

    X = build_X(df)
    y = df["is_pass"].astype(int).values
    if len(np.unique(y)) < 2:
        raise SystemExit("Need both pass and fail labels in is_pass")

    m = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    m.fit(X, y)
    coef = m.coef_[0].astype(float)
    intercept = float(m.intercept_[0])
    if coef[0] >= 0:
        print("Warning: severity coefficient is not negative; check data or regularization.")

    pred = m.predict(X)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, m.predict_proba(X)[:, 1])
    except ValueError:
        auc = float("nan")
    print(f"Train accuracy={acc:.3f} roc_auc={auc:.3f} intercept={intercept:.4f} coef={coef}")

    out = {
        "schema_version": 1,
        "intercept": intercept,
        "coef": coef.tolist(),
        "feature_names": [
            "severity",
            "age_norm",
            "num_problem_sounds_norm",
            "hist_pass_rate",
            "hist_mean_severity",
            "level_intermediate",
            "level_advanced",
        ],
        "clip_low": CLIP_LOW,
        "clip_high": CLIP_HIGH,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
