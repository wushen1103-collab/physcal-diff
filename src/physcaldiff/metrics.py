from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def safe_score(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return float("nan")


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2.0 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1))
    if best >= len(thresholds):
        return 0.5, float(f1[best])
    return float(thresholds[best]), float(f1[best])


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    threshold, best_f1 = best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        f"{prefix}roc_auc": safe_score(roc_auc_score, y_true, y_score),
        f"{prefix}pr_auc": safe_score(average_precision_score, y_true, y_score),
        f"{prefix}brier": safe_score(brier_score_loss, y_true, np.clip(y_score, 0.0, 1.0)),
        f"{prefix}ece": expected_calibration_error(y_true, np.clip(y_score, 0.0, 1.0)),
        f"{prefix}best_f1": best_f1,
        f"{prefix}threshold": threshold,
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}mcc": float(matthews_corrcoef(y_true, y_pred)),
        f"{prefix}positive_rate": float(y_true.mean()),
    }


def rank_to_unit_interval(score: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    order = score.argsort().argsort().astype(float)
    if not higher_is_better:
        order = (len(score) - 1.0) - order
    return (order + 1.0) / (len(score) + 1.0)

