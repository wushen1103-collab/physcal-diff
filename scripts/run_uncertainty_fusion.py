from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from physcaldiff.metrics import binary_metrics  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from analyze_crossdocked_ensemble_conformal import (  # noqa: E402
    conformal_threshold,
    load_seed_predictions,
    selection_stats,
)


def stable_unit(value: str, seed: int) -> float:
    text = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha1(text).hexdigest()[:12], 16) / float(16**12 - 1)


def make_three_way_split(frame: pd.DataFrame, seed: int, fit_frac: float, cal_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "pocket_id" in frame.columns:
        values = frame["pocket_id"].astype(str)
    else:
        values = frame.index.astype(str)
    unit = values.map(lambda x: stable_unit(x, seed)).to_numpy()
    fit = unit < fit_frac
    cal = (unit >= fit_frac) & (unit < fit_frac + cal_frac)
    test = unit >= fit_frac + cal_frac
    return fit, cal, test


def fusion_features(prob_mean: np.ndarray, prob_std: np.ndarray) -> np.ndarray:
    risk = np.clip(1.0 - prob_mean, 0.0, 1.0)
    std = np.clip(prob_std, 0.0, 1.0)
    return np.column_stack([risk, std, risk * std, risk * risk, std * std])


def score_variants(prob_mean: np.ndarray, prob_std: np.ndarray, model, x: np.ndarray) -> dict[str, np.ndarray]:
    risk = np.clip(1.0 - prob_mean, 0.0, 1.0)
    std = np.clip(prob_std, 0.0, 1.0)
    fusion = model.predict_proba(x)[:, 1]
    return {
        "mean_risk": risk,
        "std_only": std,
        "logistic_fusion": fusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-frac", type=float, default=0.4)
    parser.add_argument("--cal-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=2207)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    args = parser.parse_args()

    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    frame, score_cols, seeds = load_seed_predictions(args.pred_root)
    if "reliable" not in frame.columns:
        raise ValueError("Expected reliable column")
    frame["bad"] = 1 - frame["reliable"].astype(int)
    fit_mask, cal_mask, test_mask = make_three_way_split(frame, args.split_seed, args.fit_frac, args.cal_frac)

    metric_rows = []
    selection_rows = []
    pred_frames = []

    for col in score_cols:
        seed_cols = [f"{col}__seed_{seed}" for seed in seeds]
        prob_mean = frame[seed_cols].mean(axis=1).to_numpy(float)
        prob_std = frame[seed_cols].std(axis=1).fillna(0.0).to_numpy(float)
        x = fusion_features(prob_mean, prob_std)
        y = frame["bad"].to_numpy(int)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        )
        model.fit(x[fit_mask], y[fit_mask])

        variants = score_variants(prob_mean, prob_std, model, x)
        for name, scores in variants.items():
            test_scores = scores[test_mask]
            test_frame = frame.loc[test_mask].copy()
            metric = binary_metrics(y[test_mask], test_scores)
            metric.update({"score_col": col, "variant": name})
            metric_rows.append(metric)

            for alpha in args.alpha:
                tau, cal_bound = conformal_threshold(scores[cal_mask], y[cal_mask], alpha, "wilson95")
                accepted = test_scores <= tau
                row = selection_stats(test_frame, accepted)
                row.update(
                    {
                        "score_col": col,
                        "variant": name,
                        "alpha": alpha,
                        "tau": tau,
                        "calibration_controlled_bad_rate_at_tau": cal_bound,
                        "calibration_empirical_bad_rate_at_tau": float(frame.loc[cal_mask & (scores <= tau), "bad"].mean()) if np.any(cal_mask & (scores <= tau)) else math.nan,
                    }
                )
                selection_rows.append(row)

        pred_part = frame.copy()
        for name, scores in variants.items():
            pred_part[f"{col}__{name}"] = scores
        pred_frames.append(pred_part)

    metrics = pd.DataFrame(metric_rows)
    selection = pd.DataFrame(selection_rows)
    metrics.to_csv(report_dir / "fusion_metrics.csv", index=False)
    selection.to_csv(report_dir / "fusion_wilson_selection.csv", index=False)

    metric_view = metrics.sort_values(["roc_auc", "pr_auc"], ascending=False).head(20)
    valid = selection[selection["accepted_bad_rate"] <= selection["alpha"]].copy()
    valid_view = valid.sort_values(["alpha", "coverage"], ascending=[True, False]).groupby("alpha", as_index=False).head(12)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pred_root": str(args.pred_root),
        "score_cols": score_cols,
        "seeds": seeds,
        "n_rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "calibration_rows": int(cal_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "bad_rate_fit": float(frame.loc[fit_mask, "bad"].mean()),
        "bad_rate_calibration": float(frame.loc[cal_mask, "bad"].mean()),
        "bad_rate_test": float(frame.loc[test_mask, "bad"].mean()),
    }
    (report_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Uncertainty Fusion Audit",
        "",
        "Three-way pocket split: fit -> calibration -> test. Logistic fusion uses mean risk, std, and low-order interactions.",
        "",
        "## Metadata",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
        "",
        "## Top Bad-Sample Detection Rows",
        "```csv",
        metric_view.to_csv(index=False).strip(),
        "```",
        "",
        "## Wilson95 Selection Rows Satisfying Test Alpha",
        "```csv",
        valid_view.to_csv(index=False).strip() if len(valid_view) else "none",
        "```",
    ]
    report = "\n".join(lines)
    (report_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

