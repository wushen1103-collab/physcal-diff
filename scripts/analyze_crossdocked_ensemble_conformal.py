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

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from physcaldiff.metrics import binary_metrics, rank_to_unit_interval  # noqa: E402


META_CANDIDATES = [
    "pocket_id",
    "pdb_id",
    "chain_id",
    "sdf_file",
    "mol_idx",
    "candidate_uid",
    "generator",
    "chemotype_id",
    "vina_score",
    "gnina_score",
    "qed",
    "sa",
    "scscore",
]


def stable_hash_int(value: str, seed: int) -> int:
    text = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha1(text).hexdigest()[:12], 16)


def infer_seed(path: Path) -> str:
    for part in path.parts:
        if part.startswith("seed_"):
            return part.split("_", 1)[1]
    return path.parent.name


def choose_key_columns(frame: pd.DataFrame) -> list[str]:
    options = [
        ["candidate_uid"],
        ["pocket_id", "mol_idx"],
        ["pocket_id", "sdf_file", "mol_idx"],
        ["pdb_id", "chain_id", "mol_idx"],
    ]
    for cols in options:
        if all(col in frame.columns for col in cols):
            return cols
    return ["row_id"]


def normalize_score(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) and finite.min() >= -1e-6 and finite.max() <= 1.0 + 1e-6:
        return np.clip(arr, 0.0, 1.0)
    return rank_to_unit_interval(arr, higher_is_better=True)


def load_seed_predictions(pred_root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    paths = sorted(pred_root.glob("seed_*/oof_predictions.csv"))
    if not paths:
        raise FileNotFoundError(f"No seed_*/oof_predictions.csv under {pred_root}")

    base: pd.DataFrame | None = None
    key_cols: list[str] | None = None
    score_cols: list[str] | None = None
    seeds: list[str] = []

    for path in paths:
        seed = infer_seed(path)
        seeds.append(seed)
        frame = pd.read_csv(path)
        frame = frame.copy()
        frame["row_id"] = np.arange(len(frame))
        if key_cols is None:
            key_cols = choose_key_columns(frame)
        if score_cols is None:
            score_cols = [c for c in frame.columns if c.startswith("score_")]
            if not score_cols:
                raise ValueError(f"No score_ columns found in {path}")
        missing = [c for c in key_cols + score_cols for _ in [0] if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        keep_meta = [c for c in META_CANDIDATES if c in frame.columns and c not in key_cols]
        if "reliable" in frame.columns:
            keep_meta.append("reliable")
        if base is None:
            base = frame[key_cols + keep_meta].drop_duplicates(subset=key_cols).copy()
        for col in score_cols:
            frame[f"{col}__seed_{seed}"] = normalize_score(frame[col])
        seed_part = frame[key_cols + [f"{c}__seed_{seed}" for c in score_cols]]
        base = base.merge(seed_part, on=key_cols, how="inner")

    assert base is not None and key_cols is not None and score_cols is not None
    return base, score_cols, seeds


def split_cal_test(frame: pd.DataFrame, seed: int, frac: float) -> np.ndarray:
    if "pocket_id" in frame.columns:
        values = frame["pocket_id"].astype(str)
    else:
        values = frame.index.astype(str)
    unit = values.map(lambda x: stable_hash_int(x, seed) / float(16**12 - 1)).to_numpy()
    return unit < frac


def wilson_upper(k: np.ndarray, n: np.ndarray, z: float = 1.6448536269514722) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    phat = np.divide(k, np.maximum(n, 1.0))
    denom = 1.0 + z * z / np.maximum(n, 1.0)
    center = phat + z * z / (2.0 * np.maximum(n, 1.0))
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * np.maximum(n, 1.0))) / np.maximum(n, 1.0))
    return (center + half) / denom


def conformal_threshold(risk: np.ndarray, bad: np.ndarray, alpha: float, bound: str) -> tuple[float, float]:
    order = np.argsort(risk)
    risk_sorted = risk[order]
    bad_sorted = bad[order].astype(float)
    count = np.arange(1, len(risk_sorted) + 1)
    cum_bad = np.cumsum(bad_sorted)
    empirical_risk = cum_bad / count
    if bound == "empirical":
        controlled_risk = empirical_risk
    elif bound == "wilson95":
        controlled_risk = wilson_upper(cum_bad, count)
    else:
        raise ValueError(f"Unknown risk bound: {bound}")
    ok = np.where(controlled_risk <= alpha)[0]
    if len(ok) == 0:
        return -math.inf, math.nan
    idx = ok[-1]
    return float(risk_sorted[idx]), float(controlled_risk[idx])


def selection_stats(
    frame: pd.DataFrame,
    accepted: np.ndarray,
    bad_col: str = "bad",
) -> dict[str, float]:
    out = {
        "n": int(len(frame)),
        "n_accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
    }
    if accepted.any():
        selected = frame.loc[accepted]
        out.update(
            {
                "accepted_bad_rate": float(selected[bad_col].mean()),
                "accepted_reliable_rate": float(1.0 - selected[bad_col].mean()),
                "resampling_multiplier": float(1.0 / max(accepted.mean(), 1e-12)),
            }
        )
        for col in ["vina_score", "qed", "sa", "gnina_score"]:
            if col in selected.columns:
                out[f"accepted_mean_{col}"] = float(selected[col].mean())
        if "chemotype_id" in selected.columns and "pocket_id" in selected.columns:
            out["diversity_proxy"] = float(selected.groupby("pocket_id")["chemotype_id"].nunique().mean())
    else:
        out.update(
            {
                "accepted_bad_rate": math.nan,
                "accepted_reliable_rate": math.nan,
                "resampling_multiplier": math.inf,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--coverage", type=float, nargs="+", default=[0.10, 0.20, 0.50, 0.70, 0.80, 0.90])
    parser.add_argument("--lambda-uncertainty", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--risk-bound", nargs="+", default=["empirical", "wilson95"])
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=2206)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    frame, score_cols, seeds = load_seed_predictions(args.pred_root)
    if "reliable" not in frame.columns:
        raise ValueError("Expected a reliable column in prediction files")
    frame["bad"] = 1 - frame["reliable"].astype(int)
    is_cal = split_cal_test(frame, args.split_seed, args.calibration_fraction)

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pred_root": str(args.pred_root),
        "n_rows": int(len(frame)),
        "score_cols": score_cols,
        "seeds": seeds,
        "calibration_rows": int(is_cal.sum()),
        "test_rows": int((~is_cal).sum()),
        "bad_rate_all": float(frame["bad"].mean()),
        "bad_rate_calibration": float(frame.loc[is_cal, "bad"].mean()),
        "bad_rate_test": float(frame.loc[~is_cal, "bad"].mean()),
    }

    metric_rows = []
    conformal_rows = []
    coverage_rows = []
    scored_frames = []

    for col in score_cols:
        seed_cols = [f"{col}__seed_{seed}" for seed in seeds]
        if not all(seed_col in frame.columns for seed_col in seed_cols):
            continue
        prob_mean = frame[seed_cols].mean(axis=1).to_numpy(float)
        prob_std = frame[seed_cols].std(axis=1).fillna(0.0).to_numpy(float)
        frame[f"{col}__prob_mean"] = prob_mean
        frame[f"{col}__prob_std"] = prob_std
        std_metric = binary_metrics(frame["bad"].to_numpy(), np.clip(prob_std, 0.0, 1.0))
        std_metric.update(
            {
                "score_col": col,
                "lambda_uncertainty": math.nan,
                "score_name": f"{col}__std_only",
            }
        )
        metric_rows.append(std_metric)

        for lam in args.lambda_uncertainty:
            risk = np.clip((1.0 - prob_mean) + lam * prob_std, 0.0, 1.0)
            score_name = f"{col}__risk_plus_{lam:g}u"
            frame[score_name] = risk
            metric = binary_metrics(frame["bad"].to_numpy(), risk)
            metric.update({"score_col": col, "lambda_uncertainty": lam, "score_name": score_name})
            metric_rows.append(metric)

            cal = frame.loc[is_cal].copy()
            test = frame.loc[~is_cal].copy()
            cal_risk = risk[is_cal]
            test_risk = risk[~is_cal]
            cal_bad = cal["bad"].to_numpy(int)

            for bound in args.risk_bound:
                for alpha in args.alpha:
                    tau, cal_bound = conformal_threshold(cal_risk, cal_bad, alpha, bound)
                    accepted = test_risk <= tau
                    row = selection_stats(test, accepted)
                    row.update(
                        {
                            "score_col": col,
                            "lambda_uncertainty": lam,
                            "risk_bound": bound,
                            "alpha": alpha,
                            "tau": tau,
                            "calibration_controlled_bad_rate_at_tau": cal_bound,
                            "calibration_empirical_bad_rate_at_tau": float(cal.loc[cal_risk <= tau, "bad"].mean()) if np.any(cal_risk <= tau) else math.nan,
                        }
                    )
                    conformal_rows.append(row)

            for cov in args.coverage:
                tau = float(np.quantile(cal_risk, cov))
                accepted = test_risk <= tau
                row = selection_stats(test, accepted)
                row.update(
                    {
                        "score_col": col,
                        "lambda_uncertainty": lam,
                        "target_coverage": cov,
                        "tau": tau,
                    }
                )
                coverage_rows.append(row)

        scored_frames.append(frame)

    metrics = pd.DataFrame(metric_rows)
    conformal = pd.DataFrame(conformal_rows)
    coverage = pd.DataFrame(coverage_rows)

    metrics.to_csv(report_dir / "bad_sample_detection_metrics.csv", index=False)
    conformal.to_csv(report_dir / "conformal_selection.csv", index=False)
    coverage.to_csv(report_dir / "risk_coverage.csv", index=False)
    frame.to_csv(report_dir / "ensemble_predictions.csv", index=False)
    (report_dir / "run_metadata.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    metric_view = metrics.sort_values(["roc_auc", "pr_auc"], ascending=False).head(12)
    conformal_view = conformal.sort_values(
        ["risk_bound", "alpha", "accepted_bad_rate", "coverage"],
        ascending=[True, True, True, False],
    ).groupby(["risk_bound", "alpha"], as_index=False).head(8)
    valid_conformal = conformal[conformal["accepted_bad_rate"] <= conformal["alpha"]].copy()
    valid_conformal_view = valid_conformal.sort_values(
        ["risk_bound", "alpha", "coverage"],
        ascending=[True, True, False],
    ).groupby(["risk_bound", "alpha"], as_index=False).head(8)
    coverage_view = coverage.sort_values(
        ["target_coverage", "accepted_bad_rate"],
        ascending=[True, True],
    ).groupby("target_coverage", as_index=False).head(5)

    lines = [
        "# CrossDocked Ensemble Conformal Audit",
        "",
        "This is a bootstrap experiment using existing DiffSBDD/CrossDocked candidate predictions.",
        "It tests whether ensemble epistemic uncertainty adds usable risk-control signal.",
        "",
        "## Input Audit",
        "",
        "```csv",
        pd.DataFrame([audit]).to_csv(index=False).strip(),
        "```",
        "",
        "## Bad-Sample Detection: Top Rows",
        "",
        "```csv",
        metric_view.to_csv(index=False).strip(),
        "```",
        "",
        "## Conformal Selection: Best Rows Per Alpha",
        "",
        "```csv",
        conformal_view.to_csv(index=False).strip(),
        "```",
        "",
        "## Conformal Selection: Highest Coverage Rows That Satisfy Test Alpha",
        "",
        "```csv",
        valid_conformal_view.to_csv(index=False).strip() if len(valid_conformal_view) else "none",
        "```",
        "",
        "## Risk-Coverage: Best Rows Per Target Coverage",
        "",
        "```csv",
        coverage_view.to_csv(index=False).strip(),
        "```",
        "",
        "## Immediate Interpretation",
        "",
        "- If lambda_uncertainty > 0 improves AUROC or accepted_bad_rate, ensemble variance is useful.",
        "- If lambda_uncertainty = 0 dominates, current seed variance is weak and we need stronger uncertainty sources.",
        "- If coverage collapses at alpha=0.05, the next step should use calibrated triage rather than strict rejection.",
    ]
    report = "\n".join(lines)
    (report_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
