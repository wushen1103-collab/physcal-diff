from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


RATE_METRICS = {
    "high_risk_rate": "rdkit_physics_high_risk",
    "bond_outlier_rate": "bond_outlier",
    "high_energy_q90_rate": "high_energy_pool_q90",
    "relax_drop_q90_rate": "high_relax_drop_pool_q90",
    "relax_rmsd_q90_rate": "high_relax_rmsd_pool_q90",
}
MEAN_METRICS = {
    "qed_mean": "qed",
    "energy_per_heavy_mean": "ff_energy_per_heavy",
}


def extract_data_id(relative_file: str) -> int:
    hit = re.search(r"data(\d{3})", str(relative_file))
    if not hit:
        raise ValueError(f"Cannot parse data id from {relative_file!r}")
    return int(hit.group(1))


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(bool)


def aggregate_by_target(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["data_id"] = work["relative_file"].map(extract_data_id)
    work["bond_outlier"] = work["bond_outlier_count"].fillna(0) > 0
    rows = []
    for (generator, data_id), sub in work.groupby(["generator", "data_id"]):
        row: dict[str, float | int | str] = {
            "generator": generator,
            "data_id": int(data_id),
            "molecules": int(len(sub)),
            "files": int(sub["relative_file"].nunique()),
        }
        for metric, column in RATE_METRICS.items():
            row[metric] = float(bool_series(sub, column).mean())
        for metric, column in MEAN_METRICS.items():
            row[metric] = float(sub[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(target_summary: pd.DataFrame) -> pd.DataFrame:
    wide = target_summary.pivot(index="data_id", columns="generator")
    rows = []
    metrics = ["molecules", *RATE_METRICS.keys(), *MEAN_METRICS.keys()]
    for data_id in wide.index:
        row: dict[str, float | int] = {"data_id": int(data_id)}
        for metric in metrics:
            for arm in ["origin", "guide"]:
                key = (metric, arm)
                row[f"{arm}_{metric}"] = wide[key].loc[data_id] if key in wide else math.nan
            if metric != "molecules":
                row[f"delta_guide_minus_origin_{metric}"] = row[f"guide_{metric}"] - row[f"origin_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("data_id")


def sign_test_pvalue(delta: np.ndarray) -> tuple[int, int, float]:
    delta = delta[np.isfinite(delta)]
    pos = int((delta > 0).sum())
    neg = int((delta < 0).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, math.nan
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return pos, neg, float(min(1.0, 2.0 * tail))


def bootstrap_mean_ci(delta: np.ndarray, seed: int, reps: int) -> tuple[float, float, float]:
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(reps, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def statistics(pairwise: pd.DataFrame, seed: int, reps: int) -> pd.DataFrame:
    rows = []
    for metric in [*RATE_METRICS.keys(), *MEAN_METRICS.keys()]:
        delta_col = f"delta_guide_minus_origin_{metric}"
        delta = pairwise[delta_col].astype(float).to_numpy()
        mean_delta, ci_low, ci_high = bootstrap_mean_ci(delta, seed, reps)
        pos, neg, sign_p = sign_test_pvalue(delta)
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(np.isfinite(delta).sum()),
                "origin_mean": float(pairwise[f"origin_{metric}"].mean()),
                "guide_mean": float(pairwise[f"guide_{metric}"].mean()),
                "mean_delta_guide_minus_origin": mean_delta,
                "bootstrap95_low": ci_low,
                "bootstrap95_high": ci_high,
                "guide_worse_pairs": pos,
                "guide_better_pairs": neg,
                "two_sided_sign_p": sign_p,
            }
        )
    return pd.DataFrame(rows)


def csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.audit_csv)
    frame["ff_energy_per_heavy"] = frame["ff_energy_final"] / frame["heavy_atoms"].replace(0, np.nan)
    target_summary = aggregate_by_target(frame)
    pairwise = paired_table(target_summary)
    stats = statistics(pairwise, args.seed, args.bootstrap_reps)

    target_summary.to_csv(args.output_dir / "target_arm_summary.csv", index=False)
    pairwise.to_csv(args.output_dir / "target_pairwise_delta.csv", index=False)
    stats.to_csv(args.output_dir / "paired_statistics.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "audit_csv": str(args.audit_csv),
                "seed": args.seed,
                "bootstrap_reps": args.bootstrap_reps,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    key = stats[stats["metric"].isin(["high_risk_rate", "bond_outlier_rate", "relax_drop_q90_rate", "relax_rmsd_q90_rate", "qed_mean"])]
    worst_targets = pairwise.sort_values("delta_guide_minus_origin_high_risk_rate", ascending=False).head(12)
    lines = [
        "# Matched SYNC Pairwise Analysis",
        "",
        "Deltas are guide minus origin. Positive high-risk deltas mean guide is worse.",
        "",
        "## Paired Statistics",
        csv_block(key),
        "",
        "## Largest High-Risk Increases",
        csv_block(
            worst_targets[
                [
                    "data_id",
                    "origin_high_risk_rate",
                    "guide_high_risk_rate",
                    "delta_guide_minus_origin_high_risk_rate",
                    "origin_qed_mean",
                    "guide_qed_mean",
                ]
            ]
        ),
    ]
    (args.output_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
