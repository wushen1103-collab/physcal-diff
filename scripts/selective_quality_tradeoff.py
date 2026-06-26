from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_physics_risk_critic import (
    MAIN_GENERATORS,
    add_derived_features,
    add_physics_labels,
    assign_split,
    conformal_threshold,
    feature_sets,
    group_key,
    make_model,
)


def mean_or_nan(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def summarize_selection(frame: pd.DataFrame, accepted: np.ndarray, alpha: float, score_name: str, model: str) -> dict[str, float | str]:
    selected = frame.loc[accepted].copy()
    total = max(len(frame), 1)
    row: dict[str, float | str] = {
        "alpha": alpha,
        "score_name": score_name,
        "model": model,
        "n_test": len(frame),
        "n_accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
    }
    if len(selected) == 0:
        for col in [
            "accepted_bad_rate",
            "accepted_reliable_rate",
            "qed_mean",
            "heavy_atoms_mean",
            "ff_energy_per_heavy_mean",
            "ff_energy_drop_mean",
            "ff_relax_rmsd_mean",
            "hard_rule_risk_rate",
            "relaxation_risk_rate",
            "bond_outlier_rate",
            "internal_clash_rate",
            "diffsbdd_share",
            "molcraft_share",
            "pocketflow_share",
        ]:
            row[col] = float("nan")
        return row
    row.update(
        {
            "accepted_bad_rate": float(selected["bad"].mean()),
            "accepted_reliable_rate": float(1.0 - selected["bad"].mean()),
            "qed_mean": mean_or_nan(selected["qed"]),
            "heavy_atoms_mean": mean_or_nan(selected["heavy_atoms"]),
            "ff_energy_per_heavy_mean": mean_or_nan(selected["ff_energy_per_heavy"]),
            "ff_energy_drop_mean": mean_or_nan(selected["ff_energy_drop"]),
            "ff_relax_rmsd_mean": mean_or_nan(selected["ff_relax_rmsd"]),
            "hard_rule_risk_rate": float(selected["hard_rule_risk"].mean()),
            "relaxation_risk_rate": float(selected["relaxation_risk"].mean()),
            "bond_outlier_rate": float(selected["bond_outlier"].mean()),
            "internal_clash_rate": float(selected["internal_clash"].mean()),
        }
    )
    for generator in MAIN_GENERATORS:
        row[f"{generator}_share"] = float((selected["generator"] == generator).sum() / max(len(selected), 1))
    row["resampling_multiplier"] = float(1.0 / max(accepted.sum() / total, 1e-12))
    return row


def add_delta_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    raw = out[out["score_name"] == "raw_unfiltered"].iloc[0]
    for col in [
        "accepted_bad_rate",
        "qed_mean",
        "ff_energy_per_heavy_mean",
        "ff_energy_drop_mean",
        "ff_relax_rmsd_mean",
        "hard_rule_risk_rate",
        "relaxation_risk_rate",
    ]:
        out[f"delta_{col}_vs_raw"] = out[col].astype(float) - float(raw[col])
    out["risk_reduction_vs_raw"] = float(raw["accepted_bad_rate"]) - out["accepted_bad_rate"].astype(float)
    return out


def csv_block(df: pd.DataFrame) -> str:
    return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selective_quality_tradeoff_v0/reports"))
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.audit_csv)
    frame = frame[frame["generator"].isin(MAIN_GENERATORS)].copy()
    frame, thresholds = add_physics_labels(frame, args.quantile)
    frame = add_derived_features(frame)
    frame["split"] = assign_split(group_key(frame), args.seed)
    train = frame[frame["split"] == "train"].copy()
    cal = frame[frame["split"] == "cal"].copy()
    test = frame[frame["split"] == "test"].copy()

    rows = []
    rows.append(summarize_selection(test, np.ones(len(test), dtype=bool), 1.0, "raw_unfiltered", "none"))

    scorers: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for score_col in ["qed_risk", "cheap_rule_score", "relaxation_rule_score"]:
        scorers.append((score_col, "baseline", cal[score_col].astype(float).to_numpy(), test[score_col].astype(float).to_numpy()))

    for feature_name, model_name in [
        ("pre_relax_ff", "rf"),
        ("pre_relax_ff", "hgb"),
        ("full_audit_upper", "rf"),
    ]:
        features = feature_sets()[feature_name]
        model = make_model(model_name, seed=args.seed)
        model.fit(train[features], train["bad"])
        scorers.append(
            (
                feature_name,
                model_name,
                model.predict_proba(cal[features])[:, 1],
                model.predict_proba(test[features])[:, 1],
            )
        )

    for score_name, model_name, cal_score, test_score in scorers:
        for alpha in [0.05, 0.10, 0.20]:
            tau, controlled = conformal_threshold(cal_score, cal["bad"].to_numpy(), alpha=alpha, bound="wilson95")
            accepted = test_score <= tau
            row = summarize_selection(test, accepted, alpha, score_name, model_name)
            row["threshold"] = tau
            row["calibration_controlled_bad_rate_at_tau"] = controlled
            rows.append(row)

    table = pd.DataFrame(rows)
    table = add_delta_columns(table)
    table.to_csv(args.output_dir / "selective_quality_tradeoff.csv", index=False)

    compact = table[
        [
            "alpha",
            "score_name",
            "model",
            "coverage",
            "accepted_bad_rate",
            "risk_reduction_vs_raw",
            "qed_mean",
            "delta_qed_mean_vs_raw",
            "ff_energy_per_heavy_mean",
            "delta_ff_energy_per_heavy_mean_vs_raw",
            "ff_relax_rmsd_mean",
            "delta_ff_relax_rmsd_mean_vs_raw",
            "diffsbdd_share",
            "molcraft_share",
            "pocketflow_share",
        ]
    ].sort_values(["alpha", "accepted_bad_rate", "coverage"], ascending=[True, True, False])

    lines = [
        "# Selective Quality Trade-off v0",
        "",
        "Pooled q90 split. Raw row is the unfiltered test set. Other rows use Wilson95 thresholds calibrated on the calibration split.",
        "",
        "## Compact Table",
        csv_block(compact),
    ]
    (args.output_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
