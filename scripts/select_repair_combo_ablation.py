from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from train_physics_risk_critic import conformal_threshold
from transfer_physics_critic import (
    finite_score_mask,
    fit_models,
    model_scores,
    prepare_source,
    prepare_target,
    score_baselines,
)


SCORES = ["relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"]
RMSD_LIMITS = [1.0, 1.5, 2.0]


def stable_unit(value: str, seed: int) -> float:
    text = f"{seed}|{value}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:12], 16) / float(16**12 - 1)


def extract_data_id(relative_file: str) -> str:
    hit = re.search(r"data(\d{3})", str(relative_file))
    return hit.group(1) if hit else str(relative_file)


def target_key(frame: pd.DataFrame) -> pd.Series:
    return frame["generator"].astype(str) + "::" + frame["relative_file"].map(extract_data_id)


def attach_scores(target_audit: pd.DataFrame, repair: pd.DataFrame, models: dict[str, object]) -> pd.DataFrame:
    target_scores = score_baselines(target_audit)
    target_scores.update(model_scores(models, target_audit))
    score_frame = target_audit[["generator", "relative_file", "mol_idx"]].copy()
    for name, score in target_scores.items():
        score_frame[name] = score
    merged = repair.merge(score_frame, on=["generator", "relative_file", "mol_idx"], how="left", validate="one_to_one")
    merged["target_file"] = target_key(merged)
    merged["order_key"] = merged.groupby("target_file").cumcount()
    return merged


def build_thresholds(source_cal: pd.DataFrame, source_scores: dict[str, np.ndarray], alpha: float) -> dict[str, float]:
    thresholds = {}
    for score_name in SCORES:
        score_all = source_scores[score_name]
        valid = finite_score_mask(score_all)
        tau, _ = conformal_threshold(score_all[valid], source_cal.loc[valid, "bad"].to_numpy(), alpha, "wilson95")
        thresholds[score_name] = tau
    return thresholds


def summarize_policy(
    dataset: str,
    generator: str,
    sub: pd.DataFrame,
    policy: str,
    score_name: str,
    alpha: float | None,
    repair_rmsd_max: float | None,
    k: int,
    use_repaired_risk: bool,
    selector,
) -> dict[str, float | int | str]:
    accepted_parts = []
    target_rows = []
    for target, group in sub.sort_values(["relative_file", "mol_idx"]).groupby("target_file", sort=False):
        cand = selector(group).copy()
        accepted = cand.head(k)
        if len(accepted):
            accepted_parts.append(accepted)
        target_rows.append(
            {
                "target_file": target,
                "scanned": len(group),
                "eligible": len(cand),
                "accepted": len(accepted),
                "success_at_1": int(len(accepted) >= 1),
                "success_at_k": int(len(accepted) >= k),
            }
        )
    target_table = pd.DataFrame(target_rows)
    accepted_all = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame()
    risk_col = "repaired_high_risk_fixed" if use_repaired_risk else "raw_high_risk_fixed"
    qed_col = "repaired_qed" if use_repaired_risk else "raw_qed"
    energy_col = "repaired_ff_energy_per_heavy" if use_repaired_risk else "raw_ff_energy_per_heavy"
    row: dict[str, float | int | str] = {
        "dataset": dataset,
        "generator": generator,
        "policy": policy,
        "score_name": score_name,
        "alpha": math.nan if alpha is None else alpha,
        "repair_rmsd_max": math.nan if repair_rmsd_max is None else repair_rmsd_max,
        "k": k,
        "targets": int(len(target_table)),
        "molecules_scanned": int(target_table["scanned"].sum()) if len(target_table) else 0,
        "eligible": int(target_table["eligible"].sum()) if len(target_table) else 0,
        "accepted": int(target_table["accepted"].sum()) if len(target_table) else 0,
        "target_success_at_k": float(target_table["success_at_k"].mean()) if len(target_table) else math.nan,
        "mean_accepted_per_target": float(target_table["accepted"].mean()) if len(target_table) else math.nan,
        "raw_pool_bad_rate": float(sub["raw_high_risk_fixed"].mean()) if len(sub) else math.nan,
    }
    if len(accepted_all):
        row.update(
            {
                "accepted_bad_rate": float(accepted_all[risk_col].mean()),
                "accepted_qed_mean": float(accepted_all[qed_col].mean()),
                "accepted_energy_per_heavy_mean": float(accepted_all[energy_col].mean()),
                "accepted_repair_rmsd_mean": float(accepted_all["repair_rmsd"].mean())
                if "repair" in policy
                else math.nan,
                "attempts_per_accepted": float(target_table["scanned"].sum() / len(accepted_all)),
            }
        )
    else:
        row.update(
            {
                "accepted_bad_rate": math.nan,
                "accepted_qed_mean": math.nan,
                "accepted_energy_per_heavy_mean": math.nan,
                "accepted_repair_rmsd_mean": math.nan,
                "attempts_per_accepted": math.inf,
            }
        )
    row["risk_delta_vs_raw_pool"] = row["accepted_bad_rate"] - row["raw_pool_bad_rate"]
    return row


def evaluate_dataset(
    dataset: str,
    target_audit_path: Path,
    repair_table_path: Path,
    source_cal: pd.DataFrame,
    source_scores: dict[str, np.ndarray],
    models: dict[str, object],
    alpha: float,
    k: int,
) -> pd.DataFrame:
    target = prepare_target(target_audit_path)
    repair = pd.read_csv(repair_table_path)
    table = attach_scores(target, repair, models)
    thresholds = build_thresholds(source_cal, source_scores, alpha)
    rows = []
    for generator, sub in table.groupby("generator"):
        rows.append(
            summarize_policy(
                dataset,
                generator,
                sub,
                "raw_topk",
                "none",
                None,
                None,
                k,
                False,
                lambda group: group.sort_values("order_key"),
            )
        )
        for rmsd_limit in RMSD_LIMITS:
            rows.append(
                summarize_policy(
                    dataset,
                    generator,
                    sub,
                    "repair_rmsd_topk",
                    "none",
                    None,
                    rmsd_limit,
                    k,
                    True,
                    lambda group, limit=rmsd_limit: group[
                        group["repair_ok"] & (group["repair_rmsd"] <= limit)
                    ].sort_values("order_key"),
                )
            )
        for score_name in SCORES:
            tau = thresholds[score_name]
            rows.append(
                summarize_policy(
                    dataset,
                    generator,
                    sub,
                    "source_select_topk",
                    score_name,
                    alpha,
                    None,
                    k,
                    False,
                    lambda group, score=score_name, threshold=tau: group[
                        np.isfinite(group[score].astype(float)) & (group[score] <= threshold)
                    ].sort_values("order_key"),
                )
            )
            for rmsd_limit in RMSD_LIMITS:
                rows.append(
                    summarize_policy(
                        dataset,
                        generator,
                        sub,
                        "source_select_then_repair_topk",
                        score_name,
                        alpha,
                        rmsd_limit,
                        k,
                        True,
                        lambda group, score=score_name, threshold=tau, limit=rmsd_limit: group[
                            np.isfinite(group[score].astype(float))
                            & (group[score] <= threshold)
                            & group["repair_ok"]
                            & (group["repair_rmsd"] <= limit)
                        ].sort_values("order_key"),
                    )
                )
    return pd.DataFrame(rows)


def csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-audit", action="append", required=True, help="name=/path/to/molecule_physics_audit.csv")
    parser.add_argument("--repair-table", action="append", required=True, help="name=/path/to/repair_molecule_table.csv")
    parser.add_argument("--source-audit", type=Path, default=Path("outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/select_repair_combo_v0/reports"))
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, thresholds = prepare_source(args.source_audit, args.quantile, args.seed)
    source_train = source[source["split"] == "train"].copy()
    source_cal = source[source["split"] == "cal"].copy()
    models = fit_models(source_train, args.seed)
    source_scores = score_baselines(source_cal)
    source_scores.update(model_scores(models, source_cal))

    target_specs = dict(spec.split("=", 1) for spec in args.target_audit)
    repair_specs = dict(spec.split("=", 1) for spec in args.repair_table)
    missing = sorted(set(target_specs) ^ set(repair_specs))
    if missing:
        raise ValueError(f"Target/repair dataset mismatch: {missing}")

    frames = []
    for name in sorted(target_specs):
        frames.append(
            evaluate_dataset(
                name,
                Path(target_specs[name]),
                Path(repair_specs[name]),
                source_cal,
                source_scores,
                models,
                args.alpha,
                args.k,
            )
        )
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(args.output_dir / "select_repair_combo_summary.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "target_audits": target_specs,
                "repair_tables": repair_specs,
                "source_audit": str(args.source_audit),
                "source_thresholds": thresholds,
                "alpha": args.alpha,
                "k": args.k,
                "rmsd_limits": RMSD_LIMITS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    key = summary[
        (
            (summary["policy"] == "raw_topk")
            | ((summary["policy"] == "repair_rmsd_topk") & (summary["repair_rmsd_max"].isin([1.0, 1.5])))
            | (
                (summary["policy"] == "source_select_topk")
                & (summary["score_name"].isin(["relaxation_rule_score", "pre_relax_ff_rf"]))
            )
            | (
                (summary["policy"] == "source_select_then_repair_topk")
                & (summary["repair_rmsd_max"].isin([1.0, 1.5]))
                & (summary["score_name"].isin(["relaxation_rule_score", "pre_relax_ff_rf"]))
            )
        )
    ][
        [
            "dataset",
            "generator",
            "policy",
            "score_name",
            "repair_rmsd_max",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "risk_delta_vs_raw_pool",
            "accepted_repair_rmsd_mean",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate", "accepted"], ascending=[True, True, True, False])

    lines = [
        "# Select + Repair Combo Ablation",
        "",
        "Compares raw top-k, source-calibrated selection, bounded repair, and selection followed by bounded repair.",
        "",
        "## Key Rows",
        csv_block(key),
    ]
    (args.output_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
