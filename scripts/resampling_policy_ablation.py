from __future__ import annotations

import argparse
import json
import math
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
    split_target,
    target_group_key,
)


SCORES = ["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"]
ALPHAS = [0.05, 0.10, 0.20]
TOP_KS = [1, 2, 4]


def csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


def prepare_ordered_target(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["target_file"] = target_group_key(out)
    sort_cols = [c for c in ["relative_file", "mol_idx"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).copy()
    return out


def summarize_acceptance(
    dataset: str,
    generator: str,
    policy: str,
    score_name: str,
    calibration: str,
    alpha: float | None,
    k: int,
    sub_all: pd.DataFrame,
    accepted_parts: list[pd.DataFrame],
    target_rows: list[dict[str, float]],
    controlled_bad_rate_at_tau: float | None = None,
) -> dict[str, float | int | str]:
    target_table = pd.DataFrame(target_rows)
    accepted = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame()
    row: dict[str, float | int | str] = {
        "dataset": dataset,
        "generator": generator,
        "policy": policy,
        "score_name": score_name,
        "calibration": calibration,
        "alpha": math.nan if alpha is None else float(alpha),
        "k": int(k),
        "targets": int(len(target_table)),
        "molecules_scanned": int(target_table["scanned"].sum()) if len(target_table) else 0,
        "finite_candidates": int(target_table["finite_candidates"].sum()) if len(target_table) else 0,
        "accepted": int(target_table["accepted"].sum()) if len(target_table) else 0,
        "target_success_at_1": float(target_table["success_at_1"].mean()) if len(target_table) else math.nan,
        "target_success_at_k": float(target_table["success_at_k"].mean()) if len(target_table) else math.nan,
        "mean_accepted_per_target": float(target_table["accepted"].mean()) if len(target_table) else math.nan,
        "raw_bad_rate": float(sub_all["bad"].mean()) if len(sub_all) else math.nan,
        "controlled_bad_rate_at_tau": math.nan
        if controlled_bad_rate_at_tau is None
        else float(controlled_bad_rate_at_tau),
    }
    if len(accepted):
        row.update(
            {
                "accepted_bad_rate": float(accepted["bad"].mean()),
                "accepted_reliable_rate": float(1.0 - accepted["bad"].mean()),
                "accepted_qed_mean": float(accepted["qed"].mean()),
                "accepted_energy_per_heavy_mean": float(accepted["ff_energy_per_heavy"].mean()),
                "accepted_relax_rmsd_mean": float(accepted["ff_relax_rmsd"].mean()),
                "attempts_per_accepted": float(row["molecules_scanned"] / max(len(accepted), 1)),
            }
        )
    else:
        row.update(
            {
                "accepted_bad_rate": math.nan,
                "accepted_reliable_rate": math.nan,
                "accepted_qed_mean": math.nan,
                "accepted_energy_per_heavy_mean": math.nan,
                "accepted_relax_rmsd_mean": math.nan,
                "attempts_per_accepted": math.inf,
            }
        )
    return row


def evaluate_groups(
    dataset: str,
    generator: str,
    sub_all: pd.DataFrame,
    policy: str,
    score_name: str,
    calibration: str,
    alpha: float | None,
    k: int,
    score: np.ndarray | None = None,
    threshold: float | None = None,
    controlled_bad_rate_at_tau: float | None = None,
) -> dict[str, float | int | str]:
    sub_all = prepare_ordered_target(sub_all)
    score_map = None
    if score is not None:
        score_map = pd.Series(score, index=sub_all.index, name="_score")
    accepted_parts: list[pd.DataFrame] = []
    target_rows: list[dict[str, float]] = []
    for file_id, group in sub_all.groupby("target_file", sort=False):
        scanned = len(group)
        if policy == "raw_topk":
            candidates = group
            accepted = candidates.head(k)
        else:
            assert score_map is not None
            scored = group.join(score_map, how="left")
            scored = scored[np.isfinite(scored["_score"].astype(float))]
            candidates = scored
            if policy == "rank_topk":
                accepted = candidates.sort_values("_score", ascending=True).head(k)
            elif policy in {"source_wilson_topk", "target_wilson_topk"}:
                assert threshold is not None
                accepted = candidates[candidates["_score"] <= threshold].head(k)
            else:
                raise ValueError(policy)
        if len(accepted):
            accepted_parts.append(accepted.drop(columns=["_score"], errors="ignore"))
        target_rows.append(
            {
                "file_id": file_id,
                "scanned": scanned,
                "finite_candidates": len(candidates),
                "accepted": len(accepted),
                "success_at_1": int(len(accepted) >= 1),
                "success_at_k": int(len(accepted) >= k),
            }
        )
    return summarize_acceptance(
        dataset,
        generator,
        policy,
        score_name,
        calibration,
        alpha,
        k,
        sub_all,
        accepted_parts,
        target_rows,
        controlled_bad_rate_at_tau,
    )


def apply_raw_and_rank_policies(
    dataset: str,
    target: pd.DataFrame,
    target_scores: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows = []
    for generator, sub_all in target.groupby("generator"):
        sub_idx = target.index.get_indexer(sub_all.index)
        for k in TOP_KS:
            rows.append(evaluate_groups(dataset, generator, sub_all, "raw_topk", "none", "none", None, k))
            for score_name in SCORES:
                score_sub = target_scores[score_name][sub_idx]
                rows.append(evaluate_groups(dataset, generator, sub_all, "rank_topk", score_name, "none", None, k, score_sub))
    return rows


def apply_source_wilson_policies(
    dataset: str,
    source_cal: pd.DataFrame,
    source_scores: dict[str, np.ndarray],
    target: pd.DataFrame,
    target_scores: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows = []
    for generator, sub_all in target.groupby("generator"):
        sub_idx = target.index.get_indexer(sub_all.index)
        for score_name in SCORES:
            source_score_all = source_scores[score_name]
            valid_source = finite_score_mask(source_score_all)
            source_score = source_score_all[valid_source]
            source_eval = source_cal.loc[valid_source]
            if len(source_eval) == 0:
                continue
            score_sub = target_scores[score_name][sub_idx]
            for alpha in ALPHAS:
                threshold, controlled = conformal_threshold(source_score, source_eval["bad"].to_numpy(), alpha, "wilson95")
                for k in TOP_KS:
                    rows.append(
                        evaluate_groups(
                            dataset,
                            generator,
                            sub_all,
                            "source_wilson_topk",
                            score_name,
                            "source_wilson95",
                            alpha,
                            k,
                            score_sub,
                            threshold,
                            controlled,
                        )
                    )
    return rows


def apply_target_wilson_policies(
    dataset: str,
    target: pd.DataFrame,
    target_scores: dict[str, np.ndarray],
    seed: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    split = split_target(target, seed + 3000)
    target = target.copy()
    target["target_split"] = split
    for generator, sub_all in target.groupby("generator"):
        sub_idx = target.index.get_indexer(sub_all.index)
        for score_name in SCORES:
            score_sub = target_scores[score_name][sub_idx]
            valid = finite_score_mask(score_sub)
            sub_eval = sub_all.loc[valid].copy()
            score_eval = score_sub[valid]
            if len(sub_eval) == 0:
                continue
            cal = sub_eval[sub_eval["target_split"] == "cal"].copy()
            test = sub_eval[sub_eval["target_split"] == "test"].copy()
            if len(cal) == 0 or len(test) == 0 or cal["bad"].nunique() < 2:
                continue
            cal_pos = sub_eval.index.get_indexer(cal.index)
            test_pos = sub_eval.index.get_indexer(test.index)
            cal_score = score_eval[cal_pos]
            test_score = score_eval[test_pos]
            for alpha in ALPHAS:
                threshold, controlled = conformal_threshold(cal_score, cal["bad"].to_numpy(), alpha, "wilson95")
                for k in TOP_KS:
                    rows.append(
                        evaluate_groups(
                            dataset,
                            generator,
                            test,
                            "target_wilson_topk",
                            score_name,
                            "target_wilson95",
                            alpha,
                            k,
                            test_score,
                            threshold,
                            controlled,
                        )
                    )
    return rows


def add_raw_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    raw = out[out["policy"] == "raw_topk"][
        ["dataset", "generator", "k", "accepted_bad_rate", "accepted_qed_mean", "accepted_energy_per_heavy_mean"]
    ].rename(
        columns={
            "accepted_bad_rate": "raw_topk_bad_rate",
            "accepted_qed_mean": "raw_topk_qed_mean",
            "accepted_energy_per_heavy_mean": "raw_topk_energy_per_heavy_mean",
        }
    )
    out = out.merge(raw, on=["dataset", "generator", "k"], how="left")
    out["bad_rate_delta_vs_raw_topk"] = out["accepted_bad_rate"] - out["raw_topk_bad_rate"]
    out["relative_risk_reduction_vs_raw_topk"] = (
        out["raw_topk_bad_rate"] - out["accepted_bad_rate"]
    ) / out["raw_topk_bad_rate"].replace(0, np.nan)
    out["qed_delta_vs_raw_topk"] = out["accepted_qed_mean"] - out["raw_topk_qed_mean"]
    out["energy_delta_vs_raw_topk"] = out["accepted_energy_per_heavy_mean"] - out["raw_topk_energy_per_heavy_mean"]
    return out


def process_target(
    target_path: Path,
    dataset: str,
    source_cal: pd.DataFrame,
    source_scores: dict[str, np.ndarray],
    models: dict[str, object],
    seed: int,
) -> pd.DataFrame:
    target = prepare_target(target_path)
    target = prepare_ordered_target(target)
    target_scores = score_baselines(target)
    target_scores.update(model_scores(models, target))
    rows = []
    rows.extend(apply_raw_and_rank_policies(dataset, target, target_scores))
    rows.extend(apply_source_wilson_policies(dataset, source_cal, source_scores, target, target_scores))
    rows.extend(apply_target_wilson_policies(dataset, target, target_scores, seed))
    return add_raw_deltas(pd.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit", type=Path, default=Path("outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv"))
    parser.add_argument("--target", action="append", required=True, help="name=/path/to/molecule_physics_audit.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resampling_policy_ablation_v0/reports"))
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, thresholds = prepare_source(args.source_audit, args.quantile, args.seed)
    source_train = source[source["split"] == "train"].copy()
    source_cal = source[source["split"] == "cal"].copy()
    models = fit_models(source_train, args.seed)
    source_scores = score_baselines(source_cal)
    source_scores.update(model_scores(models, source_cal))

    target_specs = {}
    summaries = []
    for spec in args.target:
        name, path = spec.split("=", 1)
        target_specs[name] = path
        summaries.append(process_target(Path(path), name, source_cal, source_scores, models, args.seed))

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.output_dir / "policy_ablation_summary.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "source_audit": str(args.source_audit),
                "targets": target_specs,
                "quantile": args.quantile,
                "source_thresholds": thresholds,
                "seed": args.seed,
                "scores": SCORES,
                "alphas": ALPHAS,
                "top_ks": TOP_KS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fixed_k4 = summary[
        (summary["k"] == 4)
        & (
            ((summary["policy"] == "raw_topk") & (summary["score_name"] == "none"))
            | ((summary["policy"] == "rank_topk") & summary["score_name"].isin(SCORES))
        )
    ][
        [
            "dataset",
            "generator",
            "policy",
            "score_name",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
            "accepted_qed_mean",
            "accepted_energy_per_heavy_mean",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate"], ascending=[True, True, True])

    source_a10_k4 = summary[
        (summary["policy"] == "source_wilson_topk") & (summary["alpha"] == 0.10) & (summary["k"] == 4)
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate"], ascending=[True, True, True])

    target_valid_k4 = summary[
        (summary["policy"] == "target_wilson_topk")
        & (summary["k"] == 4)
        & (summary["accepted_bad_rate"] <= summary["alpha"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
        ]
    ].sort_values(["dataset", "generator", "alpha", "accepted_bad_rate"], ascending=[True, True, True, True])

    lines = [
        "# Resampling Policy Ablation v0",
        "",
        "Matched policy comparison on the same audited candidate pools. No molecules are regenerated.",
        "",
        "## Fixed Yield Top-4 Ranking",
        csv_block(fixed_k4),
        "",
        "## Source-Calibrated Wilson Alpha 0.10, Max 4 Accepted Per Target",
        csv_block(source_a10_k4),
        "",
        "## Target-Calibrated Wilson Rows Meeting Alpha, Max 4 Accepted Per Target",
        csv_block(target_valid_k4) if len(target_valid_k4) else "No target-calibrated top-4 row met alpha.",
    ]
    (args.output_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
