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


ALPHAS = [0.05, 0.10, 0.20]
RMSD_LIMITS = [0.5, 1.0, 1.5, 2.0]
KS = [1, 2, 4]
SCORES = ["relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"]
KEY_SCORE_ORDER = ["relaxation_rule_score", "pre_relax_ff_rf"]


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


def source_thresholds(source_cal: pd.DataFrame, source_scores: dict[str, np.ndarray]) -> dict[tuple[str, float], float]:
    out = {}
    for score_name in SCORES:
        score_all = source_scores[score_name]
        valid = finite_score_mask(score_all)
        for alpha in ALPHAS:
            tau, _ = conformal_threshold(score_all[valid], source_cal.loc[valid, "bad"].to_numpy(), alpha, "wilson95")
            out[(score_name, alpha)] = tau
    return out


def target_rows_for_policy(
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
) -> pd.DataFrame:
    rows = []
    risk_col = "repaired_high_risk_fixed" if use_repaired_risk else "raw_high_risk_fixed"
    qed_col = "repaired_qed" if use_repaired_risk else "raw_qed"
    energy_col = "repaired_ff_energy_per_heavy" if use_repaired_risk else "raw_ff_energy_per_heavy"
    for target, group in sub.sort_values(["relative_file", "mol_idx"]).groupby("target_file", sort=False):
        candidates = selector(group).copy()
        accepted = candidates.head(k)
        accepted_count = int(len(accepted))
        bad_count = int(accepted[risk_col].fillna(True).astype(bool).sum()) if accepted_count else 0
        rows.append(
            {
                "dataset": dataset,
                "generator": generator,
                "target_file": target,
                "policy": policy,
                "score_name": score_name,
                "alpha": math.nan if alpha is None else alpha,
                "repair_rmsd_max": math.nan if repair_rmsd_max is None else repair_rmsd_max,
                "k": k,
                "molecules_scanned": int(len(group)),
                "eligible": int(len(candidates)),
                "accepted": accepted_count,
                "bad_count": bad_count,
                "bad_per_budget": float(bad_count / k),
                "yield_per_budget": float(accepted_count / k),
                "accepted_bad_rate": float(bad_count / accepted_count) if accepted_count else math.nan,
                "accepted_qed_mean": float(accepted[qed_col].mean()) if accepted_count else math.nan,
                "accepted_energy_per_heavy_mean": float(accepted[energy_col].mean()) if accepted_count else math.nan,
                "accepted_repair_rmsd_mean": float(accepted["repair_rmsd"].mean()) if "repair" in policy and accepted_count else math.nan,
                "success_at_1": int(accepted_count >= 1),
                "success_at_k": int(accepted_count >= k),
            }
        )
    return pd.DataFrame(rows)


def policy_target_tables(
    dataset: str,
    table: pd.DataFrame,
    thresholds: dict[tuple[str, float], float],
) -> pd.DataFrame:
    frames = []
    for generator, sub in table.groupby("generator"):
        for k in KS:
            frames.append(
                target_rows_for_policy(
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
                frames.append(
                    target_rows_for_policy(
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
                for alpha in ALPHAS:
                    tau = thresholds[(score_name, alpha)]
                    frames.append(
                        target_rows_for_policy(
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
                        frames.append(
                            target_rows_for_policy(
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
    return pd.concat(frames, ignore_index=True)


def aggregate(target_policy: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "generator", "policy", "score_name", "alpha", "repair_rmsd_max", "k"]
    rows = []
    for keys, sub in target_policy.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "targets": int(len(sub)),
                "molecules_scanned": int(sub["molecules_scanned"].sum()),
                "eligible": int(sub["eligible"].sum()),
                "accepted": int(sub["accepted"].sum()),
                "target_success_at_1": float(sub["success_at_1"].mean()),
                "target_success_at_k": float(sub["success_at_k"].mean()),
                "mean_accepted_per_target": float(sub["accepted"].mean()),
                "bad_per_budget_mean": float(sub["bad_per_budget"].mean()),
                "yield_per_budget_mean": float(sub["yield_per_budget"].mean()),
                "accepted_bad_rate": float(sub["bad_count"].sum() / sub["accepted"].sum()) if sub["accepted"].sum() else math.nan,
                "accepted_qed_mean": float(sub["accepted_qed_mean"].mean()),
                "accepted_energy_per_heavy_mean": float(sub["accepted_energy_per_heavy_mean"].mean()),
                "accepted_repair_rmsd_mean": float(sub["accepted_repair_rmsd_mean"].mean()),
                "attempts_per_accepted": float(sub["molecules_scanned"].sum() / sub["accepted"].sum())
                if sub["accepted"].sum()
                else math.inf,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    raw = out[out["policy"] == "raw_topk"][
        ["dataset", "generator", "k", "bad_per_budget_mean", "yield_per_budget_mean", "accepted_bad_rate"]
    ].rename(
        columns={
            "bad_per_budget_mean": "raw_bad_per_budget_mean",
            "yield_per_budget_mean": "raw_yield_per_budget_mean",
            "accepted_bad_rate": "raw_accepted_bad_rate",
        }
    )
    out = out.merge(raw, on=["dataset", "generator", "k"], how="left")
    out["delta_bad_per_budget_vs_raw"] = out["bad_per_budget_mean"] - out["raw_bad_per_budget_mean"]
    out["delta_yield_per_budget_vs_raw"] = out["yield_per_budget_mean"] - out["raw_yield_per_budget_mean"]
    out["delta_accepted_bad_rate_vs_raw"] = out["accepted_bad_rate"] - out["raw_accepted_bad_rate"]
    return out.sort_values(["dataset", "generator", "k", "policy", "score_name", "alpha", "repair_rmsd_max"])


def bootstrap_ci(delta: np.ndarray, reps: int, seed: int) -> tuple[float, float, float]:
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(reps, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_test_p(delta: np.ndarray) -> tuple[int, int, float]:
    delta = delta[np.isfinite(delta)]
    pos = int((delta > 0).sum())
    neg = int((delta < 0).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, math.nan
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return pos, neg, float(min(1.0, 2.0 * p))


def paired_stats(target_policy: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    raw = target_policy[target_policy["policy"] == "raw_topk"][
        ["dataset", "generator", "target_file", "k", "bad_per_budget", "yield_per_budget"]
    ].rename(
        columns={
            "bad_per_budget": "raw_bad_per_budget",
            "yield_per_budget": "raw_yield_per_budget",
        }
    )
    candidates = target_policy[target_policy["policy"] != "raw_topk"].copy()
    merge_cols = ["dataset", "generator", "target_file", "k"]
    joined = candidates.merge(raw, on=merge_cols, how="left")
    joined["delta_bad_per_budget"] = joined["bad_per_budget"] - joined["raw_bad_per_budget"]
    joined["delta_yield_per_budget"] = joined["yield_per_budget"] - joined["raw_yield_per_budget"]
    group_cols = ["dataset", "generator", "policy", "score_name", "alpha", "repair_rmsd_max", "k"]
    for keys, sub in joined.groupby(group_cols, dropna=False):
        bad_mean, bad_lo, bad_hi = bootstrap_ci(sub["delta_bad_per_budget"].to_numpy(), reps, seed)
        yield_mean, yield_lo, yield_hi = bootstrap_ci(sub["delta_yield_per_budget"].to_numpy(), reps, seed + 17)
        bad_pos, bad_neg, bad_sign_p = sign_test_p(sub["delta_bad_per_budget"].to_numpy())
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_targets": int(len(sub)),
                "raw_bad_per_budget_mean": float(sub["raw_bad_per_budget"].mean()),
                "policy_bad_per_budget_mean": float(sub["bad_per_budget"].mean()),
                "mean_delta_bad_per_budget": bad_mean,
                "bad_bootstrap95_low": bad_lo,
                "bad_bootstrap95_high": bad_hi,
                "bad_delta_positive_targets": bad_pos,
                "bad_delta_negative_targets": bad_neg,
                "bad_delta_sign_p": bad_sign_p,
                "raw_yield_per_budget_mean": float(sub["raw_yield_per_budget"].mean()),
                "policy_yield_per_budget_mean": float(sub["yield_per_budget"].mean()),
                "mean_delta_yield_per_budget": yield_mean,
                "yield_bootstrap95_low": yield_lo,
                "yield_bootstrap95_high": yield_hi,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "generator", "k", "policy", "score_name", "alpha", "repair_rmsd_max"])


def csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


def key_tables(summary: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = summary[
        (summary["k"] == 4)
        & (
            (summary["policy"] == "raw_topk")
            | ((summary["policy"] == "repair_rmsd_topk") & (summary["repair_rmsd_max"].isin([1.0, 1.5])))
            | (
                (summary["policy"] == "source_select_topk")
                & (summary["alpha"] == 0.10)
                & summary["score_name"].isin(KEY_SCORE_ORDER)
            )
            | (
                (summary["policy"] == "source_select_then_repair_topk")
                & (summary["alpha"] == 0.10)
                & (summary["repair_rmsd_max"].isin([1.0, 1.5]))
                & summary["score_name"].isin(KEY_SCORE_ORDER)
            )
        )
    ][
        [
            "dataset",
            "generator",
            "policy",
            "score_name",
            "alpha",
            "repair_rmsd_max",
            "accepted",
            "target_success_at_k",
            "bad_per_budget_mean",
            "yield_per_budget_mean",
            "accepted_bad_rate",
            "delta_bad_per_budget_vs_raw",
            "accepted_repair_rmsd_mean",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate", "accepted"], ascending=[True, True, True, False])

    sensitivity = summary[
        (summary["k"] == 4)
        & (summary["policy"] == "source_select_then_repair_topk")
        & summary["score_name"].isin(KEY_SCORE_ORDER)
        & summary["repair_rmsd_max"].isin(RMSD_LIMITS)
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "repair_rmsd_max",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_per_budget_mean",
            "yield_per_budget_mean",
            "accepted_repair_rmsd_mean",
        ]
    ].sort_values(["dataset", "generator", "score_name", "alpha", "repair_rmsd_max"])

    stats_key = stats[
        (stats["k"] == 4)
        & (stats["policy"] == "source_select_then_repair_topk")
        & (stats["alpha"] == 0.10)
        & (stats["repair_rmsd_max"].isin([1.0, 1.5]))
        & stats["score_name"].isin(KEY_SCORE_ORDER)
    ][
        [
            "dataset",
            "generator",
            "policy",
            "score_name",
            "alpha",
            "repair_rmsd_max",
            "n_targets",
            "raw_bad_per_budget_mean",
            "policy_bad_per_budget_mean",
            "mean_delta_bad_per_budget",
            "bad_bootstrap95_low",
            "bad_bootstrap95_high",
            "bad_delta_sign_p",
            "raw_yield_per_budget_mean",
            "policy_yield_per_budget_mean",
            "mean_delta_yield_per_budget",
            "yield_bootstrap95_low",
            "yield_bootstrap95_high",
        ]
    ].sort_values(["dataset", "generator", "score_name", "repair_rmsd_max"])
    return main, sensitivity, stats_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-audit", action="append", required=True, help="name=/path/to/molecule_physics_audit.csv")
    parser.add_argument("--repair-table", action="append", required=True, help="name=/path/to/repair_molecule_table.csv")
    parser.add_argument("--source-audit", type=Path, default=Path("outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_policy_robustness_v0/reports"))
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, source_label_thresholds = prepare_source(args.source_audit, args.quantile, args.seed)
    source_train = source[source["split"] == "train"].copy()
    source_cal = source[source["split"] == "cal"].copy()
    models = fit_models(source_train, args.seed)
    source_scores = score_baselines(source_cal)
    source_scores.update(model_scores(models, source_cal))
    thresholds = source_thresholds(source_cal, source_scores)

    target_specs = dict(spec.split("=", 1) for spec in args.target_audit)
    repair_specs = dict(spec.split("=", 1) for spec in args.repair_table)
    missing = sorted(set(target_specs) ^ set(repair_specs))
    if missing:
        raise ValueError(f"Target/repair dataset mismatch: {missing}")

    target_frames = []
    for name in sorted(target_specs):
        target = prepare_target(Path(target_specs[name]))
        repair = pd.read_csv(repair_specs[name])
        merged = attach_scores(target, repair, models)
        target_frames.append(policy_target_tables(name, merged, thresholds))
    target_policy = pd.concat(target_frames, ignore_index=True)
    summary = aggregate(target_policy)
    stats = paired_stats(target_policy, args.bootstrap_reps, args.seed)
    main_table, sensitivity_table, stats_key = key_tables(summary, stats)

    target_policy.to_csv(args.output_dir / "target_policy_outcomes.csv", index=False)
    summary.to_csv(args.output_dir / "policy_sweep_summary.csv", index=False)
    stats.to_csv(args.output_dir / "paired_bootstrap_stats.csv", index=False)
    main_table.to_csv(args.output_dir / "final_main_policy_table.csv", index=False)
    sensitivity_table.to_csv(args.output_dir / "final_alpha_rmsd_sensitivity.csv", index=False)
    stats_key.to_csv(args.output_dir / "final_paired_stats_key.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "target_audits": target_specs,
                "repair_tables": repair_specs,
                "source_audit": str(args.source_audit),
                "source_label_thresholds": source_label_thresholds,
                "alphas": ALPHAS,
                "repair_rmsd_limits": RMSD_LIMITS,
                "ks": KS,
                "scores": SCORES,
                "bootstrap_reps": args.bootstrap_reps,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Final Policy Robustness v0",
        "",
        "Sweeps alpha, repair RMSD cap, and per-target k. Paired bootstrap is computed over targets.",
        "",
        "## Main Policy Table",
        csv_block(main_table),
        "",
        "## Alpha / RMSD Sensitivity",
        csv_block(sensitivity_table),
        "",
        "## Key Paired Bootstrap Statistics",
        csv_block(stats_key),
    ]
    (args.output_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
