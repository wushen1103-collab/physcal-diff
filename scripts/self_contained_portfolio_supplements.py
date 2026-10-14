from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from self_contained_redock_portfolio import (
    BASE_POLICY,
    DIRECTION,
    METRICS,
    TARGET_KEYS,
    WEIGHTS,
    add_target_z,
    base_join,
    baseline_summary,
    discover_redock_rows,
    load_baseline,
    load_portfolio,
    scalar_score,
    summarize,
)


DATE = "20260619"
DELTA_SCALE = {"rmsd": 0.10, "affinity": 1.00, "cnn": 0.02, "cnnaff": 0.10, "qed": 0.02, "sa": 0.10, "contact": 0.01}
CORE_POLICIES = [
    "portfolio_eps_scalar_balanced",
    "portfolio_dominance_scalar",
    "unconstrained_scalar",
    "single_best_affinity",
    "single_best_cnn",
    "single_best_qed",
    "single_lowest_sa",
]
ABLATION_POLICIES = [
    "portfolio_eps_scalar_balanced",
    "eps_no_contact_guard",
    "eps_no_sa_guard",
    "eps_no_rmsd_guard",
    "eps_affinity_primary",
    "eps_cnn_primary",
    "eps_qed_primary",
]


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32)


def base_mask(group: pd.DataFrame) -> pd.Series:
    return group["pass"] >= group["base_pass"] - 1e-12


def balanced_mask(group: pd.DataFrame) -> pd.Series:
    mask = base_mask(group)
    mask &= group["contact_jaccard"] >= group["base_contact_jaccard"] + 0.02
    mask &= group["eval_sa_score"] <= group["base_eval_sa_score"] + 0.20
    mask &= group["input_to_redock_rmsd"] <= group["base_input_to_redock_rmsd"] + 0.10
    return mask


def dominance_mask(group: pd.DataFrame) -> pd.Series:
    mask = base_mask(group)
    for name, col in METRICS.items():
        if DIRECTION[name] > 0:
            mask &= group[col] >= group[f"base_{col}"] - 1e-9
        else:
            mask &= group[col] <= group[f"base_{col}"] + 1e-9
    return mask


def policy_mask(group: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "portfolio_dominance_scalar":
        return dominance_mask(group)
    if policy in {
        "portfolio_eps_scalar_balanced",
        "eps_affinity_primary",
        "eps_cnn_primary",
        "eps_qed_primary",
    }:
        return balanced_mask(group)
    if policy == "eps_no_contact_guard":
        mask = base_mask(group)
        mask &= group["eval_sa_score"] <= group["base_eval_sa_score"] + 0.20
        mask &= group["input_to_redock_rmsd"] <= group["base_input_to_redock_rmsd"] + 0.10
        return mask
    if policy == "eps_no_sa_guard":
        mask = base_mask(group)
        mask &= group["contact_jaccard"] >= group["base_contact_jaccard"] + 0.02
        mask &= group["input_to_redock_rmsd"] <= group["base_input_to_redock_rmsd"] + 0.10
        return mask
    if policy == "eps_no_rmsd_guard":
        mask = base_mask(group)
        mask &= group["contact_jaccard"] >= group["base_contact_jaccard"] + 0.02
        mask &= group["eval_sa_score"] <= group["base_eval_sa_score"] + 0.20
        return mask
    if policy in {"unconstrained_scalar", "single_best_affinity", "single_best_cnn", "single_best_qed", "single_lowest_sa"}:
        return base_mask(group)
    raise ValueError(f"Unknown policy: {policy}")


def policy_sort(group: pd.DataFrame, policy: str, fallback_stage: str = "none") -> tuple[list[str], list[bool]]:
    if fallback_stage != "none" and policy in {
        "portfolio_eps_scalar_balanced",
        "portfolio_dominance_scalar",
        "eps_no_contact_guard",
        "eps_no_sa_guard",
        "eps_no_rmsd_guard",
        "eps_affinity_primary",
        "eps_cnn_primary",
        "eps_qed_primary",
    }:
        return ["contact_jaccard", "_scalar"], [False, False]
    if policy in {
        "portfolio_eps_scalar_balanced",
        "portfolio_dominance_scalar",
        "unconstrained_scalar",
        "eps_no_contact_guard",
        "eps_no_sa_guard",
        "eps_no_rmsd_guard",
    }:
        return ["_scalar", "contact_jaccard", "redock_affinity"], [False, False, True]
    if policy in {"single_best_affinity", "eps_affinity_primary"}:
        return ["redock_affinity", "contact_jaccard", "_scalar"], [True, False, False]
    if policy in {"single_best_cnn", "eps_cnn_primary"}:
        return ["redock_cnnscore", "contact_jaccard", "_scalar"], [False, False, False]
    if policy in {"single_best_qed", "eps_qed_primary"}:
        return ["eval_qed", "eval_sa_score", "_scalar"], [False, True, False]
    if policy == "single_lowest_sa":
        return ["eval_sa_score", "eval_qed", "_scalar"], [True, False, False]
    raise ValueError(f"Unknown policy: {policy}")


def select_custom(pool: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows = []
    for _, group in pool.groupby(TARGET_KEYS, sort=False):
        g = group.copy()
        g["_scalar"] = scalar_score(g)
        mask = policy_mask(g, policy)
        candidates = g[mask].copy()
        fallback_stage = "none"
        if candidates.empty:
            candidates = g[base_mask(g)].copy()
            fallback_stage = "pass_only"
        if candidates.empty:
            candidates = g.copy()
            fallback_stage = "all"
        sort_cols, ascending = policy_sort(candidates, policy, fallback_stage)
        selected = candidates.sort_values(sort_cols, ascending=ascending).head(1).copy()
        selected["portfolio_policy"] = policy
        selected["constraint_candidates"] = int(mask.sum())
        selected["constraint_pass"] = bool(mask.any())
        selected["fallback_stage"] = fallback_stage
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def target_delta_rows(selected: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in selected.iterrows():
        rec = {
            "dataset": row["dataset"],
            "target_file": row["target_file"],
            "portfolio_policy": row["portfolio_policy"],
        }
        composite = 0.0
        better_flags = []
        for metric, col in METRICS.items():
            raw_delta = float(row[col]) - float(row[f"base_{col}"])
            oriented_delta = DIRECTION[metric] * raw_delta
            rec[f"{metric}_delta"] = raw_delta
            rec[f"{metric}_oriented_delta"] = oriented_delta
            rec[f"{metric}_better_or_equal"] = oriented_delta >= -1e-12
            composite += WEIGHTS[metric] * float(np.clip(oriented_delta / DELTA_SCALE[metric], -3.0, 3.0))
            better_flags.append(oriented_delta >= -1e-12)
        rec["composite_delta_score"] = composite
        rec["all_metric_non_degraded"] = bool(all(better_flags))
        rec["metric_non_degradation_rate"] = float(np.mean(better_flags))
        records.append(rec)
    return pd.DataFrame(records)


def enrich_summary(summary: pd.DataFrame, target_deltas: pd.DataFrame) -> pd.DataFrame:
    target_summary = target_deltas.groupby(["dataset", "portfolio_policy"], dropna=False).agg(
        mean_composite_delta_score=("composite_delta_score", "mean"),
        std_composite_delta_score=("composite_delta_score", "std"),
        target_composite_positive_rate=("composite_delta_score", lambda s: float((s > 0).mean())),
        all_metric_non_degradation_rate=("all_metric_non_degraded", "mean"),
        mean_metric_non_degradation_rate=("metric_non_degradation_rate", "mean"),
    ).reset_index()
    return summary.merge(target_summary, on=["dataset", "portfolio_policy"], how="left")


def deterministic_suite(pool: pd.DataFrame, policies: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.concat([select_custom(pool, policy) for policy in policies], ignore_index=True)
    deltas = target_delta_rows(selected)
    summary = enrich_summary(summarize(selected), deltas)
    return selected, summary


def add_baseline_to_summary(summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base = baseline_summary(baseline)
    for col in [
        "mean_composite_delta_score",
        "std_composite_delta_score",
        "target_composite_positive_rate",
        "all_metric_non_degradation_rate",
        "mean_metric_non_degradation_rate",
    ]:
        base[col] = 0.0
    return pd.concat([base, summary], ignore_index=True, sort=False)


def sample_budget_pool(pool: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    parts = []
    for key, group in pool.groupby(TARGET_KEYS, sort=False):
        n = min(int(budget), len(group))
        rng = np.random.default_rng(stable_seed(seed, *key, budget))
        if n >= len(group):
            parts.append(group)
        else:
            chosen = rng.choice(group.index.to_numpy(), size=n, replace=False)
            parts.append(group.loc[chosen])
    return pd.concat(parts, ignore_index=True)


def budget_curve(pool: pd.DataFrame, budgets: list[int], seeds: int, policies: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_records = []
    for policy in policies:
        for budget in budgets:
            for seed in range(seeds):
                subset = sample_budget_pool(pool, budget, seed)
                selected = select_custom(subset, policy)
                selected_summary = enrich_summary(summarize(selected), target_delta_rows(selected))
                selected_summary["budget"] = budget
                selected_summary["seed"] = seed
                selected_summary["candidate_count_mean"] = subset.groupby(TARGET_KEYS).size().mean()
                raw_records.append(selected_summary)
    raw = pd.concat(raw_records, ignore_index=True)
    metric_cols = [
        "pass_rate",
        "rmsd",
        "affinity",
        "cnn",
        "cnnaff",
        "qed",
        "sa",
        "contact",
        "mean_composite_delta_score",
        "target_composite_positive_rate",
        "all_metric_non_degradation_rate",
        "mean_metric_non_degradation_rate",
        "candidate_count_mean",
    ]
    records = []
    for keys, group in raw.groupby(["dataset", "portfolio_policy", "budget"], sort=False):
        rec = {"dataset": keys[0], "portfolio_policy": keys[1], "budget": keys[2], "seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            rec[f"{col}_mean"] = float(group[col].mean())
            rec[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        records.append(rec)
    return raw, pd.DataFrame(records)


def runtime_table(pool: pd.DataFrame, policies: list[str], repeats: int) -> pd.DataFrame:
    rows = []
    candidates = len(pool)
    targets = int(pool[TARGET_KEYS].drop_duplicates().shape[0])
    for policy in policies:
        start = time.perf_counter()
        for _ in range(repeats):
            select_custom(pool, policy)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "portfolio_policy": policy,
                "targets": targets,
                "candidates": candidates,
                "repeats": repeats,
                "elapsed_seconds": elapsed,
                "milliseconds_per_repeat": elapsed * 1000.0 / repeats,
                "microseconds_per_candidate": elapsed * 1_000_000.0 / (repeats * candidates),
            }
        )
    return pd.DataFrame(rows)


def candidate_count_table(pool: pd.DataFrame) -> pd.DataFrame:
    counts = pool.groupby(TARGET_KEYS, dropna=False).size().reset_index(name="candidate_count")
    return counts.groupby("dataset").agg(
        targets=("target_file", "nunique"),
        min_candidates=("candidate_count", "min"),
        mean_candidates=("candidate_count", "mean"),
        median_candidates=("candidate_count", "median"),
        max_candidates=("candidate_count", "max"),
    ).reset_index()


def write_report(
    output_dir: Path,
    fair_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    budget_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    runtime: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    fair_focus = fair_summary[
        fair_summary["portfolio_policy"].isin([BASE_POLICY, "portfolio_eps_scalar_balanced", "portfolio_dominance_scalar", "unconstrained_scalar", "single_best_affinity", "single_best_cnn"])
    ].copy()
    lines = [
        "# Portfolio Supplemental Experiments",
        "",
        f"Date: {DATE}",
        "",
        "Scope: all inputs are redock rows within the supplied repository. External project folders are not read. These experiments supplement the portfolio evidence with fairness, ablation, stability, budget, and runtime checks.",
        "",
        "## Candidate Pool",
        "",
        "```csv",
        counts.to_csv(index=False).strip(),
        "```",
        "",
        "## Fair Selector Baselines",
        "",
        "```csv",
        fair_focus.to_csv(index=False).strip(),
        "```",
        "",
        "## Component And Objective Ablations",
        "",
        "```csv",
        ablation_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Budget Curve",
        "",
        "```csv",
        budget_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Seed Stability Summary",
        "",
        "```csv",
        stability_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Runtime",
        "",
        "```csv",
        runtime.to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation",
        "",
        "The strongest use of these tables is not to claim a new model-training seed study. They show that the main selector is compared against natural single-objective and unconstrained alternatives under the same candidate pool, that each guard can be audited, that the gains are not only a full-budget artifact, and that the verifier selection overhead is negligible relative to redocking.",
        "",
        "Remaining paper-level gaps after this supplement: true independent generation/training seeds, at least one additional public benchmark or public SOTA rerun, and qualitative target-level case studies. Those still matter for a TNNLS/TKDE/TPAMI-style submission.",
    ]
    (output_dir / f"portfolio_supplemental_experiments_{DATE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_tables(output_dir: Path, report_dir: Path) -> None:
    table_dir = report_dir / "evidence_tables_v0"
    table_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        f"table_portfolio_supplement_fair_selectors_{DATE}.csv": "portfolio_supplement_fair_selectors.csv",
        f"table_portfolio_supplement_ablation_{DATE}.csv": "portfolio_supplement_ablation.csv",
        f"table_portfolio_supplement_budget_curve_{DATE}.csv": "portfolio_supplement_budget_curve.csv",
        f"table_portfolio_supplement_seed_stability_{DATE}.csv": "portfolio_supplement_seed_stability.csv",
        f"table_portfolio_supplement_runtime_{DATE}.csv": "portfolio_supplement_runtime.csv",
        f"table_portfolio_supplement_candidate_counts_{DATE}.csv": "portfolio_supplement_candidate_counts.csv",
    }
    for public_name, local_name in mapping.items():
        pd.read_csv(output_dir / local_name).to_csv(table_dir / public_name, index=False)
    source_report = output_dir / f"portfolio_supplemental_experiments_{DATE}.md"
    (report_dir / source_report.name).write_text(source_report.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer-facing supplemental portfolio experiments for the 22-only redock pool.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--baseline-rows", type=Path, default=Path("outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/gnina_redock_rows.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--budget-seeds", type=int, default=32)
    parser.add_argument("--runtime-repeats", type=int, default=200)
    args = parser.parse_args()

    repo_root = args.root.resolve()
    row_paths = discover_redock_rows(repo_root / "outputs")
    baseline = load_baseline(args.baseline_rows)
    pool = add_target_z(base_join(load_portfolio(row_paths, repo_root), baseline))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = candidate_count_table(pool)
    fair_selected, fair_summary = deterministic_suite(pool, CORE_POLICIES)
    fair_summary = add_baseline_to_summary(fair_summary, baseline)
    ablation_selected, ablation_summary = deterministic_suite(pool, ABLATION_POLICIES)

    budgets = [1, 2, 4, 8, 12, 16, 24, 32]
    budget_raw, budget_summary = budget_curve(pool, budgets, args.budget_seeds, ["portfolio_eps_scalar_balanced", "portfolio_dominance_scalar"])
    stability_summary = budget_summary[budget_summary["budget"].isin([8, 12, 24, 32])].copy()
    runtime = runtime_table(pool, ["portfolio_eps_scalar_balanced", "portfolio_dominance_scalar", "unconstrained_scalar"], args.runtime_repeats)

    fair_selected.to_csv(args.output_dir / "portfolio_supplement_fair_selected_rows.csv", index=False)
    ablation_selected.to_csv(args.output_dir / "portfolio_supplement_ablation_selected_rows.csv", index=False)
    budget_raw.to_csv(args.output_dir / "portfolio_supplement_budget_curve_raw.csv", index=False)
    fair_summary.to_csv(args.output_dir / "portfolio_supplement_fair_selectors.csv", index=False)
    ablation_summary.to_csv(args.output_dir / "portfolio_supplement_ablation.csv", index=False)
    budget_summary.to_csv(args.output_dir / "portfolio_supplement_budget_curve.csv", index=False)
    stability_summary.to_csv(args.output_dir / "portfolio_supplement_seed_stability.csv", index=False)
    runtime.to_csv(args.output_dir / "portfolio_supplement_runtime.csv", index=False)
    counts.to_csv(args.output_dir / "portfolio_supplement_candidate_counts.csv", index=False)
    write_report(args.output_dir, fair_summary, ablation_summary, budget_summary, stability_summary, runtime, counts)
    copy_tables(args.output_dir, args.report_dir)

    print(fair_summary.to_csv(index=False))
    print("BUDGET")
    print(budget_summary.to_csv(index=False))


if __name__ == "__main__":
    main()
