from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


KEY = ["dataset", "target_file"]
ORIGINAL_KEY = ["dataset", "generator", "target_file", "relative_file", "mol_idx"]


def is_true(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def add_imports(repo: Path):
    sys.path.insert(0, str(repo / "scripts"))
    import self_contained_redock_portfolio as portfolio
    import self_contained_portfolio_supplements as supplement
    import tnnls_posebusters_ligand_quality as posechecks
    import forcefield_repair_ablation as ligand_repair
    from rdkit import Chem

    return portfolio, supplement, posechecks, ligand_repair, Chem


def original_molecule_audit(repo: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pd.read_csv(repo / "outputs/plsafe_verifier_rerank_v0/reports/plsafe_verifier_candidate_pool.csv", low_memory=False)
    selected = pd.read_csv(repo / "outputs/plsafe_verifier_rerank_v0/reports/accepted_plsafe_verifier_metrics.csv", low_memory=False)
    rows = []
    for subset_name, data in (("candidate_pool", pool), ("selected", selected)):
        for dataset, group in data.groupby("dataset", sort=False):
            qed_delta = pd.to_numeric(group["eval_qed"], errors="coerce") - pd.to_numeric(group["raw_qed_eval"], errors="coerce")
            sa_delta = pd.to_numeric(group["eval_sa_score"], errors="coerce") - pd.to_numeric(group["raw_sa_score"], errors="coerce")
            rows.append({
                "subset": subset_name, "dataset": dataset, "rows": len(group),
                "qed_changed_rows": int(qed_delta.abs().gt(1e-6).sum()),
                "qed_max_abs_change": float(qed_delta.abs().max()),
                "sa_changed_rows": int(sa_delta.abs().gt(1e-6).sum()),
                "sa_max_abs_change": float(sa_delta.abs().max()),
                "sa_mean_change": float(sa_delta.mean()),
                "unique_original_candidates": int(group[ORIGINAL_KEY].drop_duplicates().shape[0]),
            })
    by_policy = []
    for keys, group in selected.groupby(["dataset", "generator", "policy"], sort=False):
        by_policy.append({"dataset": keys[0], "generator": keys[1], "policy": keys[2],
                          "rows": len(group), "unique_candidates": int(group[ORIGINAL_KEY].drop_duplicates().shape[0]),
                          "raw_bad_fraction_of_selected": float(is_true(group["raw_high_risk_fixed"]).mean()),
                          "post_bad_fraction_of_selected": float(is_true(group["pl_repaired_high_risk_fixed"]).mean()),
                          "mean_qed": float(pd.to_numeric(group["eval_qed"], errors="coerce").mean()),
                          "mean_sa": float(pd.to_numeric(group["eval_sa_score"], errors="coerce").mean())})
    pd.DataFrame(rows).to_csv(out / "same_candidate_qed_sa_audit.csv", index=False)
    pd.DataFrame(by_policy).to_csv(out / "selected_candidate_identity.csv", index=False)
    return pool, selected


def repair_decomposition(pool: pd.DataFrame, selected: pd.DataFrame, out: Path) -> None:
    input_candidates = pool[ORIGINAL_KEY + ["raw_high_risk_fixed", "pre_relax_ff_rf"]].drop_duplicates(ORIGINAL_KEY).copy()
    input_candidates["raw_bad"] = is_true(input_candidates["raw_high_risk_fixed"])
    variants = pool[ORIGINAL_KEY + ["pl_repaired_high_risk_fixed", "pl_repair_ok", "pl_repair_rmsd", "plsafe_ok"]].copy()
    variants["repaired_good"] = ~is_true(variants["pl_repaired_high_risk_fixed"]) & is_true(variants["pl_repair_ok"]) & is_true(variants["plsafe_ok"])
    variants["pl_repair_rmsd"] = pd.to_numeric(variants["pl_repair_rmsd"], errors="coerce")
    good = variants.loc[variants["repaired_good"]].groupby(ORIGINAL_KEY, dropna=False)["pl_repair_rmsd"].min().rename("min_success_rmsd")
    candidates = input_candidates.merge(good.reset_index(), on=ORIGINAL_KEY, how="left")
    selected_keys = selected[ORIGINAL_KEY + ["policy", "raw_high_risk_fixed", "pl_repaired_high_risk_fixed"]].copy()
    selected_keys["raw_bad"] = is_true(selected_keys["raw_high_risk_fixed"])
    selected_keys["final_bad"] = is_true(selected_keys["pl_repaired_high_risk_fixed"])
    records = []
    for keys, g in selected_keys.groupby(["dataset", "generator", "policy"], sort=False):
        n_repaired = int((g["raw_bad"] & ~g["final_bad"]).sum())
        records.append({"dataset": keys[0], "generator": keys[1], "policy": keys[2],
                        "selected": len(g), "direct_good": int((~g["raw_bad"] & ~g["final_bad"]).sum()),
                        "bad_to_good_repair": n_repaired, "bad_remaining": int(g["final_bad"].sum()),
                        "repair_fraction_of_output": n_repaired / len(g)})
    pd.DataFrame(records).to_csv(out / "selected_repair_mechanism.csv", index=False)
    candidate_records = []
    for keys, g in candidates.groupby(["dataset", "generator"], sort=False):
        candidate_records.append({"dataset": keys[0], "generator": keys[1], "input_candidates": len(g),
                                  "raw_bad": int(g["raw_bad"].sum()),
                                  "raw_good": int((~g["raw_bad"]).sum()),
                                  "bad_repairable": int((g["raw_bad"] & g["min_success_rmsd"].notna()).sum()),
                                  "bad_unrepaired": int((g["raw_bad"] & g["min_success_rmsd"].isna()).sum())})
    pd.DataFrame(candidate_records).to_csv(out / "repair_vs_rejection_pool.csv", index=False)

    curves = []
    for keys, g in candidates.groupby(["dataset", "generator"], sort=False):
        for cap in (0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00):
            rescued = g["raw_bad"] & g["min_success_rmsd"].le(cap)
            retained = ~g["raw_bad"] | rescued
            matched_bad = 0
            matched_count = 0
            for _, target in g.groupby("target_file", sort=False):
                accepted_n = int(retained.loc[target.index].sum())
                if accepted_n:
                    ranked = target.sort_values(["pre_relax_ff_rf", "mol_idx"], na_position="last").head(accepted_n)
                    matched_bad += int(ranked["raw_bad"].sum())
                    matched_count += len(ranked)
            curves.append({"dataset": keys[0], "generator": keys[1], "repair_rmsd_cap": cap,
                           "input_candidates": len(g), "accepted": int(retained.sum()),
                           "direct_good": int((~g["raw_bad"]).sum()), "repaired_bad_to_good": int(rescued.sum()),
                           "rejected": int((~retained).sum()), "candidate_coverage": float(retained.mean()),
                           "repair_output_bad": 0.0 if retained.any() else math.nan,
                           "raw_risk_rank_matched_coverage_bad": matched_bad / matched_count if matched_count else math.nan})
    curve = pd.DataFrame(curves)
    curve.to_csv(out / "repair_success_coverage.csv", index=False)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    colors = ["#16697A", "#D55E00", "#4A7C59", "#7A5195", "#666666"]
    for color, ((dataset, generator), part) in zip(colors, curve.groupby(["dataset", "generator"], sort=False)):
        if generator == "targetdiff":
            continue
        ax.plot(part["candidate_coverage"], part["repaired_bad_to_good"] / part["input_candidates"],
                marker="o", ms=3.2, lw=1.1, color=color, label=f"{generator} ({dataset})")
    ax.set(xlabel="Candidate coverage", ylabel="Recovered bad candidates / input")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(color="#DDDDDD", linewidth=0.4)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(pad=1.5)
    fig.savefig(out / "repair_success_coverage.pdf")
    fig.savefig(out / "repair_success_coverage.png", dpi=600)
    plt.close(fig)


def full_pool_repair_audit(repo: Path, out: Path) -> None:
    full = pd.read_csv(repo / "outputs/protein_ligand_constrained_repair_v3/reports/protein_ligand_repair_candidate_pool.csv", low_memory=False)
    original = full[ORIGINAL_KEY + ["raw_high_risk_fixed"]].drop_duplicates(ORIGINAL_KEY).copy()
    original["raw_bad"] = is_true(original["raw_high_risk_fixed"])
    good = is_true(full["pl_repair_ok"]) & ~is_true(full["pl_repaired_high_risk_fixed"])
    good &= is_true(full["strict_interaction_pass"])
    eligible = full.loc[good, ORIGINAL_KEY + ["pl_repair_rmsd"]].copy()
    eligible["pl_repair_rmsd"] = pd.to_numeric(eligible["pl_repair_rmsd"], errors="coerce")
    min_rmsd = eligible.groupby(ORIGINAL_KEY, dropna=False)["pl_repair_rmsd"].min().rename("min_success_rmsd")
    original = original.merge(min_rmsd.reset_index(), on=ORIGINAL_KEY, how="left")
    rows = []
    for keys, part in original.groupby(["dataset", "generator"], sort=False):
        for cap in (0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00):
            rescued = part["raw_bad"] & part["min_success_rmsd"].le(cap)
            retained = ~part["raw_bad"] | rescued
            rows.append({"dataset": keys[0], "generator": keys[1], "repair_rmsd_cap": cap,
                         "input_candidates": len(part), "raw_bad": int(part["raw_bad"].sum()),
                         "raw_good": int((~part["raw_bad"]).sum()),
                         "repaired_bad_to_good": int(rescued.sum()),
                         "bad_unrepaired_or_above_cap": int((part["raw_bad"] & ~rescued).sum()),
                         "accepted": int(retained.sum()), "candidate_coverage": float(retained.mean()),
                         "post_bad_rate": 0.0 if retained.any() else math.nan})
    pd.DataFrame(rows).to_csv(out / "full_pool_repair_success_coverage.csv", index=False)


def prepare_portfolio(repo: Path, portfolio):
    baseline_file = repo / "outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/gnina_redock_rows.csv"
    baseline = portfolio.load_baseline(baseline_file)
    paths = portfolio.discover_redock_rows(repo / "outputs")
    pool = portfolio.add_target_z(portfolio.base_join(portfolio.load_portfolio(paths, repo), baseline))
    return baseline, pool, baseline_file


def sampled_group(group: pd.DataFrame, budget: int, seed: int, supplement) -> pd.DataFrame:
    if len(group) <= budget:
        return group.copy()
    raw = group[group["candidate_policy"].eq("raw_topk")].sort_values(["rank", "mol_idx"], na_position="last").head(1)
    keep = raw.index.to_list()
    remainder = group.drop(index=keep)
    rng = np.random.default_rng(supplement.stable_seed(seed, *[str(group.iloc[0][key]) for key in KEY], budget))
    chosen = rng.choice(remainder.index.to_numpy(), size=budget - len(keep), replace=False)
    return group.loc[keep + list(chosen)].copy()


def pareto_front(group: pd.DataFrame, portfolio) -> pd.DataFrame:
    columns = list(portfolio.METRICS.values())
    oriented = group[columns].to_numpy(dtype=float) * np.array([portfolio.DIRECTION[name] for name in portfolio.METRICS])
    dominant = np.zeros(len(group), dtype=bool)
    for i in range(len(group)):
        for j in range(len(group)):
            if i != j and np.all(oriented[j] >= oriented[i] - 1e-10) and np.any(oriented[j] > oriented[i] + 1e-10):
                dominant[i] = True
                break
    return group.iloc[np.flatnonzero(~dominant)]


def select_one(group: pd.DataFrame, policy_name: str, portfolio, supplement) -> tuple[pd.Series | None, int, str]:
    if policy_name == "raw_topk":
        eligible = group.loc[group["candidate_policy"].eq("raw_topk")]
        ranked = eligible.sort_values(["rank", "mol_idx"], na_position="last")
    elif policy_name == "vina_topk":
        eligible = group
        ranked = eligible.sort_values(["redock_affinity", "mol_idx"], na_position="last")
    elif policy_name == "uniform_weighting":
        eligible = group
        ranked = eligible.assign(_uniform=eligible[[f"z_{name}" for name in portfolio.METRICS]].mean(axis=1)).sort_values("_uniform", ascending=False)
    elif policy_name == "pareto_only":
        eligible = pareto_front(group, portfolio)
        ranked = eligible.assign(_uniform=eligible[[f"z_{name}" for name in portfolio.METRICS]].mean(axis=1)).sort_values("_uniform", ascending=False)
    elif policy_name == "physcal_balanced":
        row = supplement.select_custom(group, "portfolio_eps_scalar_balanced").iloc[0]
        return row, int(row["constraint_candidates"]), str(row["fallback_stage"])
    else:
        raise ValueError(policy_name)
    return (ranked.iloc[0] if len(ranked) else None), len(eligible), "none"


def metric_record(row: pd.Series | None, dataset: str, target: str, policy_name: str, budget: int, seed: int, eligible: int, fallback: str):
    record = {"dataset": dataset, "target_file": target, "policy": policy_name, "budget": budget, "seed": seed,
              "selected": bool(row is not None), "eligible_candidates": eligible, "fallback_stage": fallback,
              "score_budget_matched": True, "total_compute_time_matched": False}
    if row is not None:
        for col in ("input_to_redock_rmsd", "redock_affinity", "redock_cnnscore", "redock_cnnaffinity",
                    "eval_qed", "eval_sa_score", "contact_jaccard", "pass"):
            record[col] = float(row[col])
        record["candidate_policy"] = str(row["candidate_policy"])
        record["mol_idx"] = str(row["mol_idx"])
    return record


def same_budget_replay(pool: pd.DataFrame, portfolio, supplement, out: Path, seeds: int):
    records = []
    policies = ("raw_topk", "vina_topk", "pareto_only", "uniform_weighting", "physcal_balanced")
    for seed in range(seeds):
        for key, group in pool.groupby(KEY, sort=False):
            sample = sampled_group(group, 24, seed, supplement)
            sample = portfolio.add_target_z(sample)
            for policy_name in policies:
                row, eligible, fallback = select_one(sample, policy_name, portfolio, supplement)
                records.append(metric_record(row, key[0], key[1], policy_name, 24, seed, eligible, fallback))
    rows = pd.DataFrame(records)
    rows.to_csv(out / "matched_score_budget_rows.csv", index=False)
    summary = rows.groupby(["dataset", "policy"], dropna=False).agg(
        target_seed_pairs=("target_file", "size"), selection_coverage=("selected", "mean"),
        redock_stability_pass=("pass", "mean"), input_to_redock_rmsd=("input_to_redock_rmsd", "mean"),
        vina=("redock_affinity", "mean"), cnn=("redock_cnnscore", "mean"),
        qed=("eval_qed", "mean"), sa=("eval_sa_score", "mean"), contact=("contact_jaccard", "mean"),
        eligible_mean=("eligible_candidates", "mean"),
        fallback_rate=("fallback_stage", lambda s: float((s != "none").mean())),
    ).reset_index()
    summary.to_csv(out / "matched_score_budget_summary.csv", index=False)
    return rows, summary


def physical_guard(group: pd.DataFrame) -> pd.Series:
    values = pd.Series(np.nan, index=group.index, dtype=object)
    for col in ("pl_repaired_high_risk_fixed", "repaired_high_risk_fixed", "raw_high_risk_fixed"):
        if col in group:
            values = values.where(values.notna(), group[col])
    return values.notna() & ~is_true(values)


def reference_free_replay(pool: pd.DataFrame, portfolio, supplement, out: Path):
    rows = []
    for key, group in pool.groupby(KEY, sort=False):
        original = supplement.select_custom(group, "portfolio_eps_scalar_balanced").iloc[0]
        no_rmsd = group.copy()
        no_rmsd["_score_no_rmsd"] = sum(
            portfolio.WEIGHTS[name] * no_rmsd[f"z_{name}"] for name in portfolio.METRICS if name != "rmsd"
        )
        guard = physical_guard(no_rmsd)
        eligible = guard & no_rmsd["contact_jaccard"].ge(no_rmsd["base_contact_jaccard"] + 0.02)
        eligible &= no_rmsd["eval_sa_score"].le(no_rmsd["base_eval_sa_score"] + 0.20)
        stage = "strict"
        if not eligible.any():
            eligible = guard
            stage = "physical_only"
        selected = no_rmsd.loc[eligible].sort_values("_score_no_rmsd", ascending=False).head(1)
        if selected.empty:
            stage = "abstain"
        for policy_name, row, fallback in (("published_selector", original, str(original["fallback_stage"])),
                                            ("no_rmsd_no_reference", selected.iloc[0] if len(selected) else None, stage)):
            rec = metric_record(row, key[0], key[1], policy_name, len(group), 0, int(eligible.sum()), fallback)
            rec["native_ligand_reference_used"] = False
            rec["input_to_redock_rmsd_used_in_selection"] = policy_name == "published_selector"
            rows.append(rec)
    result = pd.DataFrame(rows)
    result.to_csv(out / "reference_free_replay_rows.csv", index=False)
    summary = result.groupby(["dataset", "policy"], dropna=False).agg(
        targets=("target_file", "size"), coverage=("selected", "mean"),
        rmsd_posthoc=("input_to_redock_rmsd", "mean"), vina=("redock_affinity", "mean"),
        cnn=("redock_cnnscore", "mean"), qed=("eval_qed", "mean"),
        sa=("eval_sa_score", "mean"), contact=("contact_jaccard", "mean"),
        fallback_fraction=("fallback_stage", lambda s: float((~s.isin(["none", "strict"])).mean())),
    ).reset_index()
    summary.to_csv(out / "reference_free_replay_summary.csv", index=False)


def portfolio_sensitivity(pool: pd.DataFrame, portfolio, out: Path):
    settings = [
        ("balanced", portfolio.WEIGHTS, 0.02, 0.20, 0.10),
        ("uniform", {name: 1.0 for name in portfolio.METRICS}, 0.02, 0.20, 0.10),
        ("no_contact_weight", {**portfolio.WEIGHTS, "contact": 0.0}, 0.02, 0.20, 0.10),
        ("no_sa_weight", {**portfolio.WEIGHTS, "sa": 0.0}, 0.02, 0.20, 0.10),
        ("contact_0", portfolio.WEIGHTS, 0.00, 0.20, 0.10),
        ("contact_0p05", portfolio.WEIGHTS, 0.05, 0.20, 0.10),
        ("sa_0p10", portfolio.WEIGHTS, 0.02, 0.10, 0.10),
        ("sa_0p40", portfolio.WEIGHTS, 0.02, 0.40, 0.10),
        ("rmsd_0", portfolio.WEIGHTS, 0.02, 0.20, 0.00),
        ("rmsd_0p20", portfolio.WEIGHTS, 0.02, 0.20, 0.20),
    ]
    records = []
    for key, group in pool.groupby(KEY, sort=False):
        for setting, weights, contact_min, sa_max, rmsd_max in settings:
            eligible = group["pass"].ge(group["base_pass"])
            eligible &= group["contact_jaccard"].ge(group["base_contact_jaccard"] + contact_min)
            eligible &= group["eval_sa_score"].le(group["base_eval_sa_score"] + sa_max)
            eligible &= group["input_to_redock_rmsd"].le(group["base_input_to_redock_rmsd"] + rmsd_max)
            family = pareto_front(group.loc[eligible], portfolio) if eligible.any() else group.iloc[0:0]
            choice = group.loc[eligible]
            fallback = "none"
            if choice.empty:
                choice = group.loc[group["pass"].ge(group["base_pass"])]
                fallback = "pass_only"
            if choice.empty:
                choice = group
                fallback = "all"
            score = sum(weights[name] * choice[f"z_{name}"] for name in portfolio.METRICS)
            selected = choice.assign(_score=score).sort_values(["_score", "contact_jaccard"], ascending=[False, False]).iloc[0]
            rec = metric_record(selected, key[0], key[1], setting, len(group), 0, int(eligible.sum()), fallback)
            rec["pareto_family_size_strict"] = len(family)
            records.append(rec)
    rows = pd.DataFrame(records)
    rows.to_csv(out / "portfolio_sensitivity_rows.csv", index=False)
    summary = rows.groupby(["dataset", "policy"], dropna=False).agg(
        targets=("target_file", "size"), fallback_rate=("fallback_stage", lambda s: float((s != "none").mean())),
        strict_eligible_targets=("eligible_candidates", lambda s: float((s > 0).mean())),
        mean_strict_pareto_size=("pareto_family_size_strict", "mean"),
        vina=("redock_affinity", "mean"), qed=("eval_qed", "mean"),
        sa=("eval_sa_score", "mean"), contact=("contact_jaccard", "mean"),
        rmsd=("input_to_redock_rmsd", "mean"),
    ).reset_index()
    summary.to_csv(out / "portfolio_sensitivity_summary.csv", index=False)


def independent_posebusters(repo: Path, out: Path, posechecks, ligand_repair, chem) -> None:
    baseline = pd.read_csv(repo / "outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/gnina_redock_rows.csv", low_memory=False)
    balanced = pd.read_csv(repo / "outputs/self_contained_redock_portfolio_v0/self_contained_portfolio_selected_rows.csv", low_memory=False)
    frame = pd.concat([
        baseline[baseline["policy"].eq("raw_topk")].assign(arm="raw_topk"),
        balanced[balanced["portfolio_policy"].eq("portfolio_eps_scalar_balanced")].assign(arm="physcal_balanced"),
    ], ignore_index=True, sort=False)
    frame = frame[frame["dataset"].isin(["bindingmoad", "prospective"])].copy()
    frame = frame.sort_values(["dataset", "target_file", "arm"]).drop_duplicates(["dataset", "target_file", "arm"])
    rows = []
    ligand_sdfs = out / "ligand_only_sdf"
    ligand_sdfs.mkdir(exist_ok=True)
    for _, row in frame.iterrows():
        local = None
        for col in ("export_sdf", "redock_sdf"):
            value = row.get(col)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                path = repo / value
                try:
                    path.resolve().relative_to(repo)
                except ValueError:
                    continue
                if path.is_file():
                    local = path
                    break
        checks = posechecks.bust_one(local) if local else {"posebusters_error": "missing_22_local_sdf"}
        passed = all(bool(checks.get(col, False)) for col in posechecks.PASS_COLUMNS)
        rows.append({"dataset": row["dataset"], "target_file": row["target_file"], "arm": row["arm"],
                     "sdf_available": local is not None, "posebusters_mol_pass": passed,
                     "error": checks.get("posebusters_error", ""),
                     "failed_checks": ";".join(col for col in posechecks.PASS_COLUMNS if not bool(checks.get(col, False)))})
        if row["arm"] == "raw_topk":
            repaired_path = None
            error = ""
            try:
                if local is None:
                    raise ValueError("missing_22_local_raw_sdf")
                mol = next((m for m in chem.SDMolSupplier(str(local), removeHs=False) if m is not None), None)
                if mol is None:
                    raise ValueError("unreadable_raw_sdf")
                repaired, ff_name, ok, *_ = ligand_repair.minimize_copy(mol, 250)
                if not ok or repaired is None:
                    raise ValueError(f"forcefield_failed:{ff_name}")
                repaired_path = ligand_sdfs / f"{row['dataset']}__{str(row['target_file']).replace('/', '_')}__ligand_only.sdf"
                writer = chem.SDWriter(str(repaired_path))
                writer.write(repaired)
                writer.close()
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
            ligand_checks = posechecks.bust_one(repaired_path) if repaired_path else {"posebusters_error": error}
            ligand_passed = all(bool(ligand_checks.get(col, False)) for col in posechecks.PASS_COLUMNS)
            rows.append({"dataset": row["dataset"], "target_file": row["target_file"], "arm": "ligand_only_mmff_uff",
                         "sdf_available": repaired_path is not None, "posebusters_mol_pass": ligand_passed,
                         "error": ligand_checks.get("posebusters_error", ""),
                         "failed_checks": ";".join(col for col in posechecks.PASS_COLUMNS if not bool(ligand_checks.get(col, False)))})
    results = pd.DataFrame(rows)
    results.to_csv(out / "independent_posebusters_rows.csv", index=False)
    summary = results.groupby(["dataset", "arm"]).agg(targets=("target_file", "size"),
        sdf_available=("sdf_available", "sum"), pass_count=("posebusters_mol_pass", "sum"),
        pass_rate=("posebusters_mol_pass", "mean")).reset_index()
    summary.to_csv(out / "independent_posebusters_summary.csv", index=False)
    comparisons = []
    for dataset, sub in results.groupby("dataset"):
        wide = sub.pivot(index="target_file", columns="arm", values="posebusters_mol_pass")
        if "physcal_balanced" not in wide:
            continue
        for baseline_arm in ("raw_topk", "ligand_only_mmff_uff"):
            if baseline_arm not in wide:
                continue
            paired = wide[[baseline_arm, "physcal_balanced"]].dropna()
            gained = int((~paired[baseline_arm].astype(bool) & paired["physcal_balanced"].astype(bool)).sum())
            lost = int((paired[baseline_arm].astype(bool) & ~paired["physcal_balanced"].astype(bool)).sum())
            comparisons.append({"dataset": dataset, "baseline": baseline_arm, "paired_targets": len(paired),
                                "gained_pass": gained, "lost_pass": lost,
                                "exact_two_sided_p": float(binomtest(gained, gained + lost, 0.5).pvalue) if gained + lost else 1.0})
    pd.DataFrame(comparisons).to_csv(out / "independent_posebusters_paired.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-seeds", type=int, default=10)
    parser.add_argument("--skip-posebusters", action="store_true")
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    portfolio, supplement, posechecks, ligand_repair, chem = add_imports(repo)
    repair_pool, selected = original_molecule_audit(repo, out)
    repair_decomposition(repair_pool, selected, out)
    full_pool_repair_audit(repo, out)
    print("repair and chemical identity audit complete", flush=True)
    baseline, pool, _ = prepare_portfolio(repo, portfolio)
    matched, _ = same_budget_replay(pool, portfolio, supplement, out, args.budget_seeds)
    print(f"matched score budget replay complete: {len(matched)} rows", flush=True)
    reference_free_replay(pool, portfolio, supplement, out)
    portfolio_sensitivity(pool, portfolio, out)
    print("reference-free replay and sensitivity complete", flush=True)
    if not args.skip_posebusters:
        independent_posebusters(repo, out, posechecks, ligand_repair, chem)
        print("PoseBusters independent ligand checks complete", flush=True)
    (out / "protocol.json").write_text(json.dumps({
        "repository": str(repo), "score_budget": 24, "selection_output_per_target": 1,
        "matched_budget_scope": "identical precomputed redocked candidate-row sample for every selector; raw top-k uses raw candidate rows within the same sampled table",
        "matched_total_compute_time": False,
        "reference_free_scope": "no native ligand reference in published selector; new replay also removes input-to-redock RMSD and its pass proxy from selection",
        "physical_validation": "PoseBusters mol mode on 22-local SDFs; ligand-only arm freshly minimized from raw top-1 with existing 250-iteration MMFF/UFF function; protein mode and MD unavailable without 22-local target receptor trajectories",
        "repair_curve": "same 22-local candidate table; a raw bad candidate counts as recovered only when a graph-preserving repaired variant passes the recorded PL-safe check",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
