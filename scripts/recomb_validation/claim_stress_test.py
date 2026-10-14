from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


METRICS = {
    "pass": 1,
    "input_to_redock_rmsd": -1,
    "redock_affinity": -1,
    "redock_cnnscore": 1,
    "eval_qed": 1,
    "eval_sa_score": -1,
    "contact_jaccard": 1,
}


def as_bool(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))


def paired_binary(frame: pd.DataFrame, outcome: str, out: Path) -> pd.DataFrame:
    records = []
    for dataset, section in frame.groupby("dataset"):
        wide = section.pivot(index="target_file", columns="arm", values=outcome).dropna()
        for comparator in ("raw_topk", "ligand_only_mmff_uff"):
            base = as_bool(wide[comparator])
            method = as_bool(wide["physcal_balanced"])
            gains = int((~base & method).sum())
            losses = int((base & ~method).sum())
            differences = method.to_numpy(dtype=int) - base.to_numpy(dtype=int)
            rng = np.random.default_rng(20261014)
            samples = rng.integers(0, len(differences), size=(10000, len(differences)))
            lower, upper = np.quantile(differences[samples].mean(axis=1), [0.025, 0.975])
            records.append({"dataset": dataset, "outcome": outcome, "comparator": comparator,
                            "paired_targets": len(wide), "method_pass": int(method.sum()),
                            "comparator_pass": int(base.sum()), "gains": gains, "losses": losses,
                            "risk_difference": float(method.mean() - base.mean()),
                            "difference_ci95_lower": float(lower), "difference_ci95_upper": float(upper),
                            "exact_two_sided_p": float(binomtest(gains, gains + losses, 0.5).pvalue) if gains + losses else 1.0})
    result = pd.DataFrame(records)
    result.to_csv(out / f"paired_{outcome}.csv", index=False)
    return result


def external_checks(repo: Path, protein_dir: Path, postprocess_dir: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    protein = pd.read_csv(protein_dir / "protein_posebusters_rows.csv", low_memory=False)
    mol = pd.read_csv(postprocess_dir / "independent_posebusters_rows.csv", low_memory=False)
    keys = ["dataset", "target_file", "arm"]
    joined = protein.merge(mol[keys + ["posebusters_mol_pass"]], on=keys, how="left", validate="one_to_one")
    joined["protein_pass"] = as_bool(joined["protein_pass"])
    joined["posebusters_mol_pass"] = as_bool(joined["posebusters_mol_pass"])
    joined["aligned"] = as_bool(joined["aligned"])
    joined["mol_cond_loaded"] = as_bool(joined["mol_cond_loaded"])
    joined["combined_pass"] = joined["protein_pass"] & joined["posebusters_mol_pass"]
    joined.to_csv(out / "external_checks_joined.csv", index=False)
    usable = joined.loc[joined["aligned"] & joined["mol_cond_loaded"]].copy()
    complete = usable.groupby(["dataset", "target_file"])["arm"].nunique().eq(3)
    keys_complete = set(complete[complete].index)
    usable = usable.loc[usable.apply(lambda row: (row["dataset"], row["target_file"]) in keys_complete, axis=1)]
    summary = usable.groupby(["dataset", "arm"]).agg(
        targets=("target_file", "size"), mol_pass=("posebusters_mol_pass", "mean"),
        protein_pass=("protein_pass", "mean"), combined_pass=("combined_pass", "mean"),
        nearest_protein_angstrom=("nearest_protein_angstrom", "median"),
    ).reset_index()
    summary.to_csv(out / "external_checks_summary.csv", index=False)
    paired = pd.concat([paired_binary(usable, outcome, out) for outcome in ("posebusters_mol_pass", "protein_pass", "combined_pass")], ignore_index=True)
    paired.to_csv(out / "external_checks_all_paired.csv", index=False)
    return summary, paired


def bootstrap_target_mean(target_differences: np.ndarray, rng: np.random.Generator, replications: int = 5000) -> tuple[float, float]:
    samples = rng.integers(0, len(target_differences), size=(replications, len(target_differences)))
    boot = target_differences[samples].mean(axis=1)
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def matched_budget(postprocess_dir: Path, out: Path) -> pd.DataFrame:
    rows = pd.read_csv(postprocess_dir / "matched_score_budget_rows.csv", low_memory=False)
    rng = np.random.default_rng(20261014)
    records = []
    for dataset, sub in rows.groupby("dataset"):
        for comparator in ("raw_topk", "vina_topk", "pareto_only", "uniform_weighting"):
            for metric, direction in METRICS.items():
                pair = sub[sub["policy"].isin(("physcal_balanced", comparator))]
                wide = pair.pivot(index=["target_file", "seed"], columns="policy", values=metric).dropna()
                paired_diff = pd.to_numeric(wide["physcal_balanced"], errors="coerce") - pd.to_numeric(wide[comparator], errors="coerce")
                by_target = paired_diff.groupby(level="target_file").mean().dropna().to_numpy(dtype=float)
                if len(by_target) == 0:
                    continue
                lower, upper = bootstrap_target_mean(by_target, rng)
                records.append({"dataset": dataset, "comparator": comparator, "metric": metric,
                                "better_direction": "higher" if direction > 0 else "lower",
                                "n_targets": len(by_target), "n_target_seed_pairs": len(wide),
                                "method_minus_comparator": float(by_target.mean()),
                                "ci95_lower": lower, "ci95_upper": upper,
                                "directionally_better": bool(by_target.mean() * direction > 0),
                                "ci95_excludes_zero": bool(lower > 0 or upper < 0)})
    result = pd.DataFrame(records)
    result.to_csv(out / "matched_budget_target_bootstrap.csv", index=False)
    return result


def risk_summary(risk_dir: Path, out: Path) -> pd.DataFrame:
    rows = pd.read_csv(risk_dir / "risk_calibration.csv")
    q90 = rows[rows["quantile"].eq(0.9) & rows["features"].eq("full")].copy()
    q90["nonempty"] = q90["accepted_count"].gt(0)
    q90["exceeds_alpha"] = q90["nonempty"] & q90["accepted_bad"].gt(q90["alpha"])
    summary = q90.groupby(["method", "alpha"]).agg(
        splits=("seed", "nunique"), nonempty=("nonempty", "sum"),
        exceeded=("exceeds_alpha", "sum"), mean_coverage=("coverage", "mean"),
        mean_bad_when_nonempty=("accepted_bad", "mean"),
    ).reset_index()
    summary.to_csv(out / "risk_calibration_q90_claim_check.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protein-dir", type=Path)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protein_dir = args.protein_dir.resolve() if args.protein_dir else repo / "outputs/recomb_protein_validation_20261014"
    postprocess_dir = repo / "outputs/recomb_postprocess_validation_20261013"
    risk_dir = repo / "outputs/recomb_risk_validation_20261013"
    external, paired = external_checks(repo, protein_dir, postprocess_dir, out)
    matched = matched_budget(postprocess_dir, out)
    risk = risk_summary(risk_dir, out)
    protocol = {"bootstrap_unit": "target; ten repeated subsamples averaged within target before resampling",
                "bootstrap_repetitions": 5000, "bootstrap_seed": 20261014,
                "external_paired_difference_ci": "target-paired nonparametric percentile bootstrap, 10000 replicates",
                "external_target_set": "only target triples with aligned public RCSB PDB and all three arms",
                "external_protein_caveat": "Public RCSB-derived protein coordinates are not the original prepared pocket; alignment exclusions are recorded in source rows",
                "protein_result_directory": str(protein_dir),
                "matched_budget_caveat": "precomputed scored rows, not matched generation/docking wall time; raw top-k has one raw row"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("EXTERNAL\n", external.to_string(index=False), "\nPAIRED\n", paired.to_string(index=False), flush=True)
    print("MATCHED\n", matched.to_string(index=False), "\nRISK\n", risk.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
