from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protein_posebusters_validation import local_file
from variant_protein_checks import check_one


POLICIES = ("raw_topk", "vina_topk", "pareto_only", "uniform_weighting", "physcal_balanced")


def evaluate(item: dict[str, object]) -> dict[str, object]:
    result = check_one({"dataset": item["dataset"], "target_file": item["target_file"],
                        "arm": item["policy"], "pdb_id": item["pdb_id"], "chain": item["chain"],
                        "sdf_path": item["sdf_path"]}, item["pocket_path"])
    return {"dataset": item["dataset"], "target_file": item["target_file"],
            "sdf_path": item["sdf_path"], "pocket_path": item["pocket_path"],
            "aligned": result["aligned"], "mol_pass": result["mol_pass"],
            "protein_pass": result["protein_pass"], "combined_pass": result["combined_pass"],
            "no_clashes": result.get("no_clashes", False),
            "no_volume_clash": result.get("no_volume_clash", False), "error": result["error"]}


def prepare_sites(pocket_dir: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(pocket_dir / "input_manifest.csv")
    frame = frame[frame["arm"].eq("raw_topk")][["dataset", "target_file", "pdb_id", "chain"]].drop_duplicates()
    if dataset == "prospective":
        pockets = pd.read_csv(pocket_dir / "download_manifest.csv")[["target_file", "local_path"]]
        pockets = pockets.rename(columns={"local_path": "pocket_path"})
    else:
        pockets = pd.read_csv(pocket_dir / "pocket_reconstruction_manifest.csv")[["target_file", "pocket_path"]]
    return frame.merge(pockets, on="target_file", validate="one_to_one")


def replay(repo: Path, dataset: str, seeds: int) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(repo / "scripts"))
    import recomb_postprocess_validation as prior

    portfolio, supplement, *_ = prior.add_imports(repo)
    _, pool, _ = prior.prepare_portfolio(repo, portfolio)
    pool = pool[pool["dataset"].eq(dataset)].copy()
    records = []
    for seed in range(seeds):
        for (dataset_name, target), group in pool.groupby(prior.KEY, sort=False):
            sample = prior.sampled_group(group, 24, seed, supplement)
            sample = portfolio.add_target_z(sample)
            for policy in POLICIES:
                row, eligible, fallback = prior.select_one(sample, policy, portfolio, supplement)
                if row is None:
                    records.append({"dataset": dataset_name, "target_file": target,
                                    "policy": policy, "seed": seed, "selected": False})
                    continue
                sdf = local_file(repo, row.get("export_sdf")) or local_file(repo, row.get("redock_sdf"))
                records.append({"dataset": dataset_name, "target_file": target,
                                "policy": policy, "seed": seed, "selected": True,
                                "candidate_policy": row["candidate_policy"], "mol_idx": row["mol_idx"],
                                "sdf_path": str(sdf) if sdf else "", "eligible_candidates": eligible,
                                "fallback_stage": fallback, "redock_affinity": float(row["redock_affinity"]),
                                "redock_cnnscore": float(row["redock_cnnscore"]),
                                "input_to_redock_rmsd": float(row["input_to_redock_rmsd"]),
                                "eval_qed": float(row["eval_qed"]), "eval_sa_score": float(row["eval_sa_score"])})
    return pd.DataFrame(records)


def paired_bootstrap(rows: pd.DataFrame, seed: int = 22014) -> pd.DataFrame:
    targets = rows.groupby(["target_file", "policy"])["combined_pass"].mean().unstack()
    targets = targets.dropna(subset=list(POLICIES))
    rng = np.random.default_rng(seed)
    result = []
    ref = targets["physcal_balanced"].to_numpy(float)
    for policy in POLICIES:
        other = targets[policy].to_numpy(float)
        delta = ref - other
        if not len(delta):
            continue
        sample = rng.integers(0, len(delta), size=(20000, len(delta)))
        ci = np.quantile(delta[sample].mean(axis=1), [.025, .975])
        result.append({"comparator": policy, "paired_targets": len(delta),
                       "balanced_mean": float(ref.mean()), "comparator_mean": float(other.mean()),
                       "difference": float(delta.mean()), "bootstrap_ci_low": float(ci[0]),
                       "bootstrap_ci_high": float(ci[1])})
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pocket-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["bindingmoad", "prospective"], required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = replay(repo, args.dataset, args.seeds)
    selected = selected.merge(prepare_sites(args.pocket_dir.resolve(), args.dataset),
                              on=["dataset", "target_file"], validate="many_to_one")
    selected.to_csv(out / "selected_candidate_rows.csv", index=False)
    unique = selected[selected["sdf_path"].astype(bool) & selected["pocket_path"].astype(bool)]
    unique = unique.drop_duplicates(["target_file", "sdf_path", "pocket_path"])
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        checked = pd.DataFrame(executor.map(evaluate, unique.to_dict("records")))
    checked.to_csv(out / "unique_candidate_checks.csv", index=False)
    joined = selected.merge(checked.drop(columns="dataset"), on=["target_file", "sdf_path", "pocket_path"],
                            how="left", validate="many_to_one")
    joined["missing_check"] = joined["combined_pass"].isna()
    joined.to_csv(out / "matched_physical_rows.csv", index=False)
    summary = joined.groupby("policy").agg(target_seed_pairs=("selected", "size"),
        local_sdf=("sdf_path", lambda values: int(values.astype(bool).sum())),
        checked=("combined_pass", "count"), mol_pass=("mol_pass", "mean"),
        protein_pass=("protein_pass", "mean"), combined_pass=("combined_pass", "mean"),
        vina=("redock_affinity", "mean"), cnn=("redock_cnnscore", "mean"),
        qed=("eval_qed", "mean"), sa=("eval_sa_score", "mean")).reset_index()
    summary.to_csv(out / "matched_physical_summary.csv", index=False)
    paired = paired_bootstrap(joined)
    paired.to_csv(out / "paired_target_bootstrap.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "dataset": args.dataset, "targets": int(selected["target_file"].nunique()),
        "seeds": args.seeds, "score_budget": 24, "output_per_target": 1,
        "same_precomputed_scored_pool": True, "matched_runtime_or_generation": False,
        "pockets": str(args.pocket_dir.resolve()),
        "physical_check": "PoseBusters 0.2.2 molecule plus protein modules on final SDF",
        "paired_ci_unit": "target, averaging seed-specific outcomes before 20000 paired bootstrap resamples",
        "missing_values": "reported explicitly; never imputed as pass",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
