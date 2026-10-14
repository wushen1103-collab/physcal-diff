from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path

import numpy as np
import pandas as pd

from variant_protein_checks import check_one


def check(item: dict[str, object]) -> dict[str, object]:
    checked = check_one({"dataset": item["dataset"], "target_file": item["target_file"],
                         "arm": item["role"], "pdb_id": item["pdb_id"], "chain": item["chain"],
                         "sdf_path": item["sdf_path"]}, item["pocket_path"])
    return {"dataset": item["dataset"], "target_file": item["target_file"],
            "sdf_path": item["sdf_path"], "role": item["role"],
            "aligned": checked["aligned"], "mol_pass": checked["mol_pass"],
            "protein_pass": checked["protein_pass"], "combined_pass": checked["combined_pass"],
            "no_clashes": checked.get("no_clashes", False),
            "no_volume_clash": checked.get("no_volume_clash", False), "error": checked["error"]}


def pocket_map(root: Path, dataset: str) -> pd.DataFrame:
    manifest = pd.read_csv(root / "input_manifest.csv")
    sites = manifest[manifest["arm"].eq("raw_topk")][["target_file", "pdb_id", "chain"]].drop_duplicates()
    if dataset == "prospective":
        paths = pd.read_csv(root / "download_manifest.csv")[["target_file", "local_path"]]
        paths = paths.rename(columns={"local_path": "pocket_path"})
    else:
        paths = pd.read_csv(root / "pocket_reconstruction_manifest.csv")[["target_file", "pocket_path"]]
    return sites.merge(paths, on="target_file", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--bindingmoad-pockets", type=Path, required=True)
    parser.add_argument("--prospective-pockets", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs.resolve())
    pairs = pairs[pairs["raw_available"] & pairs["repaired_available"] & pairs["same_connectivity"]].copy()
    sites = pd.concat([pocket_map(args.bindingmoad_pockets.resolve(), "bindingmoad"),
                       pocket_map(args.prospective_pockets.resolve(), "prospective")], ignore_index=True)
    long = pd.concat([pairs[["dataset", "target_file", "raw_sdf"]].rename(columns={"raw_sdf": "sdf_path"}).assign(role="raw"),
                      pairs[["dataset", "target_file", "repaired_sdf"]].rename(columns={"repaired_sdf": "sdf_path"}).assign(role="repaired")],
                     ignore_index=True).drop_duplicates(["dataset", "target_file", "sdf_path", "role"])
    long = long.merge(sites, on="target_file", validate="many_to_one")
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        checked = pd.DataFrame(executor.map(check, long.to_dict("records")))
    checked.to_csv(out / "paired_posebuster_unique_sdfs.csv", index=False)
    before = checked[checked["role"].eq("raw")].drop(columns="role").rename(columns={
        key: f"raw_{key}" for key in ("aligned", "mol_pass", "protein_pass", "combined_pass", "no_clashes", "no_volume_clash", "error")})
    after = checked[checked["role"].eq("repaired")].drop(columns="role").rename(columns={
        key: f"repair_{key}" for key in ("aligned", "mol_pass", "protein_pass", "combined_pass", "no_clashes", "no_volume_clash", "error")})
    joined = pairs.merge(before, left_on=["dataset", "target_file", "raw_sdf"],
                         right_on=["dataset", "target_file", "sdf_path"], validate="many_to_one").drop(columns="sdf_path")
    joined = joined.merge(after, left_on=["dataset", "target_file", "repaired_sdf"],
                          right_on=["dataset", "target_file", "sdf_path"], validate="many_to_one").drop(columns="sdf_path")
    joined["combined_gain"] = ~joined["raw_combined_pass"] & joined["repair_combined_pass"]
    joined["combined_loss"] = joined["raw_combined_pass"] & ~joined["repair_combined_pass"]
    joined.to_csv(out / "paired_repair_effect_rows.csv", index=False)
    summaries = []
    rng = np.random.default_rng(22014)
    for dataset, group in joined.groupby("dataset"):
        by_target = group.groupby("target_file").apply(
            lambda part: float(part["repair_combined_pass"].mean() - part["raw_combined_pass"].mean()))
        deltas = by_target.to_numpy(float)
        draws = rng.integers(0, len(deltas), size=(20000, len(deltas)))
        interval = np.quantile(deltas[draws].mean(axis=1), [.025, .975])
        summaries.append({"dataset": dataset, "pairs": len(group), "source_candidates": group["raw_sdf"].nunique(),
                          "targets": group["target_file"].nunique(),
                          "raw_mol_pass": float(group["raw_mol_pass"].mean()),
                          "repair_mol_pass": float(group["repair_mol_pass"].mean()),
                          "raw_protein_pass": float(group["raw_protein_pass"].mean()),
                          "repair_protein_pass": float(group["repair_protein_pass"].mean()),
                          "raw_combined_pass": float(group["raw_combined_pass"].mean()),
                          "repair_combined_pass": float(group["repair_combined_pass"].mean()),
                          "combined_gains": int(group["combined_gain"].sum()),
                          "combined_losses": int(group["combined_loss"].sum()),
                          "paired_target_mean_difference": float(deltas.mean()),
                          "target_bootstrap_ci_low": float(interval[0]),
                          "target_bootstrap_ci_high": float(interval[1])})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "paired_repair_effect_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "scope": "22-local Raw and repaired final-SDF pairs with identical dataset, source file, target and molecule index",
        "design": "same candidate chemistry, independent PoseBusters molecule and public-pocket protein checks",
        "selection_bias": "only source candidates that have both materialized SDFs; not all generated molecules or repair attempts",
        "independence": "paired bootstrap over targets, not over correlated repair variants",
        "pockets": "public native-ligand reconstruction for BindingMOAD; exact-name public CrossDocked file for prospective",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
