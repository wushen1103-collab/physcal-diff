from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from protein_posebusters_validation import inspect_one, local_file


def check_one(item: dict[str, object], pocket: str) -> dict[str, object]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tnnls_posebusters_ligand_quality as posechecks

    chain = None if pd.isna(item["chain"]) else str(item["chain"])
    normalized_item = dict(item)
    normalized_item["chain"] = chain
    result = inspect_one(normalized_item, {(str(item["pdb_id"]), chain): pocket})
    sdf = Path(str(item["sdf_path"]))
    checks = posechecks.bust_one(sdf) if sdf.is_file() else {"posebusters_error": "missing_local_sdf"}
    result["mol_pass"] = all(bool(checks.get(name, False)) for name in posechecks.PASS_COLUMNS)
    result["combined_pass"] = bool(result["protein_pass"] and result["mol_pass"])
    result["mol_error"] = checks.get("posebusters_error", "")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pocket-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["bindingmoad", "prospective"], default="bindingmoad")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    repo, out, pocket_dir = args.repo.resolve(), args.out.resolve(), args.pocket_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(repo / "outputs/self_contained_redock_portfolio_v0/self_contained_portfolio_selected_rows.csv", low_memory=False)
    selected = selected[selected["dataset"].eq(args.dataset)].copy()
    sites = pd.read_csv(pocket_dir / "input_manifest.csv")
    sites = sites[sites["arm"].eq("raw_topk")][["target_file", "pdb_id", "chain"]].drop_duplicates("target_file")
    if args.dataset == "bindingmoad":
        pockets = pd.read_csv(pocket_dir / "pocket_reconstruction_manifest.csv")[["target_file", "pocket_path"]]
    else:
        pockets = pd.read_csv(pocket_dir / "download_manifest.csv")[["target_file", "local_path"]]
        pockets = pockets.rename(columns={"local_path": "pocket_path"})
    selected = selected.merge(sites, on="target_file", validate="many_to_one")
    selected = selected.merge(pockets, on="target_file", validate="many_to_one")
    selected["arm"] = selected["portfolio_policy"]
    selected["sdf_path"] = selected.apply(lambda row: str(local_file(repo, row.get("export_sdf")) or
        local_file(repo, row.get("redock_sdf")) or ""), axis=1)
    records = selected[["dataset", "target_file", "arm", "pdb_id", "chain", "sdf_path"]].to_dict("records")
    pocket_paths = selected["pocket_path"].tolist()
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = pd.DataFrame(executor.map(check_one, records, pocket_paths))
    rows.to_csv(out / "variant_independent_checks.csv", index=False)
    complete = rows[rows["aligned"] & rows["mol_cond_loaded"]].copy()
    summary = complete.groupby("arm").agg(n=("target_file", "size"), mol_pass=("mol_pass", "mean"),
        protein_pass=("protein_pass", "mean"), combined_pass=("combined_pass", "mean")).reset_index()
    summary.to_csv(out / "variant_summary.csv", index=False)
    wide = complete.pivot(index="target_file", columns="arm", values="combined_pass").dropna()
    method = wide["portfolio_eps_scalar_balanced"].astype(bool)
    comparisons = []
    for arm in wide.columns:
        if arm == "portfolio_eps_scalar_balanced":
            continue
        other = wide[arm].astype(bool)
        gains, losses = int((method & ~other).sum()), int((~method & other).sum())
        comparisons.append({"comparator": arm, "paired_targets": len(wide), "balanced_pass": int(method.sum()),
                            "comparator_pass": int(other.sum()), "balanced_only": gains, "comparator_only": losses,
                            "paired_difference": float(method.mean() - other.mean()),
                            "exact_two_sided_p": float(binomtest(gains, gains + losses).pvalue) if gains + losses else 1.0})
    paired = pd.DataFrame(comparisons)
    paired.to_csv(out / "variant_paired_comparisons.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({"scope": f"pre-existing selector variants on the same {complete['target_file'].nunique()} {args.dataset} targets",
        "pocket": str(pocket_dir), "mol": "PoseBusters 0.2.2 mol checks", "protein": "PoseBusters 0.2.2 dock protein checks",
        "inference": "descriptive paired exact tests, not multiplicity adjusted; no policy selected from these checks",
        "receptor_caveat": "public native-centered reconstruction" if args.dataset == "bindingmoad" else "public exact-name CrossDocked pocket"}, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
