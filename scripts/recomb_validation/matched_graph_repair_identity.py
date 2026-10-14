from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer

from protein_posebusters_validation import local_file


KEY = ["dataset", "generator", "target_file", "relative_file", "mol_idx"]


def normalized(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"available": False}
    try:
        mol = next((mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol), None)
        if mol is None:
            return {"available": False}
        mol = Chem.RemoveHs(mol)
        graph = Chem.Mol(mol)
        Chem.RemoveStereochemistry(graph)
        return {"available": True, "connectivity": Chem.MolToSmiles(graph, isomericSmiles=False),
                "qed": float(QED.qed(mol)), "sa": float(sascorer.calculateScore(mol)),
                "heavy_atoms": mol.GetNumHeavyAtoms()}
    except Exception:
        return {"available": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = repo / "outputs/plsafe_verifier_rerank_v0"
    raw = pd.read_csv(base / "poses_external_compare/export_manifest.csv", low_memory=False)
    repaired = pd.read_csv(base / "poses_external_candidate_pool/export_manifest.csv", low_memory=False)
    raw = raw[raw["policy"].eq("raw_topk") & raw["export_ok"].astype(bool)].copy()
    repaired = repaired[repaired["export_ok"].astype(bool)].copy()
    raw = raw[KEY + ["export_sdf"]].drop_duplicates(KEY)
    matched = repaired.merge(raw, on=KEY, how="inner", suffixes=("_repair", "_raw"), validate="many_to_one")
    rows = []
    for row in matched.itertuples():
        raw_path = local_file(repo, row.export_sdf_raw)
        repair_path = local_file(repo, row.export_sdf_repair)
        before, after = normalized(raw_path), normalized(repair_path)
        rows.append({**{key: getattr(row, key) for key in KEY}, **{
            "policy": row.policy, "rank": row.rank, "raw_sdf": str(raw_path or ""),
            "repaired_sdf": str(repair_path or ""),
            "raw_available": before["available"], "repaired_available": after["available"],
            "same_connectivity": bool(before.get("connectivity") and before.get("connectivity") == after.get("connectivity")),
            "raw_qed": before.get("qed", np.nan), "repaired_qed": after.get("qed", np.nan),
            "raw_sa": before.get("sa", np.nan), "repaired_sa": after.get("sa", np.nan),
            "raw_heavy_atoms": before.get("heavy_atoms", np.nan),
            "repaired_heavy_atoms": after.get("heavy_atoms", np.nan),
        }})
    result = pd.DataFrame(rows)
    if len(result):
        result["qed_abs_change"] = (result["raw_qed"] - result["repaired_qed"]).abs()
        result["sa_abs_change"] = (result["raw_sa"] - result["repaired_sa"]).abs()
    result.to_csv(out / "raw_repair_matched_graph_pairs.csv", index=False)
    summary = result.groupby("dataset").agg(pairs=("same_connectivity", "size"),
        source_candidates=("raw_sdf", "nunique"), same_connectivity=("same_connectivity", "sum"),
        qed_changed=("qed_abs_change", lambda value: int(value.gt(1e-5).sum())),
        sa_changed=("sa_abs_change", lambda value: int(value.gt(1e-5).sum())),
        missing_raw=("raw_available", lambda value: int((~value).sum())),
        missing_repaired=("repaired_available", lambda value: int((~value).sum()))).reset_index()
    summary.to_csv(out / "raw_repair_graph_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "source": "22-local raw_topk and repaired export manifests/SDFs",
        "pair_key": KEY, "normalization": "RDKit RemoveHs, strip stereo for connectivity, one SA_Score implementation",
        "scope": "only source candidates with both 22-local SDF exports; not all repaired source candidates",
        "tolerance": 1e-5,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
