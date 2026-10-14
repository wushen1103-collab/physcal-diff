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


def molecule(path: Path | None):
    if path is None:
        return None
    return next((mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None), None)


def properties(mol) -> dict[str, object]:
    if mol is None:
        return {"smiles": "", "connectivity": "", "qed": np.nan, "sa": np.nan, "heavy_atoms": np.nan}
    mol = Chem.RemoveHs(mol)
    connectivity_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(connectivity_mol)
    connectivity_mol = Chem.RemoveHs(connectivity_mol)
    return {"smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
            "connectivity": Chem.MolToSmiles(connectivity_mol, isomericSmiles=False), "qed": float(QED.qed(mol)),
            "sa": float(sascorer.calculateScore(mol)), "heavy_atoms": mol.GetNumHeavyAtoms()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = repo / "outputs/self_contained_redock_portfolio_v0/self_contained_portfolio_selected_rows.csv"
    rows = pd.read_csv(source, low_memory=False)
    rows = rows[rows["portfolio_policy"].eq("portfolio_eps_scalar_balanced") & rows["dataset"].isin(("bindingmoad", "prospective"))]
    records = []
    for row in rows.itertuples():
        export_path = local_file(repo, row.export_sdf)
        redock_path = local_file(repo, row.redock_sdf)
        before, after = properties(molecule(export_path)), properties(molecule(redock_path))
        records.append({"dataset": row.dataset, "target_file": row.target_file, "mol_idx": row.mol_idx,
                        "candidate_policy": row.candidate_policy, "reported_qed": row.eval_qed,
                        "reported_sa": row.eval_sa_score, "export_path": str(export_path or ""),
                        "redock_path": str(redock_path or ""), "export_smiles": before["smiles"],
                        "redock_smiles": after["smiles"], "same_stereochemical_smiles": bool(before["smiles"] and before["smiles"] == after["smiles"]),
                        "export_connectivity": before["connectivity"], "redock_connectivity": after["connectivity"],
                        "same_connectivity": bool(before["connectivity"] and before["connectivity"] == after["connectivity"]),
                        "export_qed": before["qed"], "redock_qed": after["qed"],
                        "export_sa": before["sa"], "redock_sa": after["sa"],
                        "export_heavy_atoms": before["heavy_atoms"], "redock_heavy_atoms": after["heavy_atoms"]})
    result = pd.DataFrame(records)
    result["reported_vs_export_qed_abs"] = (result["reported_qed"] - result["export_qed"]).abs()
    result["reported_vs_export_sa_abs"] = (result["reported_sa"] - result["export_sa"]).abs()
    result["export_vs_redock_sa_abs"] = (result["export_sa"] - result["redock_sa"]).abs()
    result.to_csv(out / "selected_graph_identity.csv", index=False)
    summary = result.groupby("dataset").agg(n=("target_file", "size"), export_available=("export_smiles", lambda x: x.ne("").sum()),
        redock_available=("redock_smiles", lambda x: x.ne("").sum()),
        same_stereochemical_smiles=("same_stereochemical_smiles", "sum"), same_connectivity=("same_connectivity", "sum"),
        reported_sa_mismatch=("reported_vs_export_sa_abs", lambda x: x.gt(1e-5).sum()),
        sa_changed_after_redock=("export_vs_redock_sa_abs", lambda x: x.gt(1e-5).sum()),
        reported_qed_mismatch=("reported_vs_export_qed_abs", lambda x: x.gt(1e-5).sum())).reset_index()
    summary.to_csv(out / "graph_identity_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({"source": str(source.relative_to(repo)),
        "scope": "selected balanced outputs only; raw pre-repair source SDF not available in 22 for every selected candidate",
        "identity": "RDKit canonical isomeric SMILES after RemoveHs", "sa": "RDKit contrib SA_Score in active environment",
        "tolerance": 1e-5}, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
