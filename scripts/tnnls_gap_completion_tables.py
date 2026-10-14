from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")

DATE = "20260619"
BASE_POLICY = "pl_pose_clash_rankfusion_w100_plsafe"
TARGET_KEYS = ["dataset", "target_file"]
METRICS = {
    "rmsd": "input_to_redock_rmsd",
    "affinity": "redock_affinity",
    "cnn": "redock_cnnscore",
    "cnnaff": "redock_cnnaffinity",
    "qed": "eval_qed",
    "sa": "eval_sa_score",
    "contact": "contact_jaccard",
}
DIRECTION = {"rmsd": -1, "affinity": -1, "cnn": 1, "cnnaff": 1, "qed": 1, "sa": -1, "contact": 1}
WEIGHTS = {"affinity": 2.0, "cnn": 1.0, "cnnaff": 0.5, "qed": 1.0, "sa": 2.0, "rmsd": 1.0, "contact": 0.5}
DELTA_SCALE = {"rmsd": 0.10, "affinity": 1.00, "cnn": 0.02, "cnnaff": 0.10, "qed": 0.02, "sa": 0.10, "contact": 0.01}
FOCUS_POLICIES = ["old_final", "portfolio_eps_scalar_balanced", "portfolio_dominance_scalar"]


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def local_path(value: object, root: Path) -> Path | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        try:
            path.relative_to(root)
        except ValueError:
            return None
        return path
    if text.startswith("outputs/") or text.startswith("./outputs/") or text.startswith("scratch/") or text.startswith("./scratch/"):
        return root / path
    return None


def read_first_mol(path: Path | None) -> tuple[Chem.Mol | None, bool, str]:
    if path is None:
        return None, False, "missing_local_path"
    if not path.exists():
        return None, False, "file_not_found"
    try:
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
    except Exception as exc:
        return None, False, f"read_error:{type(exc).__name__}"
    if mol is None:
        return None, False, "empty_or_invalid_sdf"
    try:
        Chem.SanitizeMol(mol)
        return mol, True, ""
    except Exception as exc:
        return mol, False, f"sanitize_error:{type(exc).__name__}"


def coords_array(mol: Chem.Mol) -> np.ndarray:
    if mol.GetNumConformers() == 0:
        return np.empty((0, 3), dtype=float)
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((pos.x, pos.y, pos.z))
    return np.asarray(coords, dtype=float)


def covalent_radius(atomic_num: int) -> float:
    table = Chem.GetPeriodicTable()
    try:
        radius = float(table.GetRcovalent(int(atomic_num)))
    except Exception:
        radius = 0.7
    return radius if radius > 0 else 0.7


def bond_outliers(mol: Chem.Mol, coords: np.ndarray) -> tuple[int, float, float]:
    if len(coords) == 0:
        return 0, math.nan, math.nan
    ratios = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        expected = covalent_radius(mol.GetAtomWithIdx(i).GetAtomicNum()) + covalent_radius(mol.GetAtomWithIdx(j).GetAtomicNum())
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        ratios.append(dist / max(expected, 1e-6))
    if not ratios:
        return 0, math.nan, math.nan
    outliers = int(sum(r < 0.65 or r > 1.45 for r in ratios))
    return outliers, float(np.min(ratios)), float(np.max(ratios))


def internal_clashes(mol: Chem.Mol, coords: np.ndarray) -> int:
    if len(coords) != mol.GetNumAtoms():
        return 0
    bonded = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded.add((min(i, j), max(i, j)))
    clashes = 0
    for i in range(mol.GetNumAtoms()):
        ai = mol.GetAtomWithIdx(i)
        if ai.GetAtomicNum() <= 1:
            continue
        for j in range(i + 1, mol.GetNumAtoms()):
            if (i, j) in bonded:
                continue
            aj = mol.GetAtomWithIdx(j)
            if aj.GetAtomicNum() <= 1:
                continue
            expected = covalent_radius(ai.GetAtomicNum()) + covalent_radius(aj.GetAtomicNum())
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < 0.60 * expected:
                clashes += 1
    return clashes


def forcefield_relax(mol: Chem.Mol) -> tuple[str, bool, float, float, float, float]:
    if mol.GetNumConformers() == 0:
        return "none", False, math.nan, math.nan, math.nan, math.nan
    try:
        work = Chem.Mol(mol)
        before = coords_array(work)
        if AllChem.MMFFHasAllMoleculeParams(work):
            props = AllChem.MMFFGetMoleculeProperties(work)
            ff = AllChem.MMFFGetMoleculeForceField(work, props)
            name = "MMFF"
        elif AllChem.UFFHasAllMoleculeParams(work):
            ff = AllChem.UFFGetMoleculeForceField(work)
            name = "UFF"
        else:
            return "none", False, math.nan, math.nan, math.nan, math.nan
        e0 = float(ff.CalcEnergy())
        ff.Minimize(maxIts=200)
        e1 = float(ff.CalcEnergy())
        after = coords_array(work)
        rmsd = float(np.sqrt(np.mean(np.sum((before - after) ** 2, axis=1)))) if before.shape == after.shape and len(before) else math.nan
        return name, True, e0, e1, max(0.0, e0 - e1), rmsd
    except Exception:
        return "error", False, math.nan, math.nan, math.nan, math.nan


def make_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def canonical_smiles(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    except Exception:
        return ""


def scaffold_smiles(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(mol))
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return ""


def fp_from_smiles(smiles: str):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def mean_pairwise_distance(smiles_values: list[str]) -> float:
    fps = [fp_from_smiles(s) for s in smiles_values if s]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return math.nan
    distances = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        distances.extend(1.0 - sim for sim in sims)
    return float(np.mean(distances)) if distances else math.nan


def descriptor_row(mol: Chem.Mol | None, sanitize_ok: bool, sanitize_error: str, pains_catalog: FilterCatalog) -> dict[str, object]:
    if mol is None:
        return {
            "local_sdf_ok": False,
            "sanitize_ok": False,
            "sanitize_error": sanitize_error,
            "canonical_smiles": "",
            "scaffold_smiles": "",
        }
    smiles = canonical_smiles(mol)
    scaffold = scaffold_smiles(mol)
    coords = coords_array(mol)
    bond_outlier_count, min_ratio, max_ratio = bond_outliers(mol, coords)
    clash_count = internal_clashes(mol, coords)
    ff_name, ff_ok, e0, e1, edrop, relax_rmsd = forcefield_relax(mol)
    try:
        matches = pains_catalog.GetMatches(mol) if sanitize_ok else []
    except Exception:
        matches = []
    try:
        qed = float(QED.qed(mol)) if sanitize_ok else math.nan
    except Exception:
        qed = math.nan
    try:
        ring_info = mol.GetRingInfo()
        ring_count = int(ring_info.NumRings())
    except Exception:
        ring_count = 0
    mw = float(Descriptors.MolWt(mol)) if sanitize_ok else math.nan
    logp = float(Crippen.MolLogP(mol)) if sanitize_ok else math.nan
    hbd = int(Lipinski.NumHDonors(mol)) if sanitize_ok else math.nan
    hba = int(Lipinski.NumHAcceptors(mol)) if sanitize_ok else math.nan
    rotb = int(Lipinski.NumRotatableBonds(mol)) if sanitize_ok else math.nan
    tpsa = float(rdMolDescriptors.CalcTPSA(mol)) if sanitize_ok else math.nan
    lipinski_pass = bool(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10) if sanitize_ok else False
    veber_pass = bool(rotb <= 10 and tpsa <= 140) if sanitize_ok else False
    geometry_pass = bool(
        sanitize_ok
        and len(coords) == mol.GetNumAtoms()
        and mol.GetNumAtoms() > 0
        and bond_outlier_count == 0
        and clash_count == 0
        and ff_ok
        and (not np.isfinite(relax_rmsd) or relax_rmsd <= 2.0)
    )
    return {
        "local_sdf_ok": True,
        "sanitize_ok": bool(sanitize_ok),
        "sanitize_error": sanitize_error,
        "canonical_smiles": smiles,
        "scaffold_smiles": scaffold,
        "has_3d": bool(len(coords) == mol.GetNumAtoms() and len(coords) > 0),
        "atoms": int(mol.GetNumAtoms()),
        "heavy_atoms": int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)),
        "formal_charge_abs": int(abs(Chem.GetFormalCharge(mol))) if sanitize_ok else math.nan,
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotb,
        "tpsa": tpsa,
        "ring_count": ring_count,
        "rdkit_qed": qed,
        "lipinski_pass": lipinski_pass,
        "veber_pass": veber_pass,
        "pains_count": int(len(matches)),
        "pains_pass": bool(len(matches) == 0),
        "bond_outlier_count": int(bond_outlier_count),
        "min_bond_ratio": min_ratio,
        "max_bond_ratio": max_ratio,
        "internal_clash_count": int(clash_count),
        "ff_name": ff_name,
        "ff_ok": bool(ff_ok),
        "ff_energy_initial": e0,
        "ff_energy_final": e1,
        "ff_energy_drop": edrop,
        "ff_relax_rmsd": relax_rmsd,
        "geometry_pass": geometry_pass,
        "independent_rdkit_pass": bool(lipinski_pass and veber_pass and len(matches) == 0 and geometry_pass),
    }


def load_method_rows(root: Path, baseline_rows: Path, selected_rows: Path) -> pd.DataFrame:
    base = pd.read_csv(baseline_rows, low_memory=False)
    base = base[(base["dataset"].isin(["bindingmoad", "prospective"])) & (base["policy"] == BASE_POLICY)].copy()
    base["portfolio_policy"] = "old_final"
    selected = pd.read_csv(selected_rows, low_memory=False)
    selected = selected[selected["portfolio_policy"].isin(["portfolio_eps_scalar_balanced", "portfolio_dominance_scalar"])].copy()
    rows = pd.concat([base, selected], ignore_index=True, sort=False)
    for col in METRICS.values():
        rows[col] = numeric(rows, col)
    rows = rows.dropna(subset=["dataset", "target_file"]).copy()
    rows = rows.sort_values(["dataset", "target_file", "portfolio_policy"]).drop_duplicates(
        subset=["dataset", "target_file", "portfolio_policy"], keep="first"
    )
    for col in ["export_sdf", "redock_sdf"]:
        if col not in rows:
            rows[col] = ""
        rows[f"{col}_local"] = rows[col].map(lambda value: str(local_path(value, root) or ""))
    return rows.reset_index(drop=True)


def build_quality_rows(rows: pd.DataFrame, root: Path) -> pd.DataFrame:
    catalog = make_pains_catalog()
    records = []
    for _, row in rows.iterrows():
        sdf_path = local_path(row.get("export_sdf"), root) or local_path(row.get("redock_sdf"), root)
        mol, sanitize_ok, sanitize_error = read_first_mol(sdf_path)
        rec = {
            "dataset": row["dataset"],
            "target_file": row["target_file"],
            "portfolio_policy": row["portfolio_policy"],
            "candidate_policy": row.get("candidate_policy", row.get("policy", "")),
            "mol_idx": row.get("mol_idx", ""),
            "sdf_file": str(sdf_path) if sdf_path else "",
            "redock_sdf_local": row.get("redock_sdf_local", ""),
            "export_sdf_local": row.get("export_sdf_local", ""),
        }
        for metric, col in METRICS.items():
            rec[metric] = row.get(col, math.nan)
        rec.update(descriptor_row(mol, sanitize_ok, sanitize_error, catalog))
        records.append(rec)
    return pd.DataFrame(records)


def summarize_quality(quality: pd.DataFrame) -> pd.DataFrame:
    old_smiles = set(quality.loc[quality["portfolio_policy"] == "old_final", "canonical_smiles"].dropna().astype(str))
    old_by_target = quality[quality["portfolio_policy"] == "old_final"].set_index(TARGET_KEYS)["canonical_smiles"].to_dict()
    q = quality.copy()
    q["valid_smiles"] = q["canonical_smiles"].fillna("").astype(str) != ""
    q["novel_vs_old_global"] = q["canonical_smiles"].map(lambda s: bool(s) and s not in old_smiles)
    q["changed_vs_old_target"] = q.apply(lambda r: bool(r["canonical_smiles"]) and r["canonical_smiles"] != old_by_target.get((r["dataset"], r["target_file"]), ""), axis=1)
    records = []
    for (dataset, policy), group in q.groupby(["dataset", "portfolio_policy"], sort=False):
        valid = group[group["valid_smiles"]]
        records.append(
            {
                "dataset": dataset,
                "portfolio_policy": policy,
                "rows": int(len(group)),
                "local_sdf_rate": float(group["local_sdf_ok"].fillna(False).mean()),
                "sanitize_rate": float(group["sanitize_ok"].fillna(False).mean()),
                "has3d_rate": float(group["has_3d"].fillna(False).mean()),
                "unique_smiles_rate": float(valid["canonical_smiles"].nunique() / len(valid)) if len(valid) else math.nan,
                "scaffold_unique_rate": float(valid["scaffold_smiles"].nunique() / len(valid)) if len(valid) else math.nan,
                "mean_pairwise_tanimoto_distance": mean_pairwise_distance(valid["canonical_smiles"].dropna().astype(str).tolist()),
                "novel_vs_old_global_rate": float(group["novel_vs_old_global"].mean()) if policy != "old_final" else math.nan,
                "changed_vs_old_target_rate": float(group["changed_vs_old_target"].mean()) if policy != "old_final" else math.nan,
                "lipinski_pass_rate": float(group["lipinski_pass"].fillna(False).mean()),
                "veber_pass_rate": float(group["veber_pass"].fillna(False).mean()),
                "pains_pass_rate": float(group["pains_pass"].fillna(False).mean()),
                "geometry_pass_rate": float(group["geometry_pass"].fillna(False).mean()),
                "independent_rdkit_pass_rate": float(group["independent_rdkit_pass"].fillna(False).mean()),
                "mw_mean": float(group["mw"].mean()),
                "logp_mean": float(group["logp"].mean()),
                "tpsa_mean": float(group["tpsa"].mean()),
                "rdkit_qed_mean": float(group["rdkit_qed"].mean()),
                "ff_relax_rmsd_mean": float(group["ff_relax_rmsd"].mean()),
                "internal_clash_rate": float((group["internal_clash_count"].fillna(0) > 0).mean()),
                "bond_outlier_rate": float((group["bond_outlier_count"].fillna(0) > 0).mean()),
            }
        )
    return pd.DataFrame(records)


def target_deltas(rows: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    base = rows[rows["portfolio_policy"] == "old_final"].set_index(TARGET_KEYS)
    quality_small = quality[
        TARGET_KEYS
        + [
            "portfolio_policy",
            "canonical_smiles",
            "scaffold_smiles",
            "lipinski_pass",
            "veber_pass",
            "pains_pass",
            "geometry_pass",
            "independent_rdkit_pass",
            "mw",
            "logp",
            "tpsa",
        ]
    ].copy()
    qbase = quality_small[quality_small["portfolio_policy"] == "old_final"].set_index(TARGET_KEYS)
    records = []
    for _, row in rows[rows["portfolio_policy"] != "old_final"].iterrows():
        key = (row["dataset"], row["target_file"])
        if key not in base.index:
            continue
        b = base.loc[key]
        rec = {"dataset": row["dataset"], "target_file": row["target_file"], "portfolio_policy": row["portfolio_policy"]}
        composite = 0.0
        reasons = []
        for metric, col in METRICS.items():
            raw_delta = float(row[col]) - float(b[col])
            oriented = DIRECTION[metric] * raw_delta
            rec[f"{metric}_old"] = float(b[col])
            rec[f"{metric}_new"] = float(row[col])
            rec[f"{metric}_delta"] = raw_delta
            rec[f"{metric}_better"] = bool(oriented >= -1e-12)
            composite += WEIGHTS[metric] * float(np.clip(oriented / DELTA_SCALE[metric], -3.0, 3.0))
            if oriented < -1e-12:
                reasons.append(f"{metric}_worse")
        rec["composite_delta_score"] = composite
        rec["failure_reasons"] = ";".join(reasons) if reasons else "none"
        qnew = quality_small[(quality_small["dataset"] == key[0]) & (quality_small["target_file"] == key[1]) & (quality_small["portfolio_policy"] == row["portfolio_policy"])]
        if len(qnew):
            qrow = qnew.iloc[0]
            for col in ["canonical_smiles", "scaffold_smiles", "lipinski_pass", "veber_pass", "pains_pass", "geometry_pass", "independent_rdkit_pass", "mw", "logp", "tpsa"]:
                rec[f"new_{col}"] = qrow[col]
        if key in qbase.index:
            brow = qbase.loc[key]
            rec["changed_vs_old_target"] = rec.get("new_canonical_smiles", "") != brow.get("canonical_smiles", "")
            rec["old_canonical_smiles"] = brow.get("canonical_smiles", "")
        records.append(rec)
    return pd.DataFrame(records)


def coverage_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gap_item": "independent_ligand_quality",
                "status": "completed",
                "evidence": f"table_tnnls_quality_summary_{DATE}.csv; table_tnnls_quality_rows_{DATE}.csv",
                "scope": "RDKit descriptor, Lipinski, Veber, PAINS, 3D geometry, MMFF/UFF relaxation; independent of GNINA scores",
                "remaining_gap": "not a protein-aware docking engine",
            },
            {
                "gap_item": "novelty_diversity",
                "status": "completed_partial",
                "evidence": f"table_tnnls_quality_summary_{DATE}.csv",
                "scope": "unique rate, scaffold unique rate, mean pairwise Morgan distance, changed/novel vs old final",
                "remaining_gap": "true training-set novelty requires the original training set",
            },
            {
                "gap_item": "case_and_failure_tables",
                "status": "completed",
                "evidence": f"table_tnnls_case_top_improvements_{DATE}.csv; table_tnnls_case_failure_modes_{DATE}.csv",
                "scope": "target-level composite and metric-wise deltas with failure tags",
                "remaining_gap": "figures and 3D interaction visualizations are intentionally deferred",
            },
            {
                "gap_item": "true_independent_generation_seeds",
                "status": "not_completed",
                "evidence": "none in this run",
                "scope": "current repository has candidate-subsampling seeds, not fresh generator seeds",
                "remaining_gap": "needs self-contained generator code/data inside 22 repo or public baseline package download",
            },
            {
                "gap_item": "public_sota_head_to_head",
                "status": "not_completed",
                "evidence": "coverage ledger only",
                "scope": "local outputs support old/fair selector baselines but not fresh TargetDiff/DecompDiff/DiffSBDD reruns",
                "remaining_gap": "needs public SOTA code/output import under 22 repo and unified evaluation",
            },
            {
                "gap_item": "independent_protein_aware_docking",
                "status": "blocked_in_current_repo",
                "evidence": "tool check: vina/smina/qvina/gnina/obabel missing; no local receptor PDB/PDBQT found",
                "scope": "cannot run without installing docking binaries and preparing local receptor files",
                "remaining_gap": "download/install engine and local receptors; do not read other project folders",
            },
        ]
    )


def write_report(output_dir: Path, quality_summary: pd.DataFrame, top_cases: pd.DataFrame, failure_cases: pd.DataFrame, ledger: pd.DataFrame) -> None:
    lines = [
        "# TNNLS Gap Completion Tables",
        "",
        f"Date: {DATE}",
        "",
        "Scope: this run reads only files inside the supplied repository. Absolute external source paths in historical manifests are ignored.",
        "",
        "## Independent RDKit Quality Summary",
        "",
        "```csv",
        quality_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Top Target Improvements",
        "",
        "```csv",
        top_cases.to_csv(index=False).strip(),
        "```",
        "",
        "## Failure / Boundary Cases",
        "",
        "```csv",
        failure_cases.to_csv(index=False).strip(),
        "```",
        "",
        "## Completion Ledger",
        "",
        "```csv",
        ledger.to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation",
        "",
        "The new tables close the low-cost experimental gaps: ligand-side independent quality, novelty/diversity relative to the old final, and target-level case/failure evidence. They do not yet close the expensive gaps of fresh end-to-end generation seeds, public SOTA reruns, or protein-aware independent docking because this repository currently lacks self-contained generator code, public SOTA outputs, local receptor files, and docking binaries.",
    ]
    (output_dir / f"tnnls_gap_completion_{DATE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_public_tables(output_dir: Path, report_dir: Path) -> None:
    table_dir = report_dir / "evidence_tables_v0"
    table_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        f"table_tnnls_quality_rows_{DATE}.csv": "tnnls_quality_rows.csv",
        f"table_tnnls_quality_summary_{DATE}.csv": "tnnls_quality_summary.csv",
        f"table_tnnls_target_deltas_{DATE}.csv": "tnnls_target_deltas.csv",
        f"table_tnnls_case_top_improvements_{DATE}.csv": "tnnls_case_top_improvements.csv",
        f"table_tnnls_case_failure_modes_{DATE}.csv": "tnnls_case_failure_modes.csv",
        f"table_tnnls_gap_completion_ledger_{DATE}.csv": "tnnls_gap_completion_ledger.csv",
    }
    for public_name, local_name in mapping.items():
        pd.read_csv(output_dir / local_name).to_csv(table_dir / public_name, index=False)
    source = output_dir / f"tnnls_gap_completion_{DATE}.md"
    (report_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TNNLS reviewer-facing supplemental evidence tables.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--baseline-rows", type=Path, default=Path("outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/gnina_redock_rows.csv"))
    parser.add_argument("--selected-rows", type=Path, default=Path("outputs/self_contained_redock_portfolio_v0/self_contained_portfolio_selected_rows.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--case-count", type=int, default=12)
    args = parser.parse_args()

    root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_method_rows(root, args.baseline_rows, args.selected_rows)
    quality = build_quality_rows(rows, root)
    quality_summary = summarize_quality(quality)
    deltas = target_deltas(rows, quality)
    top_cases = deltas[deltas["portfolio_policy"] == "portfolio_eps_scalar_balanced"].sort_values("composite_delta_score", ascending=False).head(args.case_count)
    failure_cases = deltas[deltas["portfolio_policy"] == "portfolio_eps_scalar_balanced"].sort_values("composite_delta_score", ascending=True).head(args.case_count)
    ledger = coverage_ledger()

    quality.to_csv(args.output_dir / "tnnls_quality_rows.csv", index=False)
    quality_summary.to_csv(args.output_dir / "tnnls_quality_summary.csv", index=False)
    deltas.to_csv(args.output_dir / "tnnls_target_deltas.csv", index=False)
    top_cases.to_csv(args.output_dir / "tnnls_case_top_improvements.csv", index=False)
    failure_cases.to_csv(args.output_dir / "tnnls_case_failure_modes.csv", index=False)
    ledger.to_csv(args.output_dir / "tnnls_gap_completion_ledger.csv", index=False)
    write_report(args.output_dir, quality_summary, top_cases, failure_cases, ledger)
    copy_public_tables(args.output_dir, args.report_dir)
    print(quality_summary.to_csv(index=False))
    print("LEDGER")
    print(ledger.to_csv(index=False))


if __name__ == "__main__":
    main()
