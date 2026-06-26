from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, QED, rdMolAlign


RDLogger.DisableLog("rdApp.*")
PT = Chem.GetPeriodicTable()


def sdf_files(root: Path, max_files: int | None) -> list[Path]:
    files = sorted(p for p in root.rglob("*.sdf") if not p.name.startswith("._"))
    return files[:max_files] if max_files else files


def covalent_radius(atomic_num: int) -> float:
    try:
        r = float(PT.GetRcovalent(int(atomic_num)))
        return r if r > 0 else 0.7
    except Exception:
        return 0.7


def sanitize_copy(mol: Chem.Mol | None) -> tuple[Chem.Mol | None, bool, str]:
    if mol is None:
        return None, False, "null_mol"
    try:
        cp = Chem.Mol(mol)
        Chem.SanitizeMol(cp)
        return cp, True, ""
    except Exception as exc:
        return Chem.Mol(mol), False, type(exc).__name__


def coords_array(mol: Chem.Mol) -> np.ndarray:
    if mol.GetNumConformers() == 0:
        return np.empty((0, 3), dtype=float)
    conf = mol.GetConformer()
    rows = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        rows.append((pos.x, pos.y, pos.z))
    return np.asarray(rows, dtype=float)


def bond_outliers(mol: Chem.Mol, coords: np.ndarray) -> tuple[int, float, float]:
    if len(coords) == 0:
        return 0, math.nan, math.nan
    ratios = []
    outliers = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ai = mol.GetAtomWithIdx(i).GetAtomicNum()
        aj = mol.GetAtomWithIdx(j).GetAtomicNum()
        expected = covalent_radius(ai) + covalent_radius(aj)
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        ratio = dist / max(expected, 1e-6)
        ratios.append(ratio)
        if ratio < 0.65 or ratio > 1.45:
            outliers += 1
    if not ratios:
        return 0, math.nan, math.nan
    return outliers, float(np.min(ratios)), float(np.max(ratios))


def internal_clashes(mol: Chem.Mol, coords: np.ndarray) -> int:
    n = mol.GetNumAtoms()
    if n < 2 or len(coords) != n:
        return 0
    bonded = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded.add((min(i, j), max(i, j)))
    clashes = 0
    for i in range(n):
        ai = mol.GetAtomWithIdx(i)
        if ai.GetAtomicNum() <= 1:
            continue
        for j in range(i + 1, n):
            aj = mol.GetAtomWithIdx(j)
            if aj.GetAtomicNum() <= 1 or (i, j) in bonded:
                continue
            expected = covalent_radius(ai.GetAtomicNum()) + covalent_radius(aj.GetAtomicNum())
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < 0.60 * expected:
                clashes += 1
    return clashes


def forcefield_energy(mol: Chem.Mol) -> tuple[str, bool, float, float, float, float]:
    if mol is None or mol.GetNumConformers() == 0:
        return "none", False, math.nan, math.nan, math.nan, math.nan
    try:
        work = Chem.Mol(mol)
        before = coords_array(work)
        if AllChem.MMFFHasAllMoleculeParams(work):
            props = AllChem.MMFFGetMoleculeProperties(work)
            ff = AllChem.MMFFGetMoleculeForceField(work, props)
            ff_name = "MMFF"
        elif AllChem.UFFHasAllMoleculeParams(work):
            ff = AllChem.UFFGetMoleculeForceField(work)
            ff_name = "UFF"
        else:
            return "none", False, math.nan, math.nan, math.nan, math.nan
        e0 = float(ff.CalcEnergy())
        ff.Minimize(maxIts=200)
        e1 = float(ff.CalcEnergy())
        try:
            rmsd = float(rdMolAlign.GetBestRMS(mol, work))
        except Exception:
            after = coords_array(work)
            rmsd = float(np.sqrt(np.mean(np.sum((before - after) ** 2, axis=1)))) if before.shape == after.shape and len(before) else math.nan
        return ff_name, True, e0, e1, max(0.0, e0 - e1), rmsd
    except Exception:
        return "error", False, math.nan, math.nan, math.nan, math.nan


def audit_file(args: tuple[str, str, str]) -> list[dict]:
    generator, root_text, path_text = args
    root = Path(root_text)
    path = Path(path_text)
    rel = str(path.relative_to(root))
    rows = []
    failure_row = {
        "generator": generator,
        "sdf_file": str(path),
        "relative_file": rel,
        "mol_idx": -1,
        "sanitize_ok": False,
        "sanitize_error": "",
        "has_3d": False,
        "atoms": 0,
        "heavy_atoms": 0,
        "formal_charge_abs": math.nan,
        "qed": math.nan,
        "bond_outlier_count": math.nan,
        "min_bond_ratio": math.nan,
        "max_bond_ratio": math.nan,
        "internal_clash_count": math.nan,
        "ff_name": "none",
        "ff_ok": False,
        "ff_energy_initial": math.nan,
        "ff_energy_final": math.nan,
        "ff_energy_drop": math.nan,
        "ff_relax_rmsd": math.nan,
    }
    try:
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    except OSError as exc:
        row = dict(failure_row)
        row["sanitize_error"] = f"file_read_error: {exc}"
        return [row]
    for idx, mol in enumerate(supplier):
        smol, sanitize_ok, sanitize_error = sanitize_copy(mol)
        if smol is None:
            row = dict(failure_row)
            row["mol_idx"] = idx
            row["sanitize_error"] = sanitize_error
            rows.append(row)
            continue
        coords = coords_array(smol)
        heavy_atoms = sum(1 for atom in smol.GetAtoms() if atom.GetAtomicNum() > 1)
        bond_outlier_count, min_bond_ratio, max_bond_ratio = bond_outliers(smol, coords)
        clash_count = internal_clashes(smol, coords)
        ff_name, ff_ok, e0, e1, edrop, rmsd = forcefield_energy(smol)
        try:
            qed = float(QED.qed(smol)) if sanitize_ok else math.nan
        except Exception:
            qed = math.nan
        rows.append(
            {
                "generator": generator,
                "sdf_file": str(path),
                "relative_file": rel,
                "mol_idx": idx,
                "sanitize_ok": bool(sanitize_ok),
                "sanitize_error": sanitize_error,
                "has_3d": bool(len(coords) == smol.GetNumAtoms() and len(coords) > 0),
                "atoms": int(smol.GetNumAtoms()),
                "heavy_atoms": int(heavy_atoms),
                "formal_charge_abs": int(abs(Chem.GetFormalCharge(smol))) if sanitize_ok else math.nan,
                "qed": qed,
                "bond_outlier_count": int(bond_outlier_count),
                "min_bond_ratio": min_bond_ratio,
                "max_bond_ratio": max_bond_ratio,
                "internal_clash_count": int(clash_count),
                "ff_name": ff_name,
                "ff_ok": bool(ff_ok),
                "ff_energy_initial": e0,
                "ff_energy_final": e1,
                "ff_energy_drop": edrop,
                "ff_relax_rmsd": rmsd,
            }
        )
    if not rows:
        row = dict(failure_row)
        row["sanitize_error"] = "empty_sdf"
        rows.append(row)
    return rows


def parse_roots(root_specs: list[str]) -> list[tuple[str, Path]]:
    roots = []
    for spec in root_specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = Path(path).name
        roots.append((name, Path(path)))
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="name=/path/to/sdf/root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-files-per-root", type=int, default=None)
    args = parser.parse_args()

    out = args.output_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    manifest_rows = []
    for name, root in parse_roots(args.root):
        files = sdf_files(root, args.max_files_per_root)
        manifest_rows.append({"generator": name, "root": str(root), "sdf_files": len(files)})
        tasks.extend((name, str(root), str(path)) for path in files)
    pd.DataFrame(manifest_rows).to_csv(out / "input_manifest.csv", index=False)

    rows: list[dict] = []
    workers = max(1, min(args.workers, os.cpu_count() or args.workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(audit_file, task) for task in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())

    frame = pd.DataFrame(rows)
    if len(frame) == 0:
        raise RuntimeError("No molecules audited")

    ff_energy_per_heavy = frame["ff_energy_final"] / frame["heavy_atoms"].replace(0, np.nan)
    frame["ff_energy_per_heavy"] = ff_energy_per_heavy
    valid_energy = ff_energy_per_heavy[np.isfinite(ff_energy_per_heavy)]
    q90 = float(valid_energy.quantile(0.90)) if len(valid_energy) else math.inf
    drop_q90 = float(frame["ff_energy_drop"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.90))
    rmsd_q90 = float(frame["ff_relax_rmsd"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.90))
    frame["high_energy_pool_q90"] = frame["ff_energy_per_heavy"] > q90
    frame["high_relax_drop_pool_q90"] = frame["ff_energy_drop"] > drop_q90
    frame["high_relax_rmsd_pool_q90"] = frame["ff_relax_rmsd"] > rmsd_q90
    frame["rdkit_physics_high_risk"] = (
        (~frame["sanitize_ok"].fillna(False))
        | (~frame["has_3d"].fillna(False))
        | (frame["bond_outlier_count"].fillna(0) > 0)
        | (frame["internal_clash_count"].fillna(0) > 0)
        | (~frame["ff_ok"].fillna(False))
        | frame["high_energy_pool_q90"].fillna(False)
        | frame["high_relax_drop_pool_q90"].fillna(False)
        | frame["high_relax_rmsd_pool_q90"].fillna(False)
    )

    frame.to_csv(out / "molecule_physics_audit.csv", index=False)
    summary = (
        frame.groupby("generator")
        .agg(
            molecules=("relative_file", "count"),
            files=("relative_file", "nunique"),
            sanitize_rate=("sanitize_ok", "mean"),
            has3d_rate=("has_3d", "mean"),
            ff_ok_rate=("ff_ok", "mean"),
            high_risk_rate=("rdkit_physics_high_risk", "mean"),
            bond_outlier_rate=("bond_outlier_count", lambda x: float((x.fillna(0) > 0).mean())),
            internal_clash_rate=("internal_clash_count", lambda x: float((x.fillna(0) > 0).mean())),
            high_energy_q90_rate=("high_energy_pool_q90", "mean"),
            relax_drop_q90_rate=("high_relax_drop_pool_q90", "mean"),
            relax_rmsd_q90_rate=("high_relax_rmsd_pool_q90", "mean"),
            qed_mean=("qed", "mean"),
            energy_per_heavy_mean=("ff_energy_per_heavy", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "generator_physics_summary.csv", index=False)

    lines = [
        "# SDF Physics Audit",
        "",
        "Pool-level q90 thresholds are used for high energy, relaxation drop, and relaxation RMSD.",
        "",
        "## Input Manifest",
        "```csv",
        pd.DataFrame(manifest_rows).to_csv(index=False).strip(),
        "```",
        "",
        "## Generator Summary",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        f"energy_per_heavy_q90,{q90}",
        f"ff_energy_drop_q90,{drop_q90}",
        f"ff_relax_rmsd_q90,{rmsd_q90}",
    ]
    report = "\n".join(lines)
    (out / "summary.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
