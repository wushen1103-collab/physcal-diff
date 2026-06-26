from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, QED, rdMolAlign

from audit_sdf_physics import (
    bond_outliers,
    coords_array,
    internal_clashes,
    parse_roots,
    sanitize_copy,
    sdf_files,
)


RMSD_LIMITS = [0.5, 1.0, 1.5, 2.0]


def reference_thresholds(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    energy_per_heavy = frame["ff_energy_final"] / frame["heavy_atoms"].replace(0, np.nan)
    return {
        "energy_per_heavy_q90": float(energy_per_heavy[np.isfinite(energy_per_heavy)].quantile(0.90)),
        "ff_energy_drop_q90": float(frame["ff_energy_drop"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.90)),
        "ff_relax_rmsd_q90": float(frame["ff_relax_rmsd"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.90)),
    }


def minimize_copy(mol: Chem.Mol, max_its: int) -> tuple[Chem.Mol | None, str, bool, float, float, float, float]:
    try:
        work = Chem.Mol(mol)
        if AllChem.MMFFHasAllMoleculeParams(work):
            props = AllChem.MMFFGetMoleculeProperties(work)
            ff = AllChem.MMFFGetMoleculeForceField(work, props)
            ff_name = "MMFF"
        elif AllChem.UFFHasAllMoleculeParams(work):
            ff = AllChem.UFFGetMoleculeForceField(work)
            ff_name = "UFF"
        else:
            return None, "none", False, math.nan, math.nan, math.nan, math.nan
        e0 = float(ff.CalcEnergy())
        ff.Minimize(maxIts=max_its)
        e1 = float(ff.CalcEnergy())
        try:
            rmsd = float(rdMolAlign.GetBestRMS(mol, work))
        except Exception:
            before = coords_array(mol)
            after = coords_array(work)
            rmsd = float(np.sqrt(np.mean(np.sum((before - after) ** 2, axis=1)))) if before.shape == after.shape and len(before) else math.nan
        return work, ff_name, True, e0, e1, max(0.0, e0 - e1), rmsd
    except Exception:
        return None, "error", False, math.nan, math.nan, math.nan, math.nan


def forcefield_probe(mol: Chem.Mol, max_its: int) -> tuple[str, bool, float, float, float, float]:
    repaired, ff_name, ok, e0, e1, drop, rmsd = minimize_copy(mol, max_its)
    return ff_name, ok, e0, e1, drop, rmsd


def physical_row(prefix: str, mol: Chem.Mol | None, thresholds: dict[str, float], max_its: int) -> dict[str, float | bool | int | str]:
    if mol is None:
        return {
            f"{prefix}_sanitize_ok": False,
            f"{prefix}_has_3d": False,
            f"{prefix}_atoms": 0,
            f"{prefix}_heavy_atoms": 0,
            f"{prefix}_qed": math.nan,
            f"{prefix}_bond_outlier": True,
            f"{prefix}_internal_clash": True,
            f"{prefix}_ff_name": "none",
            f"{prefix}_ff_ok": False,
            f"{prefix}_ff_energy_initial": math.nan,
            f"{prefix}_ff_energy_final": math.nan,
            f"{prefix}_ff_energy_drop": math.nan,
            f"{prefix}_ff_relax_rmsd": math.nan,
            f"{prefix}_ff_energy_per_heavy": math.nan,
            f"{prefix}_high_risk_fixed": True,
        }
    smol, sanitize_ok, _ = sanitize_copy(mol)
    if smol is None:
        return physical_row(prefix, None, thresholds, max_its)
    coords = coords_array(smol)
    heavy_atoms = sum(1 for atom in smol.GetAtoms() if atom.GetAtomicNum() > 1)
    bond_outlier_count, _, _ = bond_outliers(smol, coords)
    clash_count = internal_clashes(smol, coords)
    ff_name, ff_ok, e0, e1, drop, rmsd = forcefield_probe(smol, max_its)
    energy_per_heavy = e1 / heavy_atoms if heavy_atoms else math.nan
    try:
        qed = float(QED.qed(smol)) if sanitize_ok else math.nan
    except Exception:
        qed = math.nan
    high_risk = (
        (not sanitize_ok)
        or (not (len(coords) == smol.GetNumAtoms() and len(coords) > 0))
        or (bond_outlier_count > 0)
        or (clash_count > 0)
        or (not ff_ok)
        or (np.isfinite(energy_per_heavy) and energy_per_heavy > thresholds["energy_per_heavy_q90"])
        or (np.isfinite(drop) and drop > thresholds["ff_energy_drop_q90"])
        or (np.isfinite(rmsd) and rmsd > thresholds["ff_relax_rmsd_q90"])
    )
    return {
        f"{prefix}_sanitize_ok": bool(sanitize_ok),
        f"{prefix}_has_3d": bool(len(coords) == smol.GetNumAtoms() and len(coords) > 0),
        f"{prefix}_atoms": int(smol.GetNumAtoms()),
        f"{prefix}_heavy_atoms": int(heavy_atoms),
        f"{prefix}_qed": qed,
        f"{prefix}_bond_outlier": bool(bond_outlier_count > 0),
        f"{prefix}_internal_clash": bool(clash_count > 0),
        f"{prefix}_ff_name": ff_name,
        f"{prefix}_ff_ok": bool(ff_ok),
        f"{prefix}_ff_energy_initial": e0,
        f"{prefix}_ff_energy_final": e1,
        f"{prefix}_ff_energy_drop": drop,
        f"{prefix}_ff_relax_rmsd": rmsd,
        f"{prefix}_ff_energy_per_heavy": energy_per_heavy,
        f"{prefix}_high_risk_fixed": bool(high_risk),
    }


def process_file(task: tuple[str, str, str, dict[str, float], int]) -> list[dict]:
    generator, root_text, path_text, thresholds, max_its = task
    root = Path(root_text)
    path = Path(path_text)
    rel = str(path.relative_to(root))
    rows = []
    try:
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    except OSError as exc:
        return [
            {
                "generator": generator,
                "sdf_file": str(path),
                "relative_file": rel,
                "mol_idx": -1,
                "repair_ok": False,
                "repair_error": f"file_read_error:{type(exc).__name__}",
                "repair_rmsd": math.nan,
                **physical_row("raw", None, thresholds, max_its),
                **physical_row("repaired", None, thresholds, max_its),
            }
        ]
    for idx, mol in enumerate(supplier):
        smol, sanitize_ok, sanitize_error = sanitize_copy(mol)
        raw = physical_row("raw", smol, thresholds, max_its)
        repaired_mol = None
        ff_name = "none"
        repair_ok = False
        repair_e0 = repair_e1 = repair_drop = repair_rmsd = math.nan
        repair_error = "" if sanitize_ok else sanitize_error
        if smol is not None and sanitize_ok and smol.GetNumConformers() > 0:
            repaired_mol, ff_name, repair_ok, repair_e0, repair_e1, repair_drop, repair_rmsd = minimize_copy(smol, max_its)
            if not repair_ok:
                repair_error = ff_name
        repaired = physical_row("repaired", repaired_mol, thresholds, max_its)
        rows.append(
            {
                "generator": generator,
                "sdf_file": str(path),
                "relative_file": rel,
                "mol_idx": idx,
                "repair_ok": bool(repair_ok),
                "repair_error": repair_error,
                "repair_ff_name": ff_name,
                "repair_energy_initial": repair_e0,
                "repair_energy_final": repair_e1,
                "repair_energy_drop": repair_drop,
                "repair_rmsd": repair_rmsd,
                **raw,
                **repaired,
            }
        )
    if not rows:
        rows.append(
            {
                "generator": generator,
                "sdf_file": str(path),
                "relative_file": rel,
                "mol_idx": -1,
                "repair_ok": False,
                "repair_error": "empty_sdf",
                "repair_rmsd": math.nan,
                **physical_row("raw", None, thresholds, max_its),
                **physical_row("repaired", None, thresholds, max_its),
            }
        )
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for generator, sub in frame.groupby("generator"):
        row = {
            "generator": generator,
            "molecules": len(sub),
            "files": int(sub["relative_file"].nunique()),
            "repair_ok_rate": float(sub["repair_ok"].mean()),
            "repair_rmsd_mean": float(sub.loc[sub["repair_ok"], "repair_rmsd"].mean()),
            "raw_high_risk_fixed_rate": float(sub["raw_high_risk_fixed"].mean()),
            "repaired_high_risk_fixed_rate": float(sub["repaired_high_risk_fixed"].mean()),
            "raw_qed_mean": float(sub["raw_qed"].mean()),
            "repaired_qed_mean": float(sub["repaired_qed"].mean()),
            "raw_energy_per_heavy_mean": float(sub["raw_ff_energy_per_heavy"].mean()),
            "repaired_energy_per_heavy_mean": float(sub["repaired_ff_energy_per_heavy"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("generator")


def conservative_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for generator, sub in frame.groupby("generator"):
        for limit in RMSD_LIMITS:
            accepted = sub[sub["repair_ok"] & (sub["repair_rmsd"] <= limit)].copy()
            row = {
                "generator": generator,
                "repair_rmsd_max": limit,
                "molecules": len(sub),
                "accepted": len(accepted),
                "coverage": float(len(accepted) / len(sub)) if len(sub) else math.nan,
                "raw_high_risk_fixed_rate": float(sub["raw_high_risk_fixed"].mean()),
                "raw_high_risk_on_accepted": float(accepted["raw_high_risk_fixed"].mean()) if len(accepted) else math.nan,
                "repaired_high_risk_on_accepted": float(accepted["repaired_high_risk_fixed"].mean()) if len(accepted) else math.nan,
                "repair_rmsd_mean": float(accepted["repair_rmsd"].mean()) if len(accepted) else math.nan,
                "repaired_qed_mean": float(accepted["repaired_qed"].mean()) if len(accepted) else math.nan,
                "repaired_energy_per_heavy_mean": float(accepted["repaired_ff_energy_per_heavy"].mean()) if len(accepted) else math.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["generator", "repair_rmsd_max"])


def csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="name=/path/to/sdf/root")
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-files-per-root", type=int, default=None)
    parser.add_argument("--max-its", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = reference_thresholds(args.reference_audit)
    tasks = []
    manifest_rows = []
    for name, root in parse_roots(args.root):
        files = sdf_files(root, args.max_files_per_root)
        manifest_rows.append({"generator": name, "root": str(root), "sdf_files": len(files)})
        tasks.extend((name, str(root), str(path), thresholds, args.max_its) for path in files)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())

    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    conservative = conservative_table(frame)

    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "input_manifest.csv", index=False)
    frame.to_csv(args.output_dir / "repair_molecule_table.csv", index=False)
    summary.to_csv(args.output_dir / "repair_generator_summary.csv", index=False)
    conservative.to_csv(args.output_dir / "repair_conservative_policy.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "reference_audit": str(args.reference_audit),
                "thresholds": thresholds,
                "max_its": args.max_its,
                "rmsd_limits": RMSD_LIMITS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Force-Field Repair Ablation",
        "",
        "High-risk labels use the fixed thresholds from the reference audit.",
        "",
        "## Generator Summary",
        csv_block(summary),
        "",
        "## Conservative Repair Policies",
        csv_block(conservative),
    ]
    (args.output_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
