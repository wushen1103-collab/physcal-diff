from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial import cKDTree

from protein_posebusters_validation import inputs, inspect_one


def native_code(path: str) -> str:
    match = re.search(r"bindingmoad_\d+_[0-9a-z]{4}_([0-9a-z]{3})_pocket10\.pdb", Path(path).name.lower())
    return match.group(1).upper() if match else ""


def fetch_pdb(pdb_id: str, folder: Path) -> tuple[str, str]:
    target = folder / f"{pdb_id}.pdb"
    if target.is_file() and target.stat().st_size > 100:
        return pdb_id, ""
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
                                         headers={"User-Agent": "PhysCAL-independent-validation/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            if b"ATOM  " not in data:
                raise ValueError("not_a_pdb_file")
            target.write_bytes(data)
            return pdb_id, ""
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(attempt + 1)
    return pdb_id, error


def atom_xyz(line: str) -> tuple[float, float, float] | None:
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        return None


def build_pocket(target: str, pdb_id: str, code: str, raw_sdf: Path, pdb_file: Path, out: Path) -> dict[str, object]:
    result = {"target_file": target, "pdb_id": pdb_id, "native_ligand_code": code, "pocket_path": "", "error": ""}
    mol = next((mol for mol in Chem.SDMolSupplier(str(raw_sdf), removeHs=False) if mol is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        result["error"] = "raw_ligand_unreadable"
        return result
    raw_xyz = np.asarray(mol.GetConformer().GetPositions())
    raw_xyz = raw_xyz[[atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()]]
    lines = pdb_file.read_text(encoding="ascii", errors="replace").splitlines()
    ligand_residues: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    protein_lines: list[tuple[str, tuple[str, str, str], tuple[float, float, float]]] = []
    for line in lines:
        if line[16:17] not in (" ", "A"):
            continue
        if line.startswith("HETATM") and line[17:20].strip().upper() == code:
            xyz = atom_xyz(line)
            if xyz is not None and line[76:78].strip().upper() not in ("H", "D"):
                key = (line[21:22], line[22:26], line[26:27])
                ligand_residues.setdefault(key, []).append(xyz)
        elif line.startswith("ATOM  "):
            xyz = atom_xyz(line)
            if xyz is not None:
                key = (line[21:22], line[22:26], line[26:27])
                protein_lines.append((line, key, xyz))
    if not ligand_residues:
        result["error"] = "native_ligand_code_not_found"
        return result
    raw_centroid = raw_xyz.mean(axis=0)
    options = [(float(np.linalg.norm(np.asarray(coords).mean(axis=0) - raw_centroid)), key)
               for key, coords in ligand_residues.items()]
    ligand_distance, native_key = min(options)
    result["raw_native_centroid_distance_angstrom"] = ligand_distance
    result["native_residue"] = ":".join(native_key)
    if ligand_distance > 20.0:
        result["error"] = "raw_native_coordinate_frame_mismatch"
        return result
    native_xyz = np.asarray(ligand_residues[native_key], dtype=float)
    protein_xyz = np.asarray([item[2] for item in protein_lines], dtype=float)
    near = cKDTree(native_xyz).query(protein_xyz, k=1)[0] <= 10.0
    residue_keys = {protein_lines[i][1] for i in np.flatnonzero(near)}
    kept = [line for line, key, _ in protein_lines if key in residue_keys]
    if len(kept) < 20:
        result["error"] = f"too_few_pocket_atoms:{len(kept)}"
        return result
    file = out / f"{target.replace('::', '_')}_{pdb_id}_{code}_pocket10.pdb"
    file.write_text("\n".join(kept + ["END"]) + "\n", encoding="ascii")
    result.update({"pocket_path": str(file), "pocket_atoms": len(kept), "pocket_residues": len(residue_keys)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repo, out, previous = args.repo.resolve(), args.out.resolve(), args.previous.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = inputs(repo, previous)
    frame = frame[frame["dataset"].eq("bindingmoad")].copy()
    if args.smoke:
        frame = frame[frame["target_file"].isin(frame["target_file"].drop_duplicates().head(3))]
    baseline = pd.read_csv(repo / "outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/gnina_redock_rows.csv", low_memory=False)
    baseline = baseline[baseline["policy"].eq("raw_topk") & baseline["dataset"].eq("bindingmoad")]
    baseline = baseline[["target_file", "pocket_path"]].drop_duplicates("target_file")
    baseline["native_ligand_code"] = baseline["pocket_path"].map(native_code)
    frame = frame.merge(baseline[["target_file", "native_ligand_code"]], on="target_file", validate="many_to_one")
    frame.to_csv(out / "input_manifest.csv", index=False)
    raw_folder = out / "rcsb_original_pdb"
    raw_folder.mkdir(exist_ok=True)
    ids = sorted(frame["pdb_id"].unique())
    with futures.ThreadPoolExecutor(max_workers=min(args.workers, 8)) as executor:
        fetched = list(executor.map(lambda name: fetch_pdb(name, raw_folder), ids))
    pd.DataFrame(fetched, columns=["pdb_id", "download_error"]).to_csv(out / "download_manifest.csv", index=False)
    download_errors = dict(fetched)
    pocket_dir = out / "native_pocket10"
    pocket_dir.mkdir(exist_ok=True)
    pockets = []
    for target, group in frame.groupby("target_file"):
        raw = group[group["arm"].eq("raw_topk")].iloc[0]
        if download_errors.get(raw["pdb_id"]):
            pockets.append({"target_file": target, "pocket_path": "", "error": download_errors[raw["pdb_id"]]})
            continue
        pockets.append(build_pocket(target, raw["pdb_id"], raw["native_ligand_code"], Path(raw["sdf_path"]),
                                    raw_folder / f"{raw['pdb_id']}.pdb", pocket_dir))
    pocket_frame = pd.DataFrame(pockets)
    pocket_frame.to_csv(out / "pocket_reconstruction_manifest.csv", index=False)
    by_target = dict(zip(pocket_frame["target_file"], pocket_frame["pocket_path"]))
    records = frame.to_dict("records")
    receptor_maps = [{(str(row["pdb_id"]), None): by_target[row["target_file"]]} for row in records]
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = pd.DataFrame(executor.map(inspect_one, records, receptor_maps))
    rows.to_csv(out / "protein_posebusters_rows.csv", index=False)
    valid = rows[rows["aligned"] & rows["mol_cond_loaded"]].copy()
    summary = valid.groupby(["dataset", "arm"]).agg(n=("target_file", "size"), protein_pass=("protein_pass", "mean"),
        distance_pass=("no_clashes", "mean"), volume_pass=("no_volume_clash", "mean")).reset_index()
    summary.to_csv(out / "protein_posebusters_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({"receptor_source": "RCSB PDB native-ligand-centered 10 Angstrom residue-complete crop",
        "site_selection": "native ligand copy nearest raw ligand centroid, fixed for all three arms",
        "coordinate_exclusion": "raw-to-native centroid distance above 20 Angstrom; ligand-to-cropped-protein nearest distance above 12 Angstrom",
        "limitations": "public reconstruction, not the original prepared pocket; no waters/cofactors; no experimental binding validation",
        "n_targets_requested": frame["target_file"].nunique(), "n_targets_reconstructed": int(pocket_frame["pocket_path"].astype(bool).sum())}, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"pockets={pocket_frame['pocket_path'].astype(bool).sum()}/{frame['target_file'].nunique()} aligned={len(valid)}/{len(frame)}", flush=True)


if __name__ == "__main__":
    main()
