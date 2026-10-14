from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial import cKDTree
from posebusters import PoseBusters


PASS_KEYS = ("not_too_far_away", "no_clashes", "no_volume_clash")
DATASETS = ("bindingmoad", "prospective")


def local_file(repo: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or value.lower() == "nan":
        return None
    path = Path(value)
    if path.is_absolute():
        return None
    path = (repo / path).resolve()
    try:
        path.relative_to(repo)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path


def pdb_id_for(path: str, dataset: str) -> tuple[str, str | None]:
    name = Path(path).name.lower()
    if dataset == "bindingmoad":
        match = re.search(r"bindingmoad_\d+_([0-9][a-z0-9]{3})_", name)
        return (match.group(1), None) if match else ("", None)
    match = re.search(r"data\d+_([0-9][a-z0-9]{3})_([a-z0-9])_rec_", name)
    return (match.group(1), match.group(2).upper()) if match else ("", None)


def download_receptor(item: tuple[str, str | None], folder: Path) -> dict[str, object]:
    pdb_id, chain = item
    filename = folder / f"{pdb_id}_{chain or 'all'}_protein.pdb"
    if filename.is_file() and filename.stat().st_size > 100:
        return {"pdb_id": pdb_id, "chain": chain, "path": str(filename), "error": ""}
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    error = ""
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PhysCAL-independent-validation/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                lines = response.read().decode("ascii", errors="replace").splitlines()
            atoms = []
            for line in lines:
                if not line.startswith("ATOM  "):
                    continue
                if chain is not None and line[21:22] != chain:
                    continue
                if line[16:17] not in (" ", "A"):
                    continue
                atoms.append(line)
            if len(atoms) < 20:
                error = f"too_few_protein_atoms:{len(atoms)}"
                break
            filename.write_text("\n".join(atoms + ["END"]) + "\n", encoding="ascii")
            return {"pdb_id": pdb_id, "chain": chain, "path": str(filename), "error": ""}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(attempt + 1)
    return {"pdb_id": pdb_id, "chain": chain, "path": "", "error": error}


def receptor_coords(path: Path) -> np.ndarray:
    result = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith("ATOM  ") or line[76:78].strip().upper() in {"H", "D"}:
            continue
        try:
            result.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            pass
    return np.asarray(result, dtype=float)


def inspect_one(item: dict[str, object], receptor_paths: dict[tuple[str, str | None], str]) -> dict[str, object]:
    result = {key: item[key] for key in ("dataset", "target_file", "arm", "pdb_id", "chain", "sdf_path")}
    result.update({"aligned": False, "mol_cond_loaded": False, "protein_pass": False, "error": ""})
    receptor = receptor_paths.get((str(item["pdb_id"]), item["chain"]))
    if not receptor:
        result["error"] = "receptor_not_downloaded"
        return result
    sdf = Path(str(item["sdf_path"]))
    if not sdf.is_file():
        result["error"] = "ligand_sdf_missing"
        return result
    ligand = next((mol for mol in Chem.SDMolSupplier(str(sdf), removeHs=False) if mol is not None), None)
    if ligand is None or ligand.GetNumConformers() == 0:
        result["error"] = "ligand_unreadable"
        return result
    ligand_xyz = np.asarray(ligand.GetConformer().GetPositions(), dtype=float)
    ligand_xyz = ligand_xyz[[atom.GetAtomicNum() > 1 for atom in ligand.GetAtoms()]]
    protein_xyz = receptor_coords(Path(receptor))
    if len(ligand_xyz) == 0 or len(protein_xyz) == 0:
        result["error"] = "missing_heavy_atom_coordinates"
        return result
    nearest = float(cKDTree(protein_xyz).query(ligand_xyz, k=1)[0].min())
    result["nearest_protein_angstrom"] = nearest
    result["aligned"] = bool(np.isfinite(nearest) and nearest <= 12.0)
    if not result["aligned"]:
        result["error"] = "coordinate_frame_or_pocket_mismatch"
        return result
    try:
        template = PoseBusters(config="dock").config
        config = copy.deepcopy(template)
        config["modules"] = [module for module in config["modules"]
                             if module["name"] in {"Loading", "Distance to protein", "Volume overlap with protein"}]
        output = list(PoseBusters(config=config).bust((sdf,), None, Path(receptor)))
        if not output:
            raise ValueError("PoseBusters returned no output")
        checks = {name: value for _, name, value in next(iter(output[0].values()))}
        result["mol_cond_loaded"] = bool(checks.get("mol_cond_loaded", False))
        for key in PASS_KEYS:
            result[key] = bool(checks.get(key, False))
        result["protein_pass"] = result["mol_cond_loaded"] and all(result[key] for key in PASS_KEYS)
        result["raw_check_keys"] = ";".join(sorted(checks))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"
    return result


def inputs(repo: Path, previous: Path) -> pd.DataFrame:
    baseline = pd.read_csv(repo / "outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/gnina_redock_rows.csv", low_memory=False)
    selected = pd.read_csv(repo / "outputs/self_contained_redock_portfolio_v0/self_contained_portfolio_selected_rows.csv", low_memory=False)
    baseline = baseline[baseline["policy"].eq("raw_topk") & baseline["dataset"].isin(DATASETS)].copy()
    selected = selected[selected["portfolio_policy"].eq("portfolio_eps_scalar_balanced") & selected["dataset"].isin(DATASETS)].copy()
    baseline["arm"] = "raw_topk"
    selected["arm"] = "physcal_balanced"
    frame = pd.concat([baseline, selected], ignore_index=True, sort=False)
    frame = frame.sort_values(["dataset", "target_file", "arm"]).drop_duplicates(["dataset", "target_file", "arm"])
    frame["sdf_path"] = frame.apply(lambda row: str(local_file(repo, row.get("export_sdf")) or local_file(repo, row.get("redock_sdf")) or ""), axis=1)
    receptors = baseline[["dataset", "target_file", "pocket_path"]].drop_duplicates(["dataset", "target_file"])
    frame = frame.merge(receptors, on=["dataset", "target_file"], how="left", suffixes=("", "_raw"))
    frame["pocket_path"] = frame["pocket_path_raw"].where(frame["pocket_path_raw"].notna(), frame["pocket_path"])
    frame[["pdb_id", "chain"]] = frame.apply(lambda row: pd.Series(pdb_id_for(str(row["pocket_path"]), str(row["dataset"]))), axis=1)
    ligand_only = []
    for _, row in receptors.iterrows():
        path = previous / "ligand_only_sdf" / f"{row['dataset']}__{str(row['target_file']).replace('/', '_')}__ligand_only.sdf"
        pdb_id, chain = pdb_id_for(str(row["pocket_path"]), str(row["dataset"]))
        ligand_only.append({"dataset": row["dataset"], "target_file": row["target_file"], "arm": "ligand_only_mmff_uff",
                            "sdf_path": str(path), "pdb_id": pdb_id, "chain": chain})
    return pd.concat([frame[["dataset", "target_file", "arm", "sdf_path", "pdb_id", "chain"]], pd.DataFrame(ligand_only)], ignore_index=True)


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
    if args.smoke:
        frame = frame[frame["target_file"].isin(frame["target_file"].drop_duplicates().head(3))].copy()
    frame.to_csv(out / "input_manifest.csv", index=False)
    pairs = sorted(set((str(row.pdb_id), row.chain if pd.notna(row.chain) else None) for row in frame.itertuples() if row.pdb_id),
                   key=lambda pair: (pair[0], pair[1] or ""))
    folder = out / "rcsb_protein_receptors"
    folder.mkdir(exist_ok=True)
    with futures.ThreadPoolExecutor(max_workers=min(args.workers, 8)) as executor:
        downloads = list(executor.map(lambda item: download_receptor(item, folder), pairs))
    pd.DataFrame(downloads).to_csv(out / "receptor_download_manifest.csv", index=False)
    receptor_paths = {(row["pdb_id"], row["chain"]): row["path"] for row in downloads if row["path"]}
    records = frame.to_dict("records")
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(inspect_one, records, [receptor_paths] * len(records)))
    rows = pd.DataFrame(results)
    rows.to_csv(out / "protein_posebusters_rows.csv", index=False)
    usable = rows[rows["aligned"] & rows["mol_cond_loaded"]].copy()
    summary = usable.groupby(["dataset", "arm"]).agg(n=("target_file", "size"), protein_pass=("protein_pass", "mean"),
        distance_pass=("no_clashes", "mean"), volume_pass=("no_volume_clash", "mean"),
        pocket_distance_pass=("not_too_far_away", "mean")).reset_index()
    summary.to_csv(out / "protein_posebusters_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({"receptor_source": "RCSB PDB public coordinate file; ATOM records only",
        "pocket_source_not_used": "other project directories are not accessed",
        "configuration": "PoseBusters 0.2.2 dock config, loading and protein distance/volume modules only",
        "alignment": "exclude ligand-receptor nearest distances above 12 Angstrom or missing coordinates",
        "scope": "independent protein geometry proxy, not the original 10-A pocket or full protein preparation; not MD or biological validation",
        "n_requested": len(frame), "n_aligned": len(usable)}, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"aligned={len(usable)}/{len(frame)} downloads={sum(bool(d['path']) for d in downloads)}/{len(downloads)}", flush=True)


if __name__ == "__main__":
    main()
