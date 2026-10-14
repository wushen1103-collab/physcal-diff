from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial import cKDTree

from protein_posebusters_validation import inputs, inspect_one, receptor_coords


API = "https://hf-mirror.com/api/datasets/THU-ATOM/crossdocked"
RESOLVE = "https://hf-mirror.com/datasets/THU-ATOM/crossdocked/resolve"


def get_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "PhysCAL-public-benchmark-validation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def original_name(value: str) -> str:
    name = Path(value).name
    match = re.fullmatch(r"data\d+_(.+_pocket10\.pdb)", name)
    if not match:
        raise ValueError(f"Unrecognized original pocket name: {name}")
    return match.group(1)


def fetch_one(item: tuple[str, str, str, str, Path]) -> dict[str, object]:
    target, expected, remote_path, git_oid, folder = item
    output = folder / f"{target.replace('::', '_')}__{expected}"
    url = f"{RESOLVE}/{urllib.parse.quote(COMMIT)}/{urllib.parse.quote(remote_path)}"
    error = ""
    if not output.is_file() or output.stat().st_size < 100:
        for _ in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "PhysCAL-public-benchmark-validation/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    data = response.read()
                if b"ATOM  " not in data:
                    raise ValueError("No protein ATOM records")
                output.write_bytes(data)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
    if not output.is_file() or output.stat().st_size < 100:
        return {"target_file": target, "expected_name": expected, "hf_path": remote_path,
                "local_path": "", "sha256": "", "error": error}
    data = output.read_bytes()
    observed_oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if observed_oid != git_oid:
        raise ValueError(f"Git blob checksum mismatch for {remote_path}: {observed_oid} != {git_oid}")
    return {"target_file": target, "expected_name": expected, "hf_path": remote_path,
            "local_path": str(output), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "git_blob_oid": observed_oid, "error": ""}


def ligand_nearest(sdf: Path, pocket: Path) -> float:
    ligand = next((mol for mol in Chem.SDMolSupplier(str(sdf), removeHs=False) if mol is not None), None)
    if ligand is None or ligand.GetNumConformers() == 0:
        return float("nan")
    xyz = np.asarray(ligand.GetConformer().GetPositions())
    xyz = xyz[[atom.GetAtomicNum() > 1 for atom in ligand.GetAtoms()]]
    protein = receptor_coords(pocket)
    return float(cKDTree(protein).query(xyz, k=1)[0].min()) if len(xyz) and len(protein) else float("nan")


def compare_project_pocket(repo: Path, target: str, public_path: Path) -> dict[str, object]:
    data_id = int(target.rsplit("::", 1)[1])
    path = repo / "outputs/tnnls_targetdiff_data18_rescue_v0/pockets" / f"targetdiff_data{data_id:03d}_pocket10.pdb"
    if not path.is_file():
        return {"target_file": target, "project_pocket_available": False}
    a, b = receptor_coords(path), receptor_coords(public_path)
    if not len(a) or not len(b):
        return {"target_file": target, "project_pocket_available": True, "coordinate_check": "empty"}
    a_to_b = cKDTree(b).query(a, k=1)[0]
    b_to_a = cKDTree(a).query(b, k=1)[0]
    return {"target_file": target, "project_pocket_available": True,
            "project_atoms": len(a), "public_atoms": len(b),
            "project_to_public_median_angstrom": float(np.median(a_to_b)),
            "project_to_public_p95_angstrom": float(np.quantile(a_to_b, .95)),
            "public_to_project_median_angstrom": float(np.median(b_to_a))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = inputs(repo, args.previous.resolve())
    frame = frame[frame["dataset"].eq("prospective")].copy()
    if args.smoke:
        frame = frame[frame["target_file"].isin(frame["target_file"].drop_duplicates().head(2))]
    original = pd.read_csv(repo / "outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/redock_manifest.csv", low_memory=False)
    original = original[original["dataset"].eq("prospective") & original["policy"].eq("raw_topk")]
    original = original[["target_file", "pocket_path"]].drop_duplicates("target_file")
    original["expected_name"] = original["pocket_path"].map(original_name)
    if original["target_file"].nunique() != 18:
        raise ValueError("Original manifest must have exactly 18 prospective targets")
    if args.smoke:
        original = original[original["target_file"].isin(frame["target_file"].unique())]
    frame = frame.merge(original[["target_file", "expected_name"]], on="target_file", validate="many_to_one")
    frame.to_csv(out / "input_manifest.csv", index=False)

    info = get_json(API)
    global COMMIT
    COMMIT = str(info["sha"])
    tree = get_json(f"{API}/tree/{COMMIT}/crossdocked_test?recursive=true&limit=1000")
    by_name: dict[str, list[dict[str, object]]] = {}
    for record in tree:
        path = str(record["path"])
        by_name.setdefault(Path(path).name, []).append(record)
    pocket_folder = out / "public_exact_pockets"
    pocket_folder.mkdir(exist_ok=True)
    jobs = []
    for row in original.itertuples():
        matches = by_name.get(row.expected_name, [])
        if len(matches) != 1:
            raise ValueError(f"Expected one public exact-name match: {row.expected_name}; got {matches}")
        jobs.append((row.target_file, row.expected_name, str(matches[0]["path"]), str(matches[0]["oid"]), pocket_folder))
    with futures.ThreadPoolExecutor(max_workers=min(args.workers, 8)) as executor:
        downloaded = pd.DataFrame(executor.map(fetch_one, jobs))
    downloaded.to_csv(out / "download_manifest.csv", index=False)
    missing = downloaded[downloaded["local_path"].eq("")]
    if len(missing):
        raise RuntimeError(f"Missing {len(missing)} exact public pockets; see download_manifest.csv")
    lookup = dict(zip(downloaded["target_file"], downloaded["local_path"]))
    comparisons = [compare_project_pocket(repo, target, Path(path)) for target, path in lookup.items()]
    pd.DataFrame(comparisons).to_csv(out / "project_pocket_coordinate_comparison.csv", index=False)
    receptor_maps = [{(str(row.pdb_id), row.chain if pd.notna(row.chain) else None): lookup[row.target_file]}
                     for row in frame.itertuples()]
    records = frame.to_dict("records")
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = pd.DataFrame(executor.map(inspect_one, records, receptor_maps))
    rows["exact_pocket_nearest_angstrom"] = [
        ligand_nearest(Path(row.sdf_path), Path(lookup[row.target_file]))
        if Path(row.sdf_path).is_file() else float("nan") for row in frame.itertuples()
    ]
    rows.to_csv(out / "protein_posebusters_rows.csv", index=False)
    summary = rows.groupby("arm").agg(n=("target_file", "size"), aligned=("aligned", "sum"),
                                      protein_pass=("protein_pass", "sum"),
                                      molecule_loaded=("mol_cond_loaded", "sum"),
                                      distance_pass=("no_clashes", "sum"),
                                      volume_pass=("no_volume_clash", "sum")).reset_index()
    summary.to_csv(out / "protein_posebusters_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "dataset": "THU-ATOM/crossdocked crossdocked_test public release, mirror retrieval",
        "dataset_commit": COMMIT, "n_targets": original["target_file"].nunique(),
        "match_rule": "exact original CrossDocked pocket10 basename; reject missing or ambiguous matches",
        "check": "PoseBusters 0.2.2 protein-distance and volume modules, same pocket across all arms",
        "exclusion": "none; all target/arm rows reported, including coordinate mismatch",
        "claim_scope": "independent geometry proxy on exact-name public pockets; not MD or measured binding",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("commit", COMMIT, "target_pockets", len(downloaded), flush=True)


if __name__ == "__main__":
    main()
