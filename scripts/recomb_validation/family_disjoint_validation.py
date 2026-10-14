from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1

from native_pocket_posebusters import fetch_pdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recomb_risk_validation import calibrate, eval_gate, features_for, fit_scores, labels, load_original_helpers, score_metrics


def audited_pdb_id(generator: str, relative_file: str) -> str:
    if generator != "diffsbdd":
        return ""
    match = re.match(r"^([0-9][a-z0-9]{3})-", Path(relative_file).name.lower())
    return match.group(1) if match else ""


def preferred_chains(repo: Path) -> dict[str, str]:
    manifest = repo / "outputs/recomb_native_pocket_20261014/pocket_reconstruction_manifest.csv"
    if not manifest.is_file():
        return {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in pd.read_csv(manifest).itertuples():
        if not isinstance(row.pocket_path, str) or not row.pocket_path:
            continue
        for line in Path(row.pocket_path).read_text(encoding="ascii", errors="replace").splitlines():
            if line.startswith("ATOM  "):
                counts[row.pdb_id][line[21:22]] += 1
    return {pdb_id: chain_counts.most_common(1)[0][0] for pdb_id, chain_counts in counts.items()}


def parse_sequence(path: Path, preferred: str | None) -> tuple[str, str]:
    by_chain: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if line.startswith("SEQRES"):
            by_chain[line[11:12]].extend(line[19:70].split())
    sequences = {chain: "".join(seq1(res, custom_map={"UNK": "X"}) for res in residues)
                 for chain, residues in by_chain.items()}
    sequences = {chain: sequence for chain, sequence in sequences.items() if len(sequence) >= 40}
    if not sequences:
        return "", ""
    chain = preferred if preferred in sequences else max(sequences, key=lambda key: len(sequences[key]))
    return chain, sequences[chain]


def cluster_sequences(sequences: dict[str, str], identity_cutoff: float, coverage_cutoff: float):
    ids = sorted(sequences)
    parent = {item: item for item in ids}
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    edges = []
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            a, b = sequences[left], sequences[right]
            alignment = aligner.align(a, b)[0]
            blocks_a, blocks_b = alignment.aligned
            aligned = int(sum(end - start for start, end in blocks_a))
            if aligned == 0:
                continue
            coverage = aligned / min(len(a), len(b))
            if coverage < coverage_cutoff:
                continue
            matched = sum(sum(x == y for x, y in zip(a[a0:a1], b[b0:b1]))
                          for (a0, a1), (b0, b1) in zip(blocks_a, blocks_b))
            identity = matched / aligned
            if identity >= identity_cutoff:
                parent[find(right)] = find(left)
                edges.append({"left": left, "right": right, "identity": identity, "shorter_coverage": coverage})
    roots = {pdb_id: find(pdb_id) for pdb_id in ids}
    return roots, pd.DataFrame(edges, columns=["left", "right", "identity", "shorter_coverage"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.70)
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 23, 24])
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = load_original_helpers(repo)
    audit = pd.read_csv(repo / "outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv", low_memory=False)
    audit = audit[audit["generator"].isin(original.MAIN_GENERATORS)].copy()
    audit["pdb_id"] = [audited_pdb_id(generator, relative_file) for generator, relative_file
                       in zip(audit["generator"], audit["relative_file"].astype(str))]
    ids = sorted(set(audit.loc[audit["pdb_id"].ne(""), "pdb_id"]))
    source = out / "rcsb_original_pdb"
    source.mkdir(exist_ok=True)
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        downloads = list(executor.map(lambda pdb_id: fetch_pdb(pdb_id, source), ids))
    pd.DataFrame(downloads, columns=["pdb_id", "error"]).to_csv(out / "downloads.csv", index=False)
    chain_choice = preferred_chains(repo)
    seq_rows = []
    for pdb_id, error in downloads:
        chain, sequence = parse_sequence(source / f"{pdb_id}.pdb", chain_choice.get(pdb_id)) if not error else ("", "")
        seq_rows.append({"pdb_id": pdb_id, "chain": chain, "sequence": sequence, "sequence_length": len(sequence), "error": error})
    seq_frame = pd.DataFrame(seq_rows)
    seq_frame.to_csv(out / "receptor_sequences.csv", index=False)
    sequences = dict(zip(seq_frame.loc[seq_frame["sequence"].ne(""), "pdb_id"],
                         seq_frame.loc[seq_frame["sequence"].ne(""), "sequence"]))
    roots, edges = cluster_sequences(sequences, args.identity, args.coverage)
    edges.to_csv(out / "homology_edges.csv", index=False)
    seq_frame["family"] = seq_frame["pdb_id"].map(roots)
    seq_frame.to_csv(out / "receptor_families.csv", index=False)
    known = audit[audit["pdb_id"].isin(roots)].copy()
    known["family"] = known["pdb_id"].map(roots)
    records, gates = [], []
    for seed in args.seeds:
        split = original.assign_split(known["family"], seed)
        raw_train, raw_cal, raw_test = [known.loc[split.eq(part)].copy() for part in ("train", "cal", "test")]
        if min(len(raw_train), len(raw_cal), len(raw_test)) == 0:
            records.append({"seed": seed, "error": "empty_partition"})
            continue
        thresholds = original.quantile_thresholds(raw_train, 0.90)
        train, cal, test = [labels(part, thresholds, original) for part in (raw_train, raw_cal, raw_test)]
        train_scores, cal_scores, scores = fit_scores(train, cal, test, features_for("full", original), seed, args.trees, args.jobs)
        cal["receptor_id"] = cal["family"]
        test["receptor_id"] = test["family"]
        records.append({"seed": seed, "train_n": len(train), "cal_n": len(cal), "test_n": len(test),
                        "excluded_n": len(audit) - len(known), "n_families": known["family"].nunique(),
                        "train_families": train["family"].nunique(), "cal_families": cal["family"].nunique(),
                        "test_families": test["family"].nunique(),
                        "family_overlap_train_test": len(set(train["family"]) & set(test["family"])),
                        "pdb_overlap_train_test": len(set(train["pdb_id"]) & set(test["pdb_id"])),
                        "test_bad_prevalence": float(test["bad"].mean()), **thresholds,
                        **score_metrics(test["bad"], scores)})
        print(f"seed={seed} test_n={len(test)} roc_auc={records[-1]['roc_auc']:.4f}", flush=True)
        for alpha in (0.01, 0.05, 0.10, 0.20):
            calibration = calibrate(train_scores, cal, cal_scores, alpha, original, 41, 0.95)
            for method in ("wilson", "group_hoeffding"):
                threshold, bound = calibration[method]
                gates.append({"seed": seed, "alpha": alpha, "method": method, "threshold": threshold,
                              "cal_bound": bound, "cal_groups": calibration["cal_groups"],
                              **eval_gate(test, scores, threshold, original)})
    pd.DataFrame(records).to_csv(out / "family_disjoint_detection.csv", index=False)
    pd.DataFrame(gates).to_csv(out / "family_disjoint_calibration.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({"scope": "22 project audit only; RCSB public protein sequence",
        "chain_choice": "dominant chain in independently reconstructed native-ligand-centered pocket, otherwise longest SEQRES chain",
        "family_rule": f"single-linkage local sequence identity >= {args.identity} and aligned coverage of shorter chain >= {args.coverage}",
        "pdb_id_rule": "only DiffSBDD names with leading PDB token; PocketFlow timestamp and MolCRAFT data-index paths are excluded",
        "label_thresholds": "computed using training rows only per split", "seeds": args.seeds,
        "limitations": "sequence-defined clusters, not a curated SCOP/CATH fold split; unparsed PDB IDs or missing sequences excluded"}, indent=2) + "\n")
    print(f"PDB IDs={len(ids)} sequences={len(sequences)} families={len(set(roots.values()))} included={len(known)}/{len(audit)}", flush=True)


if __name__ == "__main__":
    main()
