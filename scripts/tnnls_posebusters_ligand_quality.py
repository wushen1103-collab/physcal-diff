from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from posebusters import PoseBusters

from tnnls_gap_completion_tables import DATE, FOCUS_POLICIES, load_method_rows, local_path


PASS_COLUMNS = [
    "mol_pred_loaded",
    "passes_valence_checks",
    "passes_kekulization",
    "passes_rdkit_sanity_checks",
    "all_atoms_connected",
    "bond_lengths_within_bounds",
    "bond_angles_within_bounds",
    "no_internal_clash",
    "flatness_passes",
    "energy_ratio_passes",
]


def bust_one(sdf_path: Path) -> dict[str, object]:
    record: dict[str, object] = {}
    try:
        buster = PoseBusters(config="mol")
        result = list(buster.bust((sdf_path,), None, None))
    except Exception as exc:
        return {"posebusters_error": f"{type(exc).__name__}:{exc}"}
    if not result:
        return {"posebusters_error": "empty_result"}
    checks = next(iter(result[0].values()))
    for module, name, value in checks:
        del module
        if name not in record:
            record[name] = value
    record["posebusters_error"] = ""
    return record


def build_rows(method_rows: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for _, row in method_rows.iterrows():
        sdf = local_path(row.get("export_sdf"), root) or local_path(row.get("redock_sdf"), root)
        rec = {
            "dataset": row["dataset"],
            "target_file": row["target_file"],
            "portfolio_policy": row["portfolio_policy"],
            "candidate_policy": row.get("candidate_policy", row.get("policy", "")),
            "sdf_file": str(sdf.relative_to(root)) if sdf and sdf.exists() else "",
            "sdf_available": bool(sdf and sdf.exists()),
        }
        if sdf and sdf.exists():
            rec.update(bust_one(sdf))
        else:
            rec["posebusters_error"] = "missing_local_sdf"
        for col in PASS_COLUMNS:
            rec[col] = bool(rec.get(col, False))
        rec["posebusters_mol_pass"] = all(bool(rec.get(col, False)) for col in PASS_COLUMNS)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "rows": ("target_file", "size"),
        "sdf_available_rate": ("sdf_available", "mean"),
        "posebusters_mol_pass_rate": ("posebusters_mol_pass", "mean"),
    }
    for col in PASS_COLUMNS:
        agg[f"{col}_rate"] = (col, "mean")
    numeric_cols = [
        "shortest_bond_relative_length",
        "longest_bond_relative_length",
        "most_extreme_relative_angle",
        "shortest_noncovalent_relative_distance",
        "number_clashes",
        "max_distance",
        "energy_ratio",
        "mol_pred_energy",
        "ensemble_avg_energy",
    ]
    for col in numeric_cols:
        if col in rows:
            rows[col] = pd.to_numeric(rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            agg[f"{col}_mean"] = (col, "mean")
    return rows.groupby(["dataset", "portfolio_policy"], dropna=False).agg(**agg).reset_index()


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# PoseBusters Ligand-Only Quality",
        "",
        f"Date: {DATE}",
        "",
        "Scope: PoseBusters `mol` configuration over local 22-repo SDF files only. Protein-aware checks are not run because the current 22 repository has no local receptor files.",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
    ]
    (output_dir / f"tnnls_posebusters_ligand_quality_{DATE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_public(output_dir: Path, report_dir: Path) -> None:
    table_dir = report_dir / "evidence_tables_v0"
    table_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        f"table_tnnls_posebusters_ligand_rows_{DATE}.csv": "tnnls_posebusters_ligand_rows.csv",
        f"table_tnnls_posebusters_ligand_summary_{DATE}.csv": "tnnls_posebusters_ligand_summary.csv",
    }
    for public_name, local_name in mapping.items():
        pd.read_csv(output_dir / local_name).to_csv(table_dir / public_name, index=False)
    source = output_dir / f"tnnls_posebusters_ligand_quality_{DATE}.md"
    (report_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PoseBusters ligand-only checks over 22-repo portfolio SDFs.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--baseline-rows", type=Path, default=Path("outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/gnina_redock_rows.csv"))
    parser.add_argument("--selected-rows", type=Path, default=Path("outputs/self_contained_redock_portfolio_v0/self_contained_portfolio_selected_rows.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_rows = load_method_rows(root, args.baseline_rows, args.selected_rows)
    method_rows = method_rows[method_rows["portfolio_policy"].isin(FOCUS_POLICIES)].copy()
    rows = build_rows(method_rows, root)
    summary = summarize(rows)
    rows.to_csv(args.output_dir / "tnnls_posebusters_ligand_rows.csv", index=False)
    summary.to_csv(args.output_dir / "tnnls_posebusters_ligand_summary.csv", index=False)
    write_report(args.output_dir, summary)
    copy_public(args.output_dir, args.report_dir)
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
