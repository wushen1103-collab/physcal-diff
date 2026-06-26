from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from physcaldiff.metrics import binary_metrics


def stable_unit(value: str, seed: int = 22) -> float:
    text = f"{seed}|{value}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:12], 16) / float(16**12 - 1)


def wilson_upper(k: np.ndarray, n: np.ndarray, z: float = 1.6448536269514722) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    phat = np.divide(k, np.maximum(n, 1.0))
    denom = 1.0 + z * z / np.maximum(n, 1.0)
    center = phat + z * z / (2.0 * np.maximum(n, 1.0))
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * np.maximum(n, 1.0))) / np.maximum(n, 1.0))
    return (center + half) / denom


def conformal_threshold(risk: np.ndarray, bad: np.ndarray, alpha: float, bound: str) -> tuple[float, float]:
    order = np.argsort(risk)
    risk_sorted = risk[order]
    bad_sorted = bad[order].astype(float)
    count = np.arange(1, len(risk_sorted) + 1)
    cum_bad = np.cumsum(bad_sorted)
    empirical = cum_bad / count
    if bound == "empirical":
        controlled = empirical
    elif bound == "wilson95":
        controlled = wilson_upper(cum_bad, count)
    else:
        raise ValueError(f"Unknown risk bound: {bound}")
    ok = np.where(controlled <= alpha)[0]
    if len(ok) == 0:
        return -math.inf, math.nan
    idx = int(ok[-1])
    return float(risk_sorted[idx]), float(controlled[idx])


def basename(path: object) -> str:
    return Path(str(path)).name


def read_generated_risk(path: Path, risk_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["kind"].astype(str) == "generated"].copy()
    frame["source_base"] = frame["source_file"].map(basename)
    frame["mol_idx"] = frame["mol_index"].astype(int)
    frame = frame.rename(columns={"risk_prob": risk_name})
    keep = ["source_base", "mol_idx", risk_name, "qed", "lp_min", "center_dist", "contacts_lt_4_0_per_lig"]
    optional = [c for c in keep if c in frame.columns]
    return frame[optional]


def combine_seeded_diffsbdd_risk(results_root: Path) -> pd.DataFrame:
    files = [
        results_root / "diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv",
        results_root / "diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores_seed2027.csv",
        results_root / "diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores_seed2028.csv",
    ]
    parts = []
    for idx, path in enumerate(files):
        if path.exists():
            parts.append(read_generated_risk(path, f"risk_prob_seed{idx}"))
    if not parts:
        return pd.DataFrame()
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["source_base", "mol_idx"], how="inner", suffixes=("", "_drop"))
        merged = merged[[c for c in merged.columns if not c.endswith("_drop")]]
    seed_cols = [c for c in merged.columns if c.startswith("risk_prob_seed")]
    merged["risk_prob"] = merged[seed_cols].mean(axis=1)
    if len(seed_cols) > 1:
        merged["risk_prob_std"] = merged[seed_cols].std(axis=1, ddof=0)
    else:
        merged["risk_prob_std"] = 0.0
    return merged


def join_diffsbdd(audit: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    risk = combine_seeded_diffsbdd_risk(results_root)
    if risk.empty:
        return risk
    sub = audit[audit["generator"] == "diffsbdd"].copy()
    sub["source_base"] = sub["relative_file"].map(basename)
    return sub.merge(risk, on=["source_base", "mol_idx"], how="inner")


def join_molcraft(audit: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    risk_path = results_root / "molcraft_crossdock_t50_n16_risk_scores.csv"
    if not risk_path.exists():
        return pd.DataFrame()
    risk = pd.read_csv(risk_path)
    risk = risk[risk["kind"].astype(str) == "generated"].copy()
    risk["data_id"] = risk["data_id"].astype(int)
    risk["raw_mol_index"] = risk["mol_index"].astype(int)
    risk = risk.rename(columns={"risk_prob": "risk_prob"})

    sub = audit[audit["generator"] == "molcraft"].copy()
    sub["data_id"] = sub["relative_file"].str.extract(r"data(\d+)/")[0].astype(int)
    sub["raw_mol_index"] = sub["relative_file"].map(lambda x: int(Path(str(x)).stem))
    keep = [
        "data_id",
        "raw_mol_index",
        "risk_prob",
        "qed",
        "lp_min",
        "center_dist",
        "contacts_lt_4_0_per_lig",
    ]
    return sub.merge(risk[[c for c in keep if c in risk.columns]], on=["data_id", "raw_mol_index"], how="inner")


def pocketflow_molfast_map(results_root: Path) -> pd.DataFrame:
    path = results_root / "pocketflow_crossdock_n16_molfast.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["raw_dir"] = frame["molecule"].astype(str).str.extract(r"-(/.*?/)$")[0]
    frame["raw_mol_index"] = frame["position"].astype(int)
    frame["source_base"] = frame["file"].map(basename)
    return frame[["raw_dir", "raw_mol_index", "source_base"]].dropna()


def join_pocketflow(audit: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    risk_path = results_root / "pocketflow_crossdock_n16_risk_scores.csv"
    if not risk_path.exists():
        return pd.DataFrame()
    risk = read_generated_risk(risk_path, "risk_prob")
    mapping = pocketflow_molfast_map(results_root)
    if mapping.empty:
        return pd.DataFrame()

    sub = audit[audit["generator"] == "pocketflow"].copy()
    sub["raw_dir"] = sub["sdf_file"].astype(str).str.replace("generated.sdf", "", regex=False)
    sub["raw_mol_index"] = sub["mol_idx"].astype(int)
    mapped = sub.merge(mapping, on=["raw_dir", "raw_mol_index"], how="inner")
    return mapped.merge(risk, on=["source_base", "mol_idx"], how="inner")


def molpilot_molfast_map(results_root: Path) -> pd.DataFrame:
    path = results_root / "molpilot_crossdock_t50_n16_molfast.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["raw_name"] = frame["molecule"].astype(str)
    frame["source_base"] = frame["file"].map(basename)
    frame["mol_idx"] = frame["position"].astype(int)
    return frame[["raw_name", "source_base", "mol_idx"]]


def join_molpilot(audit: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    risk_path = results_root / "molpilot_crossdock_t50_n16_risk_scores.csv"
    if not risk_path.exists():
        return pd.DataFrame()
    risk = read_generated_risk(risk_path, "risk_prob")
    mapping = molpilot_molfast_map(results_root)
    if mapping.empty:
        return pd.DataFrame()

    sub = audit[audit["generator"] == "molpilot"].copy()
    sub["raw_name"] = sub["relative_file"].map(lambda x: Path(str(x)).stem)
    mapped = sub.merge(mapping, on="raw_name", how="inner", suffixes=("", "_mapped"))
    return mapped.merge(risk, on=["source_base", "mol_idx"], how="inner")


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["bad"] = out["rdkit_physics_high_risk"].astype(bool).astype(int)
    if "qed" not in out.columns:
        for candidate in ["qed_x", "qed_audit", "qed_y"]:
            if candidate in out.columns:
                out["qed"] = out[candidate]
                break
    if "qed" not in out.columns:
        out["qed"] = np.nan
    out["qed_risk"] = 1.0 - out["qed"].astype(float).clip(0.0, 1.0)
    if "risk_prob_std" not in out.columns:
        out["risk_prob_std"] = 0.0
    out["risk_plus_std"] = np.clip(out["risk_prob"].astype(float) + out["risk_prob_std"].astype(float), 0.0, 1.0)
    return out


def group_col_for(frame: pd.DataFrame) -> pd.Series:
    if "generator" not in frame.columns:
        return frame.index.astype(str)
    gen = frame["generator"].iloc[0]
    if gen == "diffsbdd":
        return frame["relative_file"].astype(str)
    if gen == "molcraft":
        return frame["data_id"].astype(str)
    if gen == "pocketflow":
        return frame["raw_dir"].astype(str)
    if gen == "molpilot":
        return frame["raw_name"].astype(str)
    return frame.index.astype(str)


def summarize_metrics(frame: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    for generator, sub in frame.groupby("generator"):
        for score_col in score_cols:
            if score_col not in sub.columns or sub[score_col].nunique(dropna=True) <= 1:
                continue
            clean = sub.dropna(subset=[score_col, "bad"]).copy()
            if len(clean) == 0 or clean["bad"].nunique() < 2:
                continue
            row = {
                "generator": generator,
                "score_col": score_col,
                "n": len(clean),
                "bad_rate": float(clean["bad"].mean()),
            }
            row.update(binary_metrics(clean["bad"].to_numpy(), clean[score_col].to_numpy()))
            rows.append(row)

    pooled = frame[frame["generator"] != "molpilot"].copy()
    for score_col in score_cols:
        if score_col not in pooled.columns or pooled[score_col].nunique(dropna=True) <= 1:
            continue
        clean = pooled.dropna(subset=[score_col, "bad"]).copy()
        if len(clean) == 0 or clean["bad"].nunique() < 2:
            continue
        row = {
            "generator": "pooled_without_molpilot",
            "score_col": score_col,
            "n": len(clean),
            "bad_rate": float(clean["bad"].mean()),
        }
        row.update(binary_metrics(clean["bad"].to_numpy(), clean[score_col].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_selection(frame: pd.DataFrame, score_cols: list[str], alphas: list[float]) -> pd.DataFrame:
    rows = []
    for generator, sub in frame.groupby("generator"):
        if len(sub) < 50 or sub["bad"].nunique() < 2:
            continue
        sub = sub.copy()
        groups = group_col_for(sub)
        is_cal = groups.map(lambda x: stable_unit(str(x), seed=22) < 0.6).to_numpy()
        if is_cal.sum() == 0 or (~is_cal).sum() == 0:
            continue
        cal = sub.loc[is_cal].copy()
        test = sub.loc[~is_cal].copy()
        for score_col in score_cols:
            if score_col not in sub.columns or cal[score_col].nunique(dropna=True) <= 1:
                continue
            for bound in ["empirical", "wilson95"]:
                for alpha in alphas:
                    tau, controlled = conformal_threshold(
                        cal[score_col].astype(float).to_numpy(),
                        cal["bad"].astype(int).to_numpy(),
                        alpha,
                        bound,
                    )
                    accepted = test[score_col].astype(float).to_numpy() <= tau
                    if accepted.any():
                        accepted_bad = float(test.loc[accepted, "bad"].mean())
                        accepted_reliable = float(1.0 - accepted_bad)
                    else:
                        accepted_bad = math.nan
                        accepted_reliable = math.nan
                    coverage = float(accepted.mean())
                    rows.append(
                        {
                            "generator": generator,
                            "score_col": score_col,
                            "risk_bound": bound,
                            "alpha": alpha,
                            "threshold": tau,
                            "calibration_controlled_bad_rate_at_tau": controlled,
                            "n_cal": len(cal),
                            "n_test": len(test),
                            "cal_bad_rate": float(cal["bad"].mean()),
                            "test_bad_rate": float(test["bad"].mean()),
                            "coverage": coverage,
                            "accepted_bad_rate": accepted_bad,
                            "accepted_reliable_rate": accepted_reliable,
                            "resampling_multiplier": float(1.0 / max(coverage, 1e-12)),
                        }
                    )
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, merged: pd.DataFrame, metrics: pd.DataFrame, selection: pd.DataFrame) -> None:
    lines = [
        "# SDF Risk Alignment",
        "",
        "This table-only experiment joins legacy pocket-risk scores with the RDKit physical audit labels.",
        "",
        "## Joined Rows",
        merged.groupby("generator").size().rename("rows").reset_index().to_csv(index=False).strip(),
        "",
        "## Detection Metrics",
        metrics.sort_values(["generator", "score_col"]).to_csv(index=False).strip(),
        "",
        "## Wilson95 Alpha=0.05 Selection Rows Meeting Test Alpha",
    ]
    valid = selection[
        (selection["risk_bound"] == "wilson95")
        & (selection["alpha"] == 0.05)
        & (selection["accepted_bad_rate"] <= selection["alpha"])
    ].copy()
    if len(valid):
        lines.append(
            valid.sort_values(["generator", "coverage"], ascending=[True, False]).to_csv(index=False).strip()
        )
    else:
        lines.append("No Wilson95 alpha=0.05 row met the test alpha.")
    (out_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("external/generator_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sdf_risk_alignment_v0/reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit_csv)

    parts = [
        join_diffsbdd(audit, args.results_root),
        join_molcraft(audit, args.results_root),
        join_pocketflow(audit, args.results_root),
        join_molpilot(audit, args.results_root),
    ]
    merged = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if merged.empty:
        raise RuntimeError("No rows could be joined between SDF audit and risk scores.")
    merged = add_scores(merged)

    score_cols = ["risk_prob", "risk_prob_std", "risk_plus_std", "qed_risk"]
    metrics = summarize_metrics(merged, score_cols)
    selection = summarize_selection(merged, score_cols, alphas=[0.05, 0.10, 0.20])

    merged.to_csv(args.output_dir / "joined_sdf_risk_alignment.csv", index=False)
    metrics.to_csv(args.output_dir / "sdf_risk_detection_metrics.csv", index=False)
    selection.to_csv(args.output_dir / "sdf_risk_wilson_selection.csv", index=False)
    write_summary(args.output_dir, merged, metrics, selection)
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
