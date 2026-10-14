from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
TARGET_KEYS = ["dataset", "target_file"]
BASE_POLICY = "pl_pose_clash_rankfusion_w100_plsafe"
SAFE_COLUMNS = [
    "dataset",
    "generator",
    "target_file",
    "data_id",
    "mol_idx",
    "candidate_policy",
    "portfolio_policy",
    "source_table",
    "eval_qed",
    "eval_sa_score",
    "contact_jaccard",
    "redock_affinity",
    "redock_cnnscore",
    "redock_cnnaffinity",
    "input_to_redock_rmsd",
    "redock_rmsd_pass_2a",
    "pass",
    "redock_sdf",
    "export_sdf",
    "constraint_candidates",
    "constraint_pass",
    "fallback_stage",
]


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def discover_redock_rows(root: Path) -> list[Path]:
    rows = []
    for path in root.rglob("*redock*_rows.csv"):
        try:
            cols = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if {"redock_affinity", "redock_cnnscore", "input_to_redock_rmsd"}.issubset(cols):
            rows.append(path)
    return sorted(rows)


def load_portfolio(paths: list[Path], repo_root: Path) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_table"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No redock row files found.")
    table = pd.concat(frames, ignore_index=True, sort=False)
    table = table[table["dataset"].isin(["bindingmoad", "prospective"])].copy()
    table["candidate_policy"] = table["policy"].astype(str)
    for col in METRICS.values():
        table[col] = numeric(table, col)
    table["redock_success"] = table["redock_success"].fillna(False).astype(bool)
    table["redock_rmsd_pass_2a"] = table["redock_rmsd_pass_2a"].fillna(False).astype(bool)
    table["pass"] = table["redock_rmsd_pass_2a"].astype(float)
    table = table[table["redock_success"]].dropna(subset=list(METRICS.values())).copy()
    for path_col in ["redock_sdf", "export_sdf"]:
        if path_col not in table:
            table[path_col] = ""
        table[path_col] = table[path_col].fillna("").astype(str)
        localish = table[path_col].str.startswith("outputs/") | table[path_col].str.startswith("./outputs/")
        exists = table[path_col].map(lambda p: not p or (repo_root / p).exists())
        table.loc[~(localish & exists), path_col] = ""
    table = table.sort_values(["dataset", "target_file", "candidate_policy", "source_table"]).drop_duplicates(
        subset=["dataset", "target_file", "candidate_policy", "mol_idx", "redock_affinity", "redock_cnnscore", "eval_qed", "eval_sa_score"],
        keep="first",
    )
    return table.reset_index(drop=True)


def load_baseline(path: Path) -> pd.DataFrame:
    base = pd.read_csv(path, low_memory=False)
    base = base[(base["dataset"].isin(["bindingmoad", "prospective"])) & (base["policy"] == BASE_POLICY)].copy()
    for col in METRICS.values():
        base[col] = numeric(base, col)
    base["redock_rmsd_pass_2a"] = base["redock_rmsd_pass_2a"].fillna(False).astype(bool)
    base["pass"] = base["redock_rmsd_pass_2a"].astype(float)
    return base.dropna(subset=list(METRICS.values())).set_index(TARGET_KEYS)


def add_target_z(pool: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in pool.groupby(TARGET_KEYS, sort=False):
        g = group.copy()
        for name, col in METRICS.items():
            oriented = DIRECTION[name] * g[col].astype(float)
            std = oriented.std(ddof=0)
            g[f"z_{name}"] = 0.0 if not np.isfinite(std) or std == 0 else (oriented - oriented.mean()) / std
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def scalar_score(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for name, weight in WEIGHTS.items():
        score += weight * frame[f"z_{name}"]
    return score


def base_join(pool: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base = baseline.reset_index()[TARGET_KEYS + [*METRICS.values(), "pass"]].copy()
    base = base.rename(columns={col: f"base_{col}" for col in [*METRICS.values(), "pass"]})
    return pool.merge(base, on=TARGET_KEYS, how="inner")


def mask_for(group: pd.DataFrame, policy: str) -> pd.Series:
    mask = group["pass"] >= group["base_pass"]
    if policy == "portfolio_dominance_scalar":
        for name, col in METRICS.items():
            if DIRECTION[name] > 0:
                mask &= group[col] >= group[f"base_{col}"] - 1e-9
            else:
                mask &= group[col] <= group[f"base_{col}"] + 1e-9
    elif policy == "portfolio_eps_scalar_strict_rmsd":
        mask &= group["contact_jaccard"] >= group["base_contact_jaccard"] + 0.02
        mask &= group["eval_sa_score"] <= group["base_eval_sa_score"] + 0.20
        mask &= group["input_to_redock_rmsd"] <= group["base_input_to_redock_rmsd"]
    elif policy == "portfolio_eps_scalar_balanced":
        mask &= group["contact_jaccard"] >= group["base_contact_jaccard"] + 0.02
        mask &= group["eval_sa_score"] <= group["base_eval_sa_score"] + 0.20
        mask &= group["input_to_redock_rmsd"] <= group["base_input_to_redock_rmsd"] + 0.10
    elif policy == "portfolio_affinity_guarded":
        mask &= group["contact_jaccard"] >= group["base_contact_jaccard"]
        mask &= group["eval_sa_score"] <= group["base_eval_sa_score"] + 0.25
        mask &= group["input_to_redock_rmsd"] <= group["base_input_to_redock_rmsd"] + 0.10
    elif policy == "portfolio_cnn_qed_guarded":
        mask &= group["contact_jaccard"] >= group["base_contact_jaccard"]
        mask &= group["eval_sa_score"] <= group["base_eval_sa_score"] + 0.20
        mask &= group["input_to_redock_rmsd"] <= group["base_input_to_redock_rmsd"] + 0.10
    else:
        raise ValueError(policy)
    return mask


def select_policy(pool: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows = []
    for _, group in pool.groupby(TARGET_KEYS, sort=False):
        g = group.copy()
        g["_scalar"] = scalar_score(g)
        g["_cnn_qed"] = 1.2 * g["z_cnn"] + 1.0 * g["z_qed"] + 0.8 * g["z_cnnaff"] + 0.6 * g["z_contact"] + 0.5 * g["z_sa"]
        mask = mask_for(g, policy)
        candidates = g[mask].copy()
        fallback_stage = "none"
        if candidates.empty:
            candidates = g[g["pass"] >= g["base_pass"]].copy()
            fallback_stage = "pass_only"
        if candidates.empty:
            candidates = g.copy()
            fallback_stage = "all"
        if policy == "portfolio_affinity_guarded":
            sort_cols, ascending = ["redock_affinity", "contact_jaccard", "_scalar"], [True, False, False]
        elif policy == "portfolio_cnn_qed_guarded":
            sort_cols, ascending = ["_cnn_qed", "contact_jaccard", "redock_affinity"], [False, False, True]
        elif fallback_stage != "none":
            sort_cols, ascending = ["contact_jaccard", "_scalar"], [False, False]
        else:
            sort_cols, ascending = ["_scalar", "contact_jaccard"], [False, False]
        selected = candidates.sort_values(sort_cols, ascending=ascending).head(1).copy()
        selected["portfolio_policy"] = policy
        selected["constraint_candidates"] = int(mask.sum())
        selected["constraint_pass"] = bool(mask.any())
        selected["fallback_stage"] = fallback_stage
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def summarize(selected: pd.DataFrame) -> pd.DataFrame:
    return selected.groupby(["dataset", "portfolio_policy"], dropna=False).agg(
        rows=("target_file", "size"),
        pass_rate=("pass", "mean"),
        rmsd=("input_to_redock_rmsd", "mean"),
        affinity=("redock_affinity", "mean"),
        cnn=("redock_cnnscore", "mean"),
        cnnaff=("redock_cnnaffinity", "mean"),
        qed=("eval_qed", "mean"),
        sa=("eval_sa_score", "mean"),
        contact=("contact_jaccard", "mean"),
        constraint_pass_rate=("constraint_pass", "mean"),
        constraint_candidates_mean=("constraint_candidates", "mean"),
    ).reset_index()


def baseline_summary(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = baseline.reset_index().copy()
    rows["portfolio_policy"] = BASE_POLICY
    rows["constraint_pass"] = np.nan
    rows["constraint_candidates"] = np.nan
    return summarize(rows)


def delta_vs_base(summary: pd.DataFrame, base_summary_frame: pd.DataFrame) -> pd.DataFrame:
    base = base_summary_frame.set_index("dataset")
    rows = []
    for _, row in summary.iterrows():
        b = base.loc[row["dataset"]]
        rec = {"dataset": row["dataset"], "portfolio_policy": row["portfolio_policy"]}
        all_better = True
        score = 0.0
        for name in METRICS:
            raw_delta = float(row[name]) - float(b[name])
            oriented = DIRECTION[name] * raw_delta
            rec[f"{name}_delta_vs_old"] = raw_delta
            rec[f"{name}_better"] = oriented >= -1e-12
            all_better &= oriented >= -1e-12
            score += WEIGHTS[name] * float(np.clip(oriented / (0.02 if name in {"cnn", "qed", "contact"} else 0.10), -3, 3))
        rec["all_aggregate_better"] = all_better
        rec["weighted_delta_score"] = score
        rows.append(rec)
    return pd.DataFrame(rows)


def target_dominance(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        rec = {"dataset": row["dataset"], "target_file": row["target_file"], "portfolio_policy": row["portfolio_policy"]}
        for name, col in METRICS.items():
            raw_delta = float(row[col]) - float(row[f"base_{col}"])
            rec[f"{name}_better_or_equal"] = DIRECTION[name] * raw_delta >= -1e-9
        rec["pass_better_or_equal"] = row["pass"] >= row["base_pass"] - 1e-9
        rows.append(rec)
    return pd.DataFrame(rows)


def safe_selected_rows(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    for col in SAFE_COLUMNS:
        if col not in out:
            out[col] = ""
    return out[SAFE_COLUMNS].copy()


def write_report(output_dir: Path, table_paths: list[Path], pool: pd.DataFrame, combined: pd.DataFrame, delta: pd.DataFrame, dominance: pd.DataFrame) -> None:
    lines = [
        "# Self-Contained Redock Portfolio",
        "",
        "Date: 2026-06-18",
        "",
        "Scope: only files inside the supplied repository are used. The portfolio ignores historical source paths and keeps only local redock rows, local SDF paths, and metric columns.",
        "",
        "## Inputs",
        "",
        f"- Redock row files: {len(table_paths)}",
        f"- Portfolio candidates after filtering/deduplication: {len(pool)}",
        f"- BindingMOAD candidates: {int((pool['dataset'] == 'bindingmoad').sum())}",
        f"- Prospective candidates: {int((pool['dataset'] == 'prospective').sum())}",
        "",
        "## Summary",
        "",
        "```csv",
        combined.to_csv(index=False).strip(),
        "```",
        "",
        "## Delta vs Old Final",
        "",
        "```csv",
        delta.to_csv(index=False).strip(),
        "```",
        "",
        "## Target-Level Non-Degradation",
        "",
        "```csv",
        dominance.to_csv(index=False).strip(),
        "```",
    ]
    (output_dir / "self_contained_redock_portfolio.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 22-repo-only redock candidate portfolio and test constrained selectors.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--baseline-rows", type=Path, default=Path("outputs/plsafe_verifier_rerank_v0/gnina_redock_raw_old_composite/gnina_redock_rows.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--redock-rows", action="append", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.root.resolve()
    row_paths = args.redock_rows or discover_redock_rows(repo_root / "outputs")
    pool = add_target_z(base_join(load_portfolio(row_paths, repo_root), load_baseline(args.baseline_rows)))
    policies = [
        "portfolio_dominance_scalar",
        "portfolio_eps_scalar_strict_rmsd",
        "portfolio_eps_scalar_balanced",
        "portfolio_affinity_guarded",
        "portfolio_cnn_qed_guarded",
    ]
    selected = pd.concat([select_policy(pool, policy) for policy in policies], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_selected_rows(selected).to_csv(args.output_dir / "self_contained_portfolio_selected_rows.csv", index=False)
    pool_counts = pool.groupby(["dataset", "target_file"], dropna=False).size().reset_index(name="candidates")
    pool_counts.to_csv(args.output_dir / "self_contained_portfolio_candidate_counts.csv", index=False)
    summary = summarize(selected)
    base_summary_frame = baseline_summary(load_baseline(args.baseline_rows))
    combined = pd.concat([base_summary_frame, summary], ignore_index=True, sort=False)
    combined.to_csv(args.output_dir / "self_contained_portfolio_summary.csv", index=False)
    delta = delta_vs_base(summary, base_summary_frame)
    delta.to_csv(args.output_dir / "self_contained_portfolio_delta_vs_old.csv", index=False)
    dom_rows = target_dominance(selected)
    dom = dom_rows.groupby(["dataset", "portfolio_policy"], dropna=False).mean(numeric_only=True).reset_index()
    dom_rows.to_csv(args.output_dir / "self_contained_portfolio_target_dominance_rows.csv", index=False)
    dom.to_csv(args.output_dir / "self_contained_portfolio_target_dominance.csv", index=False)
    write_report(args.output_dir, row_paths, pool, combined, delta, dom)
    print(combined.to_csv(index=False))


if __name__ == "__main__":
    main()
