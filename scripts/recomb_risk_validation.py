from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


def load_original_helpers(repo: Path):
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))
    import train_physics_risk_critic as original

    return original


def labels(frame: pd.DataFrame, thresholds: dict[str, float], original) -> pd.DataFrame:
    out = original.add_derived_features(frame.copy())
    out["invalid"] = ~(original.as_bool(out["sanitize_ok"]) & original.as_bool(out["has_3d"]) & original.as_bool(out["ff_ok"]))
    out["bond"] = pd.to_numeric(out["bond_outlier_count"], errors="coerce").fillna(0).gt(0)
    out["clash"] = pd.to_numeric(out["internal_clash_count"], errors="coerce").fillna(0).gt(0)
    for name, column, threshold_name in [
        ("energy", "ff_energy_per_heavy", "energy_per_heavy_threshold"),
        ("drop", "ff_energy_drop", "energy_drop_threshold"),
        ("relax_rmsd", "ff_relax_rmsd", "relax_rmsd_threshold"),
    ]:
        out[name] = pd.to_numeric(out[column], errors="coerce").gt(thresholds[threshold_name])
    out["bad"] = out[["invalid", "bond", "clash", "energy", "drop", "relax_rmsd"]].any(axis=1).astype(int)
    return out


def features_for(name: str, original) -> list[str]:
    full = original.feature_sets()["pre_relax_ff"]
    energy = {"ff_energy_initial", "ff_energy_initial_per_heavy"}
    bond = {"min_bond_ratio", "max_bond_ratio", "bond_ratio_span", "low_bond_margin", "high_bond_margin"}
    clash = {"internal_clash_count"}
    omitted = {
        "full": set(),
        "no_energy": energy,
        "no_bond": bond,
        "no_clash": clash,
        "descriptor_only": energy | bond | clash,
    }[name]
    return [column for column in full if column not in omitted]


def fit_scores(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int, trees: int, jobs: int):
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=trees, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=seed, n_jobs=jobs,
        )),
    ])
    model.fit(train[columns], train["bad"])
    return tuple(model.predict_proba(part[columns])[:, 1] for part in (train, cal, test))


def group_means(frame: pd.DataFrame, score: np.ndarray, threshold: float, original) -> pd.DataFrame:
    groups = frame["receptor_id"] if "receptor_id" in frame else original.group_key(frame)
    grouped = pd.DataFrame({
        "group": groups.to_numpy(),
        "accepted": (score <= threshold).astype(float),
        "bad_accepted": (score <= threshold).astype(float) * frame["bad"].to_numpy(),
    }).groupby("group", sort=False)[["accepted", "bad_accepted"]].mean()
    return grouped


def calibrate(train_score, cal: pd.DataFrame, cal_score, alpha: float, original, grid_size: int, confidence: float):
    wilson_threshold, wilson_bound = original.conformal_threshold(cal_score, cal["bad"].to_numpy(), alpha, "wilson95")
    grid = np.unique(np.quantile(train_score, np.linspace(0.0, 1.0, grid_size)))
    n_groups = cal["receptor_id"].nunique() if "receptor_id" in cal else original.group_key(cal).nunique()
    margin = (1.0 + alpha) * math.sqrt(math.log(len(grid) / (1.0 - confidence)) / (2.0 * n_groups))
    certified = []
    for threshold in grid:
        group = group_means(cal, cal_score, threshold, original)
        residual_upper = float((group["bad_accepted"] - alpha * group["accepted"]).mean() + margin)
        if residual_upper <= 0.0:
            certified.append((float(threshold), residual_upper))
    valid_threshold, valid_residual = certified[-1] if certified else (-math.inf, math.nan)
    return {
        "wilson": (wilson_threshold, wilson_bound),
        "group_hoeffding": (valid_threshold, valid_residual),
        "cal_groups": n_groups,
        "grid_size": len(grid),
    }


def eval_gate(test: pd.DataFrame, score: np.ndarray, threshold: float, original) -> dict[str, float]:
    if not np.isfinite(threshold):
        return {"coverage": 0.0, "accepted_bad": math.nan, "group_coverage": 0.0, "group_accepted_bad": math.nan, "accepted_count": 0}
    accepted = score <= threshold
    grouped = group_means(test, score, threshold, original)
    group_coverage = float(grouped["accepted"].mean())
    return {
        "coverage": float(accepted.mean()),
        "accepted_bad": float(test.loc[accepted, "bad"].mean()) if accepted.any() else math.nan,
        "group_coverage": group_coverage,
        "group_accepted_bad": float(grouped["bad_accepted"].mean() / group_coverage) if group_coverage > 0 else math.nan,
        "accepted_count": int(accepted.sum()),
    }


def score_metrics(y: pd.Series, score: np.ndarray) -> dict[str, float]:
    if y.nunique() < 2:
        return {"roc_auc": math.nan, "pr_auc": math.nan}
    return {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score))}


def run_one(frame: pd.DataFrame, original, seed: int, q: float, ablation: str, trees: int, jobs: int, grid_size: int, confidence: float):
    split = original.assign_split(original.group_key(frame), seed)
    raw_train, raw_cal, raw_test = [frame.loc[split == part].copy() for part in ("train", "cal", "test")]
    thresholds = original.quantile_thresholds(raw_train, q)
    train, cal, test = [labels(part, thresholds, original) for part in (raw_train, raw_cal, raw_test)]
    columns = features_for(ablation, original)
    train_score, cal_score, test_score = fit_scores(train, cal, test, columns, seed, trees, jobs)
    record = {
        "seed": seed, "quantile": q, "features": ablation,
        "train_n": len(train), "cal_n": len(cal), "test_n": len(test),
        "train_groups": original.group_key(train).nunique(),
        "cal_groups": original.group_key(cal).nunique(),
        "test_groups": original.group_key(test).nunique(),
        "test_bad_prevalence": float(test["bad"].mean()),
        **thresholds, **score_metrics(test["bad"], test_score),
    }
    gates = []
    for alpha in (0.01, 0.05, 0.10, 0.20):
        result = calibrate(train_score, cal, cal_score, alpha, original, grid_size, confidence)
        for method in ("wilson", "group_hoeffding"):
            threshold, bound = result[method]
            gates.append({
                "seed": seed, "quantile": q, "features": ablation, "alpha": alpha,
                "method": method, "threshold": threshold, "cal_bound": bound,
                "confidence": confidence if method == "group_hoeffding" else math.nan,
                "cal_groups": result["cal_groups"], "grid_size": result["grid_size"],
                **eval_gate(test, test_score, threshold, original),
            })
    failures = []
    for name in ("invalid", "bond", "clash", "energy", "drop", "relax_rmsd"):
        failures.append({"seed": seed, "quantile": q, "features": ablation, "failure_type": name,
                         "n_positive": int(test[name].sum()), **score_metrics(test[name].astype(int), test_score)})
    return record, gates, failures


def run_loo(frame: pd.DataFrame, original, seed: int, trees: int, jobs: int, grid_size: int, confidence: float):
    records = []
    for heldout in original.MAIN_GENERATORS:
        source = frame.loc[frame["generator"] != heldout].copy()
        target = frame.loc[frame["generator"] == heldout].copy()
        source_split = original.assign_split(original.group_key(source), seed)
        raw_train = source.loc[source_split != "test"].copy()
        raw_cal = source.loc[source_split == "test"].copy()
        thresholds = original.quantile_thresholds(raw_train, 0.90)
        train, cal, test = [labels(part, thresholds, original) for part in (raw_train, raw_cal, target)]
        train_score, cal_score, test_score = fit_scores(train, cal, test, features_for("full", original), seed, trees, jobs)
        for alpha in (0.05, 0.10, 0.20):
            result = calibrate(train_score, cal, cal_score, alpha, original, grid_size, confidence)
            for method in ("wilson", "group_hoeffding"):
                threshold, _ = result[method]
                records.append({"seed": seed, "heldout_generator": heldout, "alpha": alpha, "method": method,
                                "source_calibration_only": True, **score_metrics(test["bad"], test_score),
                                **eval_gate(test, test_score, threshold, original)})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 23, 24, 25, 26])
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    original = load_original_helpers(repo)
    frame = pd.read_csv(repo / "outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv", low_memory=False)
    frame = frame[frame["generator"].isin(original.MAIN_GENERATORS)].copy()
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = [(seed, 0.90, "full") for seed in args.seeds]
    if not args.smoke:
        scenarios.extend((seed, q, "full") for seed in args.seeds[:3] for q in (0.80, 0.85, 0.95))
        scenarios.extend((seed, 0.90, name) for seed in args.seeds[:3] for name in ("no_energy", "no_bond", "no_clash", "descriptor_only"))
    records, gates, failures = [], [], []
    for seed, q, name in scenarios:
        record, gate_rows, failure_rows = run_one(frame, original, seed, q, name, args.trees, args.jobs, args.grid_size, args.confidence)
        records.append(record)
        gates.extend(gate_rows)
        failures.extend(failure_rows)
        print(f"completed seed={seed} q={q} features={name} roc={record['roc_auc']:.4f}", flush=True)
    loo = [] if args.smoke else [row for seed in args.seeds[:3] for row in run_loo(frame, original, seed, args.trees, args.jobs, args.grid_size, args.confidence)]
    pd.DataFrame(records).to_csv(args.out / "risk_detection.csv", index=False)
    pd.DataFrame(gates).to_csv(args.out / "risk_calibration.csv", index=False)
    pd.DataFrame(failures).to_csv(args.out / "risk_failure_types.csv", index=False)
    pd.DataFrame(loo).to_csv(args.out / "cross_generator_source_only.csv", index=False)
    (args.out / "protocol.json").write_text(json.dumps({
        "source": "outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv",
        "n_rows": len(frame), "seeds": args.seeds, "trees": args.trees, "grid_size": args.grid_size,
        "group_bound": "one-sided Hoeffding on per-parent-group B-alpha*A; Bonferroni over train-fixed threshold grid",
        "estimand": "bad rate after selection for a uniform parent group then a uniform candidate within group",
        "assumptions": "independent calibration parent groups; deployment groups sampled from the same distribution; fixed trained critic and threshold grid",
        "label_quantiles": "computed on training rows only for each split, including cross-generator source-only runs",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
