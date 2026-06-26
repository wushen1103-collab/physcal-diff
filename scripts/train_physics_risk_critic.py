from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physcaldiff.metrics import binary_metrics


MAIN_GENERATORS = ["diffsbdd", "molcraft", "pocketflow"]


def stable_unit(value: str, seed: int) -> float:
    text = f"{seed}|{value}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:12], 16) / float(16**12 - 1)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def wilson_upper(k: np.ndarray, n: np.ndarray, z: float = 1.6448536269514722) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    phat = np.divide(k, np.maximum(n, 1.0))
    denom = 1.0 + z * z / np.maximum(n, 1.0)
    center = phat + z * z / (2.0 * np.maximum(n, 1.0))
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * np.maximum(n, 1.0))) / np.maximum(n, 1.0))
    return (center + half) / denom


def conformal_threshold(risk: np.ndarray, bad: np.ndarray, alpha: float, bound: str = "wilson95") -> tuple[float, float]:
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
        raise ValueError(f"Unknown bound: {bound}")
    ok = np.where(controlled <= alpha)[0]
    if len(ok) == 0:
        return -math.inf, math.nan
    idx = int(ok[-1])
    return float(risk_sorted[idx]), float(controlled[idx])


def quantile_thresholds(frame: pd.DataFrame, q: float) -> dict[str, float]:
    return {
        "energy_per_heavy_threshold": float(frame["ff_energy_per_heavy"].dropna().quantile(q)),
        "energy_drop_threshold": float(frame["ff_energy_drop"].dropna().quantile(q)),
        "relax_rmsd_threshold": float(frame["ff_relax_rmsd"].dropna().quantile(q)),
    }


def add_physics_labels(frame: pd.DataFrame, q: float) -> tuple[pd.DataFrame, dict[str, float]]:
    threshold_pool = frame[frame["generator"].isin(MAIN_GENERATORS)].copy()
    thresholds = quantile_thresholds(threshold_pool, q)
    out = frame.copy()
    out["sanitize_ok_bool"] = as_bool(out["sanitize_ok"])
    out["has_3d_bool"] = as_bool(out["has_3d"])
    out["ff_ok_bool"] = as_bool(out["ff_ok"])
    out["invalid_or_failed"] = (~out["sanitize_ok_bool"]) | (~out["has_3d_bool"]) | (~out["ff_ok_bool"])
    out["bond_outlier"] = out["bond_outlier_count"].fillna(0).astype(float) > 0
    out["internal_clash"] = out["internal_clash_count"].fillna(0).astype(float) > 0
    out["high_energy"] = out["ff_energy_per_heavy"].astype(float) > thresholds["energy_per_heavy_threshold"]
    out["high_relax_drop"] = out["ff_energy_drop"].astype(float) > thresholds["energy_drop_threshold"]
    out["high_relax_rmsd"] = out["ff_relax_rmsd"].astype(float) > thresholds["relax_rmsd_threshold"]
    out["hard_rule_risk"] = out["invalid_or_failed"] | out["bond_outlier"] | out["internal_clash"]
    out["relaxation_risk"] = out["high_energy"] | out["high_relax_drop"] | out["high_relax_rmsd"]
    out["bad"] = (out["hard_rule_risk"] | out["relaxation_risk"]).astype(int)
    return out, thresholds


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    heavy = out["heavy_atoms"].astype(float).replace(0.0, np.nan)
    out["ff_energy_initial_per_heavy"] = out["ff_energy_initial"].astype(float) / heavy
    out["ff_energy_drop_per_heavy"] = out["ff_energy_drop"].astype(float) / heavy
    out["bond_ratio_span"] = out["max_bond_ratio"].astype(float) - out["min_bond_ratio"].astype(float)
    out["low_bond_margin"] = 0.65 - out["min_bond_ratio"].astype(float)
    out["high_bond_margin"] = out["max_bond_ratio"].astype(float) - 1.45
    out["qed_risk"] = 1.0 - out["qed"].astype(float).clip(0.0, 1.0)
    out["cheap_rule_score"] = (
        out["low_bond_margin"].clip(lower=0.0)
        + out["high_bond_margin"].clip(lower=0.0)
        + np.log1p(out["internal_clash_count"].fillna(0).astype(float))
    )
    out["relaxation_rule_score"] = (
        out["ff_energy_per_heavy"].rank(pct=True)
        + out["ff_energy_drop"].rank(pct=True)
        + out["ff_relax_rmsd"].rank(pct=True)
    ) / 3.0
    return out


def feature_sets() -> dict[str, list[str]]:
    descriptor = ["atoms", "heavy_atoms", "formal_charge_abs", "qed", "qed_risk"]
    cheap_geometry = descriptor + [
        "min_bond_ratio",
        "max_bond_ratio",
        "bond_ratio_span",
        "low_bond_margin",
        "high_bond_margin",
        "internal_clash_count",
    ]
    pre_relax_ff = cheap_geometry + ["ff_energy_initial_per_heavy", "ff_energy_initial"]
    full_audit_upper = pre_relax_ff + [
        "bond_outlier_count",
        "ff_energy_per_heavy",
        "ff_energy_drop",
        "ff_energy_drop_per_heavy",
        "ff_relax_rmsd",
    ]
    return {
        "descriptor_only": descriptor,
        "cheap_geometry": cheap_geometry,
        "pre_relax_ff": pre_relax_ff,
        "full_audit_upper": full_audit_upper,
    }


def make_model(name: str, seed: int):
    if name == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=32,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "hgb":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        l2_regularization=0.01,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {name}")


def group_key(frame: pd.DataFrame) -> pd.Series:
    gen = frame["generator"].astype(str)
    rel = frame["relative_file"].astype(str)
    return gen + "::" + rel.map(lambda x: str(Path(x).parent if "/" in x else x))


def assign_split(groups: pd.Series, seed: int) -> pd.Series:
    unit = groups.map(lambda x: stable_unit(str(x), seed))
    split = pd.Series("test", index=groups.index, dtype=object)
    split.loc[unit < 0.5] = "train"
    split.loc[(unit >= 0.5) & (unit < 0.7)] = "cal"
    return split


def predict_scores(model, train: pd.DataFrame, eval_frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    model.fit(train[features], train["bad"])
    if hasattr(model, "predict_proba"):
        return model.predict_proba(eval_frame[features])[:, 1]
    return model.decision_function(eval_frame[features])


def score_baseline(frame: pd.DataFrame, score_col: str) -> np.ndarray:
    return frame[score_col].astype(float).to_numpy()


def evaluate_scores(
    protocol: str,
    generator: str,
    q: float,
    score_name: str,
    model_name: str,
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    test_score: np.ndarray,
    cal_score: np.ndarray,
    selection_rows: list[dict[str, object]],
    detection_rows: list[dict[str, object]],
) -> None:
    metrics = binary_metrics(test["bad"].to_numpy(), test_score)
    detection_rows.append(
        {
            "protocol": protocol,
            "generator": generator,
            "quantile": q,
            "score_name": score_name,
            "model": model_name,
            "n_train": len(train),
            "n_cal": len(cal),
            "n_test": len(test),
            "train_bad_rate": float(train["bad"].mean()) if len(train) else math.nan,
            "cal_bad_rate": float(cal["bad"].mean()) if len(cal) else math.nan,
            "test_bad_rate": float(test["bad"].mean()) if len(test) else math.nan,
            **metrics,
        }
    )
    for bound in ["empirical", "wilson95"]:
        for alpha in [0.05, 0.10, 0.20]:
            tau, controlled = conformal_threshold(cal_score, cal["bad"].to_numpy(), alpha=alpha, bound=bound)
            accepted = test_score <= tau
            if accepted.any():
                accepted_bad = float(test.loc[accepted, "bad"].mean())
                accepted_reliable = float(1.0 - accepted_bad)
            else:
                accepted_bad = math.nan
                accepted_reliable = math.nan
            coverage = float(accepted.mean())
            selection_rows.append(
                {
                    "protocol": protocol,
                    "generator": generator,
                    "quantile": q,
                    "score_name": score_name,
                    "model": model_name,
                    "risk_bound": bound,
                    "alpha": alpha,
                    "threshold": tau,
                    "calibration_controlled_bad_rate_at_tau": controlled,
                    "n_cal": len(cal),
                    "n_test": len(test),
                    "cal_bad_rate": float(cal["bad"].mean()) if len(cal) else math.nan,
                    "test_bad_rate": float(test["bad"].mean()) if len(test) else math.nan,
                    "coverage": coverage,
                    "accepted_bad_rate": accepted_bad,
                    "accepted_reliable_rate": accepted_reliable,
                    "resampling_multiplier": float(1.0 / max(coverage, 1e-12)),
                }
            )


def run_model_grid(
    protocol: str,
    generator: str,
    q: float,
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    detection_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
    seed: int,
    model_names: list[str],
    feature_set_names: list[str],
) -> None:
    if len(test) == 0 or len(cal) == 0 or test["bad"].nunique() < 2 or cal["bad"].nunique() < 2:
        return

    for score_col in ["qed_risk", "cheap_rule_score", "relaxation_rule_score"]:
        if score_col not in test.columns or test[score_col].nunique(dropna=True) <= 1:
            continue
        evaluate_scores(
            protocol=protocol,
            generator=generator,
            q=q,
            score_name=score_col,
            model_name="baseline",
            train=train,
            cal=cal,
            test=test,
            test_score=score_baseline(test, score_col),
            cal_score=score_baseline(cal, score_col),
            selection_rows=selection_rows,
            detection_rows=detection_rows,
        )

    all_feature_sets = feature_sets()
    for feature_name in feature_set_names:
        features = all_feature_sets[feature_name]
        available = [c for c in features if c in train.columns]
        if len(available) != len(features):
            continue
        for model_name in model_names:
            if train["bad"].nunique() < 2:
                continue
            model = make_model(model_name, seed=seed)
            train_fit = train.copy()
            cal_eval = cal.copy()
            test_eval = test.copy()
            model.fit(train_fit[features], train_fit["bad"])
            cal_score = model.predict_proba(cal_eval[features])[:, 1]
            test_score = model.predict_proba(test_eval[features])[:, 1]
            evaluate_scores(
                protocol=protocol,
                generator=generator,
                q=q,
                score_name=feature_name,
                model_name=model_name,
                train=train,
                cal=cal,
                test=test,
                test_score=test_score,
                cal_score=cal_score,
                selection_rows=selection_rows,
                detection_rows=detection_rows,
            )


def run_protocols(
    frame: pd.DataFrame,
    q: float,
    seed: int,
    model_names: list[str],
    feature_set_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detection_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    main = frame[frame["generator"].isin(MAIN_GENERATORS)].copy()
    groups = group_key(main)
    main["split"] = assign_split(groups, seed)
    run_model_grid(
        protocol="pooled_group_split",
        generator="pooled_main",
        q=q,
        train=main[main["split"] == "train"].copy(),
        cal=main[main["split"] == "cal"].copy(),
        test=main[main["split"] == "test"].copy(),
        detection_rows=detection_rows,
        selection_rows=selection_rows,
        seed=seed,
        model_names=model_names,
        feature_set_names=feature_set_names,
    )

    for generator, sub in main.groupby("generator"):
        sub = sub.copy()
        sub["split"] = assign_split(group_key(sub), seed)
        run_model_grid(
            protocol="within_generator_group_split",
            generator=generator,
            q=q,
            train=sub[sub["split"] == "train"].copy(),
            cal=sub[sub["split"] == "cal"].copy(),
            test=sub[sub["split"] == "test"].copy(),
            detection_rows=detection_rows,
            selection_rows=selection_rows,
            seed=seed,
            model_names=model_names,
            feature_set_names=feature_set_names,
        )

    for heldout in MAIN_GENERATORS:
        source = main[main["generator"] != heldout].copy()
        target = main[main["generator"] == heldout].copy()
        target_groups = group_key(target)
        unit = target_groups.map(lambda x: stable_unit(str(x), seed + 1000))
        cal = target[unit < 0.5].copy()
        test = target[unit >= 0.5].copy()
        run_model_grid(
            protocol="leave_one_generator_out_target_cal",
            generator=heldout,
            q=q,
            train=source,
            cal=cal,
            test=test,
            detection_rows=detection_rows,
            selection_rows=selection_rows,
            seed=seed,
            model_names=model_names,
            feature_set_names=feature_set_names,
        )

    return pd.DataFrame(detection_rows), pd.DataFrame(selection_rows)


def csv_block(df: pd.DataFrame) -> str:
    return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def write_summary(out_dir: Path, detection: pd.DataFrame, selection: pd.DataFrame) -> None:
    top_detection = detection.sort_values(["protocol", "generator", "roc_auc"], ascending=[True, True, False])
    top_detection = top_detection.groupby(["protocol", "generator"], as_index=False).head(5)
    valid_wilson = selection[
        (selection["risk_bound"] == "wilson95")
        & (selection["alpha"] == 0.05)
        & (selection["accepted_bad_rate"] <= selection["alpha"])
    ].copy()
    valid_wilson = valid_wilson.sort_values(["protocol", "generator", "coverage"], ascending=[True, True, False])
    valid_wilson = valid_wilson.groupby(["protocol", "generator"], as_index=False).head(5)
    lines = [
        "# Physics Risk Critic v0",
        "",
        "This is a table-only critic experiment over RDKit-audited generated SDF molecules.",
        "Feature sets are intentionally separated into low-leakage descriptors, cheap geometry, pre-relaxation force-field features, and a full-audit upper bound.",
        "",
        "## Top Detection Rows",
        csv_block(
            top_detection[
                [
                    "protocol",
                    "generator",
                    "score_name",
                    "model",
                    "n_train",
                    "n_cal",
                    "n_test",
                    "test_bad_rate",
                    "roc_auc",
                    "pr_auc",
                    "ece",
                    "mcc",
                ]
            ]
        ),
        "",
        "## Wilson95 Alpha=0.05 Rows Meeting Test Alpha",
        csv_block(
            valid_wilson[
                [
                    "protocol",
                    "generator",
                    "score_name",
                    "model",
                    "coverage",
                    "accepted_bad_rate",
                    "accepted_reliable_rate",
                    "resampling_multiplier",
                    "calibration_controlled_bad_rate_at_tau",
                ]
            ]
        )
        if len(valid_wilson)
        else "No Wilson95 alpha=0.05 row met test alpha.",
    ]
    (out_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics_risk_critic_v0/reports"))
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--models", nargs="+", default=["logreg", "rf", "hgb"], choices=["logreg", "rf", "hgb"])
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(feature_sets().keys()),
        choices=list(feature_sets().keys()),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.audit_csv)
    frame = frame[frame["generator"].isin(MAIN_GENERATORS)].copy()
    frame, thresholds = add_physics_labels(frame, args.quantile)
    frame = add_derived_features(frame)

    detection, selection = run_protocols(
        frame,
        q=args.quantile,
        seed=args.seed,
        model_names=args.models,
        feature_set_names=args.feature_sets,
    )
    detection.to_csv(args.output_dir / "critic_detection_metrics.csv", index=False)
    selection.to_csv(args.output_dir / "critic_wilson_selection.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "audit_csv": str(args.audit_csv),
                "quantile": args.quantile,
                "seed": args.seed,
                "thresholds": thresholds,
                "generators": MAIN_GENERATORS,
                "feature_sets": feature_sets(),
                "models": args.models,
                "selected_feature_sets": args.feature_sets,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir, detection, selection)
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
