from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from physcaldiff.metrics import binary_metrics
from train_physics_risk_critic import (
    MAIN_GENERATORS,
    add_derived_features,
    add_physics_labels,
    assign_split,
    conformal_threshold,
    feature_sets,
    group_key,
    make_model,
)


def stable_unit(value: str, seed: int) -> float:
    text = f"{seed}|{value}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:12], 16) / float(16**12 - 1)


def target_group_key(frame: pd.DataFrame) -> pd.Series:
    return frame["generator"].astype(str) + "::" + frame["relative_file"].astype(str)


def split_target(frame: pd.DataFrame, seed: int) -> pd.Series:
    unit = target_group_key(frame).map(lambda x: stable_unit(str(x), seed))
    return pd.Series(np.where(unit < 0.5, "cal", "test"), index=frame.index)


def prepare_source(path: Path, q: float, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = pd.read_csv(path)
    frame = frame[frame["generator"].isin(MAIN_GENERATORS)].copy()
    frame, thresholds = add_physics_labels(frame, q)
    frame = add_derived_features(frame)
    frame["split"] = assign_split(group_key(frame), seed)
    return frame, thresholds


def prepare_target(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = add_derived_features(frame)
    frame["bad"] = frame["rdkit_physics_high_risk"].astype(bool).astype(int)
    return frame


def score_baselines(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    scores = {
        "qed_risk": frame["qed_risk"].astype(float).to_numpy(),
        "relaxation_rule_score": frame["relaxation_rule_score"].astype(float).to_numpy(),
    }
    return scores


def fit_models(source_train: pd.DataFrame, seed: int) -> dict[str, object]:
    features = feature_sets()["pre_relax_ff"]
    models = {}
    for model_name in ["logreg", "rf", "hgb"]:
        model = make_model(model_name, seed=seed)
        model.fit(source_train[features], source_train["bad"])
        models[f"pre_relax_ff_{model_name}"] = model
    return models


def model_scores(models: dict[str, object], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    features = feature_sets()["pre_relax_ff"]
    return {name: model.predict_proba(frame[features])[:, 1] for name, model in models.items()}


def finite_score_mask(score: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(score, dtype=float))


def selection_stats(frame: pd.DataFrame, accepted: np.ndarray) -> dict[str, float]:
    row = {
        "n": len(frame),
        "n_accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()) if len(frame) else math.nan,
    }
    if accepted.any():
        selected = frame.loc[accepted]
        row.update(
            {
                "accepted_bad_rate": float(selected["bad"].mean()),
                "accepted_reliable_rate": float(1.0 - selected["bad"].mean()),
                "accepted_qed_mean": float(selected["qed"].mean()),
                "accepted_energy_per_heavy_mean": float(selected["ff_energy_per_heavy"].mean()),
                "accepted_relax_rmsd_mean": float(selected["ff_relax_rmsd"].mean()),
            }
        )
    else:
        row.update(
            {
                "accepted_bad_rate": math.nan,
                "accepted_reliable_rate": math.nan,
                "accepted_qed_mean": math.nan,
                "accepted_energy_per_heavy_mean": math.nan,
                "accepted_relax_rmsd_mean": math.nan,
            }
        )
    row["resampling_multiplier"] = float(1.0 / max(row["coverage"], 1e-12))
    return row


def evaluate_detection(target: pd.DataFrame, score_map: dict[str, np.ndarray], dataset_name: str) -> pd.DataFrame:
    rows = []
    for generator, sub in target.groupby("generator"):
        idx = sub.index.to_numpy()
        for score_name, score in score_map.items():
            sub_score = score[target.index.get_indexer(idx)]
            valid = finite_score_mask(sub_score)
            sub_eval = sub.loc[valid]
            sub_score = sub_score[valid]
            if len(sub_eval) == 0 or sub_eval["bad"].nunique() < 2:
                continue
            metrics = binary_metrics(sub_eval["bad"].to_numpy(), sub_score)
            rows.append(
                {
                    "dataset": dataset_name,
                    "generator": generator,
                    "score_name": score_name,
                    "n": len(sub_eval),
                    "n_total": len(sub),
                    "bad_rate": float(sub_eval["bad"].mean()),
                    **metrics,
                }
            )
    for score_name, score in score_map.items():
        valid = finite_score_mask(score)
        target_eval = target.loc[valid]
        score_eval = score[valid]
        if len(target_eval) == 0 or target_eval["bad"].nunique() < 2:
            continue
        metrics = binary_metrics(target_eval["bad"].to_numpy(), score_eval)
        rows.append(
            {
                "dataset": dataset_name,
                "generator": "pooled",
                "score_name": score_name,
                "n": len(target_eval),
                "n_total": len(target),
                "bad_rate": float(target_eval["bad"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_selection(
    source_cal: pd.DataFrame,
    source_score_map: dict[str, np.ndarray],
    target: pd.DataFrame,
    target_score_map: dict[str, np.ndarray],
    dataset_name: str,
    seed: int,
) -> pd.DataFrame:
    rows = []
    target = target.copy()
    target["target_split"] = split_target(target, seed + 2000)
    alphas = [0.05, 0.10, 0.20]

    for generator, sub_all in target.groupby("generator"):
        sub_all = sub_all.copy()
        sub_idx = target.index.get_indexer(sub_all.index)
        for score_name, target_score_all in target_score_map.items():
            score_sub_all = target_score_all[sub_idx]
            valid_target = finite_score_mask(score_sub_all)
            sub_eval = sub_all.loc[valid_target].copy()
            score_sub_eval = score_sub_all[valid_target]
            source_score_all = source_score_map[score_name]
            valid_source = finite_score_mask(source_score_all)
            source_eval = source_cal.loc[valid_source].copy()
            source_score = source_score_all[valid_source]
            if len(sub_eval) == 0 or len(source_eval) == 0:
                continue
            for alpha in alphas:
                tau, controlled = conformal_threshold(source_score, source_eval["bad"].to_numpy(), alpha, "wilson95")
                accepted = score_sub_eval <= tau
                row = {
                    "dataset": dataset_name,
                    "generator": generator,
                    "score_name": score_name,
                    "calibration": "source_wilson95",
                    "alpha": alpha,
                    "threshold": tau,
                    "calibration_controlled_bad_rate_at_tau": controlled,
                    "target_bad_rate": float(sub_eval["bad"].mean()),
                    "n_total": len(sub_all),
                }
                row.update(selection_stats(sub_eval, accepted))
                rows.append(row)

            cal = sub_eval[sub_eval["target_split"] == "cal"].copy()
            test = sub_eval[sub_eval["target_split"] == "test"].copy()
            if len(cal) == 0 or len(test) == 0 or cal["bad"].nunique() < 2:
                continue
            cal_pos = sub_eval.index.get_indexer(cal.index)
            test_pos = sub_eval.index.get_indexer(test.index)
            cal_score = score_sub_eval[cal_pos]
            test_score = score_sub_eval[test_pos]
            for alpha in alphas:
                tau, controlled = conformal_threshold(cal_score, cal["bad"].to_numpy(), alpha, "wilson95")
                accepted = test_score <= tau
                row = {
                    "dataset": dataset_name,
                    "generator": generator,
                    "score_name": score_name,
                    "calibration": "target_wilson95",
                    "alpha": alpha,
                    "threshold": tau,
                    "calibration_controlled_bad_rate_at_tau": controlled,
                    "target_bad_rate": float(test["bad"].mean()),
                    "n_total": len(sub_all),
                }
                row.update(selection_stats(test, accepted))
                rows.append(row)

    return pd.DataFrame(rows)


def simulate_resampling(target: pd.DataFrame, score_map: dict[str, np.ndarray], selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = target.copy()
    target["target_file"] = target_group_key(target)
    for _, sel in selection.iterrows():
        if sel["calibration"] != "source_wilson95" or float(sel["alpha"]) not in {0.1, 0.2}:
            continue
        score_name = str(sel["score_name"])
        generator = str(sel["generator"])
        threshold = float(sel["threshold"])
        sub = target[target["generator"] == generator].copy()
        if len(sub) == 0:
            continue
        score = score_map[score_name][target.index.get_indexer(sub.index)]
        valid = finite_score_mask(score)
        sub = sub.loc[valid].copy()
        score = score[valid]
        if len(sub) == 0:
            continue
        sub["_score"] = score
        accepted_parts = []
        target_rows = []
        for file_id, group in sub.sort_values(["relative_file", "mol_idx"]).groupby("target_file"):
            scanned = len(group)
            accepted = group[group["_score"] <= threshold].head(4)
            accepted_parts.append(accepted)
            target_rows.append(
                {
                    "file_id": file_id,
                    "scanned": scanned,
                    "accepted": len(accepted),
                    "success_at_1": int(len(accepted) >= 1),
                    "success_at_4": int(len(accepted) >= 4),
                }
            )
        accepted_all = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame()
        target_table = pd.DataFrame(target_rows)
        row = {
            "dataset": sel["dataset"],
            "generator": generator,
            "score_name": score_name,
            "calibration": sel["calibration"],
            "alpha": float(sel["alpha"]),
            "targets": int(len(target_table)),
            "molecules_scanned": int(target_table["scanned"].sum()),
            "accepted": int(target_table["accepted"].sum()),
            "target_success_at_1": float(target_table["success_at_1"].mean()),
            "target_success_at_4": float(target_table["success_at_4"].mean()),
            "mean_accepted_per_target": float(target_table["accepted"].mean()),
        }
        if len(accepted_all):
            row.update(
                {
                    "accepted_bad_rate": float(accepted_all["bad"].mean()),
                    "accepted_qed_mean": float(accepted_all["qed"].mean()),
                    "accepted_energy_per_heavy_mean": float(accepted_all["ff_energy_per_heavy"].mean()),
                    "accepted_relax_rmsd_mean": float(accepted_all["ff_relax_rmsd"].mean()),
                    "attempts_per_accepted": float(target_table["scanned"].sum() / max(len(accepted_all), 1)),
                }
            )
        else:
            row.update(
                {
                    "accepted_bad_rate": math.nan,
                    "accepted_qed_mean": math.nan,
                    "accepted_energy_per_heavy_mean": math.nan,
                    "accepted_relax_rmsd_mean": math.nan,
                    "attempts_per_accepted": math.inf,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


def process_target(
    target_path: Path,
    dataset_name: str,
    source_train: pd.DataFrame,
    source_cal: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = prepare_target(target_path)
    models = fit_models(source_train, seed)
    target_scores = score_baselines(target)
    target_scores.update(model_scores(models, target))
    source_scores = score_baselines(source_cal)
    source_scores.update(model_scores(models, source_cal))

    detection = evaluate_detection(target, target_scores, dataset_name)
    selection = evaluate_selection(source_cal, source_scores, target, target_scores, dataset_name, seed)
    resampling = simulate_resampling(target, target_scores, selection)
    return detection, selection, resampling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=Path("outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv"),
    )
    parser.add_argument("--target", action="append", required=True, help="name=/path/to/molecule_physics_audit.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics_critic_transfer_v0/reports"))
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, thresholds = prepare_source(args.source_audit, args.quantile, args.seed)
    source_train = source[source["split"] == "train"].copy()
    source_cal = source[source["split"] == "cal"].copy()

    detections = []
    selections = []
    resamplings = []
    target_specs = {}
    for spec in args.target:
        name, path = spec.split("=", 1)
        target_specs[name] = path
        detection, selection, resampling = process_target(Path(path), name, source_train, source_cal, args.seed)
        detections.append(detection)
        selections.append(selection)
        resamplings.append(resampling)

    detection_table = pd.concat(detections, ignore_index=True)
    selection_table = pd.concat(selections, ignore_index=True)
    resampling_table = pd.concat(resamplings, ignore_index=True)

    detection_table.to_csv(args.output_dir / "transfer_detection_metrics.csv", index=False)
    selection_table.to_csv(args.output_dir / "transfer_wilson_selection.csv", index=False)
    resampling_table.to_csv(args.output_dir / "transfer_resampling_simulation.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "source_audit": str(args.source_audit),
                "targets": target_specs,
                "quantile": args.quantile,
                "source_thresholds": thresholds,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    top_detection = (
        detection_table.sort_values(["dataset", "generator", "roc_auc"], ascending=[True, True, False])
        .groupby(["dataset", "generator"], as_index=False)
        .head(4)
    )
    valid_selection = selection_table[
        (selection_table["calibration"] == "target_wilson95")
        & (selection_table["accepted_bad_rate"] <= selection_table["alpha"])
    ].copy()
    valid_selection = (
        valid_selection.sort_values(["dataset", "generator", "alpha", "coverage"], ascending=[True, True, True, False])
        .groupby(["dataset", "generator", "alpha"], as_index=False)
        .head(4)
    )

    lines = [
        "# Physics Critic Transfer v0",
        "",
        "Source critic is trained on the main DiffSBDD/MolCraft/PocketFlow audit. Targets are held out sampling-side audit pools.",
        "",
        "## Top Detection Rows",
        csv_block(top_detection),
        "",
        "## Target-Calibrated Wilson Rows Meeting Test Alpha",
        csv_block(valid_selection) if len(valid_selection) else "No target-calibrated row met test alpha.",
        "",
        "## Source-Calibrated Resampling Simulation",
        csv_block(resampling_table.sort_values(["dataset", "generator", "alpha", "score_name"])),
    ]
    (args.output_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
