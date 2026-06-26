from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def csv_block(df: pd.DataFrame) -> str:
    return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--old-root", type=Path, default=Path("external/reference_baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evidence_tables_v0"))
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    conformal_root = args.repo_root / "outputs/crossdocked_ensemble_conformal_v1_wilson/reports"
    metrics = read_csv(conformal_root / "bad_sample_detection_metrics.csv")
    conformal = read_csv(conformal_root / "conformal_selection.csv")
    risk_cov = read_csv(conformal_root / "risk_coverage.csv")

    detection_keep = [
        ("score_hgb_reliamol_novinascore", 0.0),
        ("score_hgb_reliamol_novinascore", 0.5),
        ("score_rf_reliamol_novinascore", 0.0),
        ("score_mlp_reliamol_novinascore", 0.0),
        ("score_mlp_reliamol_novinascore", 0.5),
        ("score_qed_sa", 0.0),
        ("score_qed", 0.0),
    ]
    rows = []
    for score, lam in detection_keep:
        hit = metrics[(metrics["score_col"] == score) & (metrics["lambda_uncertainty"] == lam)]
        if len(hit):
            rows.append(hit.iloc[0])
    table_detection = pd.DataFrame(rows)[
        ["score_col", "lambda_uncertainty", "roc_auc", "pr_auc", "brier", "ece", "f1", "mcc"]
    ]
    table_detection.to_csv(out / "table_crossdocked_bad_sample_detection.csv", index=False)

    table_conformal = (
        conformal[
            (conformal["risk_bound"] == "wilson95")
            & (conformal["alpha"] == 0.05)
            & (conformal["accepted_bad_rate"] <= conformal["alpha"])
        ]
        .sort_values("coverage", ascending=False)
        .head(12)[
            [
                "score_col",
                "lambda_uncertainty",
                "coverage",
                "accepted_bad_rate",
                "accepted_reliable_rate",
                "resampling_multiplier",
                "calibration_controlled_bad_rate_at_tau",
            ]
        ]
    )
    table_conformal.to_csv(out / "table_wilson05_selective_generation.csv", index=False)

    table_risk_coverage = (
        risk_cov[
            risk_cov["score_col"].isin(
                [
                    "score_hgb_reliamol_novinascore",
                    "score_rf_reliamol_novinascore",
                    "score_mlp_reliamol_novinascore",
                    "score_qed_sa",
                ]
            )
            & risk_cov["lambda_uncertainty"].isin([0.0, 0.5])
            & risk_cov["target_coverage"].isin([0.5, 0.7, 0.9])
        ][
            [
                "score_col",
                "lambda_uncertainty",
                "target_coverage",
                "coverage",
                "accepted_bad_rate",
                "accepted_reliable_rate",
                "resampling_multiplier",
            ]
        ]
        .sort_values(["target_coverage", "accepted_bad_rate", "coverage"], ascending=[True, True, False])
    )
    table_risk_coverage.to_csv(out / "table_risk_coverage.csv", index=False)

    dude_metrics = read_csv(args.old_root / "outputs/dude_public_reliability_v2_multiseed/lot_metrics_summary.csv")
    table_dude = dude_metrics[
        dude_metrics["method"].isin(
            [
                "vina_rank",
                "qed_sa_vina",
                "mlp_pose_shape",
                "mlp_reliamol_noleak",
                "rf_noleak",
                "hgb_noleak",
            ]
        )
    ][["method", "roc_auc.mean", "roc_auc.std", "pr_auc.mean", "pr_auc.std", "ece.mean", "ece.std"]]
    table_dude.to_csv(out / "table_external_dude_metrics.csv", index=False)

    posebusters = read_csv(
        args.old_root / "outputs/diffsbdd_crossdocked_posebusters_molfast_v0/reports/posebusters_metric_summary.csv"
    )
    table_posebusters = posebusters[
        posebusters["method"].isin(
            [
                "qed",
                "qed_sa",
                "mlp_pose_shape",
                "mlp_reliamol_novinascore",
                "rf_reliamol_novinascore",
                "hgb_reliamol_novinascore",
            ]
        )
    ][["method", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "ece_mean", "ece_std"]]
    table_posebusters.to_csv(out / "table_posebusters_metrics.csv", index=False)

    ablation = read_csv(args.old_root / "outputs/diffsbdd_crossdocked_feature_ablation_v0/reports/ablation_metric_summary.csv")
    table_ablation = ablation[
        ablation["method"].isin(
            [
                "qed_sa",
                "rf_ligand_descriptor",
                "rf_internal_geometry",
                "rf_pocket_geometry",
                "rf_geometry_only",
                "rf_ligand_pocket",
                "rf_full_reliamol_novinascore",
            ]
        )
    ][["method", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "ece_mean", "ece_std"]]
    table_ablation.to_csv(out / "table_feature_ablation.csv", index=False)

    sdf_sensitivity = read_csv(args.repo_root / "outputs/sdf_physics_sensitivity_v0/reports/physics_label_sensitivity.csv")
    table_sdf_sensitivity = sdf_sensitivity[
        sdf_sensitivity["generator"].isin(["diffsbdd", "molcraft", "pocketflow"])
    ][
        [
            "quantile",
            "generator",
            "n",
            "physics_high_risk_rate",
            "hard_rule_risk_rate",
            "quantile_energy_risk_rate",
            "delta_vs_pocketflow",
            "multiplier_vs_pocketflow",
        ]
    ].sort_values(["quantile", "physics_high_risk_rate"], ascending=[True, False])
    table_sdf_sensitivity.to_csv(out / "table_sdf_physics_label_sensitivity.csv", index=False)

    sdf_components = read_csv(args.repo_root / "outputs/sdf_physics_sensitivity_v0/reports/physics_component_rates.csv")
    table_sdf_components = sdf_components[
        (sdf_components["quantile"] == 0.9)
        & sdf_components["generator"].isin(["diffsbdd", "molcraft", "pocketflow"])
    ][
        [
            "generator",
            "n",
            "invalid_or_failed_rate",
            "bond_outlier_rate",
            "internal_clash_rate",
            "high_energy_rate",
            "high_relax_drop_rate",
            "high_relax_rmsd_rate",
        ]
    ].sort_values("generator")
    table_sdf_components.to_csv(out / "table_sdf_physics_component_q90.csv", index=False)

    sdf_alignment = read_csv(args.repo_root / "outputs/sdf_risk_alignment_v0/reports/sdf_risk_detection_metrics.csv")
    table_sdf_alignment = sdf_alignment[
        sdf_alignment["generator"].isin(["diffsbdd", "molcraft", "pocketflow", "pooled_without_molpilot"])
        & sdf_alignment["score_col"].isin(["risk_prob", "qed_risk"])
    ][
        [
            "generator",
            "score_col",
            "n",
            "bad_rate",
            "roc_auc",
            "pr_auc",
            "brier",
            "ece",
            "f1",
            "mcc",
        ]
    ].sort_values(["generator", "score_col"])
    table_sdf_alignment.to_csv(out / "table_sdf_risk_alignment_detection.csv", index=False)

    critic_q90 = read_csv(args.repo_root / "outputs/physics_risk_critic_v0/reports/critic_detection_metrics.csv")
    critic_keep_scores = ["qed_risk", "relaxation_rule_score", "pre_relax_ff", "full_audit_upper"]
    critic_detection = critic_q90[
        critic_q90["score_name"].isin(critic_keep_scores)
        & critic_q90["protocol"].isin(["pooled_group_split", "leave_one_generator_out_target_cal"])
    ].copy()
    critic_detection = (
        critic_detection.sort_values(["protocol", "generator", "score_name", "roc_auc"], ascending=[True, True, True, False])
        .groupby(["protocol", "generator", "score_name"], as_index=False)
        .head(1)
    )
    table_critic_detection = critic_detection[
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
    ].sort_values(["protocol", "generator", "roc_auc"], ascending=[True, True, False])
    table_critic_detection.to_csv(out / "table_physics_risk_critic_detection_q90.csv", index=False)

    critic_sel_q90 = read_csv(args.repo_root / "outputs/physics_risk_critic_v0/reports/critic_wilson_selection.csv")
    critic_sel = critic_sel_q90[
        (critic_sel_q90["risk_bound"] == "wilson95")
        & critic_sel_q90["score_name"].isin(critic_keep_scores)
        & critic_sel_q90["protocol"].isin(["pooled_group_split", "leave_one_generator_out_target_cal"])
        & (critic_sel_q90["accepted_bad_rate"] <= critic_sel_q90["alpha"])
    ].copy()
    critic_sel = (
        critic_sel.sort_values(["protocol", "generator", "alpha", "score_name", "coverage"], ascending=[True, True, True, True, False])
        .groupby(["protocol", "generator", "alpha", "score_name"], as_index=False)
        .head(1)
    )
    table_critic_sel = critic_sel[
        [
            "protocol",
            "generator",
            "alpha",
            "score_name",
            "model",
            "coverage",
            "accepted_bad_rate",
            "test_bad_rate",
            "resampling_multiplier",
        ]
    ].sort_values(["protocol", "generator", "alpha", "coverage"], ascending=[True, True, True, False])
    table_critic_sel.to_csv(out / "table_physics_risk_critic_wilson_q90.csv", index=False)

    sensitivity_rows = []
    for quantile, rel_path in [
        (0.80, "outputs/physics_risk_critic_q80_v0/reports/critic_detection_metrics.csv"),
        (0.90, "outputs/physics_risk_critic_v0/reports/critic_detection_metrics.csv"),
        (0.95, "outputs/physics_risk_critic_q95_v0/reports/critic_detection_metrics.csv"),
    ]:
        det = read_csv(args.repo_root / rel_path)
        det = det[
            (det["protocol"] == "pooled_group_split")
            & (det["generator"] == "pooled_main")
            & det["score_name"].isin(["relaxation_rule_score", "pre_relax_ff", "full_audit_upper"])
        ].copy()
        det = (
            det.sort_values(["score_name", "roc_auc"], ascending=[True, False])
            .groupby("score_name", as_index=False)
            .head(1)
        )
        det["quantile"] = quantile
        sensitivity_rows.append(det)
    table_critic_sensitivity = pd.concat(sensitivity_rows, ignore_index=True)[
        ["quantile", "score_name", "model", "test_bad_rate", "roc_auc", "pr_auc", "ece", "mcc"]
    ].sort_values(["quantile", "roc_auc"], ascending=[True, False])
    table_critic_sensitivity.to_csv(out / "table_physics_risk_critic_quantile_sensitivity.csv", index=False)

    quality_tradeoff = read_csv(args.repo_root / "outputs/selective_quality_tradeoff_v0/reports/selective_quality_tradeoff.csv")
    table_quality_tradeoff = quality_tradeoff[
        quality_tradeoff["score_name"].isin(["raw_unfiltered", "qed_risk", "relaxation_rule_score", "pre_relax_ff", "full_audit_upper"])
    ][
        [
            "alpha",
            "score_name",
            "model",
            "coverage",
            "accepted_bad_rate",
            "risk_reduction_vs_raw",
            "qed_mean",
            "delta_qed_mean_vs_raw",
            "ff_energy_per_heavy_mean",
            "delta_ff_energy_per_heavy_mean_vs_raw",
            "ff_relax_rmsd_mean",
            "delta_ff_relax_rmsd_mean_vs_raw",
            "diffsbdd_share",
            "molcraft_share",
            "pocketflow_share",
        ]
    ].sort_values(["alpha", "accepted_bad_rate", "coverage"], ascending=[True, True, False])
    table_quality_tradeoff.to_csv(out / "table_selective_quality_tradeoff_q90.csv", index=False)

    sampling_audit = read_csv(args.repo_root / "outputs/sampling_guidance_sdf_audit_v0/reports/generator_physics_summary.csv")
    table_sampling_audit = sampling_audit[
        [
            "generator",
            "molecules",
            "files",
            "sanitize_rate",
            "has3d_rate",
            "ff_ok_rate",
            "high_risk_rate",
            "bond_outlier_rate",
            "internal_clash_rate",
            "high_energy_q90_rate",
            "relax_drop_q90_rate",
            "relax_rmsd_q90_rate",
            "qed_mean",
            "energy_per_heavy_mean",
        ]
    ].sort_values("high_risk_rate")
    table_sampling_audit.to_csv(out / "table_sampling_guidance_sdf_audit.csv", index=False)

    matched_guidance = read_csv(args.repo_root / "outputs/syncguide_origin_matched_audit_v0/reports/generator_physics_summary.csv")
    table_matched_guidance = matched_guidance[
        [
            "generator",
            "molecules",
            "files",
            "sanitize_rate",
            "has3d_rate",
            "ff_ok_rate",
            "high_risk_rate",
            "bond_outlier_rate",
            "internal_clash_rate",
            "high_energy_q90_rate",
            "relax_drop_q90_rate",
            "relax_rmsd_q90_rate",
            "qed_mean",
            "energy_per_heavy_mean",
        ]
    ].sort_values("generator")
    table_matched_guidance.to_csv(out / "table_syncguide_origin_matched_audit.csv", index=False)

    matched_guidance_50 = read_csv(args.repo_root / "outputs/syncguide_origin_matched_audit_50_v0/reports/generator_physics_summary.csv")
    table_matched_guidance_50 = matched_guidance_50[
        [
            "generator",
            "molecules",
            "files",
            "sanitize_rate",
            "has3d_rate",
            "ff_ok_rate",
            "high_risk_rate",
            "bond_outlier_rate",
            "internal_clash_rate",
            "high_energy_q90_rate",
            "relax_drop_q90_rate",
            "relax_rmsd_q90_rate",
            "qed_mean",
            "energy_per_heavy_mean",
        ]
    ].sort_values("generator")
    table_matched_guidance_50.to_csv(out / "table_syncguide_origin_matched50_audit.csv", index=False)

    matched_pairwise_50 = read_csv(args.repo_root / "outputs/syncguide_origin_matched_audit_50_v0/reports/pairwise/paired_statistics.csv")
    table_matched_pairwise_50 = matched_pairwise_50[
        matched_pairwise_50["metric"].isin(
            [
                "high_risk_rate",
                "bond_outlier_rate",
                "relax_drop_q90_rate",
                "relax_rmsd_q90_rate",
                "qed_mean",
            ]
        )
    ][
        [
            "metric",
            "n_pairs",
            "origin_mean",
            "guide_mean",
            "mean_delta_guide_minus_origin",
            "bootstrap95_low",
            "bootstrap95_high",
            "guide_worse_pairs",
            "guide_better_pairs",
            "two_sided_sign_p",
        ]
    ]
    table_matched_pairwise_50.to_csv(out / "table_syncguide_origin_matched50_pairwise_stats.csv", index=False)

    transfer_root = args.repo_root / "outputs/physics_critic_transfer_v0/reports"
    transfer_detection = read_csv(transfer_root / "transfer_detection_metrics.csv")
    transfer_keep_scores = ["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb", "pre_relax_ff_logreg"]
    table_transfer_detection = transfer_detection[
        transfer_detection["score_name"].isin(transfer_keep_scores)
        & transfer_detection["generator"].isin(["pooled", "syncguide", "targetdiff", "pocket2mol", "origin", "guide"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "n",
            "n_total",
            "bad_rate",
            "roc_auc",
            "pr_auc",
            "brier",
            "ece",
            "mcc",
        ]
    ].sort_values(["dataset", "generator", "roc_auc"], ascending=[True, True, False])
    table_transfer_detection.to_csv(out / "table_physics_critic_transfer_detection.csv", index=False)

    transfer_selection = read_csv(transfer_root / "transfer_wilson_selection.csv")
    transfer_target_sel = transfer_selection[
        (transfer_selection["calibration"] == "target_wilson95")
        & transfer_selection["score_name"].isin(["relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
        & (transfer_selection["accepted_bad_rate"] <= transfer_selection["alpha"])
    ].copy()
    transfer_target_sel = (
        transfer_target_sel.sort_values(
            ["dataset", "generator", "alpha", "coverage"], ascending=[True, True, True, False]
        )
        .groupby(["dataset", "generator", "alpha"], as_index=False)
        .head(3)
    )
    table_transfer_target_sel = transfer_target_sel[
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "n_total",
            "n",
            "coverage",
            "target_bad_rate",
            "accepted_bad_rate",
            "accepted_qed_mean",
            "accepted_energy_per_heavy_mean",
            "accepted_relax_rmsd_mean",
            "resampling_multiplier",
        ]
    ].sort_values(["dataset", "generator", "alpha", "accepted_bad_rate", "coverage"], ascending=[True, True, True, True, False])
    table_transfer_target_sel.to_csv(out / "table_physics_critic_transfer_target_wilson.csv", index=False)

    transfer_resampling = read_csv(transfer_root / "transfer_resampling_simulation.csv")
    table_transfer_resampling = transfer_resampling[
        transfer_resampling["score_name"].isin(["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "targets",
            "molecules_scanned",
            "accepted",
            "target_success_at_4",
            "mean_accepted_per_target",
            "accepted_bad_rate",
            "accepted_qed_mean",
            "accepted_energy_per_heavy_mean",
            "accepted_relax_rmsd_mean",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "alpha", "accepted_bad_rate"], ascending=[True, True, True, True])
    table_transfer_resampling.to_csv(out / "table_physics_critic_transfer_source_resampling.csv", index=False)

    transfer50_root = args.repo_root / "outputs/physics_critic_transfer_matched50_v0/reports"
    transfer50_detection = read_csv(transfer50_root / "transfer_detection_metrics.csv")
    table_transfer50_detection = transfer50_detection[
        transfer50_detection["score_name"].isin(transfer_keep_scores)
        & transfer50_detection["generator"].isin(["pooled", "origin", "guide"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "n",
            "n_total",
            "bad_rate",
            "roc_auc",
            "pr_auc",
            "brier",
            "ece",
            "mcc",
        ]
    ].sort_values(["dataset", "generator", "roc_auc"], ascending=[True, True, False])
    table_transfer50_detection.to_csv(out / "table_physics_critic_transfer_matched50_detection.csv", index=False)

    transfer50_resampling = read_csv(transfer50_root / "transfer_resampling_simulation.csv")
    table_transfer50_resampling = transfer50_resampling[
        (transfer50_resampling["alpha"] == 0.10)
        & transfer50_resampling["score_name"].isin(["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "targets",
            "molecules_scanned",
            "accepted",
            "target_success_at_4",
            "accepted_bad_rate",
            "accepted_qed_mean",
            "accepted_energy_per_heavy_mean",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate"], ascending=[True, True, True])
    table_transfer50_resampling.to_csv(out / "table_physics_critic_transfer_matched50_source_wilson10.csv", index=False)

    policy_ablation = read_csv(args.repo_root / "outputs/resampling_policy_ablation_v0/reports/policy_ablation_summary.csv")
    table_policy_fixed_top4 = policy_ablation[
        (policy_ablation["k"] == 4)
        & (
            ((policy_ablation["policy"] == "raw_topk") & (policy_ablation["score_name"] == "none"))
            | (
                (policy_ablation["policy"] == "rank_topk")
                & policy_ablation["score_name"].isin(["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
            )
        )
    ][
        [
            "dataset",
            "generator",
            "policy",
            "score_name",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
            "accepted_qed_mean",
            "accepted_energy_per_heavy_mean",
            "accepted_relax_rmsd_mean",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate"], ascending=[True, True, True])
    table_policy_fixed_top4.to_csv(out / "table_resampling_policy_fixed_top4.csv", index=False)

    table_policy_source_a10 = policy_ablation[
        (policy_ablation["policy"] == "source_wilson_topk")
        & (policy_ablation["alpha"] == 0.10)
        & (policy_ablation["k"] == 4)
        & policy_ablation["score_name"].isin(["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate"], ascending=[True, True, True])
    table_policy_source_a10.to_csv(out / "table_resampling_policy_source_wilson10_top4.csv", index=False)

    table_policy_target = policy_ablation[
        (policy_ablation["policy"] == "target_wilson_topk")
        & (policy_ablation["k"] == 4)
        & (policy_ablation["accepted_bad_rate"] <= policy_ablation["alpha"])
        & policy_ablation["score_name"].isin(["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
        ]
    ].sort_values(["dataset", "generator", "alpha", "accepted_bad_rate"], ascending=[True, True, True, True])
    table_policy_target.to_csv(out / "table_resampling_policy_target_wilson_top4.csv", index=False)

    policy50 = read_csv(args.repo_root / "outputs/resampling_policy_ablation_matched50_v0/reports/policy_ablation_summary.csv")
    table_policy50_fixed_top4 = policy50[
        (policy50["k"] == 4)
        & (
            ((policy50["policy"] == "raw_topk") & (policy50["score_name"] == "none"))
            | (
                (policy50["policy"] == "rank_topk")
                & policy50["score_name"].isin(["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
            )
        )
    ][
        [
            "dataset",
            "generator",
            "policy",
            "score_name",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
            "accepted_qed_mean",
            "accepted_energy_per_heavy_mean",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate"], ascending=[True, True, True])
    table_policy50_fixed_top4.to_csv(out / "table_resampling_policy_matched50_fixed_top4.csv", index=False)

    table_policy50_source_a10 = policy50[
        (policy50["policy"] == "source_wilson_topk")
        & (policy50["alpha"] == 0.10)
        & (policy50["k"] == 4)
        & policy50["score_name"].isin(["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate"], ascending=[True, True, True])
    table_policy50_source_a10.to_csv(out / "table_resampling_policy_matched50_source_wilson10_top4.csv", index=False)

    table_policy50_target = policy50[
        (policy50["policy"] == "target_wilson_topk")
        & (policy50["k"] == 4)
        & (policy50["accepted_bad_rate"] <= policy50["alpha"])
        & policy50["score_name"].isin(["qed_risk", "relaxation_rule_score", "pre_relax_ff_rf", "pre_relax_ff_hgb"])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_rate_delta_vs_raw_topk",
            "relative_risk_reduction_vs_raw_topk",
        ]
    ].sort_values(["dataset", "generator", "alpha", "accepted_bad_rate"], ascending=[True, True, True, True])
    table_policy50_target.to_csv(out / "table_resampling_policy_matched50_target_wilson_top4.csv", index=False)

    repair50 = read_csv(args.repo_root / "outputs/forcefield_repair_matched50_v0/reports/repair_generator_summary.csv")
    table_repair50_summary = repair50[
        [
            "generator",
            "molecules",
            "files",
            "repair_ok_rate",
            "repair_rmsd_mean",
            "raw_high_risk_fixed_rate",
            "repaired_high_risk_fixed_rate",
            "raw_qed_mean",
            "repaired_qed_mean",
            "raw_energy_per_heavy_mean",
            "repaired_energy_per_heavy_mean",
        ]
    ].sort_values("generator")
    table_repair50_summary.to_csv(out / "table_forcefield_repair_matched50_summary.csv", index=False)

    repair50_policy = read_csv(args.repo_root / "outputs/forcefield_repair_matched50_v0/reports/repair_conservative_policy.csv")
    table_repair50_policy = repair50_policy[
        repair50_policy["repair_rmsd_max"].isin([0.5, 1.0, 1.5, 2.0])
    ][
        [
            "generator",
            "repair_rmsd_max",
            "molecules",
            "accepted",
            "coverage",
            "raw_high_risk_fixed_rate",
            "raw_high_risk_on_accepted",
            "repaired_high_risk_on_accepted",
            "repair_rmsd_mean",
            "repaired_qed_mean",
        ]
    ].sort_values(["generator", "repair_rmsd_max"])
    table_repair50_policy.to_csv(out / "table_forcefield_repair_matched50_policy.csv", index=False)

    repair_sampling = read_csv(args.repo_root / "outputs/forcefield_repair_sampling_guidance_v0/reports/repair_generator_summary.csv")
    table_repair_sampling_summary = repair_sampling[
        [
            "generator",
            "molecules",
            "files",
            "repair_ok_rate",
            "repair_rmsd_mean",
            "raw_high_risk_fixed_rate",
            "repaired_high_risk_fixed_rate",
            "raw_qed_mean",
            "repaired_qed_mean",
            "raw_energy_per_heavy_mean",
            "repaired_energy_per_heavy_mean",
        ]
    ].sort_values("generator")
    table_repair_sampling_summary.to_csv(out / "table_forcefield_repair_sampling_summary.csv", index=False)

    repair_sampling_policy = read_csv(args.repo_root / "outputs/forcefield_repair_sampling_guidance_v0/reports/repair_conservative_policy.csv")
    table_repair_sampling_policy = repair_sampling_policy[
        repair_sampling_policy["repair_rmsd_max"].isin([1.0, 1.5, 2.0])
    ][
        [
            "generator",
            "repair_rmsd_max",
            "molecules",
            "accepted",
            "coverage",
            "raw_high_risk_fixed_rate",
            "raw_high_risk_on_accepted",
            "repaired_high_risk_on_accepted",
            "repair_rmsd_mean",
            "repaired_qed_mean",
        ]
    ].sort_values(["generator", "repair_rmsd_max"])
    table_repair_sampling_policy.to_csv(out / "table_forcefield_repair_sampling_policy.csv", index=False)

    combo = read_csv(args.repo_root / "outputs/select_repair_combo_v0/reports/select_repair_combo_summary.csv")
    combo_keep = (
        (combo["policy"] == "raw_topk")
        | ((combo["policy"] == "repair_rmsd_topk") & combo["repair_rmsd_max"].isin([1.0, 1.5]))
        | (
            (combo["policy"] == "source_select_topk")
            & combo["score_name"].isin(["relaxation_rule_score", "pre_relax_ff_rf"])
        )
        | (
            (combo["policy"] == "source_select_then_repair_topk")
            & combo["repair_rmsd_max"].isin([1.0, 1.5])
            & combo["score_name"].isin(["relaxation_rule_score", "pre_relax_ff_rf"])
        )
    )
    table_combo = combo[combo_keep][
        [
            "dataset",
            "generator",
            "policy",
            "score_name",
            "repair_rmsd_max",
            "targets",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "risk_delta_vs_raw_pool",
            "accepted_repair_rmsd_mean",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate", "accepted"], ascending=[True, True, True, False])
    table_combo.to_csv(out / "table_select_repair_combo_key.csv", index=False)

    final_root = args.repo_root / "outputs/final_policy_robustness_v0/reports"
    final_main = read_csv(final_root / "final_main_policy_table.csv")
    table_final_main = final_main[
        [
            "dataset",
            "generator",
            "policy",
            "score_name",
            "alpha",
            "repair_rmsd_max",
            "accepted",
            "target_success_at_k",
            "bad_per_budget_mean",
            "yield_per_budget_mean",
            "accepted_bad_rate",
            "delta_bad_per_budget_vs_raw",
            "accepted_repair_rmsd_mean",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "accepted_bad_rate", "accepted"], ascending=[True, True, True, False])
    table_final_main.to_csv(out / "table_final_policy_main.csv", index=False)

    final_sens = read_csv(final_root / "final_alpha_rmsd_sensitivity.csv")
    table_final_sens = final_sens[
        (final_sens["dataset"].isin(["matched50", "sampling"]))
        & final_sens["score_name"].isin(["pre_relax_ff_rf", "relaxation_rule_score"])
        & final_sens["alpha"].isin([0.05, 0.10, 0.20])
        & final_sens["repair_rmsd_max"].isin([1.0, 1.5, 2.0])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "repair_rmsd_max",
            "accepted",
            "target_success_at_k",
            "accepted_bad_rate",
            "bad_per_budget_mean",
            "yield_per_budget_mean",
            "accepted_repair_rmsd_mean",
        ]
    ].sort_values(["dataset", "generator", "score_name", "alpha", "repair_rmsd_max"])
    table_final_sens.to_csv(out / "table_final_policy_alpha_rmsd_sensitivity.csv", index=False)

    final_stats = read_csv(final_root / "final_paired_stats_key.csv")
    table_final_stats = final_stats[
        [
            "dataset",
            "generator",
            "score_name",
            "alpha",
            "repair_rmsd_max",
            "n_targets",
            "raw_bad_per_budget_mean",
            "policy_bad_per_budget_mean",
            "mean_delta_bad_per_budget",
            "bad_bootstrap95_low",
            "bad_bootstrap95_high",
            "bad_delta_sign_p",
            "raw_yield_per_budget_mean",
            "policy_yield_per_budget_mean",
            "mean_delta_yield_per_budget",
            "yield_bootstrap95_low",
            "yield_bootstrap95_high",
        ]
    ].sort_values(["dataset", "generator", "score_name", "repair_rmsd_max"])
    table_final_stats.to_csv(out / "table_final_policy_paired_bootstrap.csv", index=False)

    final_summary = read_csv(final_root / "policy_sweep_summary.csv")
    table_final_k = final_summary[
        (final_summary["policy"] == "source_select_then_repair_topk")
        & (final_summary["alpha"] == 0.10)
        & (final_summary["repair_rmsd_max"] == 1.5)
        & final_summary["score_name"].isin(["pre_relax_ff_rf", "relaxation_rule_score"])
        & final_summary["k"].isin([1, 2, 4])
    ][
        [
            "dataset",
            "generator",
            "score_name",
            "k",
            "accepted",
            "target_success_at_k",
            "bad_per_budget_mean",
            "yield_per_budget_mean",
            "accepted_bad_rate",
            "attempts_per_accepted",
        ]
    ].sort_values(["dataset", "generator", "score_name", "k"])
    table_final_k.to_csv(out / "table_final_policy_k_sensitivity.csv", index=False)

    lines = [
        "# Evidence Tables v0",
        "",
        "These tables are generated without plotting. They are paper-material candidates, not final claims.",
        "",
        "## CrossDocked Bad-Sample Detection",
        csv_block(table_detection),
        "",
        "## Wilson95 Selective Generation at Alpha = 0.05",
        csv_block(table_conformal),
        "",
        "## Risk-Coverage",
        csv_block(table_risk_coverage.head(36)),
        "",
        "## External DUD-E Reliability Stress Test",
        csv_block(table_dude),
        "",
        "## PoseBusters-Oriented Label Stress Test",
        csv_block(table_posebusters),
        "",
        "## Feature Ablation",
        csv_block(table_ablation),
        "",
        "## SDF Physics Label Sensitivity",
        csv_block(table_sdf_sensitivity),
        "",
        "## SDF Physics Component Rates at q90",
        csv_block(table_sdf_components),
        "",
        "## Legacy Risk Score Alignment to RDKit Physical Risk",
        csv_block(table_sdf_alignment),
        "",
        "## Dedicated Physics Risk Critic Detection at q90",
        csv_block(table_critic_detection),
        "",
        "## Dedicated Physics Risk Critic Wilson Selection at q90",
        csv_block(table_critic_sel),
        "",
        "## Physics Risk Critic Quantile Sensitivity",
        csv_block(table_critic_sensitivity),
        "",
        "## Selective Quality Trade-off at q90",
        csv_block(table_quality_tradeoff),
        "",
        "## Sampling-Side Guidance SDF Audit",
        csv_block(table_sampling_audit),
        "",
        "## Matched SYNC Guide vs Origin Audit",
        csv_block(table_matched_guidance),
        "",
        "## Matched SYNC Guide vs Origin Audit, 50 Targets",
        csv_block(table_matched_guidance_50),
        "",
        "## Matched SYNC Guide vs Origin Pairwise Statistics, 50 Targets",
        csv_block(table_matched_pairwise_50),
        "",
        "## Physics Critic Transfer Detection",
        csv_block(table_transfer_detection),
        "",
        "## Physics Critic Transfer Target-Calibrated Wilson Selection",
        csv_block(table_transfer_target_sel),
        "",
        "## Physics Critic Transfer Source-Calibrated Resampling Simulation",
        csv_block(table_transfer_resampling),
        "",
        "## Physics Critic Transfer Detection, Matched50",
        csv_block(table_transfer50_detection),
        "",
        "## Physics Critic Transfer Source Wilson Alpha 0.10, Matched50",
        csv_block(table_transfer50_resampling),
        "",
        "## Resampling Policy Ablation: Fixed Top-4 Ranking",
        csv_block(table_policy_fixed_top4),
        "",
        "## Resampling Policy Ablation: Source Wilson Alpha 0.10 Top-4",
        csv_block(table_policy_source_a10),
        "",
        "## Resampling Policy Ablation: Target Wilson Top-4 Rows Meeting Alpha",
        csv_block(table_policy_target),
        "",
        "## Resampling Policy Ablation Matched50: Fixed Top-4 Ranking",
        csv_block(table_policy50_fixed_top4),
        "",
        "## Resampling Policy Ablation Matched50: Source Wilson Alpha 0.10 Top-4",
        csv_block(table_policy50_source_a10),
        "",
        "## Resampling Policy Ablation Matched50: Target Wilson Top-4 Rows Meeting Alpha",
        csv_block(table_policy50_target),
        "",
        "## Force-Field Repair Ablation Matched50 Summary",
        csv_block(table_repair50_summary),
        "",
        "## Force-Field Repair Ablation Matched50 Conservative Policies",
        csv_block(table_repair50_policy),
        "",
        "## Force-Field Repair Ablation Sampling Summary",
        csv_block(table_repair_sampling_summary),
        "",
        "## Force-Field Repair Ablation Sampling Conservative Policies",
        csv_block(table_repair_sampling_policy),
        "",
        "## Select + Repair Combo Ablation Key Rows",
        csv_block(table_combo),
        "",
        "## Final Policy Main Table",
        csv_block(table_final_main),
        "",
        "## Final Policy Alpha/RMSD Sensitivity",
        csv_block(table_final_sens),
        "",
        "## Final Policy Paired Bootstrap",
        csv_block(table_final_stats),
        "",
        "## Final Policy k Sensitivity",
        csv_block(table_final_k),
    ]
    (out / "paper_tables.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((out / "paper_tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
