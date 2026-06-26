from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_GENERATORS = {"diffsbdd", "molcraft", "pocketflow"}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(mask.mean())


def quantile_thresholds(frame: pd.DataFrame, q: float) -> dict[str, float]:
    return {
        "energy_per_heavy_threshold": float(frame["ff_energy_per_heavy"].dropna().quantile(q)),
        "energy_drop_threshold": float(frame["ff_energy_drop"].dropna().quantile(q)),
        "relax_rmsd_threshold": float(frame["ff_relax_rmsd"].dropna().quantile(q)),
    }


def add_component_flags(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
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
    out["quantile_energy_risk"] = out["high_energy"] | out["high_relax_drop"] | out["high_relax_rmsd"]
    out["physics_high_risk"] = out["hard_rule_risk"] | out["quantile_energy_risk"]
    return out


def summarize_for_quantile(frame: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_pool = frame[frame["generator"].isin(MAIN_GENERATORS)].copy()
    thresholds = quantile_thresholds(threshold_pool, q)
    flagged = add_component_flags(frame, thresholds)

    rows = []
    component_rows = []
    for generator, sub in flagged.groupby("generator"):
        rows.append(
            {
                "quantile": q,
                "generator": generator,
                "n": len(sub),
                **thresholds,
                "physics_high_risk_rate": safe_rate(sub["physics_high_risk"]),
                "hard_rule_risk_rate": safe_rate(sub["hard_rule_risk"]),
                "quantile_energy_risk_rate": safe_rate(sub["quantile_energy_risk"]),
                "qed_mean": float(sub["qed"].mean()),
                "energy_per_heavy_mean": float(sub["ff_energy_per_heavy"].mean()),
            }
        )
        component_rows.append(
            {
                "quantile": q,
                "generator": generator,
                "n": len(sub),
                "invalid_or_failed_rate": safe_rate(sub["invalid_or_failed"]),
                "bond_outlier_rate": safe_rate(sub["bond_outlier"]),
                "internal_clash_rate": safe_rate(sub["internal_clash"]),
                "high_energy_rate": safe_rate(sub["high_energy"]),
                "high_relax_drop_rate": safe_rate(sub["high_relax_drop"]),
                "high_relax_rmsd_rate": safe_rate(sub["high_relax_rmsd"]),
            }
        )
    summary = pd.DataFrame(rows)
    components = pd.DataFrame(component_rows)

    baseline = summary[summary["generator"] == "pocketflow"][["quantile", "physics_high_risk_rate"]]
    baseline = baseline.rename(columns={"physics_high_risk_rate": "pocketflow_high_risk_rate"})
    summary = summary.merge(baseline, on="quantile", how="left")
    summary["delta_vs_pocketflow"] = summary["physics_high_risk_rate"] - summary["pocketflow_high_risk_rate"]
    summary["multiplier_vs_pocketflow"] = summary["physics_high_risk_rate"] / summary[
        "pocketflow_high_risk_rate"
    ].replace(0.0, np.nan)
    return summary, components


def csv_block(df: pd.DataFrame) -> str:
    return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def write_summary(out_dir: Path, sensitivity: pd.DataFrame, components: pd.DataFrame) -> None:
    ordered = sensitivity.sort_values(["quantile", "physics_high_risk_rate"], ascending=[True, False])
    q90_components = components[components["quantile"] == 0.9].sort_values("generator")
    lines = [
        "# SDF Physics Label Sensitivity",
        "",
        "Quantile thresholds are computed on the three main generators only: diffsbdd, molcraft, pocketflow.",
        "MolPilot is kept in the table but should be treated as tiny-sample diagnostic evidence.",
        "",
        "## High-Risk Rate Sensitivity",
        csv_block(
            ordered[
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
            ]
        ),
        "",
        "## Component Rates at q90",
        csv_block(q90_components),
    ]
    (out_dir / "summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sdf_physics_sensitivity_v0/reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.audit_csv)

    summaries = []
    components = []
    for q in [0.80, 0.90, 0.95]:
        summary, component = summarize_for_quantile(frame, q)
        summaries.append(summary)
        components.append(component)
    sensitivity = pd.concat(summaries, ignore_index=True)
    component_table = pd.concat(components, ignore_index=True)

    sensitivity.to_csv(args.output_dir / "physics_label_sensitivity.csv", index=False)
    component_table.to_csv(args.output_dir / "physics_component_rates.csv", index=False)
    write_summary(args.output_dir, sensitivity, component_table)
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
