# PhysCAL-Diff: Code and Evidence

This repository accompanies PhysCAL-Diff, a post-generation physical-reliability selection and repair method for structure-based molecular generation. It contains analysis code and compact evidence for the reported comparisons. The experiment inputs are not included where they consist of third-party checkpoints or large generated pose and docking files.

## Contents

- `src/physcaldiff/`: shared metric utilities.
- `scripts/`: table-generation, risk-critic, selective-risk, and repair-analysis scripts.
- `scripts/recomb_risk_validation.py`: train-only label-threshold, feature-ablation, and empirical risk-budget evaluation.
- `scripts/recomb_validation/`: family-disjoint, independent PoseBusters, same-candidate graph, and matched scored-budget checks.
- `scripts/recomb_postprocess_validation.py` and `scripts/self_contained_*`: selector replay and its portfolio dependencies.
- `evidence/main_tables/`: compact evidence for the original main tables.
- `evidence/recomb_validation/`: later risk and independent-validation summaries, including the five-split train-only risk results. Historical retrospective table rows remain in `evidence/main_tables/` for auditability.
- `configs/example_paths.yaml`: placeholder paths for external generator outputs and reference baselines.

The repository does not redistribute large generated molecule pools, docking outputs, prepared receptor files, or third-party checkpoints. The compact CSVs can be inspected without those inputs. Re-running the full pipeline requires the source files described below.

## Minimal Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD/scripts"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="$PWD\src;$PWD\scripts"
```

## Reproducing Core Tables

The CSV files in `evidence/main_tables/` are the compact audit tables used for manuscript reporting. They cover:

- target-matched public-checkpoint baselines;
- no-Vina reliability critic detection and selective-risk control;
- same-pool selection and repair;
- Pareto-style public-source comparison;
- seed stability, orthogonal validation, and composite bootstrap checks.

To regenerate the paper-facing evidence tables from a full local experiment tree:

```bash
python scripts/collect_evidence_tables.py \
  --repo-root . \
  --old-root external/reference_baselines \
  --output-dir reports/evidence_tables
```

To rerun the SDF physical-risk audit and downstream alignment, provide a directory of generated SDF files and the corresponding risk-score CSVs:

```bash
python scripts/audit_sdf_physics.py \
  --input-root external/generated_sdf \
  --output-dir outputs/sdf_physics_audit/reports

python scripts/evaluate_sdf_risk_alignment.py \
  --audit-csv outputs/sdf_physics_audit/reports/molecule_physics_audit.csv \
  --results-root external/generator_results \
  --output-dir outputs/sdf_risk_alignment/reports
```

## External Inputs

Expected external inputs are standard outputs from public SBDD generators and their post-generation evaluation files, including generated SDF files, docking summaries, and optional public-checkpoint baseline results. The scripts use command-line arguments rather than fixed machine paths.

For train-only risk-threshold replication, place the physical-audit CSV at
`outputs/sdf_physics_audit_full_v0/reports/molecule_physics_audit.csv` and run:

```bash
python scripts/recomb_risk_validation.py --repo . --out reports/recomb_risk --jobs 8
```

The corresponding five-split risk detection and calibration CSVs are already in
`evidence/recomb_validation/`. The pooled group split is by generator/parent
structure, not by receptor sequence family. Family-disjoint results use only the
mapped DiffSBDD subset and are reported separately. The Wilson gate is an
empirical operating point, not a finite-sample risk certificate. The
independent physical checks use public receptor structures and precomputed
scored candidate pools; they do not establish equal end-to-end runtime or
experimental binding validity. The stored prospective SA aggregate is omitted
from the updated manuscript because four selected exports do not reproduce
that value under a single standardized SDF calculation.

The validation scripts accept paths to the external experiment tree with
`--repo` and other command-line flags. The summary CSVs document the results
where full third-party inputs are not redistributed.

For the family-disjoint audit, use the same audit CSV and run
`python scripts/recomb_validation/family_disjoint_validation.py --repo . --out reports/family_disjoint --jobs 8`.
For the matched scored-budget structural check, the receptor directory must
contain `input_manifest.csv` and either `pocket_reconstruction_manifest.csv`
(BindingMOAD) or `download_manifest.csv` (prospective). The repository root
must also contain the scored candidate-pool CSVs and selected SDFs expected by
`scripts/recomb_postprocess_validation.py`. Run, for example,
`python scripts/recomb_validation/matched_budget_physical_validation.py --repo . --pocket-dir external/public_pockets --dataset bindingmoad --out reports/matched_physical --workers 8`.
