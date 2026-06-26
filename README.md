# PhysCAL-Diff Anonymous Release

This repository accompanies an anonymous submission on physically reliable post-selection and repair for structure-based molecule generation. It contains the core analysis scripts and a compact evidence bundle for the main reported tables.

## Contents

- `src/physcaldiff/`: shared metric utilities.
- `scripts/`: table-generation, risk-critic, selective-risk, and repair-analysis scripts.
- `evidence/main_tables/`: compact CSV evidence used by the main manuscript and essential supplementary checks.
- `configs/example_paths.yaml`: placeholder paths for external generator outputs and reference baselines.

The repository intentionally does not include large generated molecule pools, docking outputs, or third-party model checkpoints. Place those files under local paths of your choice and pass the paths through script arguments.

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

## Notes for Review

This anonymous bundle is designed to make the reported comparisons auditable without exposing author identity. After review, the release can be expanded with full raw molecule pools, environment lock files, and permanent repository metadata.
