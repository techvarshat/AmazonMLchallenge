# Entity Resolution Pipeline

This project reconstructs a production-oriented entity-resolution pipeline for the Amazon ML challenge. The first working experiment is a LightGBM-only pipeline that preserves the challenge schema and keeps training and inference in Colab.

## Layout

- `config/config.yaml` – dataset and model defaults
- `src/` – modular pipeline code
- `data/` – raw, processed, and candidate caches
- `models/` – trained LightGBM model and metadata
- `outputs/` – generated submission artifacts

## LightGBM-only workflow

### Local smoke test (no large training)

```bash
cd entity_resolution
python -m src.lightgbm_pipeline --mode train --data-root ../6ab10eb3b23ba_student_resource/student_resource --output-dir outputs --model-dir models --artifact-dir artifacts --limit 5000
```

### Train the persisted LightGBM model in Colab

```bash
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/amazon_ml')
PROJECT_ROOT = DRIVE_ROOT / 'entity_resolution'
RAW_DATA_ROOT = DRIVE_ROOT / 'data' / 'raw'

!python "{PROJECT_ROOT}/build_persisted_model.py" \
  --data-root "{RAW_DATA_ROOT}" \
  --output-dir "{DRIVE_ROOT}/outputs" \
  --model-dir "{DRIVE_ROOT}/models" \
  --artifact-dir "{DRIVE_ROOT}/artifacts"
```

### Run final inference and create the challenge files in Colab

```bash
!python "{PROJECT_ROOT}/run_final_submission.py" \
  --data-root "{RAW_DATA_ROOT}" \
  --output-dir "{DRIVE_ROOT}/outputs" \
  --model-dir "{DRIVE_ROOT}/models" \
  --artifact-dir "{DRIVE_ROOT}/artifacts"
```

### Validate the generated files

```bash
from pathlib import Path
from src.validation import validate_candidate_file, validate_matching_file

out = Path('/content/drive/MyDrive/amazon_ml/outputs')
source1_ids = [str(v) for v in pd.read_csv('/content/drive/MyDrive/amazon_ml/data/raw/test_source1.tsv', sep='\t')['entity_id']]
validate_candidate_file(out / 'candidate_pairs.tsv', source1_ids)
validate_matching_file(out / 'matching_results.tsv', source1_ids)
```

## Exact submission files

The pipeline writes the required challenge files:

- `candidate_pairs.tsv`
- `matching_results.tsv`

Both files use the tab-separated schema enforced by the project validators:

- `source1_entity_id\tcandidate_entity_ids`
- `source1_entity_id\tmatched_entity_ids`

## Key stages

1. Data audit and schema discovery
2. Normalization and multi-view pair features
3. Exact and token blocking
4. LightGBM training on validated pair rows
5. Threshold selection on validation probabilities only
6. Final inference and challenge output generation

## Important notes

- This first experiment uses LightGBM only and intentionally excludes XGBoost, CatBoost, and ensemble models.
- Labels are generated from the challenge ground truth mapping, and the matching threshold is chosen only from validation data.
- Google Drive paths are configurable so the same code works for local smoke tests and Colab execution.
