# Entity Resolution Pipeline

This project reconstructs a production-oriented entity-resolution pipeline for the Amazon ML hackathon business matching challenge.

## Layout

- `config/config.yaml` – dataset and model defaults
- `src/` – modular pipeline code
- `data/` – raw, processed, candidate, and embedding caches
- `models/` – trained models and checkpoints
- `outputs/` – generated submission artifacts

## Quick start

```bash
python -m src.main_pipeline --data-root ../6ab10eb3b23ba_student_resource/student_resource --output-dir outputs --limit 50000
```

## Key stages

1. Data audit and schema discovery
2. Normalization and multi-view representations
3. Exact and token blocking
4. Candidate recall evaluation
5. Pair feature generation
6. Baseline tabular models and validation
7. Optional hard-negative mining and reranking
8. Final threshold search and submission validation

## Important notes

- The pipeline intentionally keeps early retrieval high recall and later stages conservative for precision.
- Country is treated as a feature, not a hard-coded filter.
- The implementation is designed to work in Colab and local machines with a cache-first workflow.
