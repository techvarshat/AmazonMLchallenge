# Dataset profile and architecture

## Verified dataset facts

- Source files use a shared schema: `entity_id`, `business_name`, `business_address`, `country`.
- Training counts:
  - train_source1.tsv: 2,206,821 rows
  - train_source2.tsv: 5,034,616 rows
  - train_source3.tsv: 5,285,603 rows
- Test counts:
  - test_source1.tsv: 1,732,544 rows
  - test_source2.tsv: 4,887,273 rows
  - test_source3.tsv: 5,082,316 rows
- Ground truth file: `train_ground_truth.tsv` with `source1_entity_id` and `matched_entity_ids`.
- Country distribution is strongly US/India in train and includes France in test. Country is a feature, not a hard rule.
- Address missingness is roughly 3.3% in source 2 and 3 and zero in source 1.
- The dataset is a many-to-one / multi-match problem. Some Source 1 entities match several Source 2 and Source 3 records.

## Architecture

1. Audit and schema mapping
2. Normalization with multiple views
3. Candidate generation by exact, token, and TF-IDF retrieval
4. Candidate recall evaluation
5. Pair feature engineering
6. Baseline logistic and tree models
7. Hard negative mining and reranking
8. Threshold optimization and final submission validation

## Operational note

This project is staged to start with high-recall blocking before precision-oriented classifying and thresholding, matching the challenge objective and the F0.5 metric.
