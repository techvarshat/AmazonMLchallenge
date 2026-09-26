import numpy as np
import pandas as pd
from pathlib import Path

from src.main_pipeline import _normalize_frame, _build_truth_map, _build_training_rows
from src.evaluation import macro_f05_score
from src.train_logistic import train_logistic_model
from src.train_lgbm import train_lightgbm_model
from sklearn.model_selection import train_test_split


def read_tsv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', engine='python', on_bad_lines='skip', dtype={'entity_id': 'string'}, nrows=nrows)


root = Path(r'c:\Users\SMILE\Documents\amazom ml\6ab10eb3b23ba_student_resource\student_resource')
train_dir = root / 'dataset' / 'train'
truth = read_tsv(train_dir / 'train_ground_truth.tsv')

for sample_size in [50, 100, 200, 300, 500, 800, 1000]:
    pos = truth[
        truth['matched_entity_ids'].notna()
        & truth['matched_entity_ids'].astype(str).str.strip().ne('')
    ].head(sample_size).copy()
    neg = truth[
        truth['matched_entity_ids'].isna()
        | truth['matched_entity_ids'].astype(str).str.strip().eq('')
    ].head(sample_size).copy()
    if pos.empty or neg.empty:
        continue
    selected_truth = pd.concat([pos, neg], ignore_index=True)
    selected_s1 = selected_truth['source1_entity_id'].astype(str).tolist()

    s1 = _normalize_frame(read_tsv(train_dir / 'train_source1.tsv'))
    s1 = s1[s1['entity_id'].astype(str).isin(selected_s1)].copy()
    s2 = _normalize_frame(read_tsv(train_dir / 'train_source2.tsv', nrows=2000))
    s3 = _normalize_frame(read_tsv(train_dir / 'train_source3.tsv', nrows=2000))
    target = pd.concat([s2, s3], ignore_index=True)

    rows, labels = _build_training_rows(s1, target, _build_truth_map(selected_truth), max_candidates=20)
    y = np.asarray(labels, dtype=int)
    if np.unique(y).size < 2:
        continue

    X = pd.DataFrame(rows).fillna(0)
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42, stratify=y)
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]

    log_model = train_logistic_model(X_train.to_dict('records'), y_train)
    lgb_model = train_lightgbm_model(X_train.to_dict('records'), y_train)
    log_pred = log_model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val)
    log_acc = float((log_pred == y_val).mean())
    lgb_acc = float((lgb_pred == y_val).mean())
    log_prob = log_model.predict_proba(X_val)[:, 1]
    lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
    log_f05 = float(macro_f05_score(y_val, log_prob))
    lgb_f05 = float(macro_f05_score(y_val, lgb_prob))

    print(f'subset_size={sample_size}')
    print(f'rows={len(X)} positives={int(y.sum())} negatives={int((1-y).sum())}')
    print(f'train={len(X_train)} val={len(X_val)}')
    print(f'logistic_accuracy={log_acc:.4f}')
    print(f'lightgbm_accuracy={lgb_acc:.4f}')
    print(f'logistic_f05={log_f05:.4f}')
    print(f'lightgbm_f05={lgb_f05:.4f}')
    break
else:
    raise SystemExit('No balanced sample with both classes found.')
