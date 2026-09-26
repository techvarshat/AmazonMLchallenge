import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.main_pipeline import _normalize_frame, _build_truth_map, _build_training_rows
from src.train_logistic import train_logistic_model
from src.train_lgbm import train_lightgbm_model
from src.evaluation import macro_f05_score


def read_tsv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep='\t',
        engine='python',
        on_bad_lines='skip',
        dtype={'entity_id': 'string'},
        nrows=nrows,
    )


def evaluate_subset(sample_size: int, root: Path) -> None:
    train_dir = root / 'dataset' / 'train'
    truth = read_tsv(train_dir / 'train_ground_truth.tsv')
    truth_map = _build_truth_map(truth)

    s1 = _normalize_frame(read_tsv(train_dir / 'train_source1.tsv'))
    s2 = _normalize_frame(read_tsv(train_dir / 'train_source2.tsv'))
    s3 = _normalize_frame(read_tsv(train_dir / 'train_source3.tsv'))
    target = pd.concat([s2, s3], ignore_index=True)

    sampled = s1.sample(n=min(sample_size, len(s1)), random_state=42)
    rows, labels = _build_training_rows(sampled, target, truth_map, max_candidates=20)
    y = np.asarray(labels, dtype=int)
    if np.unique(y).size < 2:
        print(f'sample_size={sample_size}: no valid positive/negative label split generated; skipping.')
        return

    counts = np.unique(y, return_counts=True)[1]
    if counts.min() < 4:
        print(f'sample_size={sample_size}: minority class count too small for stratified split ({counts.tolist()}); skipping.')
        return

    X = pd.DataFrame(rows).fillna(0)
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42, stratify=y)
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]

    if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
        print(f'sample_size={sample_size}: train/val split still leaves only one class; skipping.')
        return

    print(f'## sample_size={sample_size} rows={len(X)} positives={int(y.sum())} negatives={int((1-y).sum())}')
    for name, model in [
        ('logistic', train_logistic_model(X_train.to_dict('records'), y_train)),
        ('lightgbm', train_lightgbm_model(X_train.to_dict('records'), y_train)),
    ]:
        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]
        tp = np.sum((pred == 1) & (y_val == 1))
        fp = np.sum((pred == 1) & (y_val == 0))
        fn = np.sum((pred == 0) & (y_val == 1))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = float((pred == y_val).mean())
        f05 = float(macro_f05_score(y_val, prob))
        print(
            f'{name}: accuracy={accuracy:.4f}, precision={precision:.4f}, '
            f'recall={recall:.4f}, f05={f05:.4f}, tp={tp}, fp={fp}, fn={fn}'
        )


def main() -> None:
    root = Path(r'c:\Users\SMILE\Documents\amazom ml\6ab10eb3b23ba_student_resource\student_resource')
    for sample_size in [200, 500, 1000, 2000, 5000, 10000]:
        evaluate_subset(sample_size, root)
        break

    print('Done. If no numbers were printed, no valid stratified subset was found for the current data slice.')


if __name__ == '__main__':
    main()
