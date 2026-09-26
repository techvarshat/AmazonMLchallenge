import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.main_pipeline import _normalize_frame, _build_truth_map, _build_training_rows
from src.evaluation import macro_f05_score
from src.train_logistic import train_logistic_model
from src.train_lgbm import train_lightgbm_model
from src.train_xgb import train_xgboost_model
from src.train_catboost import train_catboost_model


def read_tsv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep='\t',
        engine='python',
        on_bad_lines='skip',
        dtype={'entity_id': 'string'},
        nrows=nrows,
    )


def evaluate_subset(sample_size: int, root: Path) -> dict | None:
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
        return None

    counts = np.unique(y, return_counts=True)[1]
    if counts.min() < 4:
        return None

    X = pd.DataFrame(rows).fillna(0)
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42, stratify=y)
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]
    if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
        return None

    models = {
        'logistic': train_logistic_model(X_train.to_dict('records'), y_train),
        'lightgbm': train_lightgbm_model(X_train.to_dict('records'), y_train),
        'xgboost': train_xgboost_model(X_train.to_dict('records'), y_train),
        'catboost': train_catboost_model(X_train.to_dict('records'), y_train),
    }

    results = {}
    for name, model in models.items():
        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]
        tp = np.sum((pred == 1) & (y_val == 1))
        fp = np.sum((pred == 1) & (y_val == 0))
        fn = np.sum((pred == 0) & (y_val == 1))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = float((pred == y_val).mean())
        f05 = float(macro_f05_score(y_val, prob))
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f05': f05,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        }

    prob_stack = np.stack([model.predict_proba(X_val)[:, 1] for model in models.values()], axis=0)
    ensemble_prob = prob_stack.mean(axis=0)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    tp = np.sum((ensemble_pred == 1) & (y_val == 1))
    fp = np.sum((ensemble_pred == 1) & (y_val == 0))
    fn = np.sum((ensemble_pred == 0) & (y_val == 1))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    results['ensemble'] = {
        'accuracy': float((ensemble_pred == y_val).mean()),
        'precision': precision,
        'recall': recall,
        'f05': float(macro_f05_score(y_val, ensemble_prob)),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }
    return {'sample_size': sample_size, 'rows': len(X), 'positive_count': int(y.sum()), 'negative_count': int((1-y).sum()), 'results': results}


def main() -> None:
    root = Path(r'c:\Users\SMILE\Documents\amazom ml\6ab10eb3b23ba_student_resource\student_resource')
    best = None
    for sample_size in [200, 500, 1000, 2000, 5000]:
        result = evaluate_subset(sample_size, root)
        if result is None:
            continue
        if best is None or result['results']['lightgbm']['f05'] > best['results']['lightgbm']['f05']:
            best = result
        print(f"sample_size={sample_size} rows={result['rows']} positives={result['positive_count']} negatives={result['negative_count']}")
        for name, metrics in result['results'].items():
            print(
                f"  {name}: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, f05={metrics['f05']:.4f}, tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}"
            )
        if best is result:
            print(f"  CURRENT_LEADER={best['results']['lightgbm']['f05']:.4f}")
        print()
        break

    if best is None:
        raise SystemExit('No valid stratified subset could be generated for the sampled candidate pairs.')
    best_name = max(best['results'], key=lambda k: best['results'][k]['f05'])
    print(f'BEST_BY_F05={best_name} f05={best["results"][best_name]["f05"]:.4f}')


if __name__ == '__main__':
    main()
