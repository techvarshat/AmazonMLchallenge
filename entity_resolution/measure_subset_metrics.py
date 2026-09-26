import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.main_pipeline import _normalize_frame, _build_truth_map, _build_training_rows
from src.data_loader import load_ground_truth, load_source_table
from src.evaluation import macro_f05_score
from src.train_logistic import train_logistic_model
from src.train_lgbm import train_lightgbm_model


def main() -> None:
    root = Path(r"c:\Users\SMILE\Documents\amazom ml\6ab10eb3b23ba_student_resource\student_resource")
    train_dir = root / "dataset" / "train"

    for nrows in [50, 100, 200, 500, 1000]:
        truth = load_ground_truth(train_dir / "train_ground_truth.tsv").head(nrows)
        s1 = _normalize_frame(load_source_table(train_dir / "train_source1.tsv", nrows=nrows))
        s2 = _normalize_frame(load_source_table(train_dir / "train_source2.tsv", nrows=nrows))
        s3 = _normalize_frame(load_source_table(train_dir / "train_source3.tsv", nrows=nrows))
        target = pd.concat([s2, s3], ignore_index=True)
        truth_map = _build_truth_map(truth)
        rows, labels = _build_training_rows(s1, target, truth_map, max_candidates=10)
        y = np.asarray(labels, dtype=int)
        if np.unique(y).size < 2:
            continue

        X = pd.DataFrame(rows).fillna(0)
        train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42, stratify=y)
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        logistic_model = train_logistic_model(X_train.to_dict("records"), y_train)
        lgb_model = train_lightgbm_model(X_train.to_dict("records"), y_train)
        log_pred = logistic_model.predict(X_val)
        lgb_pred = lgb_model.predict(X_val)
        log_acc = float((log_pred == y_val).mean())
        lgb_acc = float((lgb_pred == y_val).mean())
        log_prob = logistic_model.predict_proba(X_val)[:, 1]
        lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
        log_f05 = float(macro_f05_score(y_val, log_prob))
        lgb_f05 = float(macro_f05_score(y_val, lgb_prob))

        print(f"subset_nrows={nrows}")
        print(f"raw_rows={len(X)} positives={int(y.sum())} negatives={int((1-y).sum())}")
        print(f"train_size={len(X_train)} val_size={len(X_val)}")
        print(f"logistic_accuracy={log_acc:.4f}")
        print(f"lightgbm_accuracy={lgb_acc:.4f}")
        print(f"logistic_f05={log_f05:.4f}")
        print(f"lightgbm_f05={lgb_f05:.4f}")
        return

    raise SystemExit("No subset with both positive and negative labels was found in the sampled training rows.")


if __name__ == "__main__":
    main()
