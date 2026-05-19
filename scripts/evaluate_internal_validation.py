from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import train_model


ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    frame = train_model.load_master()
    target = train_model.resolve_target(frame)
    feature_columns, removed = train_model.select_final_features(frame)
    features = frame[feature_columns].copy()
    labels = frame[target].astype(int).copy()

    split_seed = 42
    cv_seed = 42

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.40,
        stratify=labels,
        random_state=split_seed,
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=split_seed,
    )

    calibrated = train_model.build_calibrated_pipeline(x_train, feature_columns)
    calibrated.fit(x_train, y_train)

    val_prob = calibrated.predict_proba(x_val)[:, 1]
    threshold_payload = train_model.threshold_candidates(y_val.to_numpy(), val_prob)
    threshold = float(threshold_payload["f1_threshold"])

    test_prob = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = train_model.metrics_at_threshold(y_test, test_prob, threshold)
    val_metrics = train_model.metrics_at_threshold(y_val, val_prob, threshold)

    split_payload = {
        "dataset_rows": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "removed_feature_count": int(len(removed)),
        "target": target,
        "split_seed": split_seed,
        "threshold_selection": "f1-optimal on validation (600 rows)",
        "threshold": threshold,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "validation_rows": int(len(x_val)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "test_metrics": to_builtin(test_metrics),
        "validation_metrics": to_builtin(val_metrics),
    }

    split_path = REPORTS_DIR / "internal_train_test_validation_metrics.json"
    split_path.write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    cv_metrics = {
        "roc_auc": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "pr_auc": [],
        "balanced_accuracy": [],
    }
    fold_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(features, labels), start=1):
        x_tr = features.iloc[train_idx].copy()
        y_tr = labels.iloc[train_idx].copy()
        x_te = features.iloc[test_idx].copy()
        y_te = labels.iloc[test_idx].copy()

        fold_model = train_model.build_calibrated_pipeline(x_tr, feature_columns)
        fold_model.fit(x_tr, y_tr)

        fold_train_prob = fold_model.predict_proba(x_tr)[:, 1]
        fold_threshold = float(train_model.threshold_candidates(y_tr.to_numpy(), fold_train_prob)["f1_threshold"])
        fold_test_prob = fold_model.predict_proba(x_te)[:, 1]
        fold_metrics = train_model.metrics_at_threshold(y_te, fold_test_prob, fold_threshold)
        fold_metrics["fold"] = fold_idx
        fold_metrics["threshold"] = fold_threshold
        fold_rows.append(to_builtin(fold_metrics))
        for metric_name in cv_metrics:
            cv_metrics[metric_name].append(float(fold_metrics[metric_name]))

    cv_payload = {
        "dataset_rows": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "target": target,
        "cv_folds": 5,
        "cv_seed": cv_seed,
        "fold_metrics": fold_rows,
        "summary": {metric_name: mean_std(values) for metric_name, values in cv_metrics.items()},
    }
    cv_path = REPORTS_DIR / "internal_5fold_cv_metrics.json"
    cv_path.write_text(json.dumps(cv_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(split_path)
    print(cv_path)


if __name__ == "__main__":
    main()
