from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "recovery_index"
DATA_FILE = PROJECT_DIR / "data" / "model_dataset.csv"
MODELS_DIR = PROJECT_DIR / "models"

TARGET = "failed_within_12m_from_anchor"
MODEL_NAME = "risk_model_12m"
MODEL_FILE = MODELS_DIR / f"{MODEL_NAME}.joblib"
SUMMARY_FILE = MODELS_DIR / "risk_model_summary.json"
FEATURES_FILE = MODELS_DIR / "risk_model_features.json"

COURT_WINDOW_FIELDS = [
    "count",
    "claim_sum",
    "claim_max",
    "defendant_count",
    "plaintiff_count",
    "defendant_claim_sum",
    "plaintiff_claim_sum",
    "active_months",
]

COURT_WINDOW_FEATURES_24M = [
    *[
        f"{prefix}_{field}"
        for prefix in ("last_anchor_window12", "last_anchor_window24", "last_anchor_prev12")
        for field in COURT_WINDOW_FIELDS
    ],
    *[
        f"last_anchor_window12_vs_prev12_{field}_{suffix}"
        for field in COURT_WINDOW_FIELDS
        for suffix in ("diff", "log_change")
    ],
]

PROFILE_SCALE_FEATURES = [
    "profile_capital_sum",
    "profile_headcount",
    "profile_msp_category",
]

EXCLUDED_MODEL_FEATURES = {
    "okved_main",
}

def load_master() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE, dtype={"company_inn": "string", "company_ogrn": "string"}, low_memory=False)

def load_feature_payload() -> dict[str, Any]:
    return json.loads(FEATURES_FILE.read_text(encoding="utf-8"))

def select_final_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    payload = load_feature_payload()
    requested = []
    for column in list(payload["features"]) + PROFILE_SCALE_FEATURES + COURT_WINDOW_FEATURES_24M:
        if column not in requested:
            requested.append(column)
    features = [
        column
        for column in requested
        if column not in EXCLUDED_MODEL_FEATURES and column in frame.columns and not frame[column].isna().all()
    ]
    removed = [column for column in requested if column not in features]
    return features, removed

def build_pipeline(frame: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

def build_calibrated_pipeline(frame: pd.DataFrame, feature_columns: list[str]) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=build_pipeline(frame, feature_columns),
        method="sigmoid",
        cv=5,
        ensemble=True,
    )

def threshold_candidates(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = (probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
                "f1": float(f1_score(y_true, predicted, zero_division=0)),
            }
        )
    return {
        "accuracy_threshold": max(rows, key=lambda row: row["accuracy"])["threshold"],
        "balanced_accuracy_threshold": max(rows, key=lambda row: row["balanced_accuracy"])["threshold"],
        "f1_threshold": max(rows, key=lambda row: row["f1"])["threshold"],
    }

def metrics_at_threshold(y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "true_negative": int(matrix[0][0]),
        "false_positive": int(matrix[0][1]),
        "false_negative": int(matrix[1][0]),
        "true_positive": int(matrix[1][1]),
        "confusion_matrix": matrix.tolist(),
    }

def build_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    return (
        pd.DataFrame({"feature": preprocessor.get_feature_names_out(), "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_master()
    features, removed = select_final_features(frame)
    subset = frame[["company_inn", "company_name", "final_status_label"] + features + [TARGET]].dropna(subset=[TARGET]).reset_index(drop=True)
    x = subset[features]
    y = subset[TARGET].astype(int)
    train_idx, test_idx = train_test_split(np.arange(len(subset)), test_size=0.2, random_state=42, stratify=y)

    split_pipeline = build_calibrated_pipeline(subset, features)
    split_pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
    train_probability = split_pipeline.predict_proba(x.iloc[train_idx])[:, 1]
    thresholds = threshold_candidates(y.iloc[train_idx].to_numpy(), train_probability)
    test_probability = split_pipeline.predict_proba(x.iloc[test_idx])[:, 1]
    threshold_metrics = {
        name: metrics_at_threshold(y.iloc[test_idx].to_numpy(), test_probability, threshold)
        for name, threshold in thresholds.items()
    }

    final_pipeline = build_calibrated_pipeline(subset, features)
    final_pipeline.fit(x, y)
    joblib.dump(final_pipeline, MODEL_FILE)
    importance_pipeline = build_pipeline(subset, features)
    importance_pipeline.fit(x, y)
    importance = build_importance(importance_pipeline)

    feature_payload = load_feature_payload()
    feature_payload.update(
        {
            "model_name": MODEL_NAME,
            "target": TARGET,
            "feature_set": "combined_last_maxsafe_12m_24m_prev12_comparison",
            "feature_count": len(features),
            "features": features,
            "removed_features": removed,
        }
    )
    FEATURES_FILE.write_text(json.dumps(feature_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "model_name": MODEL_NAME,
        "model_file": str(MODEL_FILE.relative_to(ROOT_DIR)),
        "dataset_file": str(DATA_FILE.relative_to(ROOT_DIR)),
        "target": TARGET,
        "algorithm": "GradientBoostingClassifier",
        "probability_calibration": {"method": "sigmoid", "cv": 5},
        "hyperparameters": {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 4},
        "rows": int(len(subset)),
        "feature_count": len(features),
        "positive_rate": float(y.mean()),
        "split": {
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_positive_rate": float(y.iloc[train_idx].mean()),
            "test_positive_rate": float(y.iloc[test_idx].mean()),
        },
        "thresholds_selected_on_train": thresholds,
        "service_threshold_name": "f1_threshold",
        "service_threshold": thresholds["f1_threshold"],
        "holdout_metrics_by_threshold": threshold_metrics,
        "top_features": importance.head(25).to_dict("records"),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
