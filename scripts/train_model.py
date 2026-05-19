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

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "model_dataset.csv"
MODELS_DIR = ROOT_DIR / "models"

TARGET = "failed_within_12m_from_assessment"
LEGACY_TARGET = "failed_within_12m_from_anchor"
MODEL_NAME = "risk_model_12m"
MODEL_FILE = MODELS_DIR / f"{MODEL_NAME}.joblib"
SUMMARY_FILE = MODELS_DIR / "risk_model_summary.json"
FEATURES_FILE = MODELS_DIR / "risk_model_features.json"
SOURCE_DEFENDANT_COUNT24 = "last_anchor_window24_defendant_count"
DEFENDANT_COUNT24_BUCKET = "last_anchor_window24_defendant_count_bucket"

COURT_WINDOW_FIELDS = [
    "count",
    "claim_sum",
    "claim_max",
    "defendant_count",
    "defendant_claim_sum",
    "defendant_claim_max",
    "active_months",
    "defendant_active_months",
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
    "raw_licenses_count",
]

CLAIM_PRESSURE_FEATURES = [
    f"{prefix}_{field}"
    for prefix in (
        "last_anchor_window12",
        "last_anchor_window24",
        "last_burst",
        "maxsafe",
    )
    for field in (
        "signed_claim_pressure",
        "plaintiff_offset_share",
    )
]

SIGNED_RATIO_FEATURES = [
    "last_burst_claim_pressure_log_signed",
    "last_burst_claim_pressure_signed_ratio",
    "last_burst_claim_baseline_tiny_flag",
    "maxsafe_claim_pressure_log_signed",
    "maxsafe_claim_pressure_signed_ratio",
    "maxsafe_claim_baseline_tiny_flag",
    "maxsafe_case_pressure_log_signed",
    "maxsafe_case_pressure_signed_ratio",
    "maxsafe_to_last_defendant_claim_log_signed",
    "maxsafe_to_last_defendant_claim_signed_ratio",
]

RESTORED_LAST_BURST_FEATURES = [
    "last_burst_months",
    "last_burst_total_cases",
    "last_burst_peak_cases",
    "last_burst_last_month_cases",
    "last_burst_cases_per_month",
    "last_burst_defendant_cases",
    "last_burst_plaintiff_cases",
    "last_burst_defendant_claim_sum",
    "last_burst_defendant_claim_max",
    "last_burst_defendant_peak_cases",
    "last_burst_defendant_peak_claim_sum",
    "last_burst_defendant_peak_claim_max",
    "last_burst_defendant_last_month_cases",
    "last_burst_defendant_last_month_claim_sum",
    "last_burst_defendant_cases_per_month",
    "last_burst_defendant_claim_per_month",
    "last_burst_defendant_avg_claim_per_case",
    "last_burst_defendant_peak_claim_share",
    "last_burst_plaintiff_claim_sum",
]

MAXSAFE_BURST_FEATURES = [
    "maxsafe_months",
    "maxsafe_total_cases",
    "maxsafe_peak_cases",
    "maxsafe_last_month_cases",
    "maxsafe_cases_per_month",
    "maxsafe_defendant_cases",
    "maxsafe_plaintiff_cases",
    "maxsafe_defendant_claim_sum",
    "maxsafe_defendant_claim_max",
    "maxsafe_defendant_peak_cases",
    "maxsafe_defendant_peak_claim_sum",
    "maxsafe_defendant_peak_claim_max",
    "maxsafe_defendant_last_month_cases",
    "maxsafe_defendant_last_month_claim_sum",
    "maxsafe_defendant_cases_per_month",
    "maxsafe_defendant_claim_per_month",
    "maxsafe_defendant_avg_claim_per_case",
    "maxsafe_defendant_peak_claim_share",
    "maxsafe_plaintiff_claim_sum",
]

ENFORCEMENT_FEATURES = [
    f"{prefix}_{field}"
    for prefix in ("last_anchor_enf", "maxsafe_anchor_enf")
    for field in (
        "count_total",
        "debt_total",
        "remaining_total",
        "debt_max",
        "count_12m",
        "debt_12m",
        "count_18m",
        "debt_18m",
        "has_any",
    )
]

ROLE_MIXED_COURT_FEATURES = {
    *[
        f"{prefix}_{field}"
        for prefix in (
            "last_anchor_window12",
            "last_anchor_window24",
            "last_anchor_prev12",
            "maxsafe_window12",
            "maxsafe_window24",
            "maxsafe_prev12",
        )
        for field in ("count", "claim_sum", "claim_max", "active_months")
    ],
    *[
        f"{prefix}_{field}"
        for prefix in ("last_burst", "maxsafe")
        for field in (
            "total_cases",
            "total_claim_sum",
            "peak_cases",
            "peak_claim_sum",
            "peak_claim_max",
            "last_month_cases",
            "last_month_claim_sum",
            "cases_per_month",
            "claim_per_month",
            "avg_claim_per_case",
            "peak_claim_share",
            "case_pressure_log_signed",
            "case_pressure_signed_ratio",
            "case_baseline_tiny_flag",
            "claim_pressure_log_signed",
            "claim_pressure_signed_ratio",
            "claim_baseline_tiny_flag",
        )
    ],
    *[
        f"last_anchor_window12_vs_prev12_{field}_{suffix}"
        for field in ("count", "claim_sum", "claim_max", "active_months")
        for suffix in ("diff", "log_change")
    ],
}

ALLOWED_PLAINTIFF_FEATURES = {
    "last_burst_plaintiff_cases",
    "last_burst_plaintiff_claim_sum",
    "maxsafe_plaintiff_cases",
    "maxsafe_plaintiff_claim_sum",
}

EXCLUDED_MODEL_FEATURES = {
    "okved_main",
    "raw_licenses_count",
    "last_burst_claim_ratio",
    "maxsafe_claim_ratio",
    "maxsafe_case_ratio",
    "safe_history_max_to_last_defendant_claim_ratio",
} | ROLE_MIXED_COURT_FEATURES

def is_heavy_plaintiff_feature(column: str) -> bool:
    if "plaintiff" not in column:
        return False
    if column in ALLOWED_PLAINTIFF_FEATURES:
        return False
    if column.endswith("_plaintiff_offset_share"):
        return False
    return True

def load_master() -> pd.DataFrame:
    frame = pd.read_csv(DATA_FILE, dtype={"company_inn": "string", "company_ogrn": "string"}, low_memory=False)
    return augment_feature_frame(frame)

def augment_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if SOURCE_DEFENDANT_COUNT24 not in frame.columns:
        return frame
    result = frame.copy()
    counts = pd.to_numeric(result[SOURCE_DEFENDANT_COUNT24], errors="coerce").fillna(0)
    result[DEFENDANT_COUNT24_BUCKET] = pd.cut(
        counts,
        bins=[-1, 1, 3, 10, 30, 100, float("inf")],
        labels=["0-1", "2-3", "4-10", "11-30", "31-100", "100+"],
    ).astype("string")
    return result

def load_feature_payload() -> dict[str, Any]:
    return json.loads(FEATURES_FILE.read_text(encoding="utf-8"))

def resolve_target(frame: pd.DataFrame) -> str:
    if TARGET in frame.columns:
        return TARGET
    if LEGACY_TARGET in frame.columns:
        return LEGACY_TARGET
    raise KeyError(f"Dataset must contain '{TARGET}' or legacy '{LEGACY_TARGET}'.")

def select_final_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    payload = load_feature_payload()
    requested = []
    for column in (
        list(payload["features"])
        + PROFILE_SCALE_FEATURES
        + COURT_WINDOW_FEATURES_24M
        + CLAIM_PRESSURE_FEATURES
        + SIGNED_RATIO_FEATURES
        + RESTORED_LAST_BURST_FEATURES
        + MAXSAFE_BURST_FEATURES
        + ENFORCEMENT_FEATURES
    ):
        if column not in requested:
            requested.append(column)
    requested = [column for column in requested if column != SOURCE_DEFENDANT_COUNT24]
    for column in (DEFENDANT_COUNT24_BUCKET,):
        if column not in requested:
            requested.append(column)
    features = []
    for column in requested:
        if (
            column in EXCLUDED_MODEL_FEATURES
            or is_heavy_plaintiff_feature(column)
            or column not in frame.columns
            or frame[column].isna().all()
        ):
            continue
        features.append(column)
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
    target = resolve_target(frame)
    features, removed = select_final_features(frame)
    subset = frame[["company_inn", "company_name", "final_status_label"] + features + [target]].dropna(subset=[target]).reset_index(drop=True)
    x = subset[features]
    y = subset[target].astype(int)

    indices = np.arange(len(subset))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.40, random_state=42, stratify=y
    )
    test_idx, val_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=y.iloc[temp_idx]
    )

    split_pipeline = build_calibrated_pipeline(subset, features)
    split_pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])

    val_probability = split_pipeline.predict_proba(x.iloc[val_idx])[:, 1]
    thresholds = threshold_candidates(y.iloc[val_idx].to_numpy(), val_probability)

    test_probability = split_pipeline.predict_proba(x.iloc[test_idx])[:, 1]
    test_metrics_by_threshold = {
        name: metrics_at_threshold(y.iloc[test_idx].to_numpy(), test_probability, threshold)
        for name, threshold in thresholds.items()
    }
    validation_metrics_by_threshold = {
        name: metrics_at_threshold(y.iloc[val_idx].to_numpy(), val_probability, threshold)
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
            "target": target,
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
        "target": target,
        "algorithm": "GradientBoostingClassifier",
        "probability_calibration": {"method": "sigmoid", "cv": 5},
        "hyperparameters": {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 4},
        "rows": int(len(subset)),
        "feature_count": len(features),
        "positive_rate": float(y.mean()),
        "split": {
            "scheme": "60/20/20 stratified, random_state=42, threshold tuned on validation",
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "validation_rows": int(len(val_idx)),
            "train_positive_rate": float(y.iloc[train_idx].mean()),
            "test_positive_rate": float(y.iloc[test_idx].mean()),
            "validation_positive_rate": float(y.iloc[val_idx].mean()),
        },
        "threshold_selection_strategy": "f1-optimal on validation (600 rows, held out from training)",
        "thresholds_selected_on_validation": thresholds,
        "service_threshold_name": "f1_threshold",
        "service_threshold": thresholds["f1_threshold"],
        "test_metrics_by_threshold": test_metrics_by_threshold,
        "validation_metrics_by_threshold": validation_metrics_by_threshold,
        "top_features": importance.head(25).to_dict("records"),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
