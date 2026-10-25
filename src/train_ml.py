"""
HarmoniX | Phase 1 Classical Machine Learning Modeling & Tuning
Author: Bhoomi

This module handles:
1. Loading physics and acoustic feature matrix (features_physics.csv).
2. Leakage-free data partitioning (train/val splits).
3. Pipeline construction with median imputation and StandardScaler to prevent fold leakage.
4. Baseline Ridge Regression modeling and hyperparameter tuning.
5. Non-linear ensemble modeling with Random Forest Regressor.
6. Gradient boosting with XGBoost Regressor.
7. Discretized stratified cross-validation for balanced target distribution.
8. Validation set benchmark evaluation with domain-bounded predictions (MAE, RMSE, R2, Pearson r).
9. Model serialization (.joblib) and results logging.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import joblib

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HarmoniX.TrainML")


class MLConfig:
    """Project paths and training configurations."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_physics.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    TARGET_COL = "valence"
    SPLIT_COL = "split"
    RANDOM_STATE = 42

    EXPECTED_FEATURE_COUNT = 34


def load_dataset(features_path: Path) -> pd.DataFrame:
    """Load the extracted physics features dataset."""
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found at: {features_path}\n"
            "Please ensure feature_extraction.py has been executed."
        )
    df = pd.read_csv(features_path)
    logger.info(f"Loaded dataset from {features_path.name} with shape {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Identify the 34 physics & acoustic feature columns from the DataFrame."""
    feature_cols = [
        col for col in df.columns
        if col.startswith(("mfcc_", "spectral_", "rms_", "zcr_", "chroma_"))
        or col in ("hnr_estimate_db", "sensory_dissonance_estimate")
    ]

    if len(feature_cols) != MLConfig.EXPECTED_FEATURE_COUNT:
        logger.warning(
            f"Expected {MLConfig.EXPECTED_FEATURE_COUNT} features, but found {len(feature_cols)}. "
            f"Using available {len(feature_cols)} acoustic features."
        )
    else:
        logger.info(f"Verified all {len(feature_cols)} acoustic physics features.")

    return feature_cols


def prepare_leakage_free_splits(
    df: pd.DataFrame, feature_cols: List[str], target_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate dataset into Train, Validation, and Test partitions based on 'split' column.
    Guarantees strict separation:
    - Train split used for scaling and hyperparameter tuning via CV.
    - Val split used for model selection and validation benchmark.
    - Test split is preserved untouched for final evaluation.
    """
    if MLConfig.SPLIT_COL not in df.columns:
        raise ValueError(f"Required split column '{MLConfig.SPLIT_COL}' not found in dataset.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    train_df = df[df[MLConfig.SPLIT_COL] == "train"].copy().reset_index(drop=True)
    val_df = df[df[MLConfig.SPLIT_COL] == "val"].copy().reset_index(drop=True)
    test_df = df[df[MLConfig.SPLIT_COL] == "test"].copy().reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("Train split is empty. Check split generation.")
    if len(val_df) == 0:
        raise ValueError("Validation split is empty. Check split generation.")

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    X_test = test_df[feature_cols].values if len(test_df) > 0 else np.empty((0, len(feature_cols)))
    y_test = test_df[target_col].values if len(test_df) > 0 else np.empty((0,))

    logger.info(
        f"Data Partition Summary: "
        f"Train={len(X_train)} samples, "
        f"Val={len(X_val)} samples, "
        f"Test={len(X_test)} samples (reserved for final evaluation)"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_balanced_cv_splitter(y_train: np.ndarray, n_splits: int = 5):
    """
    Generate cross-validation splits. If sample size permits, uses quantile-binned
    stratification so every fold contains an even distribution of low, medium,
    and high valence songs. Falls back gracefully to standard KFold for small sets.
    """
    actual_splits = min(n_splits, max(2, len(y_train)))

    num_bins = min(actual_splits, max(2, len(y_train) // actual_splits))
    try:
        y_binned = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")
        if len(np.unique(y_binned)) >= 2:
            min_bin_count = np.min(np.bincount(y_binned))
            if min_bin_count >= 2:
                skf_splits = min(actual_splits, min_bin_count)
                logger.info(f"Using Stratified K-Fold with {skf_splits} folds across {len(np.unique(y_binned))} valence bins.")
                skf = StratifiedKFold(n_splits=skf_splits, shuffle=True, random_state=MLConfig.RANDOM_STATE)
                return list(skf.split(np.zeros(len(y_train)), y_binned))
    except Exception:
        pass

    logger.info(f"Using standard K-Fold with {actual_splits} folds.")
    kf = KFold(n_splits=actual_splits, shuffle=True, random_state=MLConfig.RANDOM_STATE)
    return kf


def predict_bounded(model_pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions and enforce valid valence bounds [0.0, 1.0].
    Valence is inherently bounded; clipping removes unphysical boundary overshoot.
    """
    raw_pred = model_pipe.predict(X)
    return np.clip(raw_pred, 0.0, 1.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics: MAE, RMSE, R2, Pearson r."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        r_val, _ = pearsonr(y_true, y_pred)
        pearson_r = float(r_val)
    else:
        pearson_r = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Pearson_r": round(pearson_r, 4)
    }


def train_and_tune_ridge(
    X_train: np.ndarray, y_train: np.ndarray, cv
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train baseline Ridge Regression with StandardScaler inside Pipeline to prevent leakage.
    Tunes L2 regularization parameter alpha via GridSearchCV.
    """
    logger.info("--- Tuning Model 1: Ridge Regression (Baseline Regularized Linear) ---")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=MLConfig.RANDOM_STATE))
    ])

    param_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    best_cv_rmse = float(-grid_search.best_score_)
    logger.info(f"Ridge Best Params: {grid_search.best_params_}")
    logger.info(f"Ridge Best CV RMSE: {best_cv_rmse:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, best_cv_rmse


def train_and_tune_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, cv
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train Random Forest Regressor to capture non-linear relationships.
    Tunes ensemble tree depth, split criteria, estimators, and feature sampling (bagging) via GridSearchCV.
    """
    logger.info("--- Tuning Model 2: Random Forest Regressor (Non-Linear Bagging Ensemble) ---")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=MLConfig.RANDOM_STATE))
    ])

    param_grid = {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_split": [2, 4],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.5, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    best_cv_rmse = float(-grid_search.best_score_)
    logger.info(f"Random Forest Best Params: {grid_search.best_params_}")
    logger.info(f"Random Forest Best CV RMSE: {best_cv_rmse:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, best_cv_rmse


def train_and_tune_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, cv
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train XGBoost Regressor for gradient boosting.
    Tunes tree depth, shrinkage (learning rate), subsampling, and regularizers via RandomizedSearchCV.
    """
    logger.info("--- Tuning Model 3: XGBoost Regressor (Gradient Tree Boosting) ---")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            random_state=MLConfig.RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1
        ))
    ])

    param_distributions = {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [1.0, 5.0, 10.0]
    }

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=30,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=MLConfig.RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )
    random_search.fit(X_train, y_train)

    best_cv_rmse = float(-random_search.best_score_)
    logger.info(f"XGBoost Best Params: {random_search.best_params_}")
    logger.info(f"XGBoost Best CV RMSE: {best_cv_rmse:.4f}")

    return random_search.best_estimator_, random_search.best_params_, best_cv_rmse


def save_artifacts(
    models: Dict[str, Pipeline],
    best_model_name: str,
    feature_cols: List[str],
    results_df: pd.DataFrame,
    metrics_summary: Dict[str, Any]
) -> None:
    """Save trained model pipelines, scaler, feature names, and tuning metrics."""
    models_dir = MLConfig.MODELS_DIR
    reports_dir = MLConfig.REPORTS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for name, model_pipe in models.items():
        filename = f"{name.lower().replace(' ', '_')}_best.joblib"
        target_path = models_dir / filename
        joblib.dump(model_pipe, target_path)
        logger.info(f"Saved {name} model pipeline to: {target_path}")

    best_pipe = models[best_model_name]
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_pipe, best_model_path)
    logger.info(f"Saved overall best model ({best_model_name}) to: {best_model_path}")

    scaler = best_pipe.named_steps.get("scaler")
    if scaler is not None:
        scaler_path = models_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved fitted StandardScaler to: {scaler_path}")

    features_json_path = models_dir / "feature_names.json"
    with open(features_json_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved feature names list to: {features_json_path}")

    results_csv_path = models_dir / "model_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)

    report_csv_path = reports_dir / "model_tuning_results.csv"
    results_df.to_csv(report_csv_path, index=False)
    logger.info(f"Saved tuning results table to: {results_csv_path} and {report_csv_path}")

    metrics_json_path = models_dir / "model_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved metrics summary to: {metrics_json_path}")


def main():
    logger.info("=================================================================")
    logger.info("HarmoniX Phase 1 | Machine Learning Modeling & Tuning (Bhoomi)")
    logger.info("=================================================================")

    df = load_dataset(MLConfig.FEATURES_PATH)
    feature_cols = get_feature_columns(df)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_leakage_free_splits(
        df, feature_cols, MLConfig.TARGET_COL
    )

    cv = create_balanced_cv_splitter(y_train, n_splits=5)

    models: Dict[str, Pipeline] = {}
    best_params: Dict[str, Dict[str, Any]] = {}
    cv_scores: Dict[str, float] = {}

    ridge_pipe, ridge_params, ridge_cv_rmse = train_and_tune_ridge(X_train, y_train, cv)
    models["Ridge"] = ridge_pipe
    best_params["Ridge"] = ridge_params
    cv_scores["Ridge"] = ridge_cv_rmse

    rf_pipe, rf_params, rf_cv_rmse = train_and_tune_random_forest(X_train, y_train, cv)
    models["Random_Forest"] = rf_pipe
    best_params["Random_Forest"] = rf_params
    cv_scores["Random_Forest"] = rf_cv_rmse

    xgb_pipe, xgb_params, xgb_cv_rmse = train_and_tune_xgboost(X_train, y_train, cv)
    models["XGBoost"] = xgb_pipe
    best_params["XGBoost"] = xgb_params
    cv_scores["XGBoost"] = xgb_cv_rmse

    logger.info("\n=================================================================")
    logger.info("Validation Set Evaluation (Model Selection Benchmark)")
    logger.info("=================================================================")

    comparison_records = []
    metrics_summary: Dict[str, Any] = {
        "target": MLConfig.TARGET_COL,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test_reserved": len(X_test),
        "models": {}
    }

    for name, pipe in models.items():
        y_val_pred = predict_bounded(pipe, X_val)
        val_metrics = compute_metrics(y_val, y_val_pred)

        record = {
            "Model": name,
            "CV_RMSE": round(cv_scores[name], 4),
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "Val_Pearson_r": val_metrics["Pearson_r"],
            "Best_Params": str(best_params[name])
        }
        comparison_records.append(record)

        metrics_summary["models"][name] = {
            "cv_rmse": round(cv_scores[name], 4),
            "val_metrics": val_metrics,
            "best_params": {
                k.replace("model__", ""): v for k, v in best_params[name].items()
            }
        }

        logger.info(
            f"{name.upper()}: "
            f"CV RMSE={cv_scores[name]:.4f} | "
            f"Val MAE={val_metrics['MAE']:.4f} | "
            f"Val RMSE={val_metrics['RMSE']:.4f} | "
            f"Val R2={val_metrics['R2']:.4f} | "
            f"Val Pearson r={val_metrics['Pearson_r']:.4f}"
        )

    results_df = pd.DataFrame(comparison_records)

    best_row = results_df.sort_values(by=["Val_RMSE", "Val_MAE"]).iloc[0]
    best_model_name = best_row["Model"]
    metrics_summary["best_model"] = best_model_name

    logger.info("\n=================================================================")
    logger.info(f"🏆 Winning Model Selected: {best_model_name}")
    logger.info(f"   Validation RMSE: {best_row['Val_RMSE']:.4f} (MAE: {best_row['Val_MAE']:.4f})")
    logger.info("=================================================================\n")
    print(results_df.to_string(index=False))

    save_artifacts(models, best_model_name, feature_cols, results_df, metrics_summary)
    logger.info("All model artifacts and tuning results successfully saved!")


if __name__ == "__main__":
    main()
