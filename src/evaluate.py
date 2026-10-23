"""
HarmoniX | Phase 1 Model Evaluation, Benchmark & Residual Diagnostics
Author: Vani (Evaluation Lead)

This module handles:
1. Loading the untouched held-out test split (116 songs / 15% of PMEmo2019).
2. Loading serialized models (Ridge, Random Forest, XGBoost, Best Model).
3. Calculating test evaluation metrics:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Coefficient of Determination (R²)
   - Pearson Correlation (r) with 95% Fisher-z confidence interval and p-value
   - Spearman Rank Correlation (rho)
   - Explained Variance Score (EVS)
4. Multi-split comparison: Cross-Validation vs. Validation vs. Untouched Test set.
5. Generating publication-quality diagnostic visualizations (Figures 16-19):
   - Fig 16: Actual vs. Predicted Valence Scatter Plots
   - Fig 17: Residual Analysis (Residuals vs. Fitted) detecting regression to mean
   - Fig 18: Residual Normality & Q-Q Plots
   - Fig 19: Cross-Split Generalization Benchmark Bar Chart
6. Exporting results to CSV and JSON for IEEE report and presentation slides.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
    max_error
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HarmoniX.Evaluate")


class EvalConfig:
    """Project directory paths and evaluation settings."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_physics.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
    
    TARGET_COL = "valence"
    SPLIT_COL = "split"
    RANDOM_SEED = 42

    # Aesthetic styling for publication-grade figures
    PALETTE = {
        "Ridge": "#3b82f6",          # Vibrant Blue
        "Random_Forest": "#10b981",  # Emerald Green
        "XGBoost": "#f59e0b",        # Amber Gold
        "Actual": "#0f172a",         # Deep Slate
        "Ideal": "#ef4444"           # Red dashed guide
    }


def set_publication_style():
    """Apply elegant, publication-grade matplotlib parameters."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.2,
        "grid.color": "#f1f5f9",
        "grid.linestyle": "--",
        "grid.alpha": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "figure.dpi": 300
    })


def load_dataset_splits(features_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load physics features and separate into Train, Val, and Test splits."""
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at: {features_path}")

    df = pd.read_csv(features_path)
    logger.info(f"Loaded feature dataset: {df.shape} from {features_path.name}")

    feature_cols = [
        col for col in df.columns
        if col.startswith(("mfcc_", "spectral_", "rms_", "zcr_", "chroma_"))
        or col in ("hnr_estimate_db", "sensory_dissonance_estimate")
    ]
    logger.info(f"Identified {len(feature_cols)} acoustic physics features.")

    train_df = df[df[EvalConfig.SPLIT_COL] == "train"].copy().reset_index(drop=True)
    val_df = df[df[EvalConfig.SPLIT_COL] == "val"].copy().reset_index(drop=True)
    test_df = df[df[EvalConfig.SPLIT_COL] == "test"].copy().reset_index(drop=True)

    logger.info(
        f"Splits verified: Train={len(train_df)}, Val={len(val_df)}, "
        f"Untouched Test={len(test_df)} (Sum = {len(train_df) + len(val_df) + len(test_df)})"
    )
    return train_df, val_df, test_df, feature_cols


def load_trained_models(models_dir: Path) -> Dict[str, Any]:
    """Load all serialized scikit-learn / XGBoost pipeline models."""
    models = {}
    model_files = {
        "Ridge": models_dir / "ridge_best.joblib",
        "Random_Forest": models_dir / "random_forest_best.joblib",
        "XGBoost": models_dir / "xgboost_best.joblib"
    }

    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Trained model artifact missing: {path}")
        models[name] = joblib.load(path)
        logger.info(f"Loaded model artifact: {name} from {path.name}")

    return models


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute rigorous evaluation metrics on ground truth vs predicted valence.
    Includes Pearson r with Fisher-z 95% confidence intervals and p-value.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)

    # Pearson correlation with Fisher-z transformation for confidence interval
    r_val, p_val = stats.pearsonr(y_true, y_pred)
    z = np.arctanh(r_val)
    se = 1.0 / np.sqrt(len(y_true) - 3)
    z_crit = 1.95996  # 95% CI
    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)

    # Spearman rank correlation
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)

    residuals = y_true - y_pred
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "Pearson_r": round(float(r_val), 4),
        "Pearson_p": float(p_val),
        "Pearson_CI_low": round(float(ci_low), 4),
        "Pearson_CI_high": round(float(ci_high), 4),
        "Spearman_rho": round(float(spearman_rho), 4),
        "Spearman_p": float(spearman_p),
        "Explained_Variance": round(float(evs), 4),
        "Max_Error": round(float(max_err), 4),
        "Residual_Mean": round(float(res_mean), 4),
        "Residual_Std": round(float(res_std), 4)
    }


def generate_figure_16_actual_vs_predicted(
    predictions_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    test_metrics: Dict[str, Dict[str, float]],
    output_path: Path
):
    """
    Figure 16: Actual vs. Predicted Valence Scatter Plots.
    Demonstrates model alignment against the ideal 45-degree parity diagonal.
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, sharex=True)
    models = ["Ridge", "Random_Forest", "XGBoost"]
    display_names = ["Ridge Regression (Baseline)", "Random Forest Regressor", "XGBoost Regressor (Selected)"]

    for i, (model_name, ax) in enumerate(zip(models, axes)):
        y_pred = predictions_dict[model_name]
        m = test_metrics[model_name]
        color = EvalConfig.PALETTE[model_name]

        # Calculate absolute error for point coloring
        abs_err = np.abs(y_test - y_pred)

        scatter = ax.scatter(
            y_test, y_pred,
            c=abs_err, cmap="viridis",
            s=55, alpha=0.85, edgecolors="none"
        )

        # 45-degree ideal diagonal line (y = x)
        limits = [min(np.min(y_test), np.min(y_pred)) - 0.05, max(np.max(y_test), np.max(y_pred)) + 0.05]
        ax.plot(limits, limits, "--", color="#ef4444", linewidth=1.8, label="Ideal Parity ($y = \hat{y}$)")

        # Linear regression trendline through predictions
        slope, intercept, _, _, _ = stats.linregress(y_test, y_pred)
        x_vals = np.linspace(limits[0], limits[1], 100)
        ax.plot(x_vals, slope * x_vals + intercept, color="#0f172a", linewidth=1.6, label=f"Fit (Slope = {slope:.2f})")

        ax.set_title(display_names[i], pad=10)
        ax.set_xlabel("True Human Valence (PMEmo Test Ground Truth)")
        if i == 0:
            ax.set_ylabel("Predicted Valence ($\hat{y}$)")

        # Metric annotation text box
        text_str = (
            f"Test $R^2$ = {m['R2']:.4f}\n"
            f"Pearson $r$ = {m['Pearson_r']:.4f}\n"
            f"RMSE = {m['RMSE']:.4f}\n"
            f"MAE = {m['MAE']:.4f}"
        )
        ax.text(
            0.05, 0.95, text_str,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cbd5e1", alpha=0.92)
        )
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)
        ax.set_xlim(limits)
        ax.set_ylim(limits)

    fig.suptitle("Figure 16: Held-Out Test Set — True vs. Predicted Musical Valence ($N = 116$ Songs)", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Figure 16 to {output_path}")


def generate_figure_17_residual_analysis(
    predictions_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: Path
):
    """
    Figure 17: Residual Plots (Residuals e = y - y_pred vs Fitted y_pred).
    Crucial for diagnosing homoscedasticity vs heteroscedasticity and regression to mean.
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    models = ["Ridge", "Random_Forest", "XGBoost"]
    display_names = ["Ridge Regression", "Random Forest Regressor", "XGBoost Regressor"]

    for i, (model_name, ax) in enumerate(zip(models, axes)):
        y_pred = predictions_dict[model_name]
        residuals = y_test - y_pred
        color = EvalConfig.PALETTE[model_name]

        # Scatter plot of residuals
        ax.scatter(y_pred, residuals, color=color, s=45, alpha=0.75, edgecolors="white", linewidth=0.5)

        # Zero reference line
        ax.axhline(0, color="#ef4444", linestyle="--", linewidth=1.8, label="Zero Error ($e = 0$)")

        # Standard deviation bounds (+- 2 sigma)
        sigma = np.std(residuals)
        ax.axhline(2 * sigma, color="#94a3b8", linestyle=":", linewidth=1.2, label="$\pm 2\sigma$ Bound")
        ax.axhline(-2 * sigma, color="#94a3b8", linestyle=":", linewidth=1.2)

        # Lowess / Polynomial trend curve to test non-linearity
        z = np.polyfit(y_pred, residuals, 2)
        p = np.poly1d(z)
        x_sort = np.sort(y_pred)
        ax.plot(x_sort, p(x_sort), color="#0f172a", linewidth=2.0, label="Residual Trend")

        ax.set_title(display_names[i], pad=10)
        ax.set_xlabel("Predicted Valence ($\hat{y}$)")
        if i == 0:
            ax.set_ylabel("Residual Error ($y - \hat{y}$)")

        # Annotation highlighting regression to the mean
        text_str = (
            f"Residual $\mu$ = {np.mean(residuals):.4f}\n"
            f"Residual $\sigma$ = {sigma:.4f}\n"
            f"Outliers ($>2\sigma$) = {np.sum(np.abs(residuals) > 2*sigma)}"
        )
        ax.text(
            0.05, 0.95, text_str,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cbd5e1", alpha=0.92)
        )
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    fig.suptitle("Figure 17: Held-Out Test Set — Residual Error Diagnostics ($y - \hat{y}$ vs. $\hat{y}$)", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Figure 17 to {output_path}")


def generate_figure_18_residual_distribution_qq(
    predictions_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: Path
):
    """
    Figure 18: Residual Normality Distribution and Q-Q Plots.
    Validates whether the prediction residuals adhere to Gaussian assumptions.
    """
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    models = ["Ridge", "Random_Forest", "XGBoost"]
    display_names = ["Ridge Regression", "Random Forest Regressor", "XGBoost Regressor"]

    for i, model_name in enumerate(models):
        y_pred = predictions_dict[model_name]
        residuals = y_test - y_pred
        color = EvalConfig.PALETTE[model_name]

        # Top row: Error Histogram & KDE vs Theoretical Normal
        ax_hist = axes[0, i]
        sns.histplot(
            residuals, kde=True, ax=ax_hist, color=color,
            stat="density", bins=16, alpha=0.5, edgecolor="white"
        )
        # Overlay theoretical normal distribution
        mu, std = stats.norm.fit(residuals)
        xmin, xmax = ax_hist.get_xlim()
        x_axis = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x_axis, mu, std)
        ax_hist.plot(x_axis, p, "k--", linewidth=1.5, label=f"Norm ($\mu$={mu:.2f}, $\sigma$={std:.2f})")

        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = stats.shapiro(residuals)

        ax_hist.set_title(f"{display_names[i]} — Error Distribution", pad=10)
        ax_hist.set_xlabel("Residual ($y - \hat{y}$)")
        if i == 0:
            ax_hist.set_ylabel("Probability Density")
        ax_hist.text(
            0.05, 0.95, f"Shapiro-Wilk: W={shapiro_stat:.3f} (p={shapiro_p:.3f})",
            transform=ax_hist.transAxes, fontsize=9.0, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1", alpha=0.9)
        )
        ax_hist.legend(loc="upper right", frameon=True)

        # Bottom row: Q-Q Plot
        ax_qq = axes[1, i]
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        ax_qq.plot(osm, osr, "o", color=color, alpha=0.8, markersize=5)
        ax_qq.plot(osm, slope * np.array(osm) + intercept, "r--", linewidth=1.8, label=f"Ref Line ($R^2$={r**2:.3f})")

        ax_qq.set_title(f"{display_names[i]} — Normal Q-Q Plot", pad=10)
        ax_qq.set_xlabel("Theoretical Normal Quantiles")
        if i == 0:
            ax_qq.set_ylabel("Sample Ordered Residuals")
        ax_qq.legend(loc="lower right", frameon=True)

    fig.suptitle("Figure 18: Residual Normality Diagnostics — Histograms & Q-Q Plots ($N = 116$ Test Clips)", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Figure 18 to {output_path}")


def generate_figure_19_generalization_comparison(
    cv_val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Dict[str, float]],
    output_path: Path
):
    """
    Figure 19: Multi-Split Generalization Comparison (Train CV vs. Val vs. Untouched Test).
    Demonstrates model stability across held-out splits.
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    models = ["Ridge", "Random_Forest", "XGBoost"]
    model_labels = ["Ridge", "Random Forest", "XGBoost"]
    x = np.arange(len(models))
    width = 0.25

    # 1. RMSE Across Splits (Lower is better)
    cv_rmse = [cv_val_metrics["models"][m]["cv_rmse"] for m in models]
    val_rmse = [cv_val_metrics["models"][m]["val_metrics"]["RMSE"] for m in models]
    test_rmse = [test_metrics[m]["RMSE"] for m in models]

    ax1.bar(x - width, cv_rmse, width, label="5-Fold CV (Train, n=536)", color="#94a3b8")
    ax1.bar(x, val_rmse, width, label="Validation Split (n=115)", color="#3b82f6")
    ax1.bar(x + width, test_rmse, width, label="Untouched Test (n=116)", color="#10b981")

    ax1.set_title("Root Mean Squared Error (RMSE) Across Partitions", pad=10)
    ax1.set_ylabel("RMSE (Valence Scale 0 to 1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0.08, 0.15)

    # Add numeric labels above bars
    for i in range(len(models)):
        ax1.text(i - width, cv_rmse[i] + 0.0015, f"{cv_rmse[i]:.4f}", ha="center", fontsize=8.5)
        ax1.text(i, val_rmse[i] + 0.0015, f"{val_rmse[i]:.4f}", ha="center", fontsize=8.5)
        ax1.text(i + width, test_rmse[i] + 0.0015, f"{test_rmse[i]:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    # 2. R² and Pearson r on Test Split
    test_r2 = [test_metrics[m]["R2"] for m in models]
    test_pearson = [test_metrics[m]["Pearson_r"] for m in models]

    ax2.bar(x - width/2, test_r2, width, label="Test $R^2$ (Variance Explained)", color="#f59e0b")
    ax2.bar(x + width/2, test_pearson, width, label="Test Pearson Correlation ($r$)", color="#6366f1")

    ax2.set_title("Held-Out Test Fit Metrics ($R^2$ and Pearson $r$)", pad=10)
    ax2.set_ylabel("Score (Higher is better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.set_ylim(0.3, 0.8)

    for i in range(len(models)):
        ax2.text(i - width/2, test_r2[i] + 0.012, f"{test_r2[i]:.4f}", ha="center", fontsize=8.5)
        ax2.text(i + width/2, test_pearson[i] + 0.012, f"{test_pearson[i]:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("Figure 19: Model Generalization Across Train CV, Validation, and Held-Out Test Splits", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Figure 19 to {output_path}")


def run_test_evaluation():
    """Execute complete end-to-end evaluation benchmark."""
    logger.info("=" * 70)
    logger.info("HarmoniX Phase 1 — Held-Out Test Evaluation & Benchmark Engine")
    logger.info("Author: Vani (Evaluation & Report Lead)")
    logger.info("=" * 70)

    train_df, val_df, test_df, feature_cols = load_dataset_splits(EvalConfig.FEATURES_PATH)
    models = load_trained_models(EvalConfig.MODELS_DIR)

    # Load Bhoomi's tuning results from model_metrics.json
    metrics_json_path = EvalConfig.MODELS_DIR / "model_metrics.json"
    if metrics_json_path.exists():
        with open(metrics_json_path, "r") as f:
            cv_val_metrics = json.load(f)
    else:
        cv_val_metrics = {"models": {}}

    X_test = test_df[feature_cols].values
    y_test = test_df[EvalConfig.TARGET_COL].values

    predictions = {}
    test_metrics = {}

    for name, pipeline in models.items():
        logger.info(f"Evaluating {name} on {len(X_test)} untouched test clips...")
        y_pred = pipeline.predict(X_test)
        
        # Clip to valid domain bounds [0.0, 1.0] if necessary
        y_pred_bounded = np.clip(y_pred, 0.0, 1.0)
        predictions[name] = y_pred_bounded

        metrics = compute_comprehensive_metrics(y_test, y_pred_bounded)
        test_metrics[name] = metrics

    # Build master benchmark comparison DataFrame
    benchmark_rows = []
    for name in ["Ridge", "Random_Forest", "XGBoost"]:
        m_test = test_metrics[name]
        cv_rmse = cv_val_metrics.get("models", {}).get(name, {}).get("cv_rmse", np.nan)
        val_m = cv_val_metrics.get("models", {}).get(name, {}).get("val_metrics", {})

        benchmark_rows.append({
            "Model": name,
            "CV_RMSE": cv_rmse,
            "Val_MAE": val_m.get("MAE", np.nan),
            "Val_RMSE": val_m.get("RMSE", np.nan),
            "Val_R2": val_m.get("R2", np.nan),
            "Val_Pearson_r": val_m.get("Pearson_r", np.nan),
            "Test_MAE": m_test["MAE"],
            "Test_RMSE": m_test["RMSE"],
            "Test_R2": m_test["R2"],
            "Test_Pearson_r": m_test["Pearson_r"],
            "Test_Pearson_CI": f"[{m_test['Pearson_CI_low']}, {m_test['Pearson_CI_high']}]",
            "Test_Spearman_rho": m_test["Spearman_rho"],
            "Test_EVS": m_test["Explained_Variance"],
            "Test_Max_Error": m_test["Max_Error"]
        })

    benchmark_df = pd.DataFrame(benchmark_rows)

    # Save CSV outputs
    test_results_path = EvalConfig.REPORTS_DIR / "test_evaluation_results.csv"
    benchmark_df.to_csv(test_results_path, index=False)
    logger.info(f"Saved test evaluation table to: {test_results_path}")

    # Save detailed JSON metrics
    test_json_path = EvalConfig.MODELS_DIR / "test_metrics.json"
    with open(test_json_path, "w") as f:
        json.dump({
            "target": EvalConfig.TARGET_COL,
            "n_test": len(X_test),
            "metrics": test_metrics,
            "models_evaluated": list(models.keys()),
            "best_test_model": max(test_metrics, key=lambda k: test_metrics[k]["R2"])
        }, f, indent=2)
    logger.info(f"Saved test metrics JSON to: {test_json_path}")

    # Generate diagnostic figures
    fig16_path = EvalConfig.FIGURES_DIR / "16_actual_vs_predicted.png"
    generate_figure_16_actual_vs_predicted(predictions, y_test, test_metrics, fig16_path)

    fig17_path = EvalConfig.FIGURES_DIR / "17_residual_analysis.png"
    generate_figure_17_residual_analysis(predictions, y_test, fig17_path)

    fig18_path = EvalConfig.FIGURES_DIR / "18_residual_distribution_qq.png"
    generate_figure_18_residual_distribution_qq(predictions, y_test, fig18_path)

    fig19_path = EvalConfig.FIGURES_DIR / "19_model_generalization_comparison.png"
    generate_figure_19_generalization_comparison(cv_val_metrics, test_metrics, fig19_path)

    # Print summary table
    print("\n" + "=" * 95)
    print("HARMONIX PHASE 1 — TEST SET EVALUATION BENCHMARK (N = 116 SONGS)")
    print("=" * 95)
    print(f"{'Model':<16} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R²':<10} | {'Test Pearson r':<16} | {'Test Spearman ρ':<15}")
    print("-" * 95)
    for row in benchmark_rows:
        print(f"{row['Model']:<16} | {row['Test_MAE']:<10.4f} | {row['Test_RMSE']:<10.4f} | {row['Test_R2']:<10.4f} | {row['Test_Pearson_r']:<16.4f} | {row['Test_Spearman_rho']:<15.4f}")
    print("=" * 95)

    best_test = max(test_metrics, key=lambda k: test_metrics[k]["R2"])
    print(f"\nTop Performing Model on Held-Out Test Data: {best_test.upper()} (R² = {test_metrics[best_test]['R2']:.4f}, r = {test_metrics[best_test]['Pearson_r']:.4f})")
    print(f"Figures successfully written to: {EvalConfig.FIGURES_DIR}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_test_evaluation()
