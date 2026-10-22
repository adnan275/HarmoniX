"""
HarmoniX | Phase 1 Feature Analysis & Explainability
Author: Praanshu Ranjan

This module handles:
1. Loading the physics/acoustic feature table (features_physics.csv) and the tuned models.
2. Pearson and Spearman correlation of every acoustic feature with valence, with
   95% confidence intervals, Benjamini-Hochberg FDR correction across the 34 tests,
   and a partial Spearman correlation that controls for arousal.
3. Feature-feature correlation heatmaps (redundancy / multicollinearity check).
4. Focused analyses: sensory dissonance vs valence and spectral centroid vs valence,
   plus a validity check of the dissonance estimator against Plomp-Levelt theory.
5. Tree-model feature importance (impurity / gain) and permutation importance.
6. SHAP explanations of the tuned Random Forest and XGBoost models.
7. Publication-ready figures (reports/figures/) and result tables (reports/).

Leakage policy:
- Correlation statistics are computed on the training split only.
- Permutation importance and SHAP values are computed on the validation split.
- The test split is never read here; it stays reserved for the final evaluation.

All correlations are statistical associations. They do not show that a feature causes
higher or lower perceived valence.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HarmoniX.FeatureAnalysis")


class AnalysisConfig:
    """Project paths and analysis parameters."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_physics.csv"
    METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_matched_metadata.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

    TARGET_COL = "valence"
    COVARIATE_COL = "arousal"
    SPLIT_COL = "split"
    STATS_SPLIT = "train"
    EXPLAIN_SPLIT = "val"

    # Tuned pipelines saved by train_ml.py (StandardScaler -> model)
    TREE_MODEL_FILES = {
        "Random_Forest": "random_forest_best.joblib",
        "XGBoost": "xgboost_best.joblib",
    }

    RANDOM_STATE = 42
    N_PERMUTATION_REPEATS = 30
    FDR_ALPHA = 0.05
    N_DISSONANCE_PROBE_SONGS = 200
    TOP_K = 15
    DPI = 300


# Figure styling: one blue/orange pair for two-series charts, a blue-grey-red diverging
# map for signed correlations, and a single-hue blue ramp for magnitudes.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "harmonix_diverging", ["#184f95", "#5598e7", "#f0efec", "#e66767", "#b02e2e"]
)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "harmonix_sequential", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

FEATURE_LABELS = {
    "spectral_centroid_mean": "Spectral centroid",
    "spectral_bandwidth_mean": "Spectral bandwidth",
    "spectral_rolloff_mean": "Spectral roll-off",
    "spectral_contrast_mean": "Spectral contrast",
    "spectral_flatness_mean": "Spectral flatness",
    "rms_mean": "RMS energy",
    "zcr_mean": "Zero-crossing rate",
    "hnr_estimate_db": "HNR (est., dB)",
    "sensory_dissonance_estimate": "Sensory dissonance (est.)",
    "valence": "Valence",
    "arousal": "Arousal",
}

# Non-MFCC, non-chroma descriptors with a direct physical reading
PHYSICS_FEATURES = [
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "spectral_contrast_mean",
    "spectral_flatness_mean",
    "rms_mean",
    "zcr_mean",
    "hnr_estimate_db",
    "sensory_dissonance_estimate",
]

FEATURE_GROUPS = [
    "MFCC (timbre)",
    "Spectral shape",
    "Energy & temporal",
    "Chroma (pitch class)",
    "Harmonicity & dissonance",
]


def feature_label(feature: str) -> str:
    """Return a short, human-readable label for a feature column."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("mfcc_"):
        return f"MFCC {feature.split('_')[1]}"
    if feature.startswith("chroma_"):
        index = int(feature.split("_")[1]) - 1
        return f"Chroma {PITCH_CLASSES[index]}"
    return feature


def feature_group(feature: str) -> str:
    """Assign a feature column to its acoustic family."""
    if feature.startswith("mfcc_"):
        return "MFCC (timbre)"
    if feature.startswith("spectral_"):
        return "Spectral shape"
    if feature in ("rms_mean", "zcr_mean"):
        return "Energy & temporal"
    if feature.startswith("chroma_"):
        return "Chroma (pitch class)"
    return "Harmonicity & dissonance"


def apply_plot_style() -> None:
    """Shared matplotlib style: recessive hairline grid and axes, dark text."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "text.color": INK,
    })


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure to reports/figures/ at print resolution and close it."""
    AnalysisConfig.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = AnalysisConfig.FIGURES_DIR / filename
    fig.savefig(path, dpi=AnalysisConfig.DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path.relative_to(AnalysisConfig.PROJECT_ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(features_path: Path) -> pd.DataFrame:
    """Load the extracted physics features dataset."""
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found at: {features_path}\n"
            "Run data_loader.py and feature_extraction.py first."
        )
    df = pd.read_csv(features_path)
    logger.info(f"Loaded dataset from {features_path.name} with shape {df.shape}")

    for column in (AnalysisConfig.TARGET_COL, AnalysisConfig.SPLIT_COL):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in {features_path.name}.")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the acoustic feature columns in the order the models were trained on.
    Uses models/feature_names.json when present so the analysis matches train_ml.py.
    """
    names_path = AnalysisConfig.MODELS_DIR / "feature_names.json"
    if names_path.exists():
        with open(names_path) as f:
            feature_cols = json.load(f)
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Feature columns listed in {names_path.name} are missing from the "
                f"feature table: {missing}. Re-run feature_extraction.py and train_ml.py."
            )
    else:
        feature_cols = [
            col for col in df.columns
            if col.startswith(("mfcc_", "spectral_", "rms_", "zcr_", "chroma_"))
            or col in ("hnr_estimate_db", "sensory_dissonance_estimate")
        ]
    logger.info(f"Analysing {len(feature_cols)} acoustic features.")
    return feature_cols


def get_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return one named partition of the dataset."""
    split_df = df[df[AnalysisConfig.SPLIT_COL] == split_name].reset_index(drop=True)
    if len(split_df) == 0:
        raise ValueError(f"Split '{split_name}' is empty. Check split generation.")
    return split_df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def fisher_confidence_interval(r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Confidence interval for a Pearson correlation via the Fisher z-transform."""
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.5 + confidence / 2.0)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs are left in place."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    valid = np.isfinite(p_values)
    if not valid.any():
        return q_values

    p = p_values[valid]
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p-value downwards
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(scaled, 0.0, 1.0)
    q_values[valid] = adjusted
    return q_values


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    Spearman correlation between x and y after removing the linear effect of z
    from the ranks of both. Answers: does the feature still track valence among
    songs of similar arousal?
    """
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    design = np.column_stack([np.ones_like(rz), rz])

    def residualise(values: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coef

    res_x, res_y = residualise(rx), residualise(ry)
    if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
        return np.nan, np.nan

    r = float(np.corrcoef(res_x, res_y)[0, 1])
    dof = len(x) - 3
    if dof <= 0:
        return r, np.nan
    t_stat = r * np.sqrt(dof / max(1e-12, 1.0 - r ** 2))
    p = float(2.0 * stats.t.sf(abs(t_stat), dof))
    return r, p


def compute_valence_correlations(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Pearson and Spearman correlation of each feature with valence, plus 95% CI,
    FDR-adjusted q-values and the arousal-controlled partial Spearman correlation.
    """
    target = AnalysisConfig.TARGET_COL
    covariate = AnalysisConfig.COVARIATE_COL
    has_covariate = covariate in df.columns

    records = []
    for feature in feature_cols:
        columns = [feature, target] + ([covariate] if has_covariate else [])
        subset = df[columns].dropna()
        x = subset[feature].to_numpy(dtype=float)
        y = subset[target].to_numpy(dtype=float)
        n = len(subset)

        record = {
            "feature": feature,
            "label": feature_label(feature),
            "group": feature_group(feature),
            "n": n,
            "pearson_r": np.nan, "pearson_ci_low": np.nan, "pearson_ci_high": np.nan,
            "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan,
            "partial_spearman_rho_given_arousal": np.nan,
            "partial_spearman_p_given_arousal": np.nan,
        }

        if n >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            r, p_r = stats.pearsonr(x, y)
            rho, p_rho = stats.spearmanr(x, y)
            ci_low, ci_high = fisher_confidence_interval(r, n)
            record.update({
                "pearson_r": float(r), "pearson_ci_low": ci_low, "pearson_ci_high": ci_high,
                "pearson_p": float(p_r), "spearman_rho": float(rho), "spearman_p": float(p_rho),
            })
            if has_covariate:
                z = subset[covariate].to_numpy(dtype=float)
                partial_rho, partial_p = partial_spearman(x, y, z)
                record["partial_spearman_rho_given_arousal"] = partial_rho
                record["partial_spearman_p_given_arousal"] = partial_p
        else:
            logger.warning(f"Skipping correlation for '{feature}': constant or too few values.")

        records.append(record)

    corr_df = pd.DataFrame(records)
    corr_df["pearson_q_fdr"] = benjamini_hochberg(corr_df["pearson_p"].to_numpy())
    corr_df["spearman_q_fdr"] = benjamini_hochberg(corr_df["spearman_p"].to_numpy())
    corr_df["partial_q_fdr"] = benjamini_hochberg(
        corr_df["partial_spearman_p_given_arousal"].to_numpy()
    )
    corr_df["significant_fdr"] = corr_df["spearman_q_fdr"] < AnalysisConfig.FDR_ALPHA

    corr_df = corr_df.reindex(
        corr_df["spearman_rho"].abs().sort_values(ascending=False, na_position="last").index
    ).reset_index(drop=True)
    return corr_df


# ---------------------------------------------------------------------------
# Correlation figures
# ---------------------------------------------------------------------------

def plot_valence_correlations(corr_df: pd.DataFrame, n_songs: int) -> None:
    """Dot plot of Pearson (with 95% CI) and Spearman correlation for every feature."""
    plot_df = corr_df.dropna(subset=["spearman_rho"]).sort_values("spearman_rho")
    y_pos = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(7.5, max(5.0, 0.26 * len(plot_df) + 1.5)))

    ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.hlines(
        y_pos + 0.15, plot_df["pearson_ci_low"], plot_df["pearson_ci_high"],
        color=BLUE, linewidth=1.5, alpha=0.6, zorder=2
    )

    significant = plot_df["spearman_q_fdr"] < AnalysisConfig.FDR_ALPHA
    for values, offset, color, marker in (
        (plot_df["pearson_r"], 0.15, BLUE, "o"),
        (plot_df["spearman_rho"], -0.15, ORANGE, "D"),
    ):
        ax.scatter(
            values[significant], y_pos[significant.to_numpy()] + offset,
            color=color, marker=marker, s=34, zorder=3, edgecolors="white", linewidths=0.6
        )
        ax.scatter(
            values[~significant], y_pos[~significant.to_numpy()] + offset,
            facecolors="white", edgecolors=color, marker=marker, s=30, linewidths=1.3, zorder=3
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Correlation with valence")
    ax.set_title(f"Acoustic features vs valence (training split, n = {n_songs})", pad=12)

    handles = [
        Line2D([0], [0], marker="o", color=BLUE, linewidth=1.5, markeredgecolor="white",
               markersize=7, label="Pearson r (95% CI)"),
        Line2D([0], [0], marker="D", color=ORANGE, linewidth=0, markeredgecolor="white",
               markersize=6, label="Spearman ρ"),
        Line2D([0], [0], marker="o", color=INK_MUTED, linewidth=0, markerfacecolor="white",
               markersize=7, label=f"Hollow: Spearman FDR q ≥ {AnalysisConfig.FDR_ALPHA}"),
    ]
    ax.legend(handles=handles, loc="lower right")
    save_figure(fig, "04_feature_valence_correlations.png")


def cluster_order(corr: pd.DataFrame) -> List[str]:
    """Order features by hierarchical clustering on 1 - |rho| so related features sit together."""
    distance = 1.0 - corr.abs().fillna(0.0).to_numpy()
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)
    tree = linkage(squareform(distance, checks=False), method="average")
    return [corr.columns[i] for i in leaves_list(tree)]


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    filename: str,
    title: str,
    annotate: bool = False,
    reorder: bool = True,
) -> pd.DataFrame:
    """Lower-triangle Spearman correlation heatmap; returns the correlation matrix."""
    corr = df[columns].corr(method="spearman")

    if reorder:
        features = [c for c in columns if c not in (AnalysisConfig.TARGET_COL, AnalysisConfig.COVARIATE_COL)]
        targets = [c for c in columns if c not in features]
        order = cluster_order(corr.loc[features, features]) + targets
        corr = corr.loc[order, order]

    labels = [feature_label(c) for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    data = np.ma.masked_where(mask, corr.to_numpy())

    size = max(6.0, 0.28 * len(columns) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    image = ax.imshow(data, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    tick_size = 9 if len(labels) <= 14 else 7
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_size)
    ax.set_yticklabels(labels, fontsize=tick_size)
    ax.tick_params(length=0)

    # Embolden the target/covariate rows so they stand out in a large matrix
    for tick_labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick in tick_labels:
            if tick.get_text() in ("Valence", "Arousal"):
                tick.set_fontweight("bold")
                tick.set_color(INK)

    if annotate:
        values = corr.to_numpy()
        for i in range(len(labels)):
            for j in range(i + 1):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(
                        j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(value) > 0.6 else INK
                    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Spearman ρ")
    colorbar.outline.set_visible(False)
    ax.set_title(title, pad=12)
    save_figure(fig, filename)
    return corr


def plot_feature_vs_valence(
    df: pd.DataFrame,
    feature: str,
    corr_df: pd.DataFrame,
    filename: str,
    x_label: str,
) -> None:
    """
    Scatter of one feature against valence with an OLS fit (95% CI band) and
    decile-binned means. Points are coloured by arousal to expose confounding.
    """
    target = AnalysisConfig.TARGET_COL
    covariate = AnalysisConfig.COVARIATE_COL
    has_covariate = covariate in df.columns

    subset = df[[feature, target] + ([covariate] if has_covariate else [])].dropna()
    x = subset[feature].to_numpy(dtype=float)
    y = subset[target].to_numpy(dtype=float)
    n = len(subset)
    if n < 5 or np.std(x) < 1e-12:
        logger.warning(f"Not enough variation to plot '{feature}' against valence.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    if has_covariate:
        points = ax.scatter(
            x, y, c=subset[covariate], cmap=SEQUENTIAL_CMAP, s=20, alpha=0.9,
            edgecolors="white", linewidths=0.4, zorder=2
        )
        colorbar = fig.colorbar(points, ax=ax, pad=0.02)
        colorbar.set_label("Arousal")
        colorbar.outline.set_visible(False)
    else:
        ax.scatter(x, y, color=BLUE, s=20, alpha=0.8, edgecolors="white", linewidths=0.4, zorder=2)

    # OLS fit with a 95% confidence band for the mean response
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    fit = intercept + slope * xs
    residual_sd = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (n - 2))
    se_fit = residual_sd * np.sqrt(1.0 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t_crit = stats.t.ppf(0.975, n - 2)
    ax.fill_between(xs, fit - t_crit * se_fit, fit + t_crit * se_fit,
                    color=INK, alpha=0.10, linewidth=0, zorder=3)
    ax.plot(xs, fit, color=INK, linewidth=2, zorder=4)

    # Binned means reveal non-linear shape without assuming a functional form
    n_bins = int(min(10, max(2, n // 15)))
    bins = pd.qcut(subset[feature], q=n_bins, duplicates="drop")
    binned = subset.groupby(bins, observed=True).agg(
        x_mean=(feature, "mean"), y_mean=(target, "mean"), y_sem=(target, "sem")
    )
    ax.errorbar(
        binned["x_mean"], binned["y_mean"], yerr=binned["y_sem"], fmt="D", color=ORANGE,
        markersize=6, markeredgecolor="white", elinewidth=1.5, capsize=0, zorder=5
    )

    row = corr_df.set_index("feature").loc[feature]
    lines = [
        f"Pearson r = {row['pearson_r']:.2f} [{row['pearson_ci_low']:.2f}, {row['pearson_ci_high']:.2f}], "
        f"q = {row['pearson_q_fdr']:.2g}",
        f"Spearman ρ = {row['spearman_rho']:.2f}, q = {row['spearman_q_fdr']:.2g}",
    ]
    if np.isfinite(row["partial_spearman_rho_given_arousal"]):
        lines.append(
            f"Partial ρ | arousal = {row['partial_spearman_rho_given_arousal']:.2f}, "
            f"q = {row['partial_q_fdr']:.2g}"
        )
    ax.text(
        0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        color=INK, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID, alpha=0.92),
        zorder=6
    )

    handles = [
        Line2D([0], [0], color=INK, linewidth=2, label="OLS fit (95% CI)"),
        Line2D([0], [0], marker="D", color=ORANGE, linewidth=1.5, markeredgecolor="white",
               markersize=6, label=f"Binned mean ± SEM ({len(binned)} quantile bins)"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Valence (human-rated)")
    ax.set_title(f"{feature_label(feature)} vs valence (training split, n = {n})", pad=12)
    save_figure(fig, filename)


def sethares_roughness(f1: float, f2: float) -> float:
    """Plomp-Levelt roughness of two equal-amplitude pure tones (Sethares' parameterisation)."""
    scaled_difference = 0.24 * abs(f2 - f1) / (0.0207 * min(f1, f2) + 18.96)
    return float(np.exp(-3.5 * scaled_difference) - np.exp(-5.75 * scaled_difference))


def plot_dissonance_estimator_check() -> None:
    """
    Sanity check of the sensory-dissonance feature on pure-tone dyads with a known answer.
    The estimator is compared with the theoretical Plomp-Levelt curve at two registers;
    where they disagree, the feature cannot be trusted to measure roughness.
    """
    from feature_extraction import SAMPLE_RATE, estimate_sensory_dissonance

    t = np.arange(int(SAMPLE_RATE * 3.0)) / SAMPLE_RATE
    semitones = np.arange(0, 13)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for root, color in ((220.0, BLUE), (440.0, ORANGE)):
        estimated, theoretical = [], []
        for step in semitones:
            upper = root * 2 ** (step / 12)
            dyad = np.sin(2 * np.pi * root * t) + np.sin(2 * np.pi * upper * t)
            dyad = 0.95 * dyad / np.max(np.abs(dyad))
            estimated.append(estimate_sensory_dissonance(dyad, SAMPLE_RATE))
            theoretical.append(sethares_roughness(root, upper) if step > 0 else 0.0)

        ax.plot(semitones, theoretical, color=color, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.plot(semitones, estimated, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white")

    handles = [
        Line2D([0], [0], color=BLUE, linewidth=2, marker="o", markeredgecolor="white",
               label="Estimator, root A3 (220 Hz)"),
        Line2D([0], [0], color=ORANGE, linewidth=2, marker="o", markeredgecolor="white",
               label="Estimator, root A4 (440 Hz)"),
        Line2D([0], [0], color=INK_SECONDARY, linewidth=1.5, linestyle="--",
               label="Plomp-Levelt theory (same root colour)"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xticks(semitones)
    ax.set_xticklabels(["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7", "P8"])
    ax.set_xlabel("Interval above the root (pure-tone dyad)")
    ax.set_ylabel("Sensory dissonance (a.u.)")
    ax.set_title("Validity check: dissonance estimator vs Plomp-Levelt theory", pad=12)
    save_figure(fig, "15_dissonance_estimator_check.png")


def _probe_song(audio_path: str) -> Dict[str, float]:
    """Dissonance-estimator diagnostics for one song (see probe_dissonance_estimator)."""
    import librosa
    from scipy.signal import find_peaks
    from feature_extraction import HOP_LENGTH, N_FFT, SAMPLE_RATE, estimate_sensory_dissonance

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    y = 0.95 * y / max(np.max(np.abs(y)), 1e-8)

    mean_spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)).mean(axis=1)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    peaks, _ = find_peaks(mean_spectrum)
    strongest = peaks[np.argsort(mean_spectrum[peaks])[-30:]]

    window = 2 * sr
    windowed = [
        estimate_sensory_dissonance(y[start:start + window], sr)
        for start in range(0, len(y) - window + 1, window)
    ]
    return {
        "dissonance_whole_clip": estimate_sensory_dissonance(y, sr),
        "dissonance_2s_windows": float(np.nanmean(windowed)) if windowed else np.nan,
        "peak_balance": float(np.mean(mean_spectrum[strongest] / mean_spectrum[strongest].max())),
        "median_peak_hz": float(np.median(frequencies[strongest])),
    }


def probe_dissonance_estimator(stats_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    """
    What does the dissonance estimate track on real music? For a sample of training songs,
    compare the whole-clip estimate with (a) the same estimator on 2-second windows, where
    only notes that sound together can interact, and (b) "peak balance", the mean relative
    amplitude of the 30 strongest peaks of the time-averaged spectrum. The estimator weights
    each peak pair by these relative amplitudes, so a strong link to (b) means the feature
    mostly reflects how evenly energy is spread across peaks rather than interval roughness.
    """
    from joblib import Parallel, delayed

    if not AnalysisConfig.METADATA_PATH.exists():
        logger.warning("Audio metadata not found; skipping the dissonance estimator probe.")
        return None

    metadata = pd.read_csv(AnalysisConfig.METADATA_PATH)[["music_id", "audio_path"]]
    sample = stats_df[["music_id", AnalysisConfig.TARGET_COL]].merge(metadata, on="music_id")
    sample = sample[sample["audio_path"].map(lambda p: Path(str(p)).exists())]
    if len(sample) < 10:
        logger.warning("Too few audio files available; skipping the dissonance estimator probe.")
        return None

    sample = sample.sample(
        min(AnalysisConfig.N_DISSONANCE_PROBE_SONGS, len(sample)), random_state=AnalysisConfig.RANDOM_STATE
    ).reset_index(drop=True)
    logger.info(f"Probing the dissonance estimator on {len(sample)} training songs...")
    measurements = Parallel(n_jobs=-1)(delayed(_probe_song)(path) for path in sample["audio_path"])
    probe_df = pd.concat([sample.drop(columns="audio_path"), pd.DataFrame(measurements)], axis=1)
    probe_df.to_csv(AnalysisConfig.REPORTS_DIR / "dissonance_estimator_probe.csv", index=False)

    pairs = [
        ("dissonance_whole_clip", "peak_balance"),
        ("dissonance_whole_clip", "dissonance_2s_windows"),
        ("dissonance_whole_clip", AnalysisConfig.TARGET_COL),
        ("dissonance_2s_windows", AnalysisConfig.TARGET_COL),
        ("peak_balance", AnalysisConfig.TARGET_COL),
    ]
    correlations = []
    for x_col, y_col in pairs:
        subset = probe_df[[x_col, y_col]].dropna()
        rho, p = stats.spearmanr(subset[x_col], subset[y_col])
        correlations.append({"x": x_col, "y": y_col, "n": len(subset), "spearman_rho": rho, "p": p})
    return {
        "n_songs": len(probe_df),
        "median_peak_hz": float(probe_df["median_peak_hz"].median()),
        "correlations": pd.DataFrame(correlations),
    }


# ---------------------------------------------------------------------------
# Model-based importance and SHAP
# ---------------------------------------------------------------------------

def load_tree_models(n_features: int) -> Dict[str, Pipeline]:
    """Load the tuned tree-model pipelines saved by train_ml.py."""
    models: Dict[str, Pipeline] = {}
    for name, filename in AnalysisConfig.TREE_MODEL_FILES.items():
        path = AnalysisConfig.MODELS_DIR / filename
        if not path.exists():
            logger.warning(f"Model file not found, skipping {name}: {path}")
            continue
        pipe = joblib.load(path)
        n_in = getattr(pipe, "n_features_in_", n_features)
        if n_in != n_features:
            raise ValueError(
                f"{name} expects {n_in} features but the feature table has {n_features}. "
                "Re-run train_ml.py on the current features_physics.csv."
            )
        models[name] = pipe
        logger.info(f"Loaded {name} pipeline from {path.name}")
    return models


def compute_model_importance(
    pipe: Pipeline, X_val: np.ndarray, y_val: np.ndarray, feature_cols: List[str], model_name: str
) -> pd.DataFrame:
    """Native tree importance plus permutation importance (RMSE increase) on the validation split."""
    native = pipe.named_steps["model"].feature_importances_
    perm = permutation_importance(
        pipe, X_val, y_val,
        scoring="neg_root_mean_squared_error",
        n_repeats=AnalysisConfig.N_PERMUTATION_REPEATS,
        random_state=AnalysisConfig.RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "model": model_name,
        "feature": feature_cols,
        "label": [feature_label(f) for f in feature_cols],
        "group": [feature_group(f) for f in feature_cols],
        "native_importance": native,
        "permutation_rmse_increase_mean": perm.importances_mean,
        "permutation_rmse_increase_std": perm.importances_std,
    })


def plot_model_importance(importance_df: pd.DataFrame, model_name: str, filename: str) -> None:
    """Two panels: the model's own importance and validation permutation importance (top-k)."""
    top_k = AnalysisConfig.TOP_K
    native_title = "Impurity importance (MDI)" if model_name == "Random_Forest" else "Gain importance"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.32 * top_k + 1.8))

    native = importance_df.nlargest(top_k, "native_importance").iloc[::-1]
    axes[0].barh(native["label"], native["native_importance"], color=BLUE, height=0.65)
    axes[0].set_xlabel("Normalised importance")
    axes[0].set_title(native_title, fontsize=11, pad=8)

    perm = importance_df.nlargest(top_k, "permutation_rmse_increase_mean").iloc[::-1]
    axes[1].barh(
        perm["label"], perm["permutation_rmse_increase_mean"], xerr=perm["permutation_rmse_increase_std"],
        color=BLUE, height=0.65, error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.0, capsize=0)
    )
    axes[1].axvline(0.0, color=AXIS, linewidth=1.0)
    axes[1].set_xlabel("Increase in validation RMSE when shuffled (mean ± SD)")
    axes[1].set_title(
        f"Permutation importance ({AnalysisConfig.N_PERMUTATION_REPEATS} repeats)", fontsize=11, pad=8
    )

    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Feature importance: {model_name.replace('_', ' ')}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, filename)


def compute_shap_values(pipe: Pipeline, X: np.ndarray) -> Optional[np.ndarray]:
    """
    SHAP values of a tree model, computed on the preprocessed inputs the model actually
    sees (every pipeline step before the model, e.g. imputer and scaler).

    XGBoost uses its built-in TreeSHAP (pred_contribs), which is the same algorithm as
    shap.TreeExplainer but does not break when the shap and xgboost versions drift apart.
    """
    model = pipe.named_steps["model"]
    X_scaled = pipe[:-1].transform(X)

    if model.__class__.__name__ == "XGBRegressor":
        from xgboost import DMatrix

        contributions = model.get_booster().predict(DMatrix(X_scaled), pred_contribs=True)
        return np.asarray(contributions)[:, :-1]  # last column is the bias term

    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed; skipping SHAP analysis (pip install shap).")
        return None

    explainer = shap.TreeExplainer(model)
    return np.asarray(explainer.shap_values(X_scaled))


def plot_shap_summary(
    shap_values: np.ndarray, X_raw: np.ndarray, feature_cols: List[str], model_name: str, filename: str
) -> None:
    """SHAP beeswarm; colour encodes the raw feature value (low = blue, high = red)."""
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed; skipping SHAP summary plot (pip install shap).")
        return

    features = pd.DataFrame(X_raw, columns=[feature_label(f) for f in feature_cols])
    shap.summary_plot(
        shap_values, features=features, max_display=AnalysisConfig.TOP_K,
        cmap=DIVERGING_CMAP, show=False, plot_size=(8.0, 0.36 * AnalysisConfig.TOP_K + 1.8)
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("SHAP value (impact on predicted valence)", fontsize=10)
    ax.set_title(
        f"SHAP summary: {model_name.replace('_', ' ')} (validation split, n = {len(X_raw)})",
        fontsize=12, fontweight="bold", pad=12
    )
    save_figure(fig, filename)


def group_shap_importance(shap_values: np.ndarray, feature_cols: List[str]) -> pd.Series:
    """
    Mean |SHAP| per feature family. Member SHAP values are summed before taking the
    absolute value, which is valid because SHAP values are additive.
    """
    groups = np.array([feature_group(f) for f in feature_cols])
    result = {}
    for group in FEATURE_GROUPS:
        members = groups == group
        if members.any():
            result[group] = float(np.mean(np.abs(shap_values[:, members].sum(axis=1))))
    return pd.Series(result)


def plot_group_importance(group_importance: Dict[str, pd.Series]) -> None:
    """Grouped bars of mean |SHAP| per feature family, one series per model."""
    group_df = pd.DataFrame(group_importance)
    group_df = group_df.loc[group_df.mean(axis=1).sort_values().index]
    colors = [BLUE, ORANGE]
    height = 0.36
    y_pos = np.arange(len(group_df))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, model_name in enumerate(group_df.columns):
        # First model on top so the bar order matches the legend order
        offset = ((len(group_df.columns) - 1) / 2 - i) * (height + 0.04)
        ax.barh(
            y_pos + offset, group_df[model_name], height=height, color=colors[i % len(colors)],
            label=model_name.replace("_", " ")
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_df.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP| of feature family (valence units)")
    ax.set_title("Contribution of each acoustic feature family (validation split)", pad=12)
    ax.legend(loc="lower right")
    save_figure(fig, "13_shap_feature_groups.png")


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_raw: np.ndarray,
    feature_cols: List[str],
    features: List[Tuple[str, str]],
    model_name: str,
) -> None:
    """SHAP dependence panels for the physically motivated features, in physical units."""
    fig, axes = plt.subplots(1, len(features), figsize=(5.4 * len(features), 4.2), squeeze=False)
    for ax, (feature, x_label) in zip(axes[0], features):
        index = feature_cols.index(feature)
        ax.axhline(0.0, color=AXIS, linewidth=1.0, zorder=1)
        ax.scatter(
            X_raw[:, index], shap_values[:, index], color=BLUE, s=20, alpha=0.85,
            edgecolors="white", linewidths=0.4, zorder=2
        )
        ax.set_xlabel(x_label)
        ax.set_title(feature_label(feature), fontsize=11, pad=8)
    axes[0][0].set_ylabel("SHAP value (impact on predicted valence)")
    fig.suptitle(
        f"SHAP dependence: {model_name.replace('_', ' ')} (validation split)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    save_figure(fig, f"14_shap_dependence_{model_name.lower()}.png")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(
    corr_df: pd.DataFrame,
    importance_df: Optional[pd.DataFrame],
    group_importance: Dict[str, pd.Series],
    dissonance_probe: Optional[Dict[str, object]],
    n_stats: int,
    n_explain: int,
) -> None:
    """Write a short Markdown summary of the numbers behind the figures."""
    alpha = AnalysisConfig.FDR_ALPHA
    n_sig = int(corr_df["significant_fdr"].sum())
    lines = [
        "# Feature Analysis & Explainability: Summary",
        "",
        "Generated by `src/feature_analysis.py`. Correlations use the training split "
        f"(n = {n_stats}); permutation importance and SHAP use the validation split "
        f"(n = {n_explain}). The test split is not read.",
        "",
        f"{n_sig} of {corr_df['spearman_rho'].notna().sum()} features have a Spearman correlation "
        f"with valence that survives Benjamini-Hochberg FDR correction at q < {alpha}.",
        "",
        "## Strongest feature-valence associations (training split)",
        "",
        "| Feature | Pearson r [95% CI] | Spearman ρ | Spearman q | Partial ρ given arousal | Partial q |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in corr_df.head(12).iterrows():
        lines.append(
            f"| {row['label']} | {row['pearson_r']:.3f} [{row['pearson_ci_low']:.3f}, "
            f"{row['pearson_ci_high']:.3f}] | {row['spearman_rho']:.3f} | {row['spearman_q_fdr']:.2g} | "
            f"{row['partial_spearman_rho_given_arousal']:.3f} | {row['partial_q_fdr']:.2g} |"
        )

    for feature in ("sensory_dissonance_estimate", "spectral_centroid_mean"):
        row = corr_df.set_index("feature").loc[feature]
        lines += [
            "",
            f"**{row['label']}:** Pearson r = {row['pearson_r']:.3f} "
            f"[{row['pearson_ci_low']:.3f}, {row['pearson_ci_high']:.3f}], "
            f"Spearman ρ = {row['spearman_rho']:.3f} (q = {row['spearman_q_fdr']:.2g}), "
            f"partial ρ given arousal = {row['partial_spearman_rho_given_arousal']:.3f} "
            f"(q = {row['partial_q_fdr']:.2g}).",
        ]

    if dissonance_probe is not None:
        lines += [
            "",
            "## What the sensory-dissonance estimate tracks",
            "",
            f"Probe on {dissonance_probe['n_songs']} randomly sampled training songs "
            "(per-song values in `reports/dissonance_estimator_probe.csv`). Median frequency of the "
            f"30 strongest spectral peaks: {dissonance_probe['median_peak_hz']:.0f} Hz.",
            "",
            "| Quantity | Compared with | Spearman ρ | p |",
            "| :--- | :--- | ---: | ---: |",
        ]
        for _, row in dissonance_probe["correlations"].iterrows():
            lines.append(f"| {row['x']} | {row['y']} | {row['spearman_rho']:.3f} | {row['p']:.2g} |")

    if importance_df is not None and len(importance_df):
        lines += ["", "## Model-based importance (validation split)", ""]
        for model_name, model_df in importance_df.groupby("model", sort=False):
            top_perm = model_df.nlargest(5, "permutation_rmse_increase_mean")
            ranking = ", ".join(
                f"{r['label']} ({r['permutation_rmse_increase_mean']:.4f})" for _, r in top_perm.iterrows()
            )
            lines.append(f"- **{model_name.replace('_', ' ')}**, top permutation importance "
                         f"(RMSE increase): {ranking}")
            if "shap_mean_abs" in model_df:
                top_shap = model_df.nlargest(5, "shap_mean_abs")
                ranking = ", ".join(f"{r['label']} ({r['shap_mean_abs']:.4f})" for _, r in top_shap.iterrows())
                lines.append(f"- **{model_name.replace('_', ' ')}**, top mean |SHAP|: {ranking}")

    if group_importance:
        lines += ["", "## Mean |SHAP| by feature family (validation split)", ""]
        group_df = pd.DataFrame(group_importance)
        lines.append("| Family | " + " | ".join(c.replace("_", " ") for c in group_df.columns) + " |")
        lines.append("| :--- |" + " ---: |" * len(group_df.columns))
        for group, row in group_df.sort_values(group_df.columns[0], ascending=False).iterrows():
            lines.append(f"| {group} | " + " | ".join(f"{v:.4f}" for v in row) + " |")

    lines += [
        "",
        "## Reading these results",
        "",
        "- Correlations and importances are associations in this dataset, not causal effects. "
        "A feature can rank highly because it co-varies with something else listeners respond to.",
        "- Valence and arousal are correlated in PMEmo, so the partial correlation is the better guide "
        "to whether a feature relates to valence beyond general energy/intensity.",
        "- Impurity (MDI) importance is biased towards continuous, high-variance features; permutation "
        "importance and SHAP are the more reliable rankings.",
        "- Strongly correlated features (see the heatmaps) share credit: shuffling one of them barely "
        "hurts the model when another carries the same information, so a low permutation score does "
        "not mean the feature is unrelated to valence.",
        "- HNR and sensory dissonance are simplified signal-based estimates, not validated "
        "psychoacoustic measurements.",
        "- Valence is a proxy for pleasantness-related perception, not a universal measure of musical "
        "pleasantness.",
        "",
    ]

    summary_path = AnalysisConfig.REPORTS_DIR / "feature_analysis_summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info(f"Saved summary: {summary_path.relative_to(AnalysisConfig.PROJECT_ROOT)}")


def main():
    logger.info("=================================================================")
    logger.info("HarmoniX Phase 1 | Feature Analysis & Explainability (Praanshu)")
    logger.info("=================================================================")
    apply_plot_style()
    AnalysisConfig.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load features and partitions (test split deliberately not loaded)
    df = load_dataset(AnalysisConfig.FEATURES_PATH)
    feature_cols = get_feature_columns(df)
    stats_df = get_split(df, AnalysisConfig.STATS_SPLIT)
    explain_df = get_split(df, AnalysisConfig.EXPLAIN_SPLIT)
    logger.info(
        f"Statistics on '{AnalysisConfig.STATS_SPLIT}' split ({len(stats_df)} songs); "
        f"model explanations on '{AnalysisConfig.EXPLAIN_SPLIT}' split ({len(explain_df)} songs)."
    )

    # 2. Feature-valence correlations
    corr_df = compute_valence_correlations(stats_df, feature_cols)
    corr_path = AnalysisConfig.REPORTS_DIR / "feature_valence_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Saved correlation table: {corr_path.relative_to(AnalysisConfig.PROJECT_ROOT)}")
    plot_valence_correlations(corr_df, len(stats_df))

    # 3. Feature-feature correlation heatmaps
    extra_cols = [c for c in (AnalysisConfig.TARGET_COL, AnalysisConfig.COVARIATE_COL) if c in stats_df.columns]
    full_corr = plot_correlation_heatmap(
        stats_df, feature_cols + extra_cols, "05_feature_correlation_heatmap.png",
        "Spearman correlation between acoustic features (clustered, training split)"
    )
    full_corr.to_csv(AnalysisConfig.REPORTS_DIR / "feature_correlation_matrix.csv")
    plot_correlation_heatmap(
        stats_df, PHYSICS_FEATURES + extra_cols, "06_physics_feature_heatmap.png",
        "Physics-informed descriptors, valence and arousal (training split)",
        annotate=True, reorder=False
    )

    # 4. Focused analyses
    plot_feature_vs_valence(
        stats_df, "sensory_dissonance_estimate", corr_df, "07_dissonance_vs_valence.png",
        "Sensory dissonance estimate (spectral-peak roughness, a.u.)"
    )
    plot_feature_vs_valence(
        stats_df, "spectral_centroid_mean", corr_df, "08_spectral_centroid_vs_valence.png",
        "Mean spectral centroid (Hz)"
    )
    plot_dissonance_estimator_check()
    dissonance_probe = probe_dissonance_estimator(stats_df)

    # 5. Model-based importance and SHAP on the validation split
    models = load_tree_models(len(feature_cols))
    X_val = explain_df[feature_cols].to_numpy(dtype=float)
    y_val = explain_df[AnalysisConfig.TARGET_COL].to_numpy(dtype=float)

    importance_frames = []
    group_importance: Dict[str, pd.Series] = {}
    shap_by_model: Dict[str, np.ndarray] = {}
    figure_index = {"Random_Forest": (9, 11), "XGBoost": (10, 12)}

    for model_name, pipe in models.items():
        logger.info(f"--- Explaining {model_name} ---")
        importance_df = compute_model_importance(pipe, X_val, y_val, feature_cols, model_name)
        importance_number, shap_number = figure_index[model_name]
        plot_model_importance(
            importance_df, model_name, f"{importance_number:02d}_feature_importance_{model_name.lower()}.png"
        )

        shap_values = compute_shap_values(pipe, X_val)
        if shap_values is not None:
            importance_df["shap_mean_abs"] = np.abs(shap_values).mean(axis=0)
            plot_shap_summary(
                shap_values, X_val, feature_cols, model_name,
                f"{shap_number:02d}_shap_summary_{model_name.lower()}.png"
            )
            group_importance[model_name] = group_shap_importance(shap_values, feature_cols)
            shap_by_model[model_name] = shap_values
        importance_frames.append(importance_df)

    importance_all = pd.concat(importance_frames, ignore_index=True) if importance_frames else None
    if importance_all is not None:
        importance_path = AnalysisConfig.REPORTS_DIR / "feature_importance.csv"
        importance_all.to_csv(importance_path, index=False)
        logger.info(f"Saved importance table: {importance_path.relative_to(AnalysisConfig.PROJECT_ROOT)}")

    if group_importance:
        plot_group_importance(group_importance)

    # Dependence plots for the best tree model chosen by train_ml.py on validation RMSE
    if shap_by_model:
        metrics_path = AnalysisConfig.MODELS_DIR / "model_metrics.json"
        best_model = None
        if metrics_path.exists():
            with open(metrics_path) as f:
                best_model = json.load(f).get("best_model")
        if best_model not in shap_by_model:
            best_model = next(iter(shap_by_model))
        plot_shap_dependence(
            shap_by_model[best_model], X_val, feature_cols,
            [("sensory_dissonance_estimate", "Sensory dissonance estimate (a.u.)"),
             ("spectral_centroid_mean", "Mean spectral centroid (Hz)")],
            best_model,
        )

    write_summary(
        corr_df, importance_all, group_importance, dissonance_probe, len(stats_df), len(explain_df)
    )
    logger.info("Feature analysis complete.")


if __name__ == "__main__":
    main()
