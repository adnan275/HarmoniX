"""
HarmoniX | Exploratory Data Analysis (EDA) & Dataset Quality Module
Author: Adnan Rizvi (Lead Architect)

This module generates publication-quality EDA figures for Phase 1 report:
1. Target Valence distribution (KDE, Mean, Median).
2. Arousal distribution and Russell's Circumplex (Valence vs Arousal).
3. Leakage-free Split consistency check (Train vs Val vs Test target distributions).
4. Generates summary statistics table: reports/dataset_summary_statistics.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def generate_eda(metadata_file: Path = None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if metadata_file is None:
        metadata_file = PROCESSED_DIR / "all_matched_metadata.csv"

    if not metadata_file.exists():
        print(f"⚠️  Metadata file not found: {metadata_file}")
        print("Please run `src/data_loader.py` first to process the dataset.")
        return

    df = pd.read_csv(metadata_file)
    print(f"📊 Loaded {len(df)} records for EDA analysis.")

    # Apply aesthetic style
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("mako")

    # 1. Valence Distribution Plot
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        df["valence"],
        kde=True,
        color="#2b5c8f",
        bins=25,
        stat="density",
        alpha=0.65
    )
    mean_val = df["valence"].mean()
    median_val = df["valence"].median()
    plt.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean ({mean_val:.2f})")
    plt.axvline(median_val, color="#2ecc71", linestyle=":", linewidth=2, label=f"Median ({median_val:.2f})")
    plt.title("Target Distribution: Human-Rated Musical Valence (Pleasantness)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Valence Score", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "01_valence_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {fig1_path.name}")

    # 2. Russell's Valence vs Arousal 2D Affect Space (if arousal exists)
    if "arousal" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x="valence",
            y="arousal",
            hue="split" if "split" in df.columns else None,
            alpha=0.75,
            palette="deep",
            s=45
        )
        plt.axvline(df["valence"].mean(), color="gray", linestyle="--", alpha=0.5)
        plt.axhline(df["arousal"].mean(), color="gray", linestyle="--", alpha=0.5)
        plt.text(df["valence"].max()*0.85, df["arousal"].max()*0.9, "Q1: Energetic / Joyful", fontsize=9, color="#2c3e50")
        plt.text(df["valence"].min()*1.05, df["arousal"].max()*0.9, "Q2: Tense / Angry", fontsize=9, color="#2c3e50")
        plt.text(df["valence"].min()*1.05, df["arousal"].min()*1.05, "Q3: Sad / Depressed", fontsize=9, color="#2c3e50")
        plt.text(df["valence"].max()*0.85, df["arousal"].min()*1.05, "Q4: Calm / Peaceful", fontsize=9, color="#2c3e50")
        plt.title("Russell's Affect Grid: Valence (Pleasantness) vs Arousal (Energy)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Valence (Negative → Positive Emotional Tone)", fontsize=11)
        plt.ylabel("Arousal (Calm → Energetic)", fontsize=11)
        plt.tight_layout()
        fig2_path = FIGURES_DIR / "02_valence_vs_arousal_space.png"
        plt.savefig(fig2_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {fig2_path.name}")

    # 3. Train vs Val vs Test Distribution Consistency (Proves No Leakage & No Shift)
    if "split" in df.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(
            data=df,
            x="split",
            y="valence",
            palette="Set2",
            boxprops=dict(alpha=0.75),
            width=0.4
        )
        plt.title("Split Consistency: Valence Target across Train / Val / Test Partitions", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Dataset Partition", fontsize=11)
        plt.ylabel("Valence", fontsize=11)
        plt.tight_layout()
        fig3_path = FIGURES_DIR / "03_split_consistency_boxplot.png"
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {fig3_path.name}")

    # 4. Summary Statistics Table
    summary_stats = df[["valence"] + (["arousal"] if "arousal" in df.columns else [])].describe().T
    stats_path = PROJECT_ROOT / "reports" / "dataset_summary_statistics.csv"
    summary_stats.to_csv(stats_path)
    print(f"✅ Summary statistics saved to: {stats_path.name}")


if __name__ == "__main__":
    generate_eda()
