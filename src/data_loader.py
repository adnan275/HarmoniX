"""
HarmoniX | Phase 1 Data Ingestion & Splitting Module
Author: Adnan Rizvi (Lead Architect)

This module handles:
1. Verification of the raw dataset structure (PMEmo).
2. Song-level annotation loading and validation (Valence / Arousal targets).
3. Audio-to-annotation alignment and sanity checks (missing files, duration validation).
4. Song-independent, leakage-free 70/15/15 Train / Validation / Test splitting.
5. Exporting split metadata to data/processed/.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HarmoniX.DataLoader")


class DataLoaderConfig:
    """Default configuration paths and parameters."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "PMEmo"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    REPORTS_FIG_DIR = PROJECT_ROOT / "reports" / "figures"
    
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # Expected target column candidates in PMEmo
    VALENCE_TARGET_CANDIDATES = [
        "valence_mean", "Valence_mean", "valence", "Valence", "valence_mu",
        "Valence(mean)"  # PMEmo2019 static_annotations.csv
    ]
    AROUSAL_TARGET_CANDIDATES = [
        "arousal_mean", "Arousal_mean", "arousal", "Arousal", "arousal_mu",
        "Arousal(mean)"  # PMEmo2019 static_annotations.csv
    ]


def find_audio_directory(raw_dir: Path) -> Optional[Path]:
    """Locate the directory containing audio files within raw_dir."""
    candidates = [
        raw_dir / "chorus",
        raw_dir / "audio",
        raw_dir / "mp3",
        raw_dir / "PMEmo2019" / "chorus",
        raw_dir
    ]
    for cand in candidates:
        if cand.is_dir():
            audio_files = list(cand.glob("*.mp3")) + list(cand.glob("*.wav"))
            if len(audio_files) > 0:
                logger.info(f"Found {len(audio_files)} audio files in: {cand}")
                return cand
    return None


def find_annotation_file(raw_dir: Path) -> Optional[Path]:
    """Locate the static song-level annotations CSV within raw_dir."""
    candidates = [
        raw_dir / "annotations" / "static_annotations.csv",
        raw_dir / "static_annotations.csv",
        raw_dir / "PMEmo2019" / "annotations" / "static_annotations.csv",
        raw_dir / "annotations.csv",
        raw_dir / "metadata.csv"
    ]
    for cand in candidates:
        if cand.is_file():
            logger.info(f"Found annotations file at: {cand}")
            return cand
            
    # Search recursively for any CSV containing 'static' or 'annotation'
    for csv_file in raw_dir.rglob("*.csv"):
        name_lower = csv_file.name.lower()
        if "static" in name_lower or "annotation" in name_lower or "valence" in name_lower:
            logger.info(f"Discovered candidate annotation file: {csv_file}")
            return csv_file
            
    return None


def load_and_standardize_annotations(annotation_path: Path) -> pd.DataFrame:
    """Load annotations CSV and standardize column names (music_id, valence, arousal)."""
    df = pd.read_csv(annotation_path)
    logger.info(f"Loaded {len(df)} rows from {annotation_path.name}")
    
    # Standardize song/music ID column
    id_col = None
    for cand in ["music_Id", "music_id", "musicId", "song_id", "id", "Id", "song_Id"]:
        if cand in df.columns:
            id_col = cand
            break
            
    if id_col is None:
        # Fall back to first column
        id_col = df.columns[0]
        logger.warning(f"No standard ID column found. Falling back to first column: '{id_col}'")
        
    df = df.rename(columns={id_col: "music_id"})
    df["music_id"] = df["music_id"].astype(str).str.strip()

    # Standardize Valence (Target)
    valence_col = None
    for cand in DataLoaderConfig.VALENCE_TARGET_CANDIDATES:
        if cand in df.columns:
            valence_col = cand
            break
            
    if valence_col is None:
        raise ValueError(
            f"Could not identify valence target column in {annotation_path.name}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.rename(columns={valence_col: "valence"})

    # Standardize Arousal (Secondary target/covariate)
    for cand in DataLoaderConfig.AROUSAL_TARGET_CANDIDATES:
        if cand in df.columns:
            df = df.rename(columns={cand: "arousal"})
            break
            
    return df


def match_audio_with_annotations(
    audio_dir: Path, annotations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match audio files on disk with corresponding annotation rows.
    Removes unmatched audio files or annotations without audio.
    """
    audio_map: Dict[str, str] = {}
    for p in list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav")):
        stem_id = p.stem.strip()
        audio_map[stem_id] = str(p.resolve())

    matched_records = []
    unmatched_audio = 0
    unmatched_annotations = 0

    for _, row in annotations_df.iterrows():
        song_id = str(row["music_id"])
        # Check direct match or zero-padded match
        filepath = audio_map.get(song_id) or audio_map.get(song_id.zfill(4))
        if filepath:
            rec = row.to_dict()
            rec["audio_path"] = filepath
            matched_records.append(rec)
        else:
            unmatched_annotations += 1

    matched_df = pd.DataFrame(matched_records)
    total_audio_found = len(audio_map)
    matched_count = len(matched_df)
    unmatched_audio = total_audio_found - matched_count

    logger.info(
        f"Matching Summary: Total Audio = {total_audio_found}, "
        f"Total Annotations = {len(annotations_df)}, "
        f"Matched Pairs = {matched_count}"
    )
    if unmatched_audio > 0:
        logger.warning(f"{unmatched_audio} audio file(s) had no corresponding annotation.")
    if unmatched_annotations > 0:
        logger.warning(f"{unmatched_annotations} annotation(s) had no corresponding audio file on disk.")

    return matched_df


def create_leakage_free_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into song-independent Train, Validation, and Test partitions.
    Guarantees no song overlap across splits.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"
    
    # Shuffle uniquely by music_id
    rng = np.random.default_rng(random_seed)
    unique_ids = df["music_id"].unique()
    shuffled_ids = rng.permutation(unique_ids)
    
    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train:n_train + n_val])
    test_ids = set(shuffled_ids[n_train + n_val:])
    
    # Sanity check: zero overlap
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected between train and val"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected between train and test"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage detected between val and test"

    train_df = df[df["music_id"].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df["music_id"].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df["music_id"].isin(test_ids)].copy().reset_index(drop=True)

    logger.info(
        f"Dataset Split Generated: Train = {len(train_df)} ({train_ratio*100:.0f}%), "
        f"Val = {len(val_df)} ({val_ratio*100:.0f}%), "
        f"Test = {len(test_df)} ({test_ratio*100:.0f}%)"
    )
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Save clean split CSV tables to processed data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train_split.csv"
    val_path = output_dir / "val_split.csv"
    test_path = output_dir / "test_split.csv"
    full_path = output_dir / "all_matched_metadata.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    combined = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test")
    ], ignore_index=True)
    combined.to_csv(full_path, index=False)

    logger.info(f"Splits successfully written to: {output_dir}")


def run_pipeline() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Execute complete dataset ingestion and splitting pipeline."""
    raw_dir = DataLoaderConfig.RAW_DATA_DIR
    processed_dir = DataLoaderConfig.PROCESSED_DATA_DIR
    
    logger.info("Starting HarmoniX Phase 1 Data Ingestion Pipeline...")
    audio_dir = find_audio_directory(raw_dir)
    annotation_file = find_annotation_file(raw_dir)

    if audio_dir is None or annotation_file is None:
        logger.warning(
            f"Dataset not fully present in {raw_dir}.\n"
            f"Expected: Audio files (.mp3/.wav) in {raw_dir}/chorus/ or {raw_dir}/ "
            f"and annotations in {raw_dir}/annotations/static_annotations.csv\n"
            f"Please place PMEmo files into {raw_dir} and rerun."
        )
        return None

    annotations_df = load_and_standardize_annotations(annotation_file)
    matched_df = match_audio_with_annotations(audio_dir, annotations_df)

    if len(matched_df) == 0:
        logger.error("0 audio files could be matched to annotation entries. Please check file naming.")
        return None

    train_df, val_df, test_df = create_leakage_free_splits(
        matched_df,
        train_ratio=DataLoaderConfig.TRAIN_RATIO,
        val_ratio=DataLoaderConfig.VAL_RATIO,
        test_ratio=DataLoaderConfig.TEST_RATIO,
        random_seed=DataLoaderConfig.RANDOM_SEED
    )
    save_splits(train_df, val_df, test_df, processed_dir)
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()
