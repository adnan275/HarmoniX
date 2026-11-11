"""
HarmoniX | Phase 2 Deep Learning & Hybrid Fusion Training Pipeline
Author: Bhoomi

This module handles:
1. Loading leakage-free dataset splits (data/processed/all_matched_metadata.csv).
2. Generating 128-bin log-Mel spectrogram windows (3.0s, 50% overlap) via
   Aditya's spectrogram_engine and dynamic_windowing modules.
3. Extracting the agreed top eight Phase 1 acoustic descriptors per window.
4. Train-only median imputation and StandardScaler normalization (zero leakage).
5. Training-only SpecAugment time/frequency masking (validation/test unaugmented).
6. Numerical stability verification for CompositeValenceLoss on small and
   near-constant batches.
7. Validation-driven tuning of composite loss weight lambda in:
       L = MSE + lambda * (1 - Pearson_r)
8. Controlled ablation comparing CNN-only vs. Hybrid Late-Fusion architectures.
9. Reporting song-level R2, RMSE, MAE, Pearson r, and Spearman rho with split counts.
10. Saving versioned checkpoints, preprocessing metadata, and reproducible logs.
"""

import argparse
import copy
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import librosa
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from src.dynamic_windowing import (
    HOP_SECONDS,
    WINDOW_SECONDS,
    create_audio_windows,
    windows_to_cnn_tensor,
)
from src.feature_extraction import (
    HOP_LENGTH as FEAT_HOP_LENGTH,
    N_FFT as FEAT_N_FFT,
    estimate_sensory_dissonance,
    safe_mean,
)
from src.models_cnn import (
    CHECKPOINT_VERSION,
    SPECTROGRAM_CONFIG,
    TOP_8_ACOUSTIC_FEATURES,
    CompositeValenceLoss,
    build_model,
    get_device,
    get_device_info,
    save_checkpoint,
)
from src.multicultural_loader import load_and_normalize
from src.spectrogram_engine import SAMPLE_RATE, apply_spec_augment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HarmoniX.TrainCNN")


class TrainCNNConfig:
    """Default paths and training hyperparameters for Phase 2 experiments."""

    METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_matched_metadata.csv"
    CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_window_cache.pt"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    SELECTED_FEATURES = TOP_8_ACOUSTIC_FEATURES
    DEFAULT_TARGETS = ("valence",)
    DEFAULT_LAMBDA_GRID = (0.0, 0.05, 0.10, 0.25)

    EPOCHS = 40
    PATIENCE = 12
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EMBEDDING_DIM = 64
    FUSION_HIDDEN_DIM = 64
    DROPOUT_CONV = 0.15
    DROPOUT_FC = 0.25
    RANDOM_SEED = 42


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def extract_window_acoustic_features(
    window_waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, float]:
    """
    Extract the agreed top eight Phase 1 acoustic descriptors from a 3-second window.

    Uses the same DSP formulation as src/feature_extraction.py so window-level
    training features align with runtime inference features.
    """
    y = np.asarray(window_waveform, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0

    if y.size == 0 or peak <= 1e-8:
        return {feat: 0.0 for feat in TOP_8_ACOUSTIC_FEATURES}

    y = 0.95 * y / peak

    mfcc1 = librosa.feature.mfcc(
        y=y, sr=sample_rate, n_mfcc=1, n_fft=FEAT_N_FFT, hop_length=FEAT_HOP_LENGTH
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sample_rate, n_fft=FEAT_N_FFT, hop_length=FEAT_HOP_LENGTH
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sample_rate, n_fft=FEAT_N_FFT, hop_length=FEAT_HOP_LENGTH
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sample_rate, n_fft=FEAT_N_FFT, hop_length=FEAT_HOP_LENGTH
    )
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sample_rate, n_fft=FEAT_N_FFT, hop_length=FEAT_HOP_LENGTH
    )
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=FEAT_N_FFT, hop_length=FEAT_HOP_LENGTH
    )
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FEAT_N_FFT, hop_length=FEAT_HOP_LENGTH
    )
    dissonance = estimate_sensory_dissonance(y, sample_rate)

    return {
        "mfcc_1_mean": safe_mean(mfcc1),
        "spectral_flatness_mean": safe_mean(flatness),
        "zcr_mean": safe_mean(zcr),
        "spectral_contrast_mean": safe_mean(contrast),
        "spectral_centroid_mean": safe_mean(centroid),
        "spectral_rolloff_mean": safe_mean(rolloff),
        "spectral_bandwidth_mean": safe_mean(bandwidth),
        "sensory_dissonance_estimate": float(dissonance) if np.isfinite(dissonance) else np.nan,
    }


def _compute_manifest_signature(
    metadata_df: pd.DataFrame, target_names: Sequence[str]
) -> str:
    """Create a lightweight signature to detect dataset or configuration changes."""
    hasher = hashlib.sha256()
    subset = metadata_df[["music_id", "split", "audio_path", *target_names]].sort_values(
        by=["split", "music_id"]
    )
    hasher.update(subset.to_csv(index=False).encode("utf-8"))
    hasher.update(json.dumps(SPECTROGRAM_CONFIG, sort_keys=True).encode("utf-8"))
    hasher.update(json.dumps(TOP_8_ACOUSTIC_FEATURES).encode("utf-8"))
    return hasher.hexdigest()[:16]


def verify_split_integrity(metadata_df: pd.DataFrame) -> Dict[str, int]:
    """Validate required columns and guarantee zero music_id overlap across splits."""
    required = {"music_id", "audio_path", "split", "valence"}
    missing = required - set(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    split_ids: Dict[str, set] = {}
    counts: Dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        sub = metadata_df[metadata_df["split"] == split_name]
        ids = set(sub["music_id"].astype(str).tolist())
        split_ids[split_name] = ids
        counts[split_name] = len(sub)

    if counts["train"] == 0 or counts["val"] == 0:
        raise ValueError(f"Train and val splits must be non-empty. Found: {counts}")

    assert len(split_ids["train"] & split_ids["val"]) == 0, "Train/Val leakage detected"
    assert len(split_ids["train"] & split_ids["test"]) == 0, "Train/Test leakage detected"
    assert len(split_ids["val"] & split_ids["test"]) == 0, "Val/Test leakage detected"
    return counts


def build_or_load_window_dataset(
    metadata_path: Path,
    target_names: Sequence[str],
    cache_path: Path,
    rebuild_cache: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Load audio files per split, create 3s overlapping windows, compute 128-bin
    log-Mel spectrogram tensors and top-8 acoustic features per window.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata manifest not found at {metadata_path}. "
            "Run src/data_loader.py first."
        )

    metadata_df = pd.read_csv(metadata_path)
    for col in target_names:
        if col not in metadata_df.columns:
            raise ValueError(f"Target column '{col}' missing in {metadata_path}")

    split_song_counts = verify_split_integrity(metadata_df)
    signature = _compute_manifest_signature(metadata_df, target_names)

    if cache_path.exists() and not rebuild_cache:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("signature") == signature:
            logger.info(
                f"Loaded cached window tensors from {cache_path.name} "
                f"(signature={signature})."
            )
            return cached["splits"]
        logger.info("Dataset signature changed; rebuilding Phase 2 window cache...")

    logger.info(
        f"Extracting 3.0s overlapping windows (50% hop) across "
        f"{split_song_counts['train']} train / {split_song_counts['val']} val / "
        f"{split_song_counts['test']} test songs..."
    )

    splits_data: Dict[str, Dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        sub_df = metadata_df[metadata_df["split"] == split_name].reset_index(drop=True)
        spec_tensors: List[torch.Tensor] = []
        acoustic_rows: List[List[float]] = []
        target_rows: List[List[float]] = []
        music_ids: List[str] = []
        window_indices: List[int] = []
        song_count = 0

        for _, row in sub_df.iterrows():
            audio_path = Path(str(row["audio_path"]))
            if not audio_path.exists():
                logger.warning(f"Skipping missing audio file: {audio_path}")
                continue

            waveform, sr, _ = load_and_normalize(audio_path)
            windows = create_audio_windows(waveform, sample_rate=sr)
            song_specs = windows_to_cnn_tensor(windows)

            song_id = str(row["music_id"])
            song_targets = [float(row[t]) for t in target_names]

            for w_idx, w_wave in enumerate(windows):
                feat_dict = extract_window_acoustic_features(w_wave, sample_rate=sr)
                acoustic_rows.append([feat_dict[col] for col in TOP_8_ACOUSTIC_FEATURES])
                target_rows.append(song_targets)
                music_ids.append(song_id)
                window_indices.append(w_idx)

            spec_tensors.append(song_specs)
            song_count += 1

        if spec_tensors:
            specs_cat = torch.cat(spec_tensors, dim=0).to(torch.float32)
            acoustics_arr = np.asarray(acoustic_rows, dtype=np.float32)
            targets_arr = np.asarray(target_rows, dtype=np.float32)
        else:
            specs_cat = torch.empty((0, 1, 128, 126), dtype=torch.float32)
            acoustics_arr = np.empty((0, len(TOP_8_ACOUSTIC_FEATURES)), dtype=np.float32)
            targets_arr = np.empty((0, len(target_names)), dtype=np.float32)

        splits_data[split_name] = {
            "spectrograms": specs_cat,
            "acoustics_raw": acoustics_arr,
            "targets": targets_arr,
            "music_ids": music_ids,
            "window_indices": window_indices,
            "n_songs": song_count,
            "n_windows": int(specs_cat.shape[0]),
        }
        logger.info(
            f"Prepared split '{split_name}': "
            f"{song_count} songs -> {specs_cat.shape[0]} windows "
            f"{tuple(specs_cat.shape)}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "splits": splits_data}, cache_path)
    logger.info(f"Saved window tensor cache to {cache_path}")
    return splits_data


def fit_and_apply_acoustic_scaler(
    splits_data: Dict[str, Dict[str, Any]],
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Fit median imputer + StandardScaler strictly on training windows and apply
    to validation and test splits without data leakage.
    """
    scaler_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X_train_raw = splits_data["train"]["acoustics_raw"]
    splits_data["train"]["acoustics_scaled"] = scaler_pipeline.fit_transform(
        X_train_raw
    ).astype(np.float32)

    for split_name in ("val", "test"):
        X_raw = splits_data[split_name]["acoustics_raw"]
        if len(X_raw) > 0:
            splits_data[split_name]["acoustics_scaled"] = scaler_pipeline.transform(
                X_raw
            ).astype(np.float32)
        else:
            splits_data[split_name]["acoustics_scaled"] = X_raw.astype(np.float32)

    imputer = scaler_pipeline.named_steps["imputer"]
    scaler = scaler_pipeline.named_steps["scaler"]
    preprocessing_metadata = {
        "acoustic_features": list(TOP_8_ACOUSTIC_FEATURES),
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "imputer_strategy": "median",
        "imputer_statistics": [float(v) for v in imputer.statistics_],
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "fitted_on": f"train_split ({splits_data['train']['n_windows']} windows)",
    }
    return scaler_pipeline, preprocessing_metadata


class WindowAudioDataset(Dataset):
    """Dataset of 3-second spectrogram windows paired with scaled acoustic features."""

    def __init__(self, split_dict: Dict[str, Any]):
        self.spectrograms: torch.Tensor = split_dict["spectrograms"]
        self.acoustics: torch.Tensor = torch.from_numpy(split_dict["acoustics_scaled"])
        self.targets: torch.Tensor = torch.from_numpy(split_dict["targets"])
        self.music_ids: List[str] = split_dict["music_ids"]

    def __len__(self) -> int:
        return int(self.spectrograms.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        return (
            self.spectrograms[idx],
            self.acoustics[idx],
            self.targets[idx],
            self.music_ids[idx],
        )


def make_dataloader(
    dataset: WindowAudioDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader avoiding trailing 1-sample batches during training."""
    drop_last = False
    if shuffle and len(dataset) > batch_size and (len(dataset) % batch_size == 1):
        drop_last = True
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        drop_last=drop_last,
    )


def verify_composite_loss_stability(device: torch.device) -> Dict[str, Any]:
    """
    Verify numerical stability of CompositeValenceLoss across normal,
    small-batch, constant-target, and constant-prediction edge cases.
    """
    criterion = CompositeValenceLoss(lambda_corr=0.1)
    cases: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "normal_batch_8": (
            torch.tensor([[0.2], [0.4], [0.6], [0.8], [0.3], [0.5], [0.7], [0.9]], device=device),
            torch.tensor([[0.25], [0.35], [0.55], [0.85], [0.28], [0.52], [0.74], [0.88]], device=device),
        ),
        "single_sample_b1": (
            torch.tensor([[0.42]], device=device),
            torch.tensor([[0.65]], device=device),
        ),
        "small_batch_b2": (
            torch.tensor([[0.30], [0.70]], device=device),
            torch.tensor([[0.25], [0.75]], device=device),
        ),
        "constant_targets": (
            torch.tensor([[0.20], [0.50], [0.80], [0.40]], device=device),
            torch.tensor([[0.50], [0.50], [0.50], [0.50]], device=device),
        ),
        "near_constant_targets": (
            torch.tensor([[0.20], [0.50], [0.80], [0.40]], device=device),
            torch.tensor([[0.50], [0.50 + 1e-10], [0.50 - 1e-10], [0.50]], device=device),
        ),
        "constant_predictions": (
            torch.tensor([[0.50], [0.50], [0.50], [0.50]], device=device),
            torch.tensor([[0.20], [0.40], [0.70], [0.90]], device=device),
        ),
    }

    report: Dict[str, Any] = {}
    all_passed = True
    for case_name, (preds_raw, targs) in cases.items():
        preds = preds_raw.clone().detach().requires_grad_(True)
        comps = criterion.compute_components(preds, targs)
        loss = comps["total_loss"]
        loss.backward()

        loss_finite = bool(torch.isfinite(loss).item())
        grad_finite = bool(preds.grad is not None and torch.isfinite(preds.grad).all().item())
        passed = loss_finite and grad_finite
        all_passed = all_passed and passed

        report[case_name] = {
            "total_loss": round(float(loss.detach().cpu().item()), 6),
            "mse_loss": round(float(comps["mse_loss"].detach().cpu().item()), 6),
            "corr_loss": round(float(comps["corr_loss"].detach().cpu().item()), 6),
            "corr_valid": bool(comps["corr_valid"]),
            "loss_finite": loss_finite,
            "grad_finite": grad_finite,
            "passed": passed,
        }

    if not all_passed:
        raise RuntimeError(f"CompositeValenceLoss stability verification failed: {report}")

    logger.info("CompositeValenceLoss numerical stability verified across all 6 edge cases.")
    return {"all_passed": all_passed, "cases": report}


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute R2, RMSE, MAE, Pearson r, and Spearman rho safely."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        pearson_r = float(pearsonr(y_true, y_pred)[0])
        spearman_rho = float(spearmanr(y_true, y_pred)[0])
    else:
        pearson_r = 0.0
        spearman_rho = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Pearson_r": round(pearson_r, 4),
        "Spearman_rho": round(spearman_rho, 4),
    }


def evaluate_split(
    model: torch.nn.Module,
    dataset: WindowAudioDataset,
    criterion: CompositeValenceLoss,
    device: torch.device,
    target_names: Sequence[str],
    split_name: str,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Evaluate model on validation or test split without SpecAugment.
    Aggregates window predictions by music_id mean for song-level metrics.
    """
    if len(dataset) == 0:
        return {"split": split_name, "n_songs": 0, "n_windows": 0, "metrics": {}}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    preds_all: List[np.ndarray] = []
    targs_all: List[np.ndarray] = []
    ids_all: List[str] = []
    total_loss_sum = 0.0
    total_mse_sum = 0.0
    n_items = 0

    with torch.no_grad():
        for spec_b, ac_b, targ_b, id_b in loader:
            spec_b = spec_b.to(device)
            ac_b = ac_b.to(device)
            targ_b = targ_b.to(device)

            # Validation/test inputs are never SpecAugmented.
            preds = model(spec_b, ac_b)
            comps = criterion.compute_components(preds, targ_b)

            bs = spec_b.shape[0]
            total_loss_sum += float(comps["total_loss"].item()) * bs
            total_mse_sum += float(comps["mse_loss"].item()) * bs
            n_items += bs

            preds_all.append(preds.detach().cpu().numpy())
            targs_all.append(targ_b.detach().cpu().numpy())
            ids_all.extend(list(id_b))

    preds_np = np.vstack(preds_all)
    targs_np = np.vstack(targs_all)

    # Aggregate window-level predictions back to song level by music_id.
    song_rows = pd.DataFrame({"music_id": ids_all})
    for idx, t_name in enumerate(target_names):
        song_rows[f"true_{t_name}"] = targs_np[:, idx]
        song_rows[f"pred_{t_name}"] = preds_np[:, idx]

    song_agg = song_rows.groupby("music_id", sort=False).mean(numeric_only=True)

    primary_target = target_names[0]
    song_metrics = compute_regression_metrics(
        song_agg[f"true_{primary_target}"].values,
        song_agg[f"pred_{primary_target}"].values,
    )

    per_target_metrics: Dict[str, Dict[str, float]] = {}
    for t_name in target_names:
        per_target_metrics[t_name] = compute_regression_metrics(
            song_agg[f"true_{t_name}"].values,
            song_agg[f"pred_{t_name}"].values,
        )

    return {
        "split": split_name,
        "n_songs": int(len(song_agg)),
        "n_windows": int(n_items),
        "window_loss": round(total_loss_sum / max(1, n_items), 6),
        "window_mse": round(total_mse_sum / max(1, n_items), 6),
        "primary_target": primary_target,
        "metrics": song_metrics,
        "per_target_metrics": per_target_metrics,
    }


def train_single_run(
    model_type: str,
    train_ds: WindowAudioDataset,
    val_ds: WindowAudioDataset,
    device: torch.device,
    target_names: Sequence[str],
    lambda_corr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    use_specaugment: bool,
    seed: int,
    run_label: str,
) -> Dict[str, Any]:
    """Train one model configuration with early stopping on song-level val RMSE."""
    set_global_seed(seed)

    model = build_model(
        model_type=model_type,
        num_acoustic_features=len(TOP_8_ACOUSTIC_FEATURES),
        embedding_dim=TrainCNNConfig.EMBEDDING_DIM,
        fusion_hidden_dim=TrainCNNConfig.FUSION_HIDDEN_DIM,
        dropout_conv=TrainCNNConfig.DROPOUT_CONV,
        dropout_fc=TrainCNNConfig.DROPOUT_FC,
        target_names=target_names,
        use_sigmoid=True,
    ).to(device)

    criterion = CompositeValenceLoss(lambda_corr=lambda_corr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_val_r = -float("inf")
    best_epoch = 0
    best_val_eval: Dict[str, Any] = {}
    epochs_no_improve = 0
    history_rows: List[Dict[str, Any]] = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_items = 0

        for spec_b, ac_b, targ_b, _ in train_loader:
            # Apply SpecAugment strictly during training batches on CPU/tensor
            # before moving to device, keeping validation/test unaugmented.
            if use_specaugment:
                spec_b = apply_spec_augment(spec_b, training=True)

            spec_b = spec_b.to(device)
            ac_b = ac_b.to(device)
            targ_b = targ_b.to(device)

            optimizer.zero_grad()
            preds = model(spec_b, ac_b)
            comps = criterion.compute_components(preds, targ_b)
            loss = comps["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = spec_b.shape[0]
            train_loss_sum += float(loss.detach().item()) * bs
            train_mse_sum += float(comps["mse_loss"].detach().item()) * bs
            train_items += bs

        val_eval = evaluate_split(
            model=model,
            dataset=val_ds,
            criterion=criterion,
            device=device,
            target_names=target_names,
            split_name="val",
            batch_size=max(32, batch_size),
        )
        val_metrics = val_eval["metrics"]
        val_rmse = val_metrics["RMSE"]
        val_r = val_metrics["Pearson_r"]

        scheduler.step(val_rmse)

        improved = (val_rmse < best_val_rmse - 1e-5) or (
            abs(val_rmse - best_val_rmse) <= 1e-5 and val_r > best_val_r
        )
        if improved:
            best_val_rmse = val_rmse
            best_val_r = val_r
            best_epoch = epoch
            best_val_eval = val_eval
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        history_rows.append({
            "run_label": run_label,
            "model_type": model_type,
            "lambda_corr": lambda_corr,
            "epoch": epoch,
            "train_loss": round(train_loss_sum / max(1, train_items), 6),
            "train_mse": round(train_mse_sum / max(1, train_items), 6),
            "val_window_loss": val_eval["window_loss"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "val_R2": val_metrics["R2"],
            "val_Pearson_r": val_metrics["Pearson_r"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs or epochs_no_improve == patience:
            logger.info(
                f"[{run_label}] Epoch {epoch:02d}/{epochs:02d} | "
                f"TrainLoss={train_loss_sum / max(1, train_items):.4f} | "
                f"Val RMSE={val_metrics['RMSE']:.4f} | "
                f"Val MAE={val_metrics['MAE']:.4f} | "
                f"Val R2={val_metrics['R2']:.4f} | "
                f"Val r={val_metrics['Pearson_r']:.4f}"
            )

        if epochs_no_improve >= patience:
            logger.info(
                f"[{run_label}] Early stopping triggered at epoch {epoch} "
                f"(best epoch={best_epoch}, Val RMSE={best_val_rmse:.4f})."
            )
            break

    elapsed_sec = round(time.time() - t_start, 2)
    model.load_state_dict(best_state)
    model.eval()

    return {
        "run_label": run_label,
        "model_type": model_type,
        "lambda_corr": lambda_corr,
        "model": model,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed_sec,
        "val_eval": best_val_eval,
        "history": history_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HarmoniX Phase 2: Train 2D-CNN and Hybrid Late-Fusion models."
    )
    parser.add_argument(
        "--metadata-path", type=Path, default=TrainCNNConfig.METADATA_PATH
    )
    parser.add_argument("--cache-path", type=Path, default=TrainCNNConfig.CACHE_PATH)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TrainCNNConfig.DEFAULT_TARGETS),
        help="Explicit target column(s), e.g., valence or valence arousal",
    )
    parser.add_argument("--epochs", type=int, default=TrainCNNConfig.EPOCHS)
    parser.add_argument("--patience", type=int, default=TrainCNNConfig.PATIENCE)
    parser.add_argument("--batch-size", type=int, default=TrainCNNConfig.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TrainCNNConfig.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=TrainCNNConfig.WEIGHT_DECAY)
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        default=list(TrainCNNConfig.DEFAULT_LAMBDA_GRID),
        help="Candidate lambda weights for MSE + lambda * (1 - Pearson_r)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument("--seed", type=int, default=TrainCNNConfig.RANDOM_SEED)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-specaugment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device_info = get_device_info(args.device)
    device = get_device(args.device)
    use_specaugment = not args.no_specaugment

    logger.info("=================================================================")
    logger.info("HarmoniX Phase 2 | PyTorch 2D-CNN & Hybrid Fusion Training (Bhoomi)")
    logger.info("=================================================================")
    logger.info(
        f"Compute Device: {device_info['selected_device'].upper()} "
        f"({device_info['reason']}, PyTorch {device_info['torch_version']})"
    )
    logger.info(f"Explicit Targets: {args.targets}")
    logger.info(f"Top-8 Acoustic Features: {TOP_8_ACOUSTIC_FEATURES}")
    logger.info(f"Training-only SpecAugment enabled: {use_specaugment}")

    # 1. Verify composite loss numerical stability on edge cases.
    stability_report = verify_composite_loss_stability(device)

    # 2. Load/build spectrogram windows & window-level top-8 acoustic descriptors.
    splits_data = build_or_load_window_dataset(
        metadata_path=args.metadata_path,
        target_names=args.targets,
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
    )

    # 3. Fit acoustic feature scaler strictly on training split (zero leakage).
    scaler_pipeline, preprocessing_metadata = fit_and_apply_acoustic_scaler(splits_data)
    preprocessing_metadata["use_specaugment_train_only"] = use_specaugment
    preprocessing_metadata["target_names"] = list(args.targets)

    train_ds = WindowAudioDataset(splits_data["train"])
    val_ds = WindowAudioDataset(splits_data["val"])
    test_ds = WindowAudioDataset(splits_data["test"])

    # 4. Tune composite objective weight lambda on validation split using Hybrid Fusion.
    logger.info("\n--- Step 1: Tuning Composite Loss Weight (lambda) on Validation Split ---")
    lambda_runs: Dict[float, Dict[str, Any]] = {}
    lambda_records: List[Dict[str, Any]] = []
    all_history: List[Dict[str, Any]] = []

    for lam in args.lambda_grid:
        run_res = train_single_run(
            model_type="hybrid_fusion",
            train_ds=train_ds,
            val_ds=val_ds,
            device=device,
            target_names=args.targets,
            lambda_corr=float(lam),
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_specaugment=use_specaugment,
            seed=args.seed,
            run_label=f"hybrid_lambda_{lam}",
        )
        lambda_runs[float(lam)] = run_res
        all_history.extend(run_res["history"])

        vm = run_res["val_eval"]["metrics"]
        lambda_records.append({
            "model_type": "hybrid_fusion",
            "lambda_corr": float(lam),
            "best_epoch": run_res["best_epoch"],
            "n_val_songs": run_res["val_eval"]["n_songs"],
            "n_val_windows": run_res["val_eval"]["n_windows"],
            "Val_MAE": vm["MAE"],
            "Val_RMSE": vm["RMSE"],
            "Val_R2": vm["R2"],
            "Val_Pearson_r": vm["Pearson_r"],
            "Val_Spearman_rho": vm["Spearman_rho"],
            "train_time_sec": run_res["elapsed_sec"],
        })

    lambda_df = pd.DataFrame(lambda_records).sort_values(
        by=["Val_RMSE", "Val_Pearson_r"], ascending=[True, False]
    )
    best_lambda = float(lambda_df.iloc[0]["lambda_corr"])
    hybrid_best_run = lambda_runs[best_lambda]
    hybrid_mse_run = lambda_runs.get(0.0)

    logger.info(f"\nValidation-selected lambda*: {best_lambda}")
    print("\nLambda Tuning Summary (Validation Split):")
    print(lambda_df.to_string(index=False))

    # 5. Controlled Ablation: Train CNN-only baseline with identical split & lambda*.
    logger.info(
        f"\n--- Step 2: Training CNN-Only Baseline for Ablation (lambda={best_lambda}) ---"
    )
    cnn_only_run = train_single_run(
        model_type="cnn_only",
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        target_names=args.targets,
        lambda_corr=best_lambda,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_specaugment=use_specaugment,
        seed=args.seed,
        run_label=f"cnn_only_lambda_{best_lambda}",
    )
    all_history.extend(cnn_only_run["history"])

    # 6. Evaluate frozen models on Validation and Held-Out Test Splits.
    logger.info("\n--- Step 3: Held-Out Evaluation & CNN-Only vs. Hybrid Fusion Comparison ---")
    ablation_candidates: List[Tuple[str, Dict[str, Any]]] = [
        ("CNN_Only", cnn_only_run),
        ("Hybrid_Fusion", hybrid_best_run),
    ]
    if hybrid_mse_run is not None and best_lambda != 0.0:
        ablation_candidates.append(("Hybrid_Fusion_MSE_Baseline", hybrid_mse_run))

    comparison_records: List[Dict[str, Any]] = []
    detailed_metrics: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "target_names": list(args.targets),
        "selected_features": list(TOP_8_ACOUSTIC_FEATURES),
        "device_info": device_info,
        "split_counts": {
            split_name: {
                "n_songs": splits_data[split_name]["n_songs"],
                "n_windows": splits_data[split_name]["n_windows"],
            }
            for split_name in ("train", "val", "test")
        },
        "best_lambda_corr": best_lambda,
        "loss_stability_check": stability_report,
        "models": {},
    }

    for model_label, run_obj in ablation_candidates:
        crit = CompositeValenceLoss(lambda_corr=run_obj["lambda_corr"])
        val_res = evaluate_split(
            model=run_obj["model"],
            dataset=val_ds,
            criterion=crit,
            device=device,
            target_names=args.targets,
            split_name="val",
        )
        test_res = evaluate_split(
            model=run_obj["model"],
            dataset=test_ds,
            criterion=crit,
            device=device,
            target_names=args.targets,
            split_name="test",
        )
        run_obj["test_eval"] = test_res

        vm = val_res["metrics"]
        tm = test_res["metrics"]
        comparison_records.append({
            "Model": model_label,
            "lambda_corr": run_obj["lambda_corr"],
            "Best_Epoch": run_obj["best_epoch"],
            "n_train_songs": splits_data["train"]["n_songs"],
            "n_val_songs": val_res["n_songs"],
            "n_test_songs": test_res["n_songs"],
            "Val_MAE": vm["MAE"],
            "Val_RMSE": vm["RMSE"],
            "Val_R2": vm["R2"],
            "Val_Pearson_r": vm["Pearson_r"],
            "Val_Spearman_rho": vm["Spearman_rho"],
            "Test_MAE": tm.get("MAE"),
            "Test_RMSE": tm.get("RMSE"),
            "Test_R2": tm.get("R2"),
            "Test_Pearson_r": tm.get("Pearson_r"),
            "Test_Spearman_rho": tm.get("Spearman_rho"),
            "Target_R2_gt_0.50_Val": bool(vm["R2"] > 0.50),
            "Target_R2_gt_0.50_Test": bool(tm.get("R2", -1.0) > 0.50),
        })

        detailed_metrics["models"][model_label] = {
            "model_type": run_obj["model_type"],
            "lambda_corr": run_obj["lambda_corr"],
            "best_epoch": run_obj["best_epoch"],
            "train_time_sec": run_obj["elapsed_sec"],
            "val_evaluation": val_res,
            "test_evaluation": test_res,
        }

    comparison_df = pd.DataFrame(comparison_records)
    print("\nPhase 2 Ablation & Held-Out Benchmark:")
    print(comparison_df.to_string(index=False))

    # Select winning model strictly on validation RMSE (with Pearson r tie-breaker).
    val_winner_row = comparison_df[
        comparison_df["Model"].isin(["CNN_Only", "Hybrid_Fusion"])
    ].sort_values(by=["Val_RMSE", "Val_Pearson_r"], ascending=[True, False]).iloc[0]
    winner_name = str(val_winner_row["Model"])
    winner_run = cnn_only_run if winner_name == "CNN_Only" else hybrid_best_run
    detailed_metrics["best_validation_model"] = winner_name

    # 7. Save versioned checkpoints, scaler, preprocessing metadata, and logs.
    models_dir = TrainCNNConfig.MODELS_DIR
    reports_dir = TrainCNNConfig.REPORTS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_train_meta = {
        "seed": args.seed,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "device_info": device_info,
        "split_counts": detailed_metrics["split_counts"],
    }

    save_checkpoint(
        models_dir / "cnn_only_best.pt",
        model=cnn_only_run["model"],
        selected_features=[],
        preprocessing_metadata=preprocessing_metadata,
        training_metadata={
            **base_train_meta,
            "lambda_corr": cnn_only_run["lambda_corr"],
            "best_epoch": cnn_only_run["best_epoch"],
        },
        val_metrics=cnn_only_run["val_eval"],
        test_metrics=cnn_only_run["test_eval"],
    )

    save_checkpoint(
        models_dir / "hybrid_fusion_best.pt",
        model=hybrid_best_run["model"],
        selected_features=TOP_8_ACOUSTIC_FEATURES,
        preprocessing_metadata=preprocessing_metadata,
        training_metadata={
            **base_train_meta,
            "lambda_corr": hybrid_best_run["lambda_corr"],
            "best_epoch": hybrid_best_run["best_epoch"],
        },
        val_metrics=hybrid_best_run["val_eval"],
        test_metrics=hybrid_best_run["test_eval"],
    )

    save_checkpoint(
        models_dir / "best_phase2_model.pt",
        model=winner_run["model"],
        selected_features=(
            TOP_8_ACOUSTIC_FEATURES if winner_name == "Hybrid_Fusion" else []
        ),
        preprocessing_metadata=preprocessing_metadata,
        training_metadata={
            **base_train_meta,
            "selected_by": "lowest_val_rmse",
            "winner_label": winner_name,
            "lambda_corr": winner_run["lambda_corr"],
            "best_epoch": winner_run["best_epoch"],
        },
        val_metrics=winner_run["val_eval"],
        test_metrics=winner_run["test_eval"],
    )

    joblib.dump(scaler_pipeline, models_dir / "phase2_acoustic_scaler.joblib")

    metadata_handoff = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "best_validation_model": winner_name,
        "checkpoints": {
            "cnn_only": "models/cnn_only_best.pt",
            "hybrid_fusion": "models/hybrid_fusion_best.pt",
            "best_phase2": "models/best_phase2_model.pt",
        },
        "scaler_path": "models/phase2_acoustic_scaler.joblib",
        "spectrogram_config": SPECTROGRAM_CONFIG,
        "preprocessing_metadata": preprocessing_metadata,
        "device_info": device_info,
    }
    with open(models_dir / "phase2_metadata.json", "w") as f:
        json.dump(metadata_handoff, f, indent=2)

    with open(models_dir / "phase2_model_metrics.json", "w") as f:
        json.dump(detailed_metrics, f, indent=2)

    lambda_df.to_csv(reports_dir / "phase2_lambda_tuning.csv", index=False)
    pd.DataFrame(all_history).to_csv(
        reports_dir / "phase2_training_history.csv", index=False
    )
    comparison_df.to_csv(reports_dir / "phase2_cnn_ablation_results.csv", index=False)

    logger.info("=================================================================")
    logger.info(f"Phase 2 Training Complete | Validation Winner: {winner_name}")
    logger.info(f"Saved checkpoints and metadata to: {models_dir}")
    logger.info(f"Saved tuning/ablation logs to: {reports_dir}")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
