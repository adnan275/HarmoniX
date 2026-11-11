"""
HarmoniX | Phase 2 PyTorch 2D-CNN & Hybrid Late-Fusion Architectures
Author: Bhoomi

This module defines:
1. Device selection with Apple Silicon MPS acceleration and CPU fallback.
2. Agreed top eight Phase 1 acoustic descriptors for late fusion.
3. SpectrogramCNNBackbone: 2D-CNN blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout)
   converting (B, 1, 128, 126) log-Mel spectrograms into learned embeddings.
4. CNNOnlyRegressor: Spectrogram-only baseline for ablation against fusion.
5. HybridFusionRegressor: Late-fusion architecture concatenating learned CNN embeddings
   with standardized top-8 Phase 1 acoustic descriptors.
6. CompositeValenceLoss: Numerically stable MSE + lambda * (1 - Pearson r) objective.
7. Versioned checkpoint serialization and loading utilities for downstream inference,
   Grad-CAM explainability, and evaluation.

Shared Contract (Section 5.2):
    model(spectrogram_batch, acoustic_features) -> predictions (or (predictions, embeddings))
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.spectrogram_engine import (
    EXPECTED_FRAMES,
    FMAX,
    FMIN,
    HOP_LENGTH,
    MEL_SCALE,
    N_FFT,
    N_MELS,
    POWER,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SEGMENT_SECONDS,
    TOP_DB,
    WIN_LENGTH,
    WINDOW,
)

CHECKPOINT_VERSION = "2.0.0"

# Top eight Phase 1 descriptors selected from reports/feature_valence_correlations.csv
# and reports/feature_importance.csv (excluding key-dependent chroma averages):
# includes agreed MFCC, dissonance, flatness, and spectral-contrast features.
TOP_8_ACOUSTIC_FEATURES: List[str] = [
    "mfcc_1_mean",
    "spectral_flatness_mean",
    "zcr_mean",
    "spectral_contrast_mean",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "spectral_bandwidth_mean",
    "sensory_dissonance_estimate",
]

SPECTROGRAM_CONFIG: Dict[str, Any] = {
    "sample_rate": SAMPLE_RATE,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "win_length": WIN_LENGTH,
    "window": WINDOW,
    "mel_scale": MEL_SCALE,
    "fmin": FMIN,
    "fmax": FMAX,
    "power": POWER,
    "top_db": TOP_DB,
    "segment_seconds": SEGMENT_SECONDS,
    "segment_samples": SEGMENT_SAMPLES,
    "expected_frames": EXPECTED_FRAMES,
}


def get_device_info(preferred: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve compute device with Apple Silicon MPS priority and CPU fallback.

    Parameters
    ----------
    preferred : str, optional
        Device override ('auto', 'mps', 'cuda', or 'cpu'). Defaults to 'auto'.

    Returns
    -------
    dict
        Documented device selection metadata including PyTorch and backend flags.
    """
    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    cuda_available = bool(torch.cuda.is_available())

    pref = (preferred or "auto").lower().strip()
    if pref == "auto":
        if mps_available and mps_built:
            selected = "mps"
            reason = "Apple Silicon MPS backend available"
        elif cuda_available:
            selected = "cuda"
            reason = "CUDA GPU backend available"
        else:
            selected = "cpu"
            reason = "CPU fallback selected"
    elif pref == "mps":
        if mps_available and mps_built:
            selected = "mps"
            reason = "Explicitly requested MPS backend"
        else:
            selected = "cpu"
            reason = "MPS requested but unavailable; falling back to CPU"
    elif pref == "cuda":
        if cuda_available:
            selected = "cuda"
            reason = "Explicitly requested CUDA backend"
        else:
            selected = "cpu"
            reason = "CUDA requested but unavailable; falling back to CPU"
    elif pref == "cpu":
        selected = "cpu"
        reason = "Explicitly requested CPU backend"
    else:
        raise ValueError(f"Unsupported device preference: '{preferred}'")

    return {
        "selected_device": selected,
        "reason": reason,
        "torch_version": torch.__version__,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
    }


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return a PyTorch device prioritizing Apple Silicon MPS with CPU fallback."""
    info = get_device_info(preferred=preferred)
    return torch.device(info["selected_device"])


class SpectrogramCNNBackbone(nn.Module):
    """
    2D-CNN feature extractor for 128-bin log-Mel spectrogram windows.

    Input shape:  (B, 1, 128, 126) in dB scale [-TOP_DB, 0.0]
    Output shape: (B, embedding_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 64,
        dropout_conv: float = 0.15,
        dropout_fc: float = 0.25,
        top_db: float = TOP_DB,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.dropout_conv = dropout_conv
        self.dropout_fc = dropout_fc
        self.top_db = float(top_db)

        # Keep ReLU(inplace=False) so backward hooks (e.g., Grad-CAM) stay safe.
        self.block1 = self._make_conv_block(in_channels, 32, dropout=min(0.10, dropout_conv))
        self.block2 = self._make_conv_block(32, 64, dropout=dropout_conv)
        self.block3 = self._make_conv_block(64, 128, dropout=min(0.30, dropout_conv + 0.05))
        self.block4 = self._make_conv_block(128, 128, dropout=min(0.35, dropout_conv + 0.10))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.embedding_head = nn.Sequential(
            nn.Linear(128 * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_fc),
        )

    @staticmethod
    def _make_conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def normalize_input(self, spectrogram_batch: torch.Tensor) -> torch.Tensor:
        """Map log-Mel values from [-TOP_DB, 0.0] to [0.0, 1.0]."""
        x = torch.clamp(spectrogram_batch, min=-self.top_db, max=0.0)
        return (x + self.top_db) / self.top_db

    def get_target_conv_layer(self) -> nn.Module:
        """Expose the final convolutional layer for Grad-CAM visualization."""
        return self.block4[0]

    def forward_features(self, spectrogram_batch: torch.Tensor) -> torch.Tensor:
        """Compute spatial feature map before global pooling."""
        if spectrogram_batch.ndim == 3:
            spectrogram_batch = spectrogram_batch.unsqueeze(1)
        if spectrogram_batch.ndim != 4 or spectrogram_batch.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected spectrogram shape (B, {self.in_channels}, H, W), "
                f"got {tuple(spectrogram_batch.shape)}."
            )

        x = self.normalize_input(spectrogram_batch)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, spectrogram_batch: torch.Tensor) -> torch.Tensor:
        spatial_maps = self.forward_features(spectrogram_batch)
        pooled_avg = self.avg_pool(spatial_maps).flatten(1)
        pooled_max = self.max_pool(spatial_maps).flatten(1)
        pooled = torch.cat([pooled_avg, pooled_max], dim=1)
        return self.embedding_head(pooled)


class CNNOnlyRegressor(nn.Module):
    """
    Spectrogram-only 2D-CNN baseline regressor for Phase 2 ablation comparison.

    Implements the shared contract:
        model(spectrogram_batch, acoustic_features=None, return_embeddings=False)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        dropout_conv: float = 0.15,
        dropout_fc: float = 0.25,
        target_names: Sequence[str] = ("valence",),
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.model_type = "cnn_only"
        self.embedding_dim = embedding_dim
        self.dropout_conv = dropout_conv
        self.dropout_fc = dropout_fc
        self.target_names = tuple(target_names)
        self.num_targets = len(self.target_names)
        self.use_sigmoid = use_sigmoid

        self.backbone = SpectrogramCNNBackbone(
            in_channels=1,
            embedding_dim=embedding_dim,
            dropout_conv=dropout_conv,
            dropout_fc=dropout_fc,
        )

        layers: List[nn.Module] = [
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_fc),
            nn.Linear(32, self.num_targets),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*layers)

    def get_target_conv_layer(self) -> nn.Module:
        """Expose the final convolutional layer for Grad-CAM."""
        return self.backbone.get_target_conv_layer()

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "embedding_dim": self.embedding_dim,
            "dropout_conv": self.dropout_conv,
            "dropout_fc": self.dropout_fc,
            "target_names": list(self.target_names),
            "use_sigmoid": self.use_sigmoid,
            "num_acoustic_features": 0,
        }

    def forward(
        self,
        spectrogram_batch: torch.Tensor,
        acoustic_features: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embeddings = self.backbone(spectrogram_batch)
        predictions = self.regressor(embeddings)
        if return_embeddings:
            return predictions, embeddings
        return predictions


class HybridFusionRegressor(nn.Module):
    """
    Late-fusion multimodal regressor combining 2D-CNN spectrogram embeddings
    with the selected top eight Phase 1 acoustic descriptors.

    Implements the shared contract:
        model(spectrogram_batch, acoustic_features, return_embeddings=False)
    """

    def __init__(
        self,
        num_acoustic_features: int = len(TOP_8_ACOUSTIC_FEATURES),
        embedding_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout_conv: float = 0.15,
        dropout_fc: float = 0.25,
        target_names: Sequence[str] = ("valence",),
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.model_type = "hybrid_fusion"
        self.num_acoustic_features = num_acoustic_features
        self.embedding_dim = embedding_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.dropout_conv = dropout_conv
        self.dropout_fc = dropout_fc
        self.target_names = tuple(target_names)
        self.num_targets = len(self.target_names)
        self.use_sigmoid = use_sigmoid

        self.backbone = SpectrogramCNNBackbone(
            in_channels=1,
            embedding_dim=embedding_dim,
            dropout_conv=dropout_conv,
            dropout_fc=dropout_fc,
        )

        # Normalize CNN embedding magnitude so it balances cleanly with
        # StandardScaler-normalized Phase 1 acoustic features during concatenation.
        self.cnn_embedding_norm = nn.LayerNorm(embedding_dim)

        fusion_input_dim = embedding_dim + num_acoustic_features
        layers: List[nn.Module] = [
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_fc),
            nn.Linear(fusion_hidden_dim, 32),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_fc * 0.5),
            nn.Linear(32, self.num_targets),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.fusion_head = nn.Sequential(*layers)

    def get_target_conv_layer(self) -> nn.Module:
        """Expose the final convolutional layer for Grad-CAM."""
        return self.backbone.get_target_conv_layer()

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "num_acoustic_features": self.num_acoustic_features,
            "embedding_dim": self.embedding_dim,
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "dropout_conv": self.dropout_conv,
            "dropout_fc": self.dropout_fc,
            "target_names": list(self.target_names),
            "use_sigmoid": self.use_sigmoid,
        }

    def forward(
        self,
        spectrogram_batch: torch.Tensor,
        acoustic_features: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if acoustic_features is None:
            raise ValueError(
                "HybridFusionRegressor requires acoustic_features of shape "
                f"(B, {self.num_acoustic_features})."
            )
        if acoustic_features.ndim == 1:
            acoustic_features = acoustic_features.unsqueeze(0)
        if acoustic_features.shape[-1] != self.num_acoustic_features:
            raise ValueError(
                f"Expected {self.num_acoustic_features} acoustic features, "
                f"received {acoustic_features.shape[-1]}."
            )

        cnn_embeddings = self.backbone(spectrogram_batch)
        if cnn_embeddings.shape[0] != acoustic_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: spectrogram batch has {cnn_embeddings.shape[0]} "
                f"samples while acoustic_features has {acoustic_features.shape[0]}."
            )

        normalized_cnn_emb = self.cnn_embedding_norm(cnn_embeddings)
        fused = torch.cat([normalized_cnn_emb, acoustic_features], dim=1)
        predictions = self.fusion_head(fused)

        if return_embeddings:
            return predictions, cnn_embeddings
        return predictions


class CompositeValenceLoss(nn.Module):
    """
    Composite regression objective:
        L = MSE(y_pred, y_true) + lambda_corr * (1 - Pearson_r(y_pred, y_true))

    Undefined Correlation Policy
    ----------------------------
    Pearson correlation is mathematically undefined when batch size < 2 or when
    target variance is zero/near-zero (std(y_true) < min_target_std). For any
    target column where correlation is undefined, the correlation term evaluates
    to 0.0 and the objective gracefully reduces to pure MSE. When target variance
    is valid but predicted variance approaches zero, an epsilon inside the square
    root ensures finite, non-NaN gradients.
    """

    def __init__(
        self,
        lambda_corr: float = 0.1,
        eps: float = 1e-8,
        min_target_std: float = 1e-6,
    ):
        super().__init__()
        if lambda_corr < 0.0:
            raise ValueError("lambda_corr must be non-negative.")
        self.lambda_corr = float(lambda_corr)
        self.eps = float(eps)
        self.min_target_std = float(min_target_std)
        self.mse = nn.MSELoss(reduction="mean")

    def compute_components(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute total loss along with individual MSE and correlation diagnostics."""
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        mse_loss = self.mse(predictions, targets)

        batch_size = predictions.shape[0]
        if batch_size < 2 or self.lambda_corr == 0.0:
            corr_loss = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
            mean_r = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
            corr_valid = False
            total_loss = mse_loss
            return {
                "total_loss": total_loss,
                "mse_loss": mse_loss,
                "corr_loss": corr_loss,
                "pearson_r": mean_r,
                "corr_valid": corr_valid,
            }

        pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
        targ_centered = targets - targets.mean(dim=0, keepdim=True)

        var_pred = torch.sum(pred_centered ** 2, dim=0)
        var_targ = torch.sum(targ_centered ** 2, dim=0)
        cov = torch.sum(pred_centered * targ_centered, dim=0)

        # Target std check to guard against near-constant batches.
        targ_std = torch.sqrt(var_targ / batch_size)
        valid_mask = targ_std >= self.min_target_std

        if not torch.any(valid_mask):
            corr_loss = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
            mean_r = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
            corr_valid = False
            total_loss = mse_loss
        else:
            # Adding eps inside sqrt keeps gradients finite even if var_pred == 0.
            denom = torch.sqrt(var_pred[valid_mask] * var_targ[valid_mask] + self.eps)
            r_valid = torch.clamp(cov[valid_mask] / denom, min=-1.0, max=1.0)
            mean_r = r_valid.mean()
            corr_loss = 1.0 - mean_r
            corr_valid = True
            total_loss = mse_loss + self.lambda_corr * corr_loss

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "corr_loss": corr_loss,
            "pearson_r": mean_r,
            "corr_valid": corr_valid,
        }

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_components(predictions, targets)["total_loss"]


def build_model(model_type: str, **kwargs: Any) -> nn.Module:
    """Instantiate a Phase 2 CNN or Hybrid Fusion model by name."""
    normalized = model_type.lower().strip()
    if normalized == "cnn_only":
        allowed = {
            "embedding_dim",
            "dropout_conv",
            "dropout_fc",
            "target_names",
            "use_sigmoid",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return CNNOnlyRegressor(**filtered)
    if normalized == "hybrid_fusion":
        allowed = {
            "num_acoustic_features",
            "embedding_dim",
            "fusion_hidden_dim",
            "dropout_conv",
            "dropout_fc",
            "target_names",
            "use_sigmoid",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return HybridFusionRegressor(**filtered)
    raise ValueError(f"Unknown model_type '{model_type}'. Expected 'cnn_only' or 'hybrid_fusion'.")


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    selected_features: Sequence[str],
    preprocessing_metadata: Dict[str, Any],
    training_metadata: Dict[str, Any],
    val_metrics: Optional[Dict[str, Any]] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a self-contained, versioned PyTorch checkpoint with full metadata."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": model.get_config(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "selected_features": list(selected_features),
        "spectrogram_config": SPECTROGRAM_CONFIG,
        "preprocessing_metadata": preprocessing_metadata,
        "training_metadata": training_metadata,
        "val_metrics": val_metrics or {},
        "test_metrics": test_metrics or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a saved Phase 2 model checkpoint and reconstruct the matching architecture.

    Returns
    -------
    (model, metadata) : (nn.Module, dict)
        Eval-mode model placed on the resolved device and full checkpoint metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = (
        get_device(device) if (device is None or isinstance(device, str)) else device
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config = dict(payload["model_config"])
    model_type = model_config.pop("model_type")
    model = build_model(model_type, **model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(resolved_device)
    model.eval()

    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    metadata["loaded_device"] = str(resolved_device)
    return model, metadata
