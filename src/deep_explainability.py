"""
HarmoniX | Phase 2 Explainable AI & Spectrogram Heatmaps
Author: Praanshu Ranjan

Explains the Phase 2 spectrogram models (2D-CNN and hybrid late-fusion) on the
validation split. Shared contract (Phase 2 plan, Section 5.2):

    explain(model, input_window) -> attribution map and metadata

This module handles:
1. Grad-CAM on the last convolutional block, overlaid on the log-Mel spectrogram.
   Valence is a regression target, so there is no class to explain; two maps are
   produced instead: regions that RAISE the prediction and regions that LOWER it.
2. Neural-branch attribution with Integrated Gradients and DeepLIFT (Captum).
   DeepLIFT's rescale rule does not cover LayerNorm, so its completeness property
   is checked numerically and the result is reported either way.
3. Exact Shapley values for the 8 tabular (Phase 1) features of the hybrid model
   (256 coalitions, interventional baseline from training windows).
4. A modality split: how much of the hybrid prediction is attributed to the
   spectrogram vs the acoustic descriptors.
5. Validation of the attributions (Adebayo et al. 2018 randomization test,
   deletion faithfulness against shape-matched random masks, consistency across
   neighbouring windows).
6. Aggregate and per-window diagnostic summaries, publication-ready figures in
   reports/figures_phase2/, and result tables.

Policy:
- Only the validation split is explained; the test split is never loaded here.
- Attributions describe where the MODEL is sensitive in time and frequency. They do
  not identify instruments, semantic content, or causes of perceived valence.
"""

import argparse
import copy
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.feature_analysis import (
    AXIS,
    BLUE,
    DIVERGING_CMAP,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    ORANGE,
    apply_plot_style,
)
from src.models_cnn import TOP_8_ACOUSTIC_FEATURES, get_device, load_checkpoint
from src.spectrogram_engine import (
    EXPECTED_FRAMES,
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
    TOP_DB,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HarmoniX.DeepExplainability")


class ExplainConfig:
    """Paths and analysis parameters."""
    MODELS_DIR = PROJECT_ROOT / "models"
    CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_window_cache.pt"
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures_phase2"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    CHECKPOINTS = {
        "cnn_only": "cnn_only_best.pt",
        "hybrid_fusion": "hybrid_fusion_best.pt",
    }
    EXPLAIN_SPLIT = "val"

    SILENCE_DB = -float(TOP_DB)       # maps to 0.0 after the model's input normalisation
    RANDOM_SEED = 42
    N_ATTR_WINDOWS = 150              # subset for the expensive analyses
    IG_STEPS = 256                    # median completeness gap ~1% (about 10% at 32 steps)
    SHAPLEY_BACKGROUND = 32
    DELETION_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
    N_RANDOM_CONTROLS = 5
    COMPLETENESS_REL_TOL = 0.05       # |gap| / |f(x) - f(baseline)| to call a method complete
    DPI = 300


# Axes of the 128 x 126 spectrogram: Mel-bin centre frequencies and frame centres.
MEL_FREQS_HZ = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=False)
FRAME_TIMES_S = (np.arange(EXPECTED_FRAMES) * HOP_LENGTH + N_FFT / 2) / SAMPLE_RATE

FREQ_BANDS = [
    ("bass (<250 Hz)", 0.0, 250.0),
    ("low-mid (250-500 Hz)", 250.0, 500.0),
    ("mid (0.5-2 kHz)", 500.0, 2000.0),
    ("upper-mid (2-4 kHz)", 2000.0, 4000.0),
    ("high (4-11 kHz)", 4000.0, float("inf")),
]
BAND_MASKS = [(name, (MEL_FREQS_HZ >= lo) & (MEL_FREQS_HZ < hi)) for name, lo, hi in FREQ_BANDS]
TIME_SEGMENTS = [("0-1 s", 0.0, 1.0), ("1-2 s", 1.0, 2.0), ("2-3 s", 2.0, SEGMENT_SECONDS + 1e-6)]
TIME_MASKS = [(name, (FRAME_TIMES_S >= lo) & (FRAME_TIMES_S < hi)) for name, lo, hi in TIME_SEGMENTS]

FEATURE_LABELS = {
    "mfcc_1_mean": "MFCC 1",
    "spectral_flatness_mean": "Spectral flatness",
    "zcr_mean": "Zero-crossing rate",
    "spectral_contrast_mean": "Spectral contrast",
    "spectral_centroid_mean": "Spectral centroid",
    "spectral_rolloff_mean": "Spectral roll-off",
    "spectral_bandwidth_mean": "Spectral bandwidth",
    "sensory_dissonance_estimate": "Sensory dissonance (est.)",
}

CAVEAT = (
    "Attributions describe where the model is sensitive in time and frequency. They do not "
    "identify instruments or semantic content, and they are not causal explanations of "
    "perceived valence."
)


# ---------------------------------------------------------------------------
# Input handling and model helpers
# ---------------------------------------------------------------------------

def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _as_spec_batch(spec: Any) -> torch.Tensor:
    """Accept (H, W), (B, H, W) or (B, 1, H, W) and return (B, 1, H, W) float32."""
    spec = torch.as_tensor(spec, dtype=torch.float32)
    if spec.ndim == 2:
        spec = spec[None, None]
    elif spec.ndim == 3:
        spec = spec[:, None]
    if spec.ndim != 4 or tuple(spec.shape[1:]) != (1, N_MELS, EXPECTED_FRAMES):
        raise ValueError(
            f"Expected spectrogram windows of shape (B, 1, {N_MELS}, {EXPECTED_FRAMES}), "
            f"got {tuple(spec.shape)}."
        )
    return spec


def _as_acoustic_batch(acoustic: Any, n: int) -> Optional[torch.Tensor]:
    if acoustic is None:
        return None
    acoustic = torch.as_tensor(acoustic, dtype=torch.float32)
    if acoustic.ndim == 1:
        acoustic = acoustic[None]
    if acoustic.shape[0] != n:
        raise ValueError(f"Batch mismatch: {n} spectrograms but {acoustic.shape[0]} feature rows.")
    return acoustic


def is_hybrid(model: nn.Module) -> bool:
    return hasattr(model, "fusion_head")


def _head(model: nn.Module) -> nn.Sequential:
    return model.fusion_head if is_hybrid(model) else model.regressor


def _logit_layer(model: nn.Module) -> nn.Linear:
    """The last Linear layer: its output is the pre-sigmoid score (the sigmoid saturates
    and would shrink gradients near 0 and 1)."""
    return [m for m in _head(model) if isinstance(m, nn.Linear)][-1]


def _cam_layer(model: nn.Module, layer: str) -> nn.Module:
    """'relu': activation of the last conv block (textbook Grad-CAM layer).
    'conv': raw output of the last Conv2d, as exposed by get_target_conv_layer()."""
    if layer == "relu":
        return model.backbone.block4[2]
    if layer == "conv":
        return model.get_target_conv_layer()
    raise ValueError(f"Unknown Grad-CAM layer '{layer}'.")


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    specs: torch.Tensor,
    acoustics: Optional[torch.Tensor] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Valence predictions (first target) for a set of windows."""
    model.eval()
    device = _device_of(model)
    outputs = []
    for start in range(0, len(specs), batch_size):
        s = specs[start:start + batch_size].to(device)
        a = acoustics[start:start + batch_size].to(device) if acoustics is not None else None
        outputs.append(model(s, a)[:, 0].cpu())
    return torch.cat(outputs).numpy()


# ---------------------------------------------------------------------------
# 1. Grad-CAM
# ---------------------------------------------------------------------------

def gradcam_batch(
    model: nn.Module,
    specs: torch.Tensor,
    acoustics: Optional[torch.Tensor] = None,
    layer: str = "relu",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Signed Grad-CAM for a regression output.

    cam = sum_k alpha_k * A_k, alpha_k = spatial mean of d(logit)/dA_k, upsampled to the
    input resolution. Positive values mark time-frequency regions whose activations push
    the (pre-sigmoid) valence score UP; negative values push it DOWN. The textbook
    ReLU(cam) keeps only the positive part, which is the 'raising' map; the 'lowering'
    map is ReLU(-cam). Runs in eval mode (dropout off, batch-norm running statistics).
    """
    model.eval()
    device = _device_of(model)
    cam_layer, logit_layer = _cam_layer(model, layer), _logit_layer(model)
    cams, logits, preds = [], [], []

    for start in range(0, len(specs), batch_size):
        s = specs[start:start + batch_size].to(device)
        a = acoustics[start:start + batch_size].to(device) if acoustics is not None else None
        store: Dict[str, torch.Tensor] = {}
        h_act = cam_layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
        h_logit = logit_layer.register_forward_hook(lambda m, i, o: store.__setitem__("logit", o))
        try:
            with torch.enable_grad():
                out = model(s, a)
                logit = store["logit"][:, 0]
                grads = torch.autograd.grad(logit.sum(), store["act"])[0]
        finally:
            h_act.remove()
            h_logit.remove()
        act = store["act"].detach()
        alpha = grads.mean(dim=(2, 3), keepdim=True)
        cam = (alpha * act).sum(dim=1, keepdim=True)
        cam = F.interpolate(cam, size=(N_MELS, EXPECTED_FRAMES), mode="bilinear", align_corners=False)
        cams.append(cam[:, 0].cpu())
        logits.append(logit.detach().cpu())
        preds.append(out[:, 0].detach().cpu())

    return {
        "cam_signed": torch.cat(cams).numpy(),
        "logit": torch.cat(logits).numpy(),
        "prediction": torch.cat(preds).numpy(),
    }


# ---------------------------------------------------------------------------
# 2. Integrated Gradients and DeepLIFT (neural branch)
# ---------------------------------------------------------------------------

class _ValenceOutput(nn.Module):
    """Wrap a Phase 2 model so that Captum sees one valence value per sample."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.hybrid = is_hybrid(model)

    def forward(self, spec: torch.Tensor, acoustic: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model(spec, acoustic) if self.hybrid else self.model(spec)
        return out[:, 0]


def _baselines(specs: torch.Tensor, acoustics: Optional[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """Silence spectrogram (-80 dB everywhere) and the training-mean acoustic vector
    (zeros after standardisation)."""
    spec_base = torch.full_like(specs, ExplainConfig.SILENCE_DB)
    if acoustics is None:
        return (spec_base,)
    return (spec_base, torch.zeros_like(acoustics))


def captum_attribution(
    model: nn.Module,
    specs: torch.Tensor,
    acoustics: Optional[torch.Tensor],
    method: str,
    n_steps: int = ExplainConfig.IG_STEPS,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Attribute the valence output (in valence units) relative to a silence baseline.

    Returns the spectrogram attribution (N, 128, 126), the acoustic attribution (N, 8) for
    hybrid models, f(x), f(baseline), and the completeness gap
        sum(attributions) - (f(x) - f(baseline)),
    which is 0 for a method that satisfies completeness.
    """
    from captum.attr import DeepLift, IntegratedGradients

    model.eval()
    device = _device_of(model)
    wrapper = _ValenceOutput(model).eval()
    spec_attrs, acoustic_attrs, f_x, f_b = [], [], [], []

    for start in range(0, len(specs), batch_size):
        s = specs[start:start + batch_size].to(device)
        a = acoustics[start:start + batch_size].to(device) if acoustics is not None else None
        inputs = (s,) if a is None else (s, a)
        baselines = _baselines(s, a)
        if method == "ig":
            explainer = IntegratedGradients(wrapper)
            attributions = explainer.attribute(
                inputs, baselines=baselines, n_steps=n_steps, method="gausslegendre",
                internal_batch_size=64,
            )
        elif method == "deeplift":
            explainer = DeepLift(wrapper)
            attributions = explainer.attribute(inputs, baselines=baselines)
        else:
            raise ValueError(f"Unknown attribution method '{method}'.")
        with torch.no_grad():
            f_x.append(wrapper(*inputs).cpu())
            f_b.append(wrapper(*baselines).cpu())
        spec_attrs.append(attributions[0].detach().cpu()[:, 0])
        if a is not None:
            acoustic_attrs.append(attributions[1].detach().cpu())

    spec_attr = torch.cat(spec_attrs).numpy()
    acoustic_attr = torch.cat(acoustic_attrs).numpy() if acoustic_attrs else None
    f_x_arr, f_b_arr = torch.cat(f_x).numpy(), torch.cat(f_b).numpy()
    attr_sum = spec_attr.reshape(len(spec_attr), -1).sum(axis=1)
    if acoustic_attr is not None:
        attr_sum = attr_sum + acoustic_attr.sum(axis=1)
    return {
        "spec_attr": spec_attr,
        "acoustic_attr": acoustic_attr,
        "f_x": f_x_arr,
        "f_baseline": f_b_arr,
        "attr_sum": attr_sum,
        "completeness_gap": attr_sum - (f_x_arr - f_b_arr),
    }


def completeness_report(result: Dict[str, Any]) -> Dict[str, float]:
    delta = result["f_x"] - result["f_baseline"]
    gap = result["completeness_gap"]
    rel = np.abs(gap) / np.maximum(np.abs(delta), 1e-6)
    return {
        "mean_abs_gap": float(np.mean(np.abs(gap))),
        "median_relative_gap": float(np.median(rel)),
        "fraction_within_tolerance": float(np.mean(rel <= ExplainConfig.COMPLETENESS_REL_TOL)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "satisfied": bool(np.median(rel) <= ExplainConfig.COMPLETENESS_REL_TOL),
    }


def deeplift_assumption_diagnostic(
    model: nn.Module,
    specs: torch.Tensor,
    acoustics: Optional[torch.Tensor],
) -> pd.DataFrame:
    """
    Why does DeepLIFT fail completeness on this architecture? Captum's DeepLIFT applies
    rescale rules to ReLU, Sigmoid and MaxPool2d only; LayerNorm and AdaptiveMaxPool2d are
    treated as linear. This re-runs DeepLIFT on copies of the model with those layers
    swapped for linear ones (identity / average pooling). Each variant is a different
    function, so its gap is measured against its own f(x) - f(baseline); the point is
    which layer type breaks the method, not the variants' predictive quality.
    """
    def variant(layer_norm: bool, max_pool: bool) -> nn.Module:
        m = copy.deepcopy(model)
        if layer_norm:
            m.backbone.embedding_head[1] = nn.Identity()
            if is_hybrid(m):
                m.cnn_embedding_norm = nn.Identity()
        if max_pool:
            m.backbone.max_pool = nn.AdaptiveAvgPool2d((1, 1))
        return m.eval()

    rows = []
    for label, kwargs in (
        ("deployed model", dict(layer_norm=False, max_pool=False)),
        ("LayerNorm -> identity", dict(layer_norm=True, max_pool=False)),
        ("AdaptiveMaxPool -> average pool", dict(layer_norm=False, max_pool=True)),
        ("both replaced", dict(layer_norm=True, max_pool=True)),
    ):
        report = completeness_report(captum_attribution(variant(**kwargs), specs, acoustics, "deeplift"))
        rows.append({"variant": label, "median_relative_gap": report["median_relative_gap"],
                     "fraction_within_tolerance": report["fraction_within_tolerance"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Exact Shapley values for the tabular branch
# ---------------------------------------------------------------------------

def exact_shapley_tabular(
    model: nn.Module,
    specs: torch.Tensor,
    acoustics: torch.Tensor,
    background: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Exact interventional Shapley values of the acoustic features of a hybrid model.

    The spectrogram (hence the CNN embedding) is held fixed. For a coalition S of
    features, v(S) = mean over background rows b of f(spec, x_S, b_notS). With n = 8
    there are 256 coalitions, so no sampling approximation is needed. Efficiency holds:
    sum(phi) = v(all) - v(empty) = f(x) - E_b[f(spec, b)].
    """
    if not is_hybrid(model):
        raise ValueError("Tabular Shapley values need a hybrid fusion model.")
    model.eval()
    device = _device_of(model)
    n = acoustics.shape[1]
    n_sets = 2 ** n
    masks = ((np.arange(n_sets)[:, None] >> np.arange(n)) & 1).astype(np.float32)
    popcount = masks.sum(axis=1).astype(int)
    weights = np.array([math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)])
    masks_t = torch.from_numpy(masks).to(device)
    bg = background.to(device)

    phi = np.zeros((len(specs), n), dtype=np.float64)
    v_empty, v_full, direct = np.zeros(len(specs)), np.zeros(len(specs)), np.zeros(len(specs))

    with torch.no_grad():
        for w in range(len(specs)):
            spec = specs[w:w + 1].to(device)
            x = acoustics[w].to(device)
            emb = model.cnn_embedding_norm(model.backbone(spec))
            mixed = masks_t[:, None, :] * x[None, None, :] + (1 - masks_t[:, None, :]) * bg[None, :, :]
            k = bg.shape[0]
            fused = torch.cat([emb.expand(n_sets * k, -1), mixed.reshape(n_sets * k, n)], dim=1)
            v = model.fusion_head(fused)[:, 0].reshape(n_sets, k).mean(dim=1).cpu().numpy()
            for i in range(n):
                without_i = np.where(masks[:, i] == 0)[0]
                phi[w, i] = np.sum(weights[popcount[without_i]] * (v[without_i | (1 << i)] - v[without_i]))
            v_empty[w], v_full[w] = v[0], v[-1]
            direct[w] = model(spec, x[None])[0, 0].item()

    return {
        "phi": phi,
        "v_empty": v_empty,
        "v_full": v_full,
        "prediction": direct,
        "efficiency_error": np.abs(phi.sum(axis=1) - (v_full - v_empty)),
        "consistency_error": np.abs(v_full - direct),
    }


# ---------------------------------------------------------------------------
# 4. Validation of the attributions
# ---------------------------------------------------------------------------

def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(a.ravel(), b.ravel())[0])


def cascading_randomization(
    model: nn.Module,
    specs: torch.Tensor,
    acoustics: Optional[torch.Tensor],
    seed: int = ExplainConfig.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Model-parameter randomisation test (Adebayo et al., 2018). Layers are re-initialised
    from the output towards the input; a faithful explanation must change as the model
    is destroyed. Reports the rank correlation of the randomised map with the original.
    """
    original = gradcam_batch(model, specs, acoustics)["cam_signed"]
    head = _head(model)
    stages: List[Tuple[str, List[nn.Module]]] = [
        ("head", [head]),
        ("+ embedding", [model.backbone.embedding_head]),
        ("+ block 4", [model.backbone.block4]),
        ("+ block 3", [model.backbone.block3]),
        ("+ block 2", [model.backbone.block2]),
        ("+ block 1", [model.backbone.block1]),
    ]
    rows = [{"stage": "trained", "n_layers_randomised": 0,
             "spearman_signed_mean": 1.0, "spearman_signed_sd": 0.0,
             "spearman_abs_mean": 1.0, "spearman_abs_sd": 0.0}]

    clone = copy.deepcopy(model)
    torch.manual_seed(seed)
    for depth, (name, modules) in enumerate(stages, start=1):
        for module in modules:
            for m in clone.get_submodule(_module_path(model, module)).modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        cam = gradcam_batch(clone, specs, acoustics)["cam_signed"]
        signed = [_rank_corr(o, c) for o, c in zip(original, cam)]
        absolute = [_rank_corr(np.abs(o), np.abs(c)) for o, c in zip(original, cam)]
        rows.append({
            "stage": name, "n_layers_randomised": depth,
            "spearman_signed_mean": float(np.nanmean(signed)), "spearman_signed_sd": float(np.nanstd(signed)),
            "spearman_abs_mean": float(np.nanmean(absolute)), "spearman_abs_sd": float(np.nanstd(absolute)),
        })
    return pd.DataFrame(rows)


def _module_path(root: nn.Module, target: nn.Module) -> str:
    for name, module in root.named_modules():
        if module is target:
            return name
    raise KeyError("Module not found in model.")


def _apply_mask(specs: torch.Tensor, mask: np.ndarray, baseline: str) -> torch.Tensor:
    mask_t = torch.from_numpy(mask).reshape(len(specs), 1, N_MELS, EXPECTED_FRAMES)
    if baseline == "silence":
        fill = torch.full_like(specs, ExplainConfig.SILENCE_DB)
    elif baseline == "mean":
        fill = specs.mean(dim=(1, 2, 3), keepdim=True).expand_as(specs)
    else:
        raise ValueError(f"Unknown mask baseline '{baseline}'.")
    return torch.where(mask_t, fill, specs)


def deletion_faithfulness(
    model: nn.Module,
    specs: torch.Tensor,
    acoustics: Optional[torch.Tensor],
    cam_signed: np.ndarray,
    song_ids: Sequence[str],
    seed: int = ExplainConfig.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Mask the top-k% of pixels the map calls 'raising' (or 'lowering') and measure the
    change in predicted valence. Control: the same mask circularly shifted to random
    positions, which keeps its area and shape. If the map is faithful, removing 'raising'
    regions should lower the prediction more than the shifted controls, and removing
    'lowering' regions should raise it more.

    Windows of one song are correlated, so the paired Wilcoxon tests are run on per-song
    means (n = number of distinct songs), not on windows.
    """
    rng = np.random.default_rng(seed)
    n = len(specs)
    ids = np.asarray(song_ids)

    def song_level_p(delta_a: np.ndarray, delta_b: np.ndarray, alternative: str) -> Tuple[float, int]:
        per_song = pd.DataFrame({"id": ids, "a": delta_a, "b": delta_b}).groupby("id").mean()
        return float(stats.wilcoxon(per_song["a"], per_song["b"], alternative=alternative).pvalue), len(per_song)
    base = predict_batch(model, specs, acoustics)
    flat = cam_signed.reshape(n, -1)
    order = np.argsort(flat, axis=1)
    rows = []

    for fraction in ExplainConfig.DELETION_FRACTIONS:
        n_pix = int(round(fraction * flat.shape[1]))
        raise_mask = np.zeros_like(flat, dtype=bool)
        lower_mask = np.zeros_like(flat, dtype=bool)
        np.put_along_axis(raise_mask, order[:, -n_pix:], True, axis=1)
        np.put_along_axis(lower_mask, order[:, :n_pix], True, axis=1)
        random_masks = []
        for _ in range(ExplainConfig.N_RANDOM_CONTROLS):
            shifted = np.stack([
                np.roll(raise_mask[i].reshape(N_MELS, EXPECTED_FRAMES),
                        (int(rng.integers(N_MELS)), int(rng.integers(EXPECTED_FRAMES))), axis=(0, 1)).ravel()
                for i in range(n)
            ])
            random_masks.append(shifted)

        for baseline in ("silence", "mean"):
            d_raise = predict_batch(model, _apply_mask(specs, raise_mask, baseline), acoustics) - base
            d_lower = predict_batch(model, _apply_mask(specs, lower_mask, baseline), acoustics) - base
            d_random = np.mean(
                [predict_batch(model, _apply_mask(specs, m, baseline), acoustics) - base for m in random_masks],
                axis=0,
            )
            p_raise, n_songs = song_level_p(d_raise, d_random, "less")
            p_lower, _ = song_level_p(d_lower, d_random, "greater")
            rows.append({
                "fraction": fraction, "mask_baseline": baseline, "n_windows": n, "n_songs": n_songs,
                "delta_remove_raising": float(d_raise.mean()),
                "delta_remove_lowering": float(d_lower.mean()),
                "delta_random_shifted": float(d_random.mean()),
                "p_raising_vs_random_song_level": p_raise,
                "p_lowering_vs_random_song_level": p_lower,
            })
    return pd.DataFrame(rows)


def consistency_checks(
    cam_signed: np.ndarray,
    music_ids: Sequence[str],
    window_indices: Sequence[int],
    seed: int = ExplainConfig.RANDOM_SEED,
) -> Dict[str, Any]:
    """
    (a) Do neighbouring windows of one song (50% overlap) get more similar frequency
        profiles than windows of different songs?
    (b) How much of each map is just the dataset-average map? (a high value means the
        heatmap is mostly an input-independent frequency prior)
    """
    rng = np.random.default_rng(seed)
    ids = np.asarray(music_ids)
    idx = np.asarray(window_indices)
    profiles = np.clip(cam_signed, 0, None).mean(axis=2)
    neighbour = [i for i in range(len(ids) - 1) if ids[i] == ids[i + 1] and idx[i + 1] == idx[i] + 1]

    def corr(i: int, j: int) -> float:
        a, b = profiles[i], profiles[j]
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else float("nan")

    same = np.array([corr(i, i + 1) for i in neighbour])
    other = []
    while len(other) < len(neighbour):
        i, j = rng.integers(len(ids), size=2)
        if ids[i] != ids[j]:
            other.append(corr(int(i), int(j)))
    other = np.array(other)
    same, other = same[np.isfinite(same)], other[np.isfinite(other)]

    mean_map = cam_signed.mean(axis=0)
    sample = rng.choice(len(cam_signed), size=min(300, len(cam_signed)), replace=False)
    vs_mean = np.array([_rank_corr(cam_signed[i], mean_map) for i in sample])
    return {
        "n_neighbour_pairs": int(len(same)),
        "profile_corr_neighbour_mean": float(same.mean()),
        "profile_corr_different_song_mean": float(other.mean()),
        "spearman_vs_mean_map_mean": float(np.nanmean(vs_mean)),
        "spearman_vs_mean_map_sd": float(np.nanstd(vs_mean)),
        "same": same,
        "other": other,
    }


# ---------------------------------------------------------------------------
# 5. Evidence-grounded diagnostic summaries
# ---------------------------------------------------------------------------

def _mass_shares(positive_map: np.ndarray, masks: List[Tuple[str, np.ndarray]], axis: int) -> List[float]:
    total = positive_map.sum()
    if total <= 0:
        return [0.0] * len(masks)
    profile = positive_map.sum(axis=axis)
    return [float(profile[mask].sum() / total) for _, mask in masks]


def diagnose_window(
    cam_signed: np.ndarray,
    prediction: float,
    shapley: Optional[np.ndarray] = None,
    feature_names: Sequence[str] = TOP_8_ACOUSTIC_FEATURES,
) -> Dict[str, Any]:
    """Concise, numbers-only description of one window's heatmap."""
    raising = np.clip(cam_signed, 0, None)
    lowering = np.clip(-cam_signed, 0, None)
    summary: Dict[str, Any] = {"prediction": float(prediction)}
    parts = [f"Predicted valence {prediction:.2f}."]

    for label, pos_map in (("raising", raising), ("lowering", lowering)):
        band_shares = _mass_shares(pos_map, BAND_MASKS, axis=1)
        time_shares = _mass_shares(pos_map, TIME_MASKS, axis=0)
        summary[f"{label}_band_shares"] = dict(zip([n for n, _ in BAND_MASKS], band_shares))
        summary[f"{label}_time_shares"] = dict(zip([n for n, _ in TIME_MASKS], time_shares))
        if pos_map.sum() <= 0:
            parts.append(f"No valence-{label} evidence in this window.")
            continue
        b, t = int(np.argmax(band_shares)), int(np.argmax(time_shares))
        parts.append(
            f"Valence-{label} evidence is concentrated in the {BAND_MASKS[b][0]} band "
            f"({band_shares[b]:.0%} of its mass) and during {TIME_MASKS[t][0]} ({time_shares[t]:.0%})."
        )

    if shapley is not None:
        order = np.argsort(-np.abs(shapley))[:2]
        drivers = ", ".join(f"{FEATURE_LABELS.get(feature_names[i], feature_names[i])} {shapley[i]:+.3f}" for i in order)
        parts.append(f"Largest acoustic-feature contributions (Shapley, valence units): {drivers}.")
        summary["top_acoustic_drivers"] = {feature_names[i]: float(shapley[i]) for i in order}

    parts.append("This describes model sensitivity, not instruments or causes.")
    summary["text"] = " ".join(parts)
    return summary


def explain(
    model: nn.Module,
    input_window: Any,
    acoustic_features: Any = None,
    methods: Sequence[str] = ("gradcam",),
    background: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Shared contract: explain(model, input_window) -> attribution map and metadata.

    input_window: log-Mel window in dB, (128, 126) or (B, 1, 128, 126), as produced by
    spectrogram_engine / dynamic_windowing. acoustic_features: standardised Phase 1
    descriptors, (8,) or (B, 8), required for hybrid models.

    methods: any of 'gradcam', 'ig', 'deeplift', 'shapley' (hybrid only; needs background).
    """
    model.eval()
    specs = _as_spec_batch(input_window)
    acoustics = _as_acoustic_batch(acoustic_features, len(specs))
    if is_hybrid(model) and acoustics is None:
        raise ValueError("Hybrid models need acoustic_features.")
    if not is_hybrid(model):
        acoustics = None

    result: Dict[str, Any] = {"maps": {}, "metadata": {}}
    cam = gradcam_batch(model, specs, acoustics)
    result["prediction"] = cam["prediction"]
    result["metadata"].update({
        "model_type": getattr(model, "model_type", type(model).__name__),
        "target": "valence",
        "gradcam_layer": "backbone.block4 activation (16 x 15), bilinear upsample to 128 x 126",
        "gradcam_score": "pre-sigmoid output",
        "mel_frequencies_hz": MEL_FREQS_HZ,
        "frame_times_s": FRAME_TIMES_S,
        "caveat": CAVEAT,
    })
    if "gradcam" in methods:
        result["maps"]["gradcam_signed"] = cam["cam_signed"]
        result["maps"]["gradcam_raising"] = np.clip(cam["cam_signed"], 0, None)
        result["maps"]["gradcam_lowering"] = np.clip(-cam["cam_signed"], 0, None)

    for method in ("ig", "deeplift"):
        if method in methods:
            attr = captum_attribution(model, specs, acoustics, method)
            result["maps"][method] = attr["spec_attr"]
            if attr["acoustic_attr"] is not None:
                result["maps"][f"{method}_acoustic"] = attr["acoustic_attr"]
            result["metadata"][f"{method}_completeness"] = completeness_report(attr)

    if "shapley" in methods:
        if background is None:
            raise ValueError("Shapley values need a background of standardised training feature rows.")
        shap_result = exact_shapley_tabular(model, specs, acoustics, background)
        result["maps"]["shapley_tabular"] = shap_result["phi"]
        result["metadata"]["shapley_efficiency_error_max"] = float(shap_result["efficiency_error"].max())

    result["summaries"] = [
        diagnose_window(cam["cam_signed"][i], cam["prediction"][i],
                        result["maps"]["shapley_tabular"][i] if "shapley_tabular" in result["maps"] else None)
        for i in range(len(specs))
    ]
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def standardise_acoustics(raw: np.ndarray, preprocessing: Dict[str, Any]) -> torch.Tensor:
    """Apply the training-set median imputation and scaling stored in the checkpoint."""
    fill = np.asarray(preprocessing["imputer_statistics"], dtype=np.float64)
    mean = np.asarray(preprocessing["scaler_mean"], dtype=np.float64)
    scale = np.asarray(preprocessing["scaler_scale"], dtype=np.float64)
    x = np.where(np.isfinite(raw), raw, fill)
    return torch.from_numpy(((x - mean) / scale).astype(np.float32))


def load_windows(cache_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """Load the cached train/val windows. The test split is dropped immediately, so no
    analysis in this module can touch it."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Window cache not found at {cache_path}. Run src/train_cnn.py first "
            "(it builds data/processed/phase2_window_cache.pt)."
        )
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    splits = cached["splits"]
    splits.pop("test", None)
    return splits, cached.get("signature")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, name: str) -> Path:
    ExplainConfig.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = ExplainConfig.FIGURES_DIR / name
    fig.savefig(path, dpi=ExplainConfig.DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path.relative_to(PROJECT_ROOT)}")
    return path


def _spectrogram_axes(ax: plt.Axes, ylabel: bool = True) -> None:
    ticks_hz = [250, 500, 1000, 2000, 4000, 8000]
    idx = [int(np.argmin(np.abs(MEL_FREQS_HZ - f))) for f in ticks_hz]
    ax.set_yticks(idx)
    ax.set_yticklabels([f"{f / 1000:g}k" for f in ticks_hz])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlim(0, SEGMENT_SECONDS)
    ax.set_xlabel("Time in window (s)")
    ax.set_ylabel("Frequency (Hz, Mel axis)" if ylabel else "")
    ax.grid(False)


def _draw_overlay(ax: plt.Axes, spec: np.ndarray, attribution: np.ndarray, percentile: float = 100.0,
                  threshold: float = 0.15, alpha: float = 0.62) -> None:
    """Spectrogram in grey, signed attribution on top (red raises valence, blue lowers it)."""
    ax.imshow(spec, origin="lower", aspect="auto", cmap="gray", vmin=-TOP_DB, vmax=0,
              extent=[0, SEGMENT_SECONDS, 0, N_MELS], interpolation="nearest")
    scale = np.percentile(np.abs(attribution), percentile) + 1e-12
    norm = np.clip(attribution / scale, -1, 1)
    ax.imshow(np.ma.masked_where(np.abs(norm) < threshold, norm), origin="lower", aspect="auto",
              cmap=DIVERGING_CMAP, vmin=-1, vmax=1, alpha=alpha,
              extent=[0, SEGMENT_SECONDS, 0, N_MELS], interpolation="bilinear")
    _spectrogram_axes(ax)


def _band_label_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(60, 11025)
    ax.set_xticks([100, 250, 500, 1000, 2000, 4000, 8000])
    ax.set_xticklabels(["100", "250", "500", "1k", "2k", "4k", "8k"])
    ax.set_xlabel("Frequency (Hz)")


def figure_examples(name: str, specs: torch.Tensor, cam: np.ndarray, pred: np.ndarray,
                    true: np.ndarray, music_ids: Sequence[str], model_type: str) -> None:
    """Three highest- and three lowest-valence songs (by human rating), one window each."""
    ids = np.asarray(music_ids)
    song_true = pd.Series(true).groupby(ids).first().sort_values()
    chosen = [("lowest-rated songs", list(song_true.index[:3])), ("highest-rated songs", list(song_true.index[-3:][::-1]))]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey=True)
    for row, (title, songs) in enumerate(chosen):
        for col, song in enumerate(songs):
            window_ids = np.where(ids == song)[0]
            i = int(window_ids[len(window_ids) // 2])
            ax = axes[row, col]
            _draw_overlay(ax, specs[i, 0].numpy(), cam[i])
            ax.set_title(f"song {song}: human {true[i]:.2f}, model {pred[i]:.2f}", fontsize=10, pad=6)
            if col > 0:
                ax.set_ylabel("")
            if row == 0:
                ax.set_xlabel("")
        axes[row, 0].annotate(title, xy=(0, 1.16), xycoords="axes fraction", fontsize=11, fontweight="bold")
    handles = [Line2D([0], [0], color=DIVERGING_CMAP(0.95), lw=6, label="raises predicted valence"),
               Line2D([0], [0], color=DIVERGING_CMAP(0.05), lw=6, label="lowers predicted valence")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Grad-CAM on log-Mel windows: {model_type.replace('_', ' ')} (validation split)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, name)


def figure_band_time(name: str, cam: np.ndarray, true: np.ndarray, model_type: str) -> pd.DataFrame:
    """Mean signed Grad-CAM by frequency and by time, lowest vs highest human-rated tercile."""
    lo, hi = np.quantile(true, [1 / 3, 2 / 3])
    groups = [("lowest-rated third", true <= lo, BLUE), ("highest-rated third", true >= hi, ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for label, mask, color in groups:
        sub = cam[mask]
        band = sub.mean(axis=2)
        time = sub.mean(axis=1)
        for ax, x, values, axis_name in ((axes[0], MEL_FREQS_HZ, band, "frequency"), (axes[1], FRAME_TIMES_S, time, "time")):
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            ax.plot(x, mean, color=color, lw=2, label=f"{label} (n = {mask.sum()} windows)")
            ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.18, lw=0)
        pos = np.clip(sub, 0, None)
        neg = np.clip(-sub, 0, None)
        for band_name, band_mask in BAND_MASKS:
            rows.append({
                "model": model_type, "group": label, "band": band_name,
                "share_of_raising_mass": float(pos.mean(axis=0)[band_mask].sum() / pos.mean(axis=0).sum()),
                "share_of_lowering_mass": float(neg.mean(axis=0)[band_mask].sum() / neg.mean(axis=0).sum()),
                "mean_signed_cam": float(sub.mean(axis=0)[band_mask].mean()),
            })
    for ax in axes:
        ax.axhline(0, color=AXIS, lw=1)
    _band_label_axis(axes[0])
    axes[0].set_ylabel("Mean signed Grad-CAM (pre-sigmoid units)")
    axes[0].set_title("By frequency", fontsize=11, pad=8)
    axes[1].set_xlabel("Time in window (s)")
    axes[1].set_title("By time", fontsize=11, pad=8)
    axes[0].legend(loc="best", fontsize=8.5)
    fig.suptitle(f"Where the model finds valence evidence: {model_type.replace('_', ' ')} (mean ± 95% CI)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, name)
    return pd.DataFrame(rows)


def figure_neural_attribution(name: str, spec: np.ndarray, cam: np.ndarray, ig: np.ndarray,
                              dl: np.ndarray, gaps: Dict[str, float], header: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    axes[0].imshow(spec, origin="lower", aspect="auto", cmap="gray", vmin=-TOP_DB, vmax=0,
                   extent=[0, SEGMENT_SECONDS, 0, N_MELS])
    _spectrogram_axes(axes[0])
    axes[0].set_title("log-Mel window (dB)", fontsize=10.5, pad=6)
    dl_flag = "assumptions not met" if gaps["deeplift"] > ExplainConfig.COMPLETENESS_REL_TOL else "complete"
    panels = [(axes[1], cam, "Grad-CAM"), (axes[2], ig, "Integrated Gradients"),
              (axes[3], dl, f"DeepLIFT (gap {gaps['deeplift']:.0%}: {dl_flag})")]
    for ax, attribution, title in panels:
        _draw_overlay(ax, spec, attribution, percentile=99.0, threshold=0.2)
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.set_ylabel("")
    fig.suptitle(header, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, name)


def figure_completeness(name: str, results: Dict[str, Dict[str, Any]], model_type: str,
                        reports: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (method, res) in zip(axes, results.items()):
        x = res["f_x"] - res["f_baseline"]
        y = res["attr_sum"]
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        pad = 0.05 * (lim[1] - lim[0] + 1e-9)
        ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], color=AXIS, lw=1.2, zorder=1)
        ax.scatter(x, y, s=16, color=BLUE if method == "ig" else ORANGE, alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=2)
        rep = reports[method]
        ax.set_title(f"{'Integrated Gradients' if method == 'ig' else 'DeepLIFT'}: median gap "
                     f"{rep['median_relative_gap']:.1%}", fontsize=10.5, pad=8)
        ax.set_xlabel("f(x) − f(silence)  (valence units)")
        ax.set_ylabel("Sum of attributions")
    fig.suptitle(f"Completeness check: {model_type.replace('_', ' ')} (points on the line = complete)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, name)


def figure_randomization(name: str, table: pd.DataFrame, model_type: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(table))
    for column, label, color in (("spearman_signed", "signed map", BLUE), ("spearman_abs", "absolute map", ORANGE)):
        m, s = table[f"{column}_mean"].to_numpy(), table[f"{column}_sd"].to_numpy()
        ax.plot(x, m, color=color, lw=2, marker="o", markeredgecolor="white", ms=7, label=label)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(table["stage"], rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ with the trained-model map")
    ax.set_xlabel("Layers re-initialised (output towards input)")
    ax.set_ylim(-0.4, 1.05)
    ax.legend(loc="upper right")
    ax.set_title(f"Randomisation sanity check: {model_type.replace('_', ' ')}", pad=12)
    save_figure(fig, name)


def figure_faithfulness(name: str, table: pd.DataFrame, model_type: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, baseline, title in ((axes[0], "silence", "Masked area set to silence (−80 dB)"),
                                (axes[1], "mean", "Masked area set to the window mean")):
        sub = table[table["mask_baseline"] == baseline]
        x = sub["fraction"] * 100
        ax.plot(x, sub["delta_remove_raising"], color=ORANGE, lw=2, marker="o", markeredgecolor="white", ms=7,
                label="remove 'raising' regions")
        ax.plot(x, sub["delta_remove_lowering"], color=BLUE, lw=2, marker="o", markeredgecolor="white", ms=7,
                label="remove 'lowering' regions")
        ax.plot(x, sub["delta_random_shifted"], color=INK_MUTED, lw=2, ls="--", marker="s", markeredgecolor="white",
                ms=6, label="same-shape random region")
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlabel("Share of spectrogram masked (%)")
        ax.set_title(title, fontsize=10.5, pad=8)
    axes[0].set_ylabel("Mean change in predicted valence")
    axes[0].legend(loc="best", fontsize=8.5)
    fig.suptitle(f"Deletion faithfulness: {model_type.replace('_', ' ')} (n = {int(table['n_windows'].iloc[0])} windows, {int(table['n_songs'].iloc[0])} songs)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, name)


def figure_consistency(name: str, check: Dict[str, Any], model_type: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bins = np.linspace(-1, 1, 41)
    ax.hist(check["other"], bins=bins, color=INK_MUTED, alpha=0.55, label="windows from different songs")
    ax.hist(check["same"], bins=bins, color=ORANGE, alpha=0.65, label="neighbouring windows, same song")
    ax.set_xlabel("Correlation of frequency profiles of the Grad-CAM maps")
    ax.set_ylabel("Window pairs")
    ax.legend(loc="upper left")
    ax.set_title(f"Consistency across overlapping windows: {model_type.replace('_', ' ')}", pad=12)
    save_figure(fig, name)


def figure_shapley(name: str, phi: np.ndarray, acoustics: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    mean_abs = np.abs(phi).mean(axis=0)
    order = np.argsort(mean_abs)
    sem = np.abs(phi).std(axis=0, ddof=1) / np.sqrt(len(phi))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    axes[0].barh(np.arange(len(order)), mean_abs[order], xerr=1.96 * sem[order], color=BLUE, height=0.62,
                 error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.0, capsize=0))
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([labels[i] for i in order])
    axes[0].set_xlabel("Mean |Shapley value| (valence units, ± 95% CI)")
    axes[0].set_title("Importance", fontsize=11, pad=8)
    axes[0].grid(axis="y", visible=False)
    for row, i in enumerate(order):
        values = acoustics[:, i]
        colour = (values - values.min()) / (values.max() - values.min() + 1e-12)
        axes[1].scatter(phi[:, i], row + rng.uniform(-0.28, 0.28, len(phi)), c=colour, cmap=DIVERGING_CMAP, s=14,
                        alpha=0.8, edgecolors="none")
    axes[1].axvline(0, color=AXIS, lw=1)
    axes[1].set_xlabel("Shapley value (effect on predicted valence)")
    axes[1].set_title("Direction (red = high feature value, blue = low)", fontsize=11, pad=8)
    axes[1].grid(axis="y", visible=False)
    fig.suptitle(f"Tabular branch of the hybrid model: exact Shapley values (n = {len(phi)} windows)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, name)
    return pd.DataFrame({"feature": list(feature_names), "label": labels, "mean_abs_shapley": mean_abs,
                         "mean_signed_shapley": phi.mean(axis=0)}).sort_values("mean_abs_shapley", ascending=False)


def figure_modality(name: str, ig: Dict[str, Any], model_type: str) -> Dict[str, float]:
    spec_net = ig["spec_attr"].reshape(len(ig["spec_attr"]), -1).sum(axis=1)
    acoustic_net = ig["acoustic_attr"].sum(axis=1)
    deviation = ig["f_x"] - ig["f_baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar([0, 1], [np.abs(spec_net).mean(), np.abs(acoustic_net).mean()], color=[BLUE, ORANGE], width=0.55)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Spectrogram", "8 acoustic\ndescriptors"])
    axes[0].set_ylabel("Mean |net attribution| (valence units)")
    axes[0].set_title("Size of each modality's contribution", fontsize=11, pad=8)
    axes[0].grid(axis="x", visible=False)
    axes[1].scatter(spec_net, acoustic_net, c=deviation, cmap=DIVERGING_CMAP, s=18, alpha=0.85, edgecolors="white",
                    linewidths=0.4)
    axes[1].axhline(0, color=AXIS, lw=1)
    axes[1].axvline(0, color=AXIS, lw=1)
    axes[1].set_xlabel("Net spectrogram attribution")
    axes[1].set_ylabel("Net acoustic-feature attribution")
    axes[1].set_title("Per window (colour = f(x) − f(silence))", fontsize=11, pad=8)
    fig.suptitle(f"Modality split of the {model_type.replace('_', ' ')} model (Integrated Gradients)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, name)
    return {
        "mean_abs_spectrogram": float(np.abs(spec_net).mean()),
        "mean_abs_acoustic": float(np.abs(acoustic_net).mean()),
        "corr_spectrogram_with_deviation": float(np.corrcoef(spec_net, deviation)[0, 1]),
        "corr_acoustic_with_deviation": float(np.corrcoef(acoustic_net, deviation)[0, 1]),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def model_skill(model: nn.Module, split: Dict[str, Any], acoustics: Optional[torch.Tensor]) -> Dict[str, float]:
    """Window- and song-level fit on the explained split, for context."""
    pred = predict_batch(model, split["spectrograms"], acoustics)
    true = split["targets"][:, 0]
    frame = pd.DataFrame({"id": split["music_ids"], "pred": pred, "true": true})
    song = frame.groupby("id").mean()
    return {
        "window_r2": float(r2_score(true, pred)),
        "window_pearson": float(np.corrcoef(true, pred)[0, 1]),
        "song_r2": float(r2_score(song["true"], song["pred"])),
        "song_pearson": float(np.corrcoef(song["true"], song["pred"])[0, 1]),
        "n_windows": int(len(true)),
        "n_songs": int(len(song)),
    }


def run_for_model(model_type: str, checkpoint: Path, splits: Dict[str, Dict[str, Any]],
                  device: torch.device, seed: int) -> Dict[str, Any]:
    logger.info(f"=== {model_type}: {checkpoint.name} ===")
    model, meta = load_checkpoint(checkpoint, device=device)
    model.eval()
    preprocessing = meta["preprocessing_metadata"]
    feature_names = meta.get("selected_features", TOP_8_ACOUSTIC_FEATURES)

    val = splits[ExplainConfig.EXPLAIN_SPLIT]
    specs = val["spectrograms"].float()
    true = val["targets"][:, 0]
    music_ids, window_idx = val["music_ids"], val["window_indices"]
    hybrid = is_hybrid(model)
    acoustics = standardise_acoustics(val["acoustics_raw"], preprocessing) if hybrid else None

    results: Dict[str, Any] = {"checkpoint": checkpoint.name, "model_type": model_type,
                               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                               "checkpoint_version": meta.get("checkpoint_version"),
                               "n_val_windows": int(len(specs))}
    results["skill_on_validation"] = model_skill(model, val, acoustics)
    logger.info(f"Validation fit: {results['skill_on_validation']}")

    # Grad-CAM on every validation window
    cam_all = gradcam_batch(model, specs, acoustics)
    cam, pred = cam_all["cam_signed"], cam_all["prediction"]
    figure_examples(f"01_gradcam_examples_{model_type}.png", specs, cam, pred, true, music_ids, model_type)
    band_table = figure_band_time(f"02_gradcam_band_time_{model_type}.png", cam, true, model_type)
    band_table.to_csv(ExplainConfig.REPORTS_DIR / f"phase2_gradcam_band_attribution_{model_type}.csv", index=False)

    # Layer-choice robustness
    rng = np.random.default_rng(seed)
    subset = np.sort(rng.choice(len(specs), size=min(ExplainConfig.N_ATTR_WINDOWS, len(specs)), replace=False))
    s_sub = specs[subset]
    a_sub = acoustics[subset] if acoustics is not None else None
    conv_cam = gradcam_batch(model, s_sub, a_sub, layer="conv")["cam_signed"]
    results["gradcam_layer_agreement_spearman"] = float(np.nanmean(
        [_rank_corr(a, b) for a, b in zip(cam[subset], conv_cam)]))

    # IG and DeepLIFT
    attributions: Dict[str, Dict[str, Any]] = {}
    reports: Dict[str, Dict[str, float]] = {}
    for method in ("ig", "deeplift"):
        try:
            attributions[method] = captum_attribution(model, s_sub, a_sub, method)
            reports[method] = completeness_report(attributions[method])
            logger.info(f"{method} completeness: {reports[method]}")
        except Exception as exc:  # DeepLIFT cannot always be applied to every layer type
            logger.warning(f"{method} unavailable: {exc}")
            results[f"{method}_error"] = str(exc)
    results["completeness"] = reports
    diag_table = deeplift_assumption_diagnostic(model, s_sub, a_sub)
    diag_table.to_csv(ExplainConfig.REPORTS_DIR / f"phase2_deeplift_assumptions_{model_type}.csv", index=False)
    results["deeplift_assumption_diagnostic"] = diag_table.to_dict(orient="records")
    if attributions:
        figure_completeness(f"04_completeness_check_{model_type}.png", attributions, model_type, reports)
    if "ig" in attributions:
        ig_pool = F.adaptive_avg_pool2d(torch.from_numpy(attributions["ig"]["spec_attr"])[:, None], (16, 15))[:, 0].numpy()
        cam_pool = F.adaptive_avg_pool2d(torch.from_numpy(cam[subset])[:, None], (16, 15))[:, 0].numpy()
        results["ig_vs_gradcam_spearman_16x15"] = float(np.nanmean([_rank_corr(a, b) for a, b in zip(ig_pool, cam_pool)]))
    if "ig" in attributions and "deeplift" in attributions:
        k = int(np.argmax(np.abs(true[subset] - true[subset].mean())))
        figure_neural_attribution(
            f"03_neural_attribution_{model_type}.png", s_sub[k, 0].numpy(), cam[subset][k],
            attributions["ig"]["spec_attr"][k], attributions["deeplift"]["spec_attr"][k],
            {"deeplift": reports["deeplift"]["median_relative_gap"]},
            f"{model_type.replace('_', ' ')}: song {music_ids[subset[k]]}, human valence {true[subset][k]:.2f}, "
            f"model {pred[subset][k]:.2f}")

    # Validation of the attributions
    n_check = min(60, len(subset))
    rand_table = cascading_randomization(model, s_sub[:n_check], a_sub[:n_check] if a_sub is not None else None, seed)
    rand_table.to_csv(ExplainConfig.REPORTS_DIR / f"phase2_randomization_test_{model_type}.csv", index=False)
    figure_randomization(f"05_sanity_randomization_{model_type}.png", rand_table, model_type)
    results["randomization"] = rand_table.to_dict(orient="records")

    delete_table = deletion_faithfulness(model, s_sub, a_sub, cam[subset],
                                         [music_ids[i] for i in subset], seed)
    delete_table.to_csv(ExplainConfig.REPORTS_DIR / f"phase2_deletion_faithfulness_{model_type}.csv", index=False)
    figure_faithfulness(f"06_faithfulness_deletion_{model_type}.png", delete_table, model_type)
    results["deletion"] = delete_table.to_dict(orient="records")

    check = consistency_checks(cam, music_ids, window_idx, seed)
    figure_consistency(f"07_consistency_{model_type}.png", check, model_type)
    results["consistency"] = {k: v for k, v in check.items() if k not in ("same", "other")}

    # Tabular branch and modality split (hybrid only)
    shapley_summary = None
    if hybrid:
        train_acoustics = standardise_acoustics(splits["train"]["acoustics_raw"], preprocessing)
        bg_idx = rng.choice(len(train_acoustics), size=min(ExplainConfig.SHAPLEY_BACKGROUND, len(train_acoustics)),
                            replace=False)
        shap_out = exact_shapley_tabular(model, s_sub, a_sub, train_acoustics[bg_idx])
        results["shapley"] = {
            "n_windows": int(len(s_sub)),
            "max_efficiency_error": float(shap_out["efficiency_error"].max()),
            "max_consistency_error": float(shap_out["consistency_error"].max()),
            "background_windows": int(len(bg_idx)),
        }
        shapley_summary = figure_shapley("08_tabular_shapley_hybrid_fusion.png", shap_out["phi"],
                                         a_sub.numpy(), feature_names)
        shapley_summary.to_csv(ExplainConfig.REPORTS_DIR / "phase2_tabular_shapley_hybrid_fusion.csv", index=False)
        results["shapley"]["ranking"] = shapley_summary[["feature", "mean_abs_shapley", "mean_signed_shapley"]].to_dict(
            orient="records")
        window_table = pd.DataFrame(shap_out["phi"], columns=list(feature_names))
        window_table.insert(0, "music_id", [music_ids[i] for i in subset])
        window_table.insert(1, "window_index", [window_idx[i] for i in subset])
        window_table.to_csv(ExplainConfig.REPORTS_DIR / "phase2_tabular_shapley_windows_hybrid_fusion.csv", index=False)
        # Correlated inputs share credit; strongly anti-correlated Shapley columns mean the
        # model uses the pair as a contrast, so single-feature readings are unsafe.
        phi_corr = pd.DataFrame(shap_out["phi"], columns=list(feature_names)).corr()
        pairs = [(a, b, float(phi_corr.loc[a, b])) for i, a in enumerate(feature_names)
                 for b in feature_names[i + 1:]]
        results["shapley"]["most_anticorrelated_pairs"] = sorted(pairs, key=lambda p: p[2])[:3]
        results["shapley"]["most_correlated_pairs"] = sorted(pairs, key=lambda p: -p[2])[:3]
        if "ig" in attributions:
            results["modality_split"] = figure_modality("09_modality_split_hybrid_fusion.png",
                                                       attributions["ig"], model_type)
        results["window_diagnostics"] = [
            diagnose_window(cam[subset][i], pred[subset][i], shap_out["phi"][i], feature_names)["text"]
            for i in range(min(5, len(subset)))
        ]
    else:
        results["window_diagnostics"] = [
            diagnose_window(cam[subset][i], pred[subset][i])["text"] for i in range(min(5, len(subset)))
        ]
    return results


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HarmoniX Phase 2: explain the spectrogram models.")
    parser.add_argument("--checkpoint-dir", type=Path, default=ExplainConfig.MODELS_DIR,
                        help="Folder with cnn_only_best.pt / hybrid_fusion_best.pt")
    parser.add_argument("--cache-path", type=Path, default=ExplainConfig.CACHE_PATH)
    parser.add_argument("--models", nargs="+", default=["cnn_only", "hybrid_fusion"],
                        choices=sorted(ExplainConfig.CHECKPOINTS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"],
                        help="CPU by default so the maps are reproducible")
    parser.add_argument("--seed", type=int, default=ExplainConfig.RANDOM_SEED)
    parser.add_argument("--summary-name", default="phase2_deep_explainability_summary.json")
    parser.add_argument("--train-command", default=None,
                        help="Command that produced the checkpoints (recorded in the summary)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    apply_plot_style()
    ExplainConfig.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    logger.info("=================================================================")
    logger.info("HarmoniX Phase 2 | Explainable AI & Spectrogram Heatmaps (Praanshu)")
    logger.info("=================================================================")
    splits, cache_signature = load_windows(args.cache_path)
    logger.info(f"Explaining the '{ExplainConfig.EXPLAIN_SPLIT}' split only "
                f"({splits[ExplainConfig.EXPLAIN_SPLIT]['n_songs']} songs, "
                f"{splits[ExplainConfig.EXPLAIN_SPLIT]['n_windows']} windows).")

    summary: Dict[str, Any] = {
        "split_explained": ExplainConfig.EXPLAIN_SPLIT, "device": str(device), "seed": args.seed,
        "torch_version": torch.__version__, "window_cache_signature": cache_signature,
        "train_command": args.train_command, "caveat": CAVEAT, "models": {},
    }
    for model_type in args.models:
        checkpoint = args.checkpoint_dir / ExplainConfig.CHECKPOINTS[model_type]
        if not checkpoint.exists():
            logger.warning(f"Checkpoint not found, skipping {model_type}: {checkpoint}")
            continue
        summary["models"][model_type] = run_for_model(model_type, checkpoint, splits, device, args.seed)

    if not summary["models"]:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}. Run src/train_cnn.py first.")
    out = ExplainConfig.REPORTS_DIR / args.summary_name
    out.write_text(json.dumps(_to_jsonable(summary), indent=2))
    logger.info(f"Saved summary: {out.relative_to(PROJECT_ROOT)}")
    logger.info("Deep explainability complete.")


if __name__ == "__main__":
    main()
