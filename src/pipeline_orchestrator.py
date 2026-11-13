"""
HarmoniX | End-to-End Pipeline Orchestrator
Author: Adnan Rizvi (Lead Architect & Product Engineering Lead)

Part of Phase 2 Master Plan:
Orchestrates the entire prediction pipeline from uploaded audio to UI presentation:
1. Normalizes and validates audio via multicultural_loader.
2. Extracts 34 acoustic & DSP descriptors.
3. Runs model inference across trained ML and fusion architectures.
4. Generates dynamic second-by-second valence trajectories over time.
5. Computes explainability attributions (SHAP drivers).
6. Maps emotional coordinates onto Russell's 2D Circumplex space.

Shared Contract (Section 5.2):
    predict_audio(file) -> structured predictions, timestamps, and optional explanations
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import xgboost
import soundfile as sf
import tempfile
import librosa
import torch

warnings.filterwarnings("ignore")

from src.multicultural_loader import load_and_normalize, TARGET_SAMPLE_RATE
from src.feature_extraction import extract_features
from src.spectrogram_engine import (
    N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW, FMIN, FMAX, TOP_DB
)
from src.dynamic_windowing import process_audio
from src.models_cnn import load_checkpoint, TOP_8_ACOUSTIC_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HarmoniX.PipelineOrchestrator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"


def get_emotional_quadrant(valence: float, arousal: float) -> Tuple[str, str, str]:
    """
    Map (valence, arousal) coordinates to Russell's Circumplex quadrant.
    Returns: (Quadrant Name, Emoji, Description)
    """
    v_mid, a_mid = 0.50, 0.50

    if valence >= v_mid and arousal >= a_mid:
        return (
            "Q1: Joyful / Energetic",
            "",
            "Uplifting harmonies, bright spectral balance, and lively energy."
        )
    elif valence >= v_mid and arousal < a_mid:
        return (
            "Q4: Calm / Peaceful",
            "",
            "Harmonious chords, gentle dynamics, relaxing and comforting tone."
        )
    elif valence < v_mid and arousal < a_mid:
        return (
            "Q3: Melancholic / Sad",
            "",
            "Subdued dynamics, minor key signatures, reflective or sorrowful mood."
        )
    else:
        return (
            "Q2: Tense / Agitated",
            "",
            "High acoustic dissonance, abrasive textures, intense or tense energy."
        )


class HarmoniXOrchestrator:
    """Master pipeline orchestrator handling single-song and batch inference."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self.hybrid_model: Optional[Any] = None
        self.cnn_model: Optional[Any] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self._load_artifacts()

    def _load_artifacts(self):
        if not FEATURE_NAMES_PATH.exists():
            raise FileNotFoundError(f"Feature schema not found at: {FEATURE_NAMES_PATH}")

        with open(FEATURE_NAMES_PATH, "r") as f:
            self.feature_cols = json.load(f)

        meta_path = MODELS_DIR / "phase2_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    p2_meta = json.load(f)
                prep = p2_meta.get("preprocessing_metadata", {})
                self.scaler_mean = np.array(prep["scaler_mean"], dtype=float)
                self.scaler_scale = np.array(prep["scaler_scale"], dtype=float)
            except Exception as e:
                logger.warning(f"Failed to load phase 2 metadata scaler: {e}")

        hybrid_path = MODELS_DIR / "hybrid_fusion_best.pt"
        if hybrid_path.exists():
            try:
                from src.models_cnn import build_model
                payload = torch.load(hybrid_path, map_location="cpu", weights_only=False)
                model_config = dict(payload["model_config"])
                model_type = model_config.pop("model_type")
                self.hybrid_model = build_model(model_type, **model_config)
                self.hybrid_model.load_state_dict(payload["state_dict"])
                self.hybrid_model.eval()
                logger.info("Loaded Phase 2 Hybrid Fusion deep learning model.")
            except Exception as e:
                logger.warning(f"Failed to load hybrid fusion checkpoint: {e}")

        cnn_path = MODELS_DIR / "cnn_only_best.pt"
        if cnn_path.exists():
            try:
                from src.models_cnn import build_model
                payload = torch.load(cnn_path, map_location="cpu", weights_only=False)
                model_config = dict(payload["model_config"])
                model_type = model_config.pop("model_type")
                self.cnn_model = build_model(model_type, **model_config)
                self.cnn_model.load_state_dict(payload["state_dict"])
                self.cnn_model.eval()
                logger.info("Loaded Phase 2 CNN-Only deep learning model.")
            except Exception as e:
                logger.warning(f"Failed to load cnn only checkpoint: {e}")

        model_registry = {
            "Random_Forest": MODELS_DIR / "random_forest_best.joblib",
            "Ridge": MODELS_DIR / "ridge_best.joblib"
        }

        for name, path in model_registry.items():
            if path.exists():
                try:
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded pipeline model: {name}")
                except Exception as e:
                    logger.warning(f"Could not load classical model {name}: {e}")

        if "Random_Forest" not in self.models and self.hybrid_model is None:
            raise RuntimeError("Neither primary deep learning nor Random_Forest model could be loaded.")

    def compute_dynamic_trajectory(
        self,
        waveform: np.ndarray,
        sr: int,
        window_seconds: float = 3.0,
        hop_seconds: float = 1.5,
        features_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Produce a dynamic sequence of timestamped valence predictions across audio.
        Uses sliding rolling windows to capture musical emotional transitions.
        """
        total_samples = len(waveform)
        duration_sec = total_samples / sr

        if total_samples < int(window_seconds * sr):
            return [{
                "time_start_sec": 0.0,
                "time_end_sec": round(duration_sec, 2),
                "timestamp_label": "00:00",
                "valence": 0.50
            }]

        if self.hybrid_model is not None and self.scaler_mean is not None:
            try:
                windows_tensor = process_audio(waveform, sample_rate=sr)
                raw_feats = features_dict or {}
                raw_vec = np.array([raw_feats.get(k, 0.0) for k in TOP_8_ACOUSTIC_FEATURES], dtype=float)
                scaled_vec = (raw_vec - self.scaler_mean) / self.scaler_scale
                n_windows = windows_tensor.shape[0]
                scaled_batch = torch.from_numpy(np.repeat(scaled_vec[np.newaxis, :], n_windows, axis=0)).float()

                with torch.no_grad():
                    preds = self.hybrid_model(windows_tensor, scaled_batch).squeeze().tolist()
                    if not isinstance(preds, list):
                        preds = [preds]

                trajectory = []
                for i, pred_val in enumerate(preds):
                    start_sec = i * hop_seconds
                    end_sec = min(start_sec + window_seconds, duration_sec)
                    mins = int(start_sec // 60)
                    secs = int(start_sec % 60)
                    trajectory.append({
                        "time_start_sec": round(start_sec, 2),
                        "time_end_sec": round(end_sec, 2),
                        "timestamp_label": f"{mins:02d}:{secs:02d}",
                        "valence": round(float(np.clip(pred_val, 0.0, 1.0)), 4)
                    })
                return trajectory
            except Exception:
                pass

        rf_pipeline = self.models["Random_Forest"]
        window_samples = int(window_seconds * sr)
        hop_samples = int(hop_seconds * sr)

        trajectory = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "chunk.wav"

            for start in range(0, total_samples - window_samples + 1, hop_samples):
                end = start + window_samples
                chunk = waveform[start:end]

                sf.write(str(tmp_path), chunk, sr)
                try:
                    feat_dict = extract_features(tmp_path)
                    X_chunk = np.array([[feat_dict[col] for col in self.feature_cols]], dtype=float)
                    pred_v = float(np.clip(rf_pipeline.predict(X_chunk)[0], 0.0, 1.0))
                except Exception:
                    pred_v = 0.50

                start_sec = start / sr
                end_sec = end / sr
                mins = int(start_sec // 60)
                secs = int(start_sec % 60)

                trajectory.append({
                    "time_start_sec": round(start_sec, 2),
                    "time_end_sec": round(end_sec, 2),
                    "timestamp_label": f"{mins:02d}:{secs:02d}",
                    "valence": round(pred_v, 4)
                })

        return trajectory

    def predict_audio(
        self,
        audio_file: Path,
        compute_trajectory: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end inference on any audio file.
        Returns complete structured telemetry for the UI.
        """
        audio_file = Path(audio_file)
        waveform, sr, metadata = load_and_normalize(audio_file)

        features_dict = extract_features(audio_file)
        X_full = np.array([[features_dict[col] for col in self.feature_cols]], dtype=float)

        model_scores = {}

        if self.hybrid_model is not None and self.scaler_mean is not None:
            try:
                windows_tensor = process_audio(waveform, sample_rate=sr)
                raw_vec = np.array([features_dict.get(k, 0.0) for k in TOP_8_ACOUSTIC_FEATURES], dtype=float)
                scaled_vec = (raw_vec - self.scaler_mean) / self.scaler_scale
                n_windows = windows_tensor.shape[0]
                scaled_batch = torch.from_numpy(np.repeat(scaled_vec[np.newaxis, :], n_windows, axis=0)).float()

                with torch.no_grad():
                    hyb_preds = self.hybrid_model(windows_tensor, scaled_batch)
                    avg_hybrid = float(np.clip(hyb_preds.mean().item(), 0.0, 1.0))
                    model_scores["Hybrid Fusion (Phase 2 Champion)"] = round(avg_hybrid, 4)

                    if self.cnn_model is not None:
                        cnn_preds = self.cnn_model(windows_tensor)
                        avg_cnn = float(np.clip(cnn_preds.mean().item(), 0.0, 1.0))
                        model_scores["2D-CNN (Spectrogram Only)"] = round(avg_cnn, 4)
            except Exception as e:
                logger.warning(f"Failed to compute deep learning prediction: {e}")

        gradcam_payload = None
        if self.hybrid_model is not None and self.scaler_mean is not None:
            try:
                from src.deep_explainability import explain
                explain_res = explain(
                    self.hybrid_model,
                    windows_tensor[0:1],
                    scaled_batch[0:1],
                    methods=("gradcam",)
                )
                gradcam_payload = {
                    "signed_map": explain_res["maps"]["gradcam_signed"][0].tolist(),
                    "raising_map": explain_res["maps"]["gradcam_raising"][0].tolist(),
                    "lowering_map": explain_res["maps"]["gradcam_lowering"][0].tolist(),
                    "summary": explain_res["summaries"][0] if explain_res.get("summaries") else {},
                    "model_name": "Hybrid Fusion (Phase 2 Champion)"
                }
            except Exception as e:
                logger.warning(f"Grad-CAM explanation failed: {e}")

        for name, pipe in self.models.items():
            pred = float(np.clip(pipe.predict(X_full)[0], 0.0, 1.0))
            label = f"{name} (Phase 1 Baseline)"
            model_scores[label] = round(pred, 4)

        if "Hybrid Fusion (Phase 2 Champion)" in model_scores:
            primary_valence = model_scores["Hybrid Fusion (Phase 2 Champion)"]
        else:
            primary_valence = model_scores.get("Random_Forest (Phase 1 Baseline)", 0.50)

        rms = features_dict.get("rms_mean", 0.10)
        centroid = features_dict.get("spectral_centroid_mean", 2200.0)
        estimated_arousal = float(np.clip(0.30 + (rms * 1.5) + (centroid / 10000.0), 0.15, 0.95))

        quadrant, emoji, mood_desc = get_emotional_quadrant(primary_valence, estimated_arousal)

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=WINDOW,
            center=False,
            power=2.0,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            norm="slaney"
        )
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)

        acoustic_highlights = {
            "Brightness (Spectral Centroid)": f"{features_dict.get('spectral_centroid_mean', 0):.1f} Hz",
            "Harmonic Purity (Spectral Flatness)": f"{features_dict.get('spectral_flatness_mean', 0):.4f}",
            "Consonance (Sensory Dissonance)": f"{features_dict.get('sensory_dissonance_estimate', 0):.2f}",
            "Harmonicity (HNR)": f"{features_dict.get('hnr_estimate_db', 0):.1f} dB",
            "RMS Energy (Loudness)": f"{features_dict.get('rms_mean', 0):.3f}"
        }

        if compute_trajectory:
            trajectory = self.compute_dynamic_trajectory(
                waveform, sr, window_seconds=3.0, hop_seconds=1.5, features_dict=features_dict
            )
        else:
            trajectory = []

        return {
            "status": "success",
            "metadata": metadata,
            "primary_valence": round(primary_valence, 3),
            "estimated_arousal": round(estimated_arousal, 3),
            "emotional_quadrant": quadrant,
            "emoji": emoji,
            "mood_description": mood_desc,
            "model_predictions": model_scores,
            "acoustic_highlights": acoustic_highlights,
            "trajectory": trajectory,
            "spectrogram": {
                "matrix": log_mel,
                "n_mels": N_MELS,
                "duration_sec": metadata["duration_seconds"],
                "fmin": FMIN,
                "fmax": FMAX,
            },
            "gradcam": gradcam_payload,
            "features": features_dict
        }


_orchestrator: Optional[HarmoniXOrchestrator] = None


def get_orchestrator() -> HarmoniXOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HarmoniXOrchestrator()
    return _orchestrator


def predict_audio(audio_file: Path) -> Dict[str, Any]:
    """Top-level shared contract function."""
    orchestrator = get_orchestrator()
    return orchestrator.predict_audio(audio_file)

