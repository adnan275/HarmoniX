"""
HarmoniX | Multicultural & Multi-Genre Audio Ingestion Engine
Author: Adnan Rizvi (Lead Architect & Product Engineering Lead)

Part of Phase 2 Master Plan:
Eliminates Western/English music bias by providing:
1. Universal audio normalization (stereo-to-mono, 22,050 Hz resampling, peak normalization).
2. Robust decoding across MP3, WAV, FLAC, OGG, and M4A.
3. Multi-genre dataset scanning (Bollywood, Punjabi, Sufi, Indian Classical, Western).
4. Audio validation safeguards (detecting empty, silent, or corrupt audio files).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import soundfile as sf
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HarmoniX.MulticulturalLoader")

TARGET_SAMPLE_RATE = 22050
PEAK_TARGET = 0.95
MIN_AUDIBLE_PEAK = 1e-8


def load_and_normalize(audio_path: Path) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Standardize any audio file into a validated, mono, 22.05 kHz, peak-normalized waveform.
    Preserves original metadata (duration, sample rate, channels).
    
    Shared Contract (Section 5.2):
        load_and_normalize(file) -> waveform, sample_rate, metadata
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Inspect original metadata using soundfile or librosa
    try:
        info = sf.info(str(audio_path))
        orig_sr = info.samplerate
        orig_channels = info.channels
        orig_format = info.format
    except Exception:
        orig_sr = "unknown"
        orig_channels = "unknown"
        orig_format = audio_path.suffix.upper().replace(".", "")

    # Load and resample to TARGET_SAMPLE_RATE as mono
    try:
        y, sr = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to decode audio file '{audio_path.name}': {e}")

    # Validation checks
    if y is None or len(y) == 0:
        raise ValueError(f"Audio file '{audio_path.name}' contains zero samples (empty).")

    duration = float(len(y) / sr)
    if duration < 0.5:
        raise ValueError(f"Audio file duration ({duration:.2f}s) is too short for analysis (min 0.5s).")

    # Guard against completely silent / corrupted flatline audio
    current_peak = float(np.max(np.abs(y)))
    if current_peak < MIN_AUDIBLE_PEAK:
        raise ValueError(f"Audio file '{audio_path.name}' is completely silent (peak amplitude < {MIN_AUDIBLE_PEAK}).")

    # Peak normalization to target [-0.95, +0.95]
    y_norm = (y / current_peak) * PEAK_TARGET

    metadata = {
        "filename": audio_path.name,
        "filepath": str(audio_path.resolve()),
        "duration_seconds": round(duration, 3),
        "target_sample_rate": sr,
        "original_sample_rate": orig_sr,
        "original_channels": orig_channels,
        "original_format": orig_format,
        "original_peak": round(current_peak, 4),
        "normalized_peak": PEAK_TARGET,
        "num_samples": len(y_norm)
    }

    return y_norm, sr, metadata


def scan_multicultural_directory(directory_path: Path) -> List[Dict[str, Any]]:
    """
    Scan a directory for diverse audio tracks across genres/subfolders.
    Returns a catalog of discoverable tracks with their genre tags.
    """
    directory_path = Path(directory_path)
    if not directory_path.exists():
        return []

    supported_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    tracks = []

    for file_path in directory_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            genre = file_path.parent.name if file_path.parent != directory_path else "Unspecified"
            tracks.append({
                "title": file_path.stem.replace("_", " ").title(),
                "genre": genre.title(),
                "filename": file_path.name,
                "filepath": str(file_path.resolve()),
                "filesize_kb": round(file_path.stat().st_size / 1024, 1)
            })

    return sorted(tracks, key=lambda x: (x["genre"], x["title"]))


def create_sample_diverse_dataset(output_dir: Path) -> List[Path]:
    """
    Bootstrap a curated multi-genre test dataset with diverse musical characteristics
    (Bollywood Melancholy, High-Energy Bhangra/Punjabi, Classical Raag, Western Pop).
    Ensures the pipeline is immediately testable out-of-the-box.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genres = {
        "bollywood_melancholy": {
            "root_freq": 220.0,   # A3 root (Minor chord, introspective)
            "harmonics": [(1.0, 0.5), (1.20, 0.35), (1.50, 0.25)],  # Minor 3rd triad
            "tempo_hz": 1.2,
            "filename": "bollywood_tum_hi_ho_style.wav",
            "duration": 6.0
        },
        "punjabi_upbeat": {
            "root_freq": 261.63,  # C4 root (High energy, syncopated rhythm)
            "harmonics": [(1.0, 0.6), (1.25, 0.3), (1.50, 0.2)],   # Major 3rd triad
            "tempo_hz": 2.5,
            "filename": "punjabi_bhangra_groove.wav",
            "duration": 6.0
        },
        "indian_classical_raag": {
            "root_freq": 146.83,  # D3 Tanpura drone + meditative Sitar overtone
            "harmonics": [(1.0, 0.4), (1.333, 0.3), (1.50, 0.25), (1.875, 0.15)],
            "tempo_hz": 0.8,
            "filename": "raag_yaman_meditation.wav",
            "duration": 6.0
        },
        "western_pop": {
            "root_freq": 329.63,  # E4 Bright Pop hook
            "harmonics": [(1.0, 0.5), (1.25, 0.3), (1.50, 0.2)],
            "tempo_hz": 2.0,
            "filename": "western_pop_bright_chorus.wav",
            "duration": 6.0
        }
    }

    created_paths = []
    sr = TARGET_SAMPLE_RATE

    for genre_key, config in genres.items():
        genre_dir = output_dir / genre_key
        genre_dir.mkdir(exist_ok=True)
        file_path = genre_dir / config["filename"]

        t = np.linspace(0, config["duration"], int(sr * config["duration"]), endpoint=False)
        wave = np.zeros_like(t)

        for ratio, amp in config["harmonics"]:
            f = config["root_freq"] * ratio
            wave += amp * np.sin(2 * np.pi * f * t)

        # Apply rhythmic envelope
        envelope = 0.5 * (1 + np.sin(2 * np.pi * config["tempo_hz"] * t))
        wave = wave * (0.6 + 0.4 * envelope)

        # Smooth fades at boundaries
        fade_len = int(0.05 * sr)
        wave[:fade_len] *= np.linspace(0, 1, fade_len)
        wave[-fade_len:] *= np.linspace(1, 0, fade_len)

        # Normalize
        peak = np.max(np.abs(wave))
        if peak > 0:
            wave = (wave / peak) * PEAK_TARGET

        sf.write(str(file_path), wave, sr)
        created_paths.append(file_path)

    return created_paths
