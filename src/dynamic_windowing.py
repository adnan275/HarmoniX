
"""
HarmoniX: Dynamic Audio Windowing

Splits normalized mono waveforms into 3-second windows
with a 1.5-second hop (50% overlap).

Short and final partial windows are zero-padded so that
every segment has exactly 3 seconds of audio.

Output spectrogram tensors have shape:
    (number_of_windows, 1, 128, 126)
"""

import numpy as np
import torch

from src.spectrogram_engine import (
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SEGMENT_SECONDS,
    N_MELS,
    EXPECTED_FRAMES,
    generate_log_mel,
    to_cnn_tensor,
)


# --------------------------------------------------
# 1. WINDOWING CONFIGURATION
# --------------------------------------------------

WINDOW_SECONDS = SEGMENT_SECONDS
HOP_SECONDS = 1.5

HOP_SAMPLES = int(SAMPLE_RATE * HOP_SECONDS)


# --------------------------------------------------
# 2. CREATE AUDIO WINDOWS
# --------------------------------------------------

def create_audio_windows(waveform, sample_rate=SAMPLE_RATE):
    """
    Split a complete audio waveform into overlapping windows.

    Parameters
    ----------
    waveform : numpy.ndarray
        One-dimensional normalized mono audio waveform.
    sample_rate : int
        Must be 22050 Hz.

    Returns
    -------
    list[numpy.ndarray]
        List of fixed-size 3-second waveform segments.

    Window policy
    -------------
    - Window length: 3 seconds.
    - Hop length: 1.5 seconds.
    - Overlap: 50%.
    - Short recordings: zero-padded to 3 seconds.
    - Final incomplete window: zero-padded.
    - No unnecessary extra window for an exact 3-second track.
    """

    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono (one-dimensional).")

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, received {sample_rate} Hz."
        )

    if len(waveform) == 0:
        raise ValueError("Waveform cannot be empty.")

    if not np.all(np.isfinite(waveform)):
        raise ValueError("Waveform contains NaN or infinite values.")

    # Case 1: Recording shorter than 3 seconds.
    if len(waveform) <= SEGMENT_SAMPLES:
        padded = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        padded[:len(waveform)] = waveform
        return [padded]

    # Case 2: Recording longer than 3 seconds.
    windows = []

    # Generate all complete windows on the regular 1.5-second grid.
    starts = list(
        range(
            0,
            len(waveform) - SEGMENT_SAMPLES + 1,
            HOP_SAMPLES,
        )
    )

    for start in starts:
        end = start + SEGMENT_SAMPLES
        windows.append(waveform[start:end].copy())

    # Check whether the final complete window covers the track ending.
    last_end = starts[-1] + SEGMENT_SAMPLES

    if last_end < len(waveform):
        # Add the next regular 1.5-second position.
        final_start = starts[-1] + HOP_SAMPLES

        final_audio = waveform[final_start:]

        # Pad the incomplete final window with zeros.
        padded = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        padded[:len(final_audio)] = final_audio

        windows.append(padded)

    return windows


# --------------------------------------------------
# 3. CONVERT WINDOWS TO LOG-MEL SPECTROGRAMS
# --------------------------------------------------

def windows_to_spectrograms(windows):
    """
    Convert audio windows into log-Mel spectrograms.

    Returns
    -------
    numpy.ndarray
        Shape: (number_of_windows, 128, 126)
    """

    if not windows:
        raise ValueError("No audio windows were provided.")

    spectrograms = []

    for window in windows:
        spectrogram = generate_log_mel(
            window,
            sample_rate=SAMPLE_RATE,
        )
        spectrograms.append(spectrogram)

    return np.stack(spectrograms, axis=0)


# --------------------------------------------------
# 4. CONVERT WINDOWS TO CNN-READY BATCH
# --------------------------------------------------

def windows_to_cnn_tensor(windows):
    """
    Convert audio windows directly into a CNN-ready PyTorch tensor.

    Returns
    -------
    torch.Tensor
        Shape: (number_of_windows, 1, 128, 126)
    """

    spectrograms = windows_to_spectrograms(windows)

    tensors = [
        to_cnn_tensor(spectrogram)
        for spectrogram in spectrograms
    ]

    return torch.stack(tensors, dim=0)


# --------------------------------------------------
# 5. COMPLETE AUDIO-TO-TENSOR PIPELINE
# --------------------------------------------------

def process_audio(waveform, sample_rate=SAMPLE_RATE):
    """
    Complete pipeline:
        Waveform -> Audio windows -> Log-Mel -> CNN tensor

    Returns
    -------
    torch.Tensor
        Shape: (number_of_windows, 1, 128, 126)
    """

    windows = create_audio_windows(
        waveform,
        sample_rate=sample_rate,
    )

    return windows_to_cnn_tensor(windows)
