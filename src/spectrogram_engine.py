"""
HarmoniX: Log-Mel Spectrogram Engine

Converts normalized audio waveforms into fixed-configuration
128-bin log-Mel spectrograms suitable for CNN input.

Preprocessing must remain identical during training, validation,
testing, and inference. SpecAugment is a separate training-only step.
"""

import numpy as np
import librosa
import torch


# --------------------------------------------------
# 1. PREPROCESSING CONFIGURATION
# --------------------------------------------------

SAMPLE_RATE = 22050

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048

WINDOW = "hann"
MEL_SCALE = "slaney"

FMIN = 0.0
FMAX = SAMPLE_RATE / 2

POWER = 2.0
TOP_DB = 80.0

# A 3-second segment contains exactly 66,150 samples.
SEGMENT_SECONDS = 3.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)

# With center=False:
# 1 + floor((66150 - 2048) / 512) = 126 frames.
EXPECTED_FRAMES = 126


# --------------------------------------------------
# 2. GENERATE LOG-MEL SPECTROGRAM
# --------------------------------------------------

def generate_log_mel(waveform, sample_rate=SAMPLE_RATE):
    """
    Convert one normalized 3-second waveform into a log-Mel spectrogram.

    Parameters
    ----------
    waveform : numpy.ndarray
        One-dimensional audio waveform containing exactly 3 seconds.
    sample_rate : int
        Sampling rate of the waveform. Must be 22050 Hz.

    Returns
    -------
    numpy.ndarray
        Log-Mel spectrogram with shape (128, 126).
    """

    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono (one-dimensional).")

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected sample rate {SAMPLE_RATE} Hz, "
            f"received {sample_rate} Hz."
        )

    if len(waveform) != SEGMENT_SAMPLES:
        raise ValueError(
            f"Expected exactly {SEGMENT_SAMPLES} samples "
            f"(3 seconds), received {len(waveform)}."
        )

    if not np.all(np.isfinite(waveform)):
        raise ValueError("Waveform contains NaN or infinite values.")

    # Convert audio into a power Mel spectrogram.
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=False,
        power=POWER,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=False,
        norm="slaney",
    )

    # Convert power values into decibels.
    log_mel = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
        top_db=TOP_DB,
    )

    if log_mel.shape != (N_MELS, EXPECTED_FRAMES):
        raise RuntimeError(
            f"Unexpected spectrogram shape: {log_mel.shape}. "
            f"Expected {(N_MELS, EXPECTED_FRAMES)}."
        )

    return log_mel.astype(np.float32)


# --------------------------------------------------
# 3. CONVERT TO CNN-READY TENSOR
# --------------------------------------------------

def to_cnn_tensor(log_mel):
    """
    Convert a log-Mel spectrogram into a PyTorch tensor.

    Input shape:  (128, 126)
    Output shape: (1, 128, 126)

    The first dimension represents the single audio channel.
    """

    log_mel = np.asarray(log_mel, dtype=np.float32)

    if log_mel.shape != (N_MELS, EXPECTED_FRAMES):
        raise ValueError(
            f"Expected shape {(N_MELS, EXPECTED_FRAMES)}, "
            f"received {log_mel.shape}."
        )

    # Add the channel dimension required by CNNs.
    tensor = torch.from_numpy(log_mel).unsqueeze(0)

    return tensor


# --------------------------------------------------
# 4. SPECAUGMENT — TRAINING ONLY
# --------------------------------------------------

def apply_spec_augment(
    spectrogram,
    training=False,
    num_time_masks=2,
    num_frequency_masks=2,
    max_time_mask=20,
    max_frequency_mask=12,
):
    """
    Apply SpecAugment time and frequency masking.

    This function does NOT augment data unless training=True.

    Accepts tensors shaped:
        (128, 126)
        (1, 128, 126)
        (batch, 1, 128, 126)

    Returns a tensor with the same shape.
    """

    if not isinstance(spectrogram, torch.Tensor):
        raise TypeError("spectrogram must be a PyTorch tensor.")

    if spectrogram.ndim < 2:
        raise ValueError("Spectrogram must have at least 2 dimensions.")

    # Validation and test data must remain unchanged.
    if not training:
        return spectrogram

    augmented = spectrogram.clone()

    frequency_bins = augmented.shape[-2]
    time_frames = augmented.shape[-1]

    # Treat every leading-dimension item as an independent sample.
    samples = augmented.reshape(-1, frequency_bins, time_frames)

    for sample in samples:

        # Time masking: hide selected time frames.
        for _ in range(num_time_masks):
            mask_width = torch.randint(
                0,
                min(max_time_mask, time_frames) + 1,
                (1,),
            ).item()

            if mask_width > 0:
                start = torch.randint(
                    0,
                    time_frames - mask_width + 1,
                    (1,),
                ).item()

                sample[:, start:start + mask_width] = -TOP_DB

        # Frequency masking: hide selected Mel frequency bins.
        for _ in range(num_frequency_masks):
            mask_width = torch.randint(
                0,
                min(max_frequency_mask, frequency_bins) + 1,
                (1,),
            ).item()

            if mask_width > 0:
                start = torch.randint(
                    0,
                    frequency_bins - mask_width + 1,
                    (1,),
                ).item()

                sample[start:start + mask_width, :] = -TOP_DB

    return augmented