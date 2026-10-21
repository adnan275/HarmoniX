"""
HarmoniX — Acoustic / Physics-Informed Feature Extraction

Extracts fixed-length acoustic features from each audio file listed in
data/processed/all_matched_metadata.csv.

Output:
    data/processed/features_physics.csv

Feature schema (34 features):
    13 MFCC means
     1 Spectral centroid mean
     1 Spectral bandwidth mean
     1 Spectral rolloff mean
     1 Spectral contrast mean
     1 Spectral flatness mean
     1 RMS mean
     1 Zero-crossing rate mean
    12 Chroma STFT means
     1 HNR estimate
     1 Sensory dissonance estimate
    --------------------------------
    34 features

Metadata columns such as music_id, valence, arousal, and split are
preserved when available.
"""

from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_matched_metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_physics.csv"

SAMPLE_RATE = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512


def safe_mean(values):
    """Return a finite mean, or NaN if no valid values are available."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def estimate_hnr_db(y, sr):
    """
    Estimate harmonic-to-noise ratio in dB using autocorrelation.

    This is a simplified signal-based estimate, not a clinical or
    perceptual HNR measurement.
    """
    y = np.asarray(y, dtype=float)

    if len(y) < 3 or np.std(y) < 1e-8:
        return np.nan

    y = y - np.mean(y)
    autocorr = np.correlate(y, y, mode="full")[len(y) - 1:]

    if autocorr[0] <= 1e-12:
        return np.nan

    autocorr /= autocorr[0]

    # Search plausible fundamental frequencies between 50 and 1000 Hz.
    min_lag = max(1, int(sr / 1000))
    max_lag = min(len(autocorr) - 1, int(sr / 50))

    if max_lag <= min_lag:
        return np.nan

    peak_region = autocorr[min_lag:max_lag + 1]
    peaks, _ = find_peaks(peak_region)

    if len(peaks) == 0:
        return np.nan

    peak_index = peaks[np.argmax(peak_region[peaks])] + min_lag
    rho = float(np.clip(autocorr[peak_index], 1e-6, 0.999999))

    return float(10.0 * np.log10(rho / (1.0 - rho)))


def estimate_sensory_dissonance(y, sr):
    """
    Estimate sensory dissonance from prominent spectral peaks.

    Uses a simplified roughness model based on the interaction of
    nearby frequency components. This is a proxy, not a validated
    psychoacoustic measurement.
    """
    if len(y) < N_FFT:
        return np.nan

    spectrum = np.abs(librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    ))

    mean_spectrum = np.mean(spectrum, axis=1)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    peaks, properties = find_peaks(mean_spectrum)

    if len(peaks) < 2:
        return 0.0

    # Keep at most the 30 strongest spectral peaks.
    strongest = np.argsort(mean_spectrum[peaks])[-30:]
    peaks = peaks[strongest]

    peak_freqs = frequencies[peaks]
    peak_amps = mean_spectrum[peaks]

    # Normalize amplitudes so the score is less dependent on volume.
    peak_amps = peak_amps / (np.max(peak_amps) + 1e-12)

    roughness_sum = 0.0

    for i in range(len(peak_freqs)):
        for j in range(i + 1, len(peak_freqs)):
            f1 = peak_freqs[i]
            f2 = peak_freqs[j]

            if f1 <= 0 or f2 <= 0:
                continue

            # Approximate critical-band interaction.
            min_freq = min(f1, f2)
            freq_difference = abs(f2 - f1)
            scaled_difference = 0.24 * freq_difference / (
                0.0207 * min_freq + 18.96
            )

            interaction = (
                np.exp(-3.5 * scaled_difference)
                - np.exp(-5.75 * scaled_difference)
            )

            roughness_sum += (
                peak_amps[i] * peak_amps[j] * max(0.0, interaction)
            )

    return float(roughness_sum)


def extract_features(audio_path):
    """Extract one fixed-length feature dictionary from an audio file."""
    y, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if y.size == 0:
        raise ValueError("Audio file is empty.")

    # Peak normalization with protection against silence.
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y = 0.95 * y / peak

    # Frame-level spectral and temporal features.
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    rms = librosa.feature.rms(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    features = {}

    # 13 MFCC means.
    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = safe_mean(mfcc[i])

    # Five spectral summaries.
    features["spectral_centroid_mean"] = safe_mean(centroid)
    features["spectral_bandwidth_mean"] = safe_mean(bandwidth)
    features["spectral_rolloff_mean"] = safe_mean(rolloff)
    features["spectral_contrast_mean"] = safe_mean(contrast)
    features["spectral_flatness_mean"] = safe_mean(flatness)

    # Energy and temporal features.
    features["rms_mean"] = safe_mean(rms)
    features["zcr_mean"] = safe_mean(zcr)

    # Twelve chroma means.
    for i in range(12):
        features[f"chroma_{i + 1}_mean"] = safe_mean(chroma[i])

    # Additional harmonic and psychoacoustic estimates.
    features["hnr_estimate_db"] = estimate_hnr_db(y, sr)
    features["sensory_dissonance_estimate"] = estimate_sensory_dissonance(y, sr)

    return features


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}\n"
            "Run the existing data-loading pipeline first."
        )

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = {"music_id", "audio_path"}
    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    rows = []
    failures = []

    print(f"Found {len(metadata)} metadata rows.")
    print("Extracting acoustic features...")

    for index, row in metadata.iterrows():
        audio_path = Path(str(row["audio_path"]))

        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            features = extract_features(audio_path)

            # Keep identifiers, labels, and split assignments with features.
            result = {
                column: row[column]
                for column in metadata.columns
                if column != "audio_path"
            }
            result.update(features)
            rows.append(result)

            print(
                f"[{index + 1}/{len(metadata)}] "
                f"Processed music_id={row['music_id']}"
            )

        except Exception as exc:
            failures.append({
                "music_id": row["music_id"],
                "audio_path": str(audio_path),
                "error": str(exc),
            })
            warnings.warn(
                f"Skipping music_id={row['music_id']}: {exc}"
            )

    if not rows:
        raise RuntimeError(
            "No audio files were successfully processed. "
            "Check the audio paths and dependencies."
        )

    result_df = pd.DataFrame(rows)

    # Verify the acoustic feature dimension.
    feature_columns = [
        column for column in result_df.columns
        if column.startswith((
            "mfcc_", "spectral_", "rms_", "zcr_", "chroma_"
        ))
        or column in (
            "hnr_estimate_db",
            "sensory_dissonance_estimate"
        )
    ]

    if len(feature_columns) != 34:
        raise RuntimeError(
            f"Expected 34 acoustic features, found {len(feature_columns)}. "
            f"Feature columns: {feature_columns}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature extraction complete.")
    print(f"Successful songs: {len(result_df)}")
    print(f"Failed songs: {len(failures)}")
    print(f"Acoustic feature count: {len(feature_columns)}")
    print(f"Output saved to: {OUTPUT_PATH}")

    if failures:
        failure_path = OUTPUT_PATH.parent / "feature_extraction_errors.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"Failure details saved to: {failure_path}")

    print("\nOutput preview:")
    print(result_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
