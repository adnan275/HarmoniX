"""
HarmoniX | Phase 1 Prediction on New Audio
Author: Praanshu Ranjan

Predicts human-rated valence for any audio file with the trained Phase 1 models:
1. Extracts the same 34 acoustic features used in training (feature_extraction.py).
2. Predicts valence with the saved Ridge, Random Forest and XGBoost pipelines.
3. Repeats the prediction on consecutive 30-second segments, because the models were
   trained on 30-60 s chorus excerpts rather than full songs.
4. Explains the Random Forest prediction with SHAP: which features pushed it up or down.
5. Flags features far outside the range seen in training (|z| > 3).

Usage:
    python src/predict.py "path/to/song.mp3"

Valence is on the PMEmo scale (0 = negative, 1 = positive). The output is a model
estimate, not a probability and not a universal verdict on how pleasant a song is.
"""

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile as sf

from feature_extraction import SAMPLE_RATE, extract_features
from feature_analysis import compute_shap_values, feature_label

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILES = {
    "Ridge": "ridge_best.joblib",
    "Random_Forest": "random_forest_best.joblib",
    "XGBoost": "xgboost_best.joblib",
}
PRIMARY_MODEL = "Random_Forest"  # best on the held-out test set (models/test_metrics.json)
TRAIN_VALENCE_MEAN = 0.584       # training split, see reports/dataset_summary_statistics.csv
TEST_RMSE = 0.133                # Random Forest error on the 116 test songs


def load_models():
    """Load the saved pipelines and the feature order they were trained on."""
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_cols = json.load(f)
    models = {name: joblib.load(MODELS_DIR / filename) for name, filename in MODEL_FILES.items()}
    return models, feature_cols


def predict_bounded(pipe, X):
    """Predict and clip to the valid valence range, as train_ml.py does."""
    return np.clip(pipe.predict(X), 0.0, 1.0)


def features_for(audio_path: Path, feature_cols):
    """Feature row in training column order."""
    features = extract_features(audio_path)
    return np.array([[features[col] for col in feature_cols]], dtype=float)


def segment_predictions(audio_path: Path, pipe, feature_cols, segment_seconds: float):
    """Predict each consecutive segment; the last partial segment is kept if >= 15 s."""
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    window = int(segment_seconds * sr)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, start in enumerate(range(0, len(y), window)):
            chunk = y[start:start + window]
            if len(chunk) < 15 * sr:
                break
            chunk_path = Path(tmp) / f"segment_{i}.wav"
            sf.write(chunk_path, chunk, sr)
            X = features_for(chunk_path, feature_cols)
            results.append((start / sr, (start + len(chunk)) / sr, float(predict_bounded(pipe, X)[0])))
    return results


def describe(valence: float) -> str:
    if valence >= 0.70:
        return "clearly positive"
    if valence >= 0.55:
        return "mildly positive"
    if valence >= 0.45:
        return "neutral"
    if valence >= 0.30:
        return "mildly negative"
    return "clearly negative"


def main():
    parser = argparse.ArgumentParser(description="Predict valence for an audio file.")
    parser.add_argument("audio", type=Path, help="Path to an .mp3 or .wav file")
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    models, feature_cols = load_models()
    duration = librosa.get_duration(path=str(args.audio))
    print(f"\nHarmoniX valence prediction: {args.audio.name} ({duration:.0f} s)")
    print("=" * 64)

    # 1. Whole track
    X = features_for(args.audio, feature_cols)
    print("\nWhole track (all three models):")
    predictions = {}
    for name, pipe in models.items():
        predictions[name] = float(predict_bounded(pipe, X)[0])
        marker = "  <- primary model" if name == PRIMARY_MODEL else ""
        print(f"  {name.replace('_', ' '):<14} {predictions[name]:.3f}{marker}")
    primary = predictions[PRIMARY_MODEL]
    print(
        f"\n  Estimate: {primary:.2f} on a 0-1 scale ({describe(primary)}); "
        f"typical error +/- {TEST_RMSE:.2f}. PMEmo average: {TRAIN_VALENCE_MEAN:.2f}."
    )

    # 2. Segments (the models were trained on chorus-length excerpts)
    segments = segment_predictions(args.audio, models[PRIMARY_MODEL], feature_cols, args.segment_seconds)
    if segments:
        print(f"\n{PRIMARY_MODEL.replace('_', ' ')} on {args.segment_seconds:.0f}-second segments:")
        for start, end, value in segments:
            print(f"  {int(start // 60)}:{int(start % 60):02d}-{int(end // 60)}:{int(end % 60):02d}  "
                  f"{value:.3f}  {'#' * int(round(value * 40))}")
        values = np.array([v for _, _, v in segments])
        print(f"  Median {np.median(values):.3f}, range {values.min():.3f}-{values.max():.3f}")

    # 3. Why: SHAP for the primary model on the whole track
    shap_values = compute_shap_values(models[PRIMARY_MODEL], X)
    if shap_values is not None:
        contributions = sorted(zip(feature_cols, shap_values[0]), key=lambda item: item[1])
        print(f"\nWhy ({PRIMARY_MODEL.replace('_', ' ')}, SHAP): base value + contributions = prediction")
        print("  Pushed valence UP:")
        for col, value in reversed(contributions[-5:]):
            if value > 0:
                print(f"    {feature_label(col):<28} {value:+.3f}")
        print("  Pushed valence DOWN:")
        for col, value in contributions[:5]:
            if value < 0:
                print(f"    {feature_label(col):<28} {value:+.3f}")

    # 4. How unusual is this song compared with the training songs?
    scaler = models[PRIMARY_MODEL].named_steps["scaler"]
    z_scores = (X[0] - scaler.mean_) / scaler.scale_
    unusual = [(feature_label(col), z) for col, z in zip(feature_cols, z_scores) if abs(z) > 3]
    print("\nFeatures far outside the training range (|z| > 3):")
    if unusual:
        for label, z in unusual:
            print(f"  {label:<28} z = {z:+.1f}")
    else:
        print("  None: the song looks like the training data, so the estimate is more trustworthy.")

    print(
        "\nNote: trained on 30-60 s pop choruses from PMEmo. A full song, or a style far from that "
        "data, adds uncertainty. This is a model estimate, not a probability."
    )


if __name__ == "__main__":
    main()
