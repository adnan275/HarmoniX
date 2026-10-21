"""
HarmoniX | Sample Data Generator & Verification Tool
Author: Adnan Rizvi (Lead Architect)

This utility generates a lightweight validation sample in `data/raw/PMEmo/`:
- Creates sample musical audio clips (.wav) with consonant & dissonant harmonic properties.
- Creates sample `static_annotations.csv` matching the official PMEmo column schema.
- Allows immediate end-to-end testing of data_loader.py, feature_extraction.py, and models
  before/while downloading the full 794 PMEmo dataset archive.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PME_DIR = PROJECT_ROOT / "data" / "raw" / "PMEmo"
AUDIO_DIR = RAW_PME_DIR / "chorus"
ANNOTATION_DIR = RAW_PME_DIR / "annotations"


def create_sample_dataset(num_samples: int = 15, duration_sec: float = 5.0, sample_rate: int = 22050):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    print(f"🎵 Generating {num_samples} sample harmonic audio clips for validation...")

    for i in range(1, num_samples + 1):
        music_id = str(i)
        
        # Alternate between consonant (pleasant major chords) and dissonant (harsh clusters)
        if i % 2 == 1:
            # Consonant C-Major / G-Major Chord (Harmonious waves -> Higher Valence)
            f0 = 261.63 * (1.0 + (i % 3) * 0.25)
            wave = (
                0.5 * np.sin(2 * np.pi * f0 * t) +
                0.3 * np.sin(2 * np.pi * (f0 * 1.25) * t) +   # Major 3rd
                0.2 * np.sin(2 * np.pi * (f0 * 1.50) * t)     # Perfect 5th
            )
            valence = 0.65 + np.random.uniform(0.05, 0.25)
            arousal = 0.50 + np.random.uniform(-0.15, 0.25)
        else:
            # Dissonant Tritone / Minor 2nd Clash (Harsh waves -> Lower Valence)
            f0 = 220.0
            wave = (
                0.5 * np.sin(2 * np.pi * f0 * t) +
                0.4 * np.sin(2 * np.pi * (f0 * 1.059) * t) +  # Minor 2nd (Clash)
                0.3 * np.sin(2 * np.pi * (f0 * 1.414) * t)    # Tritone (Devil's interval)
            )
            valence = 0.20 + np.random.uniform(0.02, 0.20)
            arousal = 0.60 + np.random.uniform(-0.10, 0.20)

        # Apply envelope to prevent clicks
        fade = int(sample_rate * 0.05)
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)

        # Normalize audio to [-0.95, 0.95]
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = (wave / max_val) * 0.95

        # Save audio file
        audio_filename = f"{music_id}.wav"
        audio_filepath = AUDIO_DIR / audio_filename
        sf.write(str(audio_filepath), wave, sample_rate)

        records.append({
            "musicId": music_id,
            "valence_mean": round(float(np.clip(valence, 0.0, 1.0)), 4),
            "arousal_mean": round(float(np.clip(arousal, 0.0, 1.0)), 4),
            "valence_std": 0.05,
            "arousal_std": 0.05
        })

    # Save static annotations CSV
    df = pd.DataFrame(records)
    csv_path = ANNOTATION_DIR / "static_annotations.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Generated {num_samples} audio clips in: {AUDIO_DIR}")
    print(f"✅ Saved sample annotations ({len(df)} rows) to: {csv_path}")


if __name__ == "__main__":
    create_sample_dataset()
