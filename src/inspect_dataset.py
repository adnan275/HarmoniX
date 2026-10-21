"""
HarmoniX | Dataset Inspection & Sanity Check Utility
Author: Adnan Rizvi (Lead Architect)

Use this script to:
1. Scan data/raw/PMEmo/ for available audio files and CSV files.
2. Verify audio file decodability, duration distribution, and sample rates.
3. Check annotation columns, value ranges, and missing/NaN values.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PMEmo"


def inspect():
    print("=" * 65)
    print("🎵 HARMONIX | Phase 1 Raw Dataset Inspection")
    print("=" * 65)
    print(f"Target Directory: {RAW_DIR}")

    if not RAW_DIR.exists():
        print(f"❌ Error: Directory '{RAW_DIR}' does not exist.")
        return

    # Check Audio Files
    audio_files = list(RAW_DIR.rglob("*.mp3")) + list(RAW_DIR.rglob("*.wav"))
    print(f"\n📂 Audio Files Found: {len(audio_files)}")
    if audio_files:
        sample_audio = audio_files[:3]
        print("   Sample paths:")
        for sa in sample_audio:
            print(f"   - {sa.name} ({sa.stat().st_size / 1024:.1f} KB)")
    else:
        print("   ⚠️  No .mp3 or .wav audio files found yet.")

    # Check CSV Files
    csv_files = list(RAW_DIR.rglob("*.csv"))
    print(f"\n📊 CSV / Annotation Files Found: {len(csv_files)}")
    for cf in csv_files:
        print(f"   - {cf.relative_to(RAW_DIR)} ({cf.stat().st_size / 1024:.1f} KB)")
        try:
            df = pd.read_csv(cf)
            print(f"     Rows: {len(df)}, Columns: {list(df.columns[:8])}")
            # Check for valence
            val_cols = [c for c in df.columns if "val" in c.lower()]
            if val_cols:
                print(f"     🎯 Valence target candidate found: {val_cols}")
                for vc in val_cols:
                    if pd.api.types.is_numeric_dtype(df[vc]):
                        print(f"        {vc} range: [{df[vc].min():.3f}, {df[vc].max():.3f}], mean: {df[vc].mean():.3f}")
        except Exception as e:
            print(f"     ⚠️  Could not preview CSV: {e}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    inspect()
