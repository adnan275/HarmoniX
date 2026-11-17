#!/usr/bin/env bash
# HarmoniX Phase 2 — full PMEmo training (767 songs, 40 epochs, Apple MPS)
# Run from the HarmoniX folder:   bash run_phase2_full.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
python -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'"

RAW=data/raw/PMEmo
if [ ! -f "$RAW/annotations/static_annotations.csv" ] || [ "$(ls "$RAW/chorus" 2>/dev/null | wc -l)" -lt 700 ]; then
  echo ">> Downloading PMEmo2019 (~680 MB) ..."
  pip install -q gdown
  mkdir -p data/raw
  gdown 1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ -O data/raw/PMEmo2019.zip
  rm -rf "$RAW" && unzip -q data/raw/PMEmo2019.zip -d data/raw
  mv data/raw/PMEmo2019 "$RAW" && rm -rf data/raw/__MACOSX data/raw/PMEmo2019.zip
fi

echo ">> Building 70/15/15 song-level split (expect 536 / 115 / 116) ..."
python src/data_loader.py

echo ">> Training Phase 2 (4 lambdas + CNN-only, 40 epochs each, no early stop) ..."
python src/train_cnn.py --epochs 40 --patience 40 --device mps --rebuild-cache 2>&1 | tee reports/phase2_full_training.log
