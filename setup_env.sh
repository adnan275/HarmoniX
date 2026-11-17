#!/usr/bin/env bash
# HarmoniX — one-time Mac environment setup (creates .venv with all requirements)
# Run from the HarmoniX folder:   bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"

# requirements.txt pins numpy<2, which only has wheels up to Python 3.12.
PY=""
for c in python3.12 python3.11 python3.10; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo ">> No Python 3.10–3.12 found."
  if ! command -v brew >/dev/null 2>&1; then
    echo "   Install Homebrew first (https://brew.sh), then rerun this script."
    exit 1
  fi
  echo ">> Installing Python 3.12 via Homebrew ..."
  brew install python@3.12
  PY="$(brew --prefix)/bin/python3.12"
fi
echo ">> Using $($PY --version) at $(command -v $PY || echo $PY)"

if command -v brew >/dev/null 2>&1; then
  brew list ffmpeg >/dev/null 2>&1 || { echo ">> Installing ffmpeg (audio decoding fallback) ..."; brew install ffmpeg; }
fi

if [ ! -d .venv ]; then
  echo ">> Creating virtual environment .venv ..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt gdown

echo ">> Sanity check ..."
python - <<'PY'
import numpy, torch, librosa, soundfile, sklearn
print("numpy", numpy.__version__, "| torch", torch.__version__, "| librosa", librosa.__version__)
print("MPS available:", torch.backends.mps.is_available())
PY
echo ""
echo "Setup done. Next:"
echo "   source .venv/bin/activate"
echo "   caffeinate -i bash run_phase2_full.sh"
