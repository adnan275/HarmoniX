# 🎵 HarmoniX: The Science of Musical Perception

> **A Physics-Informed Hybrid Machine Learning and Deep Learning System for Predicting Human-Rated Musical Valence from Sound Waves**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio_DSP-blue.svg?style=flat)](https://librosa.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-red.svg?style=flat)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📖 1. Overview & Problem Statement

Music is universally perceived across cultures, invoking profound emotional responses regardless of linguistic or semantic comprehension. A listener can encounter a song in an unfamiliar language and immediately perceive it as soothing, joyful, energetic, or dissonant.

**HarmoniX** investigates a central scientific question:
> *If we remove lyrics and semantic meaning, can the physical and mathematical structure of sound waves predict how pleasant (positive valence) music sounds to human listeners?*

Sound is fundamentally physical variations in atmospheric air pressure over time. By combining **Wave Physics** ($v = f \lambda$), **Digital Signal Processing (Fourier Transforms)**, **Classical Machine Learning**, and **Deep Learning Computer Vision (Mel-Spectrograms)**, HarmoniX builds an evidence-based bridge between objective acoustics and human musical perception.

---

## 🏗️ 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph INPUT["🎵 1. Raw Audio Input"]
        A["Unseen Audio File (.mp3 / .wav)"]
        A --> B["Audio Preprocessing & Validation<br/>(Resample 22.05kHz, Mono, Normalize)"]
    end

    subgraph DUAL_BRANCH["🔬 2. Dual-Branch Feature Engineering"]
        B --> C1["Branch A: Physics & DSP Engine"]
        B --> C2["Branch B: Spectroscopy Engine"]

        C1 --> D1["Acoustic Descriptors:<br/>• Timbre: MFCCs (1-13)<br/>• Warmth: Spectral Centroid<br/>• Energy: RMS & Zero-Crossing Rate<br/>• Harmony: Chroma & Dissonance Index"]
        C2 --> D2["Log-Mel-Spectrogram Generation<br/>(2D Time-Frequency Image Heatmap)"]
    end

    subgraph MODELS["🧠 3. Predictive Modeling"]
        D1 --> E1["Phase 1: Advanced Machine Learning<br/>• Ridge Regression (Baseline)<br/>• Random Forest Regressor<br/>• XGBoost / LightGBM"]
        D2 --> E2["Phase 2: Deep Learning<br/>• PyTorch 2D CNN<br/>• Spatial Convolutions + Regularization"]
    end

    subgraph HYBRID["⚡ 4. Hybrid Multi-Modal Fusion"]
        E1 --> F["Late Fusion Ensemble<br/>(Physics Features + CNN Embeddings)"]
        E2 --> F
    end

    subgraph OUTPUT["📊 5. Output & Inference"]
        F --> G["Streamlit Interactive Application"]
        G --> H1["Continuous Valence Estimate (0.0 - 1.0)"]
        G --> H2["Visual Mel-Spectrogram Display"]
        G --> H3["Acoustic Physics Breakdown Radar"]
    end

    style INPUT fill:#1a365d,stroke:#63b3ed,stroke-width:2px,color:#fff
    style DUAL_BRANCH fill:#2c3e50,stroke:#4fd1c5,stroke-width:2px,color:#fff
    style MODELS fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#fff
    style HYBRID fill:#4a154b,stroke:#b794f4,stroke-width:2px,color:#fff
    style OUTPUT fill:#1c4532,stroke:#68d391,stroke-width:2px,color:#fff
```

---

## 👥 3. Team & Module Contribution Matrix

| Team Member | Module / Role | Primary Files | Core Focus |
| :--- | :--- | :--- | :--- |
| **Adnan Rizvi** *(Lead)* | **Architecture & Data Pipeline** | `src/data_loader.py`<br/>`src/predict.py`<br/>`app/streamlit_app.py` | Repository setup, leakage-free 70/15/15 train-test partitioning, data sanity checks, and final Streamlit web application. |
| **Garariya** | **DSP & Physics Engineering** | `src/feature_extraction.py` | Fourier transforms (FFT/STFT), MFCCs, Spectral Centroid, RMS energy, Zero-crossing rate, and sensory dissonance index. |
| **Bhoomi** | **Advanced Machine Learning** | `src/train_ml.py` | Training and tuning Ridge Regression, Random Forest, and XGBoost regressor with strict cross-validation. |
| **Praanshu Ranjan** | **Interpretability & Physics EDA** | `src/feature_analysis.py`<br/>`reports/figures/` | Feature correlation heatmaps, Dissonance vs. Valence curves, SHAP explainability, and feature importance analysis. |
| **Vani** | **Evaluation & IEEE Reporting** | `src/evaluate.py`<br/>`reports/phase1_report/` | Held-out test evaluation (MAE, RMSE, R²), residual error analysis, and Phase 1 / Phase 2 IEEE LaTeX report authoring. |

---

## 📂 4. Repository Structure

```text
AIML_project/
├── data/
│   ├── raw/PMEmo/               # Raw audio clips and static_annotations.csv
│   └── processed/               # Leakage-free train, val, and test split tables
├── src/
│   ├── data_loader.py           # Ingestion & reproducible 70/15/15 splitting
│   ├── inspect_dataset.py       # Dataset audit & decodability verification
│   ├── eda_analysis.py          # Publication-ready distribution & quadrant plots
│   ├── feature_extraction.py    # Physics & DSP acoustic feature extractor
│   ├── train_ml.py              # Classical ML modeling (Ridge, RF, XGBoost)
│   ├── feature_analysis.py      # Feature importance & physical correlation
│   ├── evaluate.py              # Test set evaluation benchmark & residual plots
│   └── predict.py               # Real-time inference engine for unseen audio
├── app/
│   └── streamlit_app.py         # Interactive UI for audio upload & live analysis
├── models/                      # Serialized trained model artifacts (.joblib, .pt)
├── reports/
│   ├── figures/                 # High-resolution plots for IEEE reports
│   └── dataset_summary_statistics.csv
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignores large binaries, venvs & raw audio
└── README.md                    # Project documentation
```

---

## ⚡ 5. Quickstart & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/adnan275/HarmoniX.git
cd HarmoniX
```

### Step 2: Setup Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run Data Verification & Splitting
```bash
python src/data_loader.py
```

### Step 4: Generate Phase 1 EDA Plots
```bash
python src/eda_analysis.py
```
*Generated plots will be saved in `reports/figures/`.*

### Step 5: Launch the Interactive Web Application (Phase 5)
```bash
streamlit run app/streamlit_app.py
```

---

## 🔬 6. Research & Evaluation Protocol

* **Evaluation Target:** Continuous human-rated **Valence** (proxy for musical pleasantness and positive emotional tone).
* **Leakage Prevention:** Song-independent train/val/test splits (70% Train, 15% Validation, 15% Test). Scalers and feature transformers are fitted **exclusively on the training split**.
* **Metrics:**
  * Mean Absolute Error (**MAE**)
  * Root Mean Squared Error (**RMSE**)
  * Coefficient of Determination (**$R^2$**)
  * Pearson Correlation Coefficient (**$r$**)

---

## ⚠️ 7. Scientific Framing & Caveats
1. **Valence vs. Universal Pleasantness:** Valence measures perceived emotional positivity vs. negativity. Musical taste is multifaceted and influenced by cultural background, familiarity, and personal psychology; this model predicts objective acoustic-driven perceived valence, not universal artistic value.
2. **Audio-Only Modeling:** Lyrics and linguistic semantics are intentionally excluded to isolate acoustic and wave mechanics.

---

## 📜 8. License & Acknowledgments
* **Primary Dataset:** PMEmo Dataset (*Hui Zhang et al.*)
* Developed as part of the **Computer Science & Artificial Intelligence (CSAI)** curriculum.
