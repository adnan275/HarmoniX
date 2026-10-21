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
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'Inter, system-ui, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b' }}}%%
flowchart TB

    subgraph S1["🎧 STAGE 1: RAW AUDIO INGESTION & DSP PIPELINE"]
        direction LR
        RAW["📁 Unseen Audio File<br/><code>.mp3</code> / <code>.wav</code> (Any Source)"]:::inputNode
        PRE["🎛️ Audio Preprocessor<br/>• Resample: 22.05 kHz<br/>• Mono Conversion<br/>• Peak Decibel Normalization"]:::inputNode
        RAW ==>|"Decoded PCM Waveform"| PRE
    end

    PRE ==>|"Continuous Time-Domain Wave x(t)"| S2A
    PRE ==>|"Windowed STFT Frames"| S2B

    subgraph S2A["🔬 BRANCH A: PHYSICS & ACOUSTIC FEATURE EXTRACTION (AML)"]
        direction TB
        subgraph P1["⚡ Wave Mechanics & Energy"]
            F_ENERGY["• RMS Energy: Wave Amplitude<br/>• Zero-Crossing Rate: Frequency Indicator (v = fλ)"]:::dspNode
        end
        subgraph P2["🎼 Timbre & Spectrum"]
            F_SPEC["• 13 MFCCs: Acoustic Texture<br/>• Spectral Centroid: Warmth / Brightness<br/>• Spectral Rolloff & Bandwidth"]:::dspNode
        end
        subgraph P3["🎶 Harmony & Consonance"]
            F_HARM["• Chroma STFT: Harmonic Pitch Distribution<br/>• Harmonic-to-Noise Ratio (HNR)<br/>• Sensory Dissonance Index (Roughness)"]:::dspNode
        end
        VEC["📊 1D Engineered Feature Vector<br/><code>[x₁, x₂, x₃, ..., xₙ] ∈ ℝ³⁴</code>"]:::featNode
        P1 & P2 & P3 ==> VEC
    end

    subgraph S2B["👁️ BRANCH B: AUDIO SPECTROSCOPY & COMPUTER VISION (DL)"]
        direction TB
        MEL["🌈 Log-Mel-Spectrogram Generator<br/>• FFT Window Size: 2048 | Hop: 512<br/>• 128 Mel-frequency Bins (Log-scale)<br/>• Output: 2D Energy Heatmap (Image)"]:::specNode
        
        subgraph CNN_ARCH["🧠 PyTorch 2D CNN Architecture"]
            direction TB
            C1["Conv2D Block 1: 32 Filters (3x3) + BatchNorm + ReLU + MaxPool"]:::dlNode
            C2["Conv2D Block 2: 64 Filters (3x3) + BatchNorm + ReLU + MaxPool"]:::dlNode
            C3["Conv2D Block 3: 128 Filters (3x3) + AdaptiveAvgPool + Dropout(0.4)"]:::dlNode
            C1 ==> C2 ==> C3
        end
        EMB["🧩 2D Latent Audio Embedding<br/><code>e_cnn ∈ ℝ⁶⁴</code>"]:::featNode
        MEL ==> CNN_ARCH ==> EMB
    end

    subgraph S3["🏆 STAGE 3: PREDICTIVE MODELING & BENCHMARKING"]
        direction LR
        subgraph ML_MODELS["Phase 1: Advanced Machine Learning"]
            M1["Linear / Ridge Regressor<br/>(Baseline Benchmark)"]:::mlNode
            M2["Random Forest Regressor<br/>(Non-linear Ensemble)"]:::mlNode
            M3["XGBoost / LightGBM<br/>(Gradient Tree Boosting)"]:::mlNode
        end
        VEC ==> ML_MODELS
    end

    subgraph S4["⚡ STAGE 4: HYBRID MULTI-MODAL FUSION (BONUS MARKS)"]
        direction TB
        FUSION{{"✨ Late-Fusion Gated Ensemble<br/>y_pred = α · ŷ_AML + (1-α) · ŷ_CNN<br/>(Subject to Held-Out Validation Tuning)"}}:::fusionNode
        ML_MODELS ==>|"Tabular Prediction ŷ_AML"| FUSION
        EMB ==>|"Visual Latent Features"| FUSION
    end

    subgraph S5["💻 STAGE 5: STREAMLIT INTERACTIVE APP & EXPLAINABILITY"]
        direction LR
        OUT1["🎯 Perceived Valence Score<br/>(0.00 to 1.00 Continuous Rating)"]:::outNode
        OUT2["📈 Real-Time Mel-Spectrogram<br/>(Interactive Visual Heatmap)"]:::outNode
        OUT3["🧭 Physics Radar Breakdown<br/>(Harmony vs Warmth vs Dynamics)"]:::outNode
        FUSION ==> OUT1 & OUT2 & OUT3
    end

    classDef inputNode fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef dspNode fill:#042f2e,stroke:#14b8a6,stroke-width:1.5px,color:#ccfbf1;
    classDef specNode fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#e0e7ff;
    classDef dlNode fill:#311042,stroke:#c084fc,stroke-width:1.5px,color:#f3e8ff;
    classDef mlNode fill:#422006,stroke:#f59e0b,stroke-width:1.5px,color:#fef3c7;
    classDef featNode fill:#1e293b,stroke:#94a3b8,stroke-width:2px,color:#f1f5f9;
    classDef fusionNode fill:#4c0519,stroke:#fb7185,stroke-width:2.5px,color:#ffe4e6;
    classDef outNode fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5;
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
