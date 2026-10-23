---
marp: true
theme: default
paginate: true
header: "HarmoniX Phase 1 — The Science of Musical Perception"
footer: "Department of CSAI • Phase 1 Evaluation"
style: |
  section {
    background-color: #0a0e17;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
  }
  h1 {
    color: #38bdf8;
    font-family: 'Outfit', sans-serif;
  }
  h2, h3 {
    color: #818cf8;
  }
  strong {
    color: #34d399;
  }
  table {
    font-size: 14px;
    border-collapse: collapse;
    width: 100%;
  }
  th {
    background-color: #1e293b;
    color: #f8fafc;
  }
  td {
    border-bottom: 1px solid #334155;
  }
---

# 🎵 HarmoniX: The Science of Musical Perception
## Phase 1: Machine Learning & Physics-Informed Audio Analysis
### Predicting Musical Valence from Sound Waves

**Authors:** Adnan Rizvi, Aditya Garariya, Bhoomi, Praanshu Ranjan, Vani Vashishtha  
**Department of Computer Science & Artificial Intelligence**

---

# 👥 Phase 1 Team & Module Distribution

1. **👑 Adnan Rizvi (Lead):** Dataset Ingestion, Quality Audit & Leak-Free 70/15/15 Splitting (`src/data_loader.py`)
2. **🔬 Aditya Garariya:** Acoustic & Physics Feature Extraction (`src/feature_extraction.py`)
3. **📊 Bhoomi:** Classical ML Modeling & Cross-Validation Tuning (`src/train_ml.py`)
4. **🔍 Praanshu Ranjan:** Statistical Feature Analysis & SHAP Explainability (`src/feature_analysis.py`)
5. **📑 Vani Vashishtha:** Held-Out Test Evaluation, Residual Diagnostics & IEEE Report (`src/evaluate.py`)

---

# 🎯 Core Research Question

> **If we remove lyrics and semantic meaning, to what extent can measurable acoustic, physical, and mathematical characteristics of sound waves predict human-rated musical pleasantness (valence)?**

* **Russell's Circumplex Model of Affect:**
  * **Arousal:** Energy, speed, loudness (easy to predict via wave power $\mathcal{E} \propto A^2$).
  * **Valence:** Positivity vs negativity (challenging; rooted in harmony, consonance, and timbre).
* **HarmoniX Approach:** Strictly evaluate physics & DSP features using classical ML on an untouched test split.

---

# 🛡️ Zero Data Leakage Architecture

* **Dataset:** PMEmo (767 verified songs with static valence/arousal annotations).
* **Song-Independent Partitioning:**
  * **Train Split (70%):** 536 songs (fitted scalers and CV grids).
  * **Validation Split (15%):** 115 songs (model selection & tuning).
  * **Untouched Test Split (15%):** 116 songs (reserved for final evaluation).
* **Zero Overlap:** Disjoint sets by `music_id`. Zero target distribution shift across splits.

---

# 🔬 34-Dimensional Acoustic Feature Vector

1. **Timbral Descriptors (13):** 13 MFCC means (MFCC 1 tracks overall spectral energy).
2. **Spectral Moments (5):** Centroid (brightness), Bandwidth, Roll-off, Contrast, and Flatness.
3. **Wave Dynamics & Energy (2):** RMS Energy ($\mathcal{E} \propto A^2$) and Zero-Crossing Rate.
4. **Harmonic Pitch Classes (12):** 12 Chroma STFT pitch class averages.
5. **Harmonic Feature (1):** Harmonic-to-Noise Ratio (HNR in dB) via autocorrelation.
6. **Psychoacoustic Feature (1):** Sensory Dissonance estimate via Plomp--Levelt summation.

---

# 🧠 Classical Machine Learning Formulations

* **1. Ridge Regression (L2 Baseline):**
  * $\mathcal{L} = \sum (y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \alpha \|\mathbf{w}\|_2^2$
  * Optimal $\alpha = 0.1$ via 5-fold CV.
* **2. Random Forest Regressor (Bagging Ensemble):**
  * 50 decorrelated trees with sub-feature sampling ($m = \sqrt{p}$).
  * Minimum leaf samples $= 2$. High resilience to label noise.
* **3. XGBoost Regressor (Gradient Tree Boosting):**
  * 150 shallow trees (depth 3), $\eta = 0.1$, $\lambda = 10.0$, subsample 0.7.

---

# 🔍 Explainability Findings & Confounders

* **Valence & Arousal Collinearity:** $\rho = 0.69$ ($p < 10^{-70}$) in PMEmo.
* **The Spectral Centroid Reality Check:**
  * Raw correlation with valence: $\rho = 0.40$ ($q < 10^{-20}$).
  * Partial correlation controlling for arousal: $\rho_{\text{partial}} = -0.002$ ($q = 0.97$)!
  * **Conclusion:** Brighter songs are not rated positive because of brightness; they are positive because they are energetic!
* **SHAP Dominance:** MFCC 1, Spectral Contrast, and Spectral Flatness explain over 70% of model predictions.

---

# 🎶 The Sensory Dissonance Paradox

* **Intuition:** Dissonance sounds harsh $\to$ should correlate negatively with valence.
* **Empirical Finding:** Sensory dissonance correlated **positively** with valence ($\rho = +0.32$).
* **Physical Probe on Real Songs:**
  * Estimator matches Plomp--Levelt theory on pure-tone dyads at 440 Hz.
  * On polyphonic pop chorus excerpts, it correlates $\rho = 0.969$ with **spectral peak fullness**.
  * **Reality:** In PMEmo, the feature tracks *rich commercial production and arrangement density*, not musical clash!

---

# 📊 Held-Out Test Evaluation Benchmark ($N = 116$ Songs)

| Model | CV RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Test Pearson $r$ | Test $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge (Baseline)** | 0.1138 | 0.4123 | 0.1075 | 0.1356 | 0.3994 | 0.6390 | 0.6514 |
| **Random Forest 🏆** | 0.1166 | 0.4320 | **0.1096** | **0.1334** | **0.4183** | **0.6709** | **0.6934** |
| **XGBoost** | 0.1174 | 0.4464 | 0.1107 | 0.1382 | 0.3765 | 0.6204 | 0.6424 |

* **Random Forest Champion:** $R^2 = 0.4183$ and $r = 0.6709$ ($p = 6.2 \times 10^{-16}$).
* Explains **41.83% of human valence variance** purely from sound waves!

---

# 📈 Diagnostic Residual Analysis

* **Homoscedasticity:** Residual errors ($y - \hat{y}$) remain uniformly bounded within $\pm 2\sigma$ across the entire range $[0, 1]$.
* **Regression to the Mean:** 
  * Songs with true valence $> 0.8$ have positive residuals (models predict $\approx 0.68$).
  * Songs with true valence $< 0.3$ have negative residuals (models predict $\approx 0.40$).
  * Standard behavior of MSE loss on subjective human perceptual labels.
* **Error Normality:** Q-Q plots and Shapiro-Wilk test ($W = 0.98$) confirm Gaussian residual distribution.

---

# 🏆 Why Random Forest Beat XGBoost on Test Data

* **Generalization Stability:**
  * Random Forest dropped only **$-0.0137$** from Val ($0.4320$) to Test ($0.4183$).
  * XGBoost dropped **$-0.0699$** from Val ($0.4464$) to Test ($0.3765$).
* **Theoretical Reason:**
  * Subjective human valence ratings contain perceptual noise.
  * **Bagging (RF)** averages 50 independent trees with sub-feature sampling ($\sqrt{p}$), drastically lowering prediction variance.
  * **Boosting (XGBoost)** sequentially fits residuals, slightly overfitting to validation quirks on small datasets ($N=536$).

---

# 🔮 Phase 2 Deep Learning Roadmap

1. **Audio Spectroscopy:** Compute 128-band Log-Mel Spectrograms directly from raw PCM waveforms.
2. **Deep 2D CNNs:** Implement PyTorch-based Convolutional Neural Networks (ResNet-18) to learn spatial-temporal audio features.
3. **Late-Fusion Multimodal Hybrid:** Concatenate 34 handcrafted physics features with CNN dense embeddings for maximum predictive power.
4. **Streamlit Interactive UI:** Web app allowing real-time song upload and affective visualization.

---

# 📑 Phase 1 Deliverables Summary

* ✅ `src/evaluate.py`: Held-out test evaluation engine.
* ✅ `reports/phase1_report.pdf`: 7-page IEEE format publication paper.
* ✅ `reports/test_evaluation_results.csv`: Verified test metrics table.
* ✅ `reports/figures/16-19`: Publication-grade actual vs. predicted, residual & Q-Q plots.
* ✅ `reports/presentation/`: Interactive slide deck.

**Thank you! Questions & Viva Discussion.**
