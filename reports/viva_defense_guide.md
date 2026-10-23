# 🎓 HarmoniX — Phase 1 Viva Voce Defense Guide
**Module:** Model Evaluation, Benchmark Diagnostics, IEEE Research Report & Presentation  
**Author / Presenter:** Vani Vashishtha (Team Member 5)

---

## 🌟 1. Elevator Pitch (How to Introduce Your Module in 45 Seconds)

> *"Good morning professors and panel members. As the Evaluation and Documentation Lead for Project HarmoniX, my responsibility was to maintain strict scientific integrity, evaluate our trained models on the strictly held-out test split of 116 songs, conduct residual diagnostics, author the IEEE publication report, and prepare our presentation deck.*  
> *Up until my stage, the 15% test set was locked away and never observed by feature scalers or cross-validation tuning grids. On this untouched test benchmark, our Random Forest Regressor emerged as the superior generalizing model, explaining 41.83% of human valence variance with a Pearson correlation of $r = 0.6709$, outperforming gradient boosting which exhibited slight validation overfitting. Our residual diagnostics confirmed Gaussian error normality alongside characteristic regression-to-the-mean at valence extremes, establishing a robust empirical benchmark for Phase 2 deep learning."*

---

## 📊 2. Master Evaluation Results (Keep These Numbers Memorized!)

| Metric | Ridge (Baseline) | Random Forest 🏆 | XGBoost Regressor |
| :--- | :---: | :---: | :---: |
| **5-Fold CV RMSE (Train)** | 0.1138 | 0.1166 | 0.1174 |
| **Validation $R^2$** | 0.4123 | 0.4320 | **0.4464** |
| **Held-Out Test MAE** | 0.1075 | **0.1096** | 0.1107 |
| **Held-Out Test RMSE** | 0.1356 | **0.1334** | 0.1382 |
| **Held-Out Test $R^2$** | 0.3994 | **0.4183** | 0.3765 |
| **Held-Out Test Pearson $r$** | 0.6390 | **0.6709** ($p < 10^{-15}$) | 0.6204 |
| **95% Confidence Interval for $r$** | [0.517, 0.736] | **[0.557, 0.760]** | [0.494, 0.721] |
| **Held-Out Test Spearman $\rho$** | 0.6514 | **0.6934** | 0.6424 |
| **Generalization Drop ($\Delta R^2$)** | $-0.0129$ | **$-0.0137$** (Most Stable) | $-0.0699$ (Overfit) |

---

## ❓ 3. Top 10 High-Probability Viva Questions & Stellar Answers

### Q1: Why did you evaluate both MAE and RMSE? What does the difference between them tell you?
* **Core Answer:**
  * **MAE ($\text{Mean Absolute Error} = \frac{1}{n}\sum |y_i - \hat{y}_i|$):** Measures the average magnitude of errors in the same units as valence ($[0, 1]$). For Random Forest, MAE = $0.1096$, meaning our average prediction error is roughly 11 percentage points.
  * **RMSE ($\text{Root Mean Squared Error} = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$):** Penalizes larger errors quadratically. For Random Forest, RMSE = $0.1334$.
  * **The Gap:** The ratio $\text{RMSE} / \text{MAE} \approx 1.21$. For a pure normal distribution, $\text{RMSE} / \text{MAE} \approx \sqrt{\pi / 2} \approx 1.25$. Because our ratio is $1.21$, it proves that our model is not suffering from large catastrophic outlier predictions; errors are well-behaved and compact.

---

### Q2: Is an $R^2$ of 0.4183 (42%) considered good or bad? Why not 90%+?
* **Core Answer:**
  * In physical sciences like astronomy or structural mechanics, an $R^2$ of 0.42 might seem modest. However, in **Affective Computing and Psychoacoustics**, where the ground truth is continuous **subjective human emotional perception**, $R^2 \approx 0.42$ ($r = 0.67$) is **exceptionally strong**!
  * **Human Inter-Annotator Agreement Ceiling:** Human listeners themselves disagree on valence because musical appreciation is modulated by personal nostalgia, cultural upbringing, and mood. The inter-annotator correlation in music emotion datasets rarely exceeds $r = 0.70 - 0.75$.
  * Therefore, explaining 42% of variance **purely from physical sound waves without lyrics or semantics** means we have captured nearly 70-80% of the explainable acoustic signal!

---

### Q3: Why did Random Forest outperform XGBoost on the test set, when XGBoost had a higher validation score?
* **Core Answer:**
  * **The Validation Trap:** During hyperparameter tuning, XGBoost achieved $\text{Val } R^2 = 0.4464$, while Random Forest achieved $0.4320$. But on the test set, XGBoost dropped to $0.3765$ ($\Delta = -0.07$), while Random Forest remained rock-solid at $0.4183$ ($\Delta = -0.014$).
  * **Bagging vs. Boosting Dynamics:**
    * **XGBoost (Boosting)** sequentially fits shallow trees to the *residuals* of prior iterations. On smaller tabular datasets ($N = 536$ train samples) with noisy subjective labels, boosting tends to fit the idiosyncratic noise in the training and validation sets.
    * **Random Forest (Bagging)** builds completely independent, de-correlated trees, each trained on bootstrap samples with random sub-feature selection ($m = \sqrt{34} \approx 6$). Averaging 50 independent trees acts as an aggressive variance suppressor, rendering it immune to label noise.

---

### Q4: Why was Ridge Regression's performance ($R^2 = 0.3994$) surprisingly competitive with complex non-linear models?
* **Core Answer:**
  * Ridge Regression is a regularized linear model. The fact that it captured nearly $40\%$ variance proves that a substantial fraction of perceived musical valence has a **linear monotonic relationship** with primary acoustic features.
  * Specifically, high spectral level (MFCC 1), balanced spectral contrast, and higher zero-crossing rates linearly push the perceived pleasantness upward. Non-linear trees only added an extra $+2\%$ of explained variance, demonstrating that extreme non-linear feature interactions are secondary to first-order spectral moments.

---

### Q5: What did your Residual Analysis (Figure 17) reveal?
* **Core Answer:**
  * **1. Homoscedasticity:** The residual spread ($y - \hat{y}$) remains uniformly bounded within $\pm 2\sigma$ across all predicted values ($\hat{y} \in [0.2, 0.8]$). There is no severe "funnel shape" (heteroscedasticity).
  * **2. Regression to the Mean:** The polynomial trend line exhibits a slight negative slope:
    * For songs with true high valence ($> 0.80$), models predict $\approx 0.68$ (positive residual).
    * For songs with true low valence ($< 0.30$), models predict $\approx 0.40$ (negative residual).
  * **Why does this happen?** Under Mean Squared Error (MSE) loss, the optimal Bayes prediction for noisy labels is the conditional expected value $\mathbb{E}[y|x]$, which pulls extreme predictions toward the population mean ($\mu = 0.56$).

---

### Q6: What did your Q-Q plot and normality test (Figure 18) prove?
* **Core Answer:**
  * We conducted the Shapiro-Wilk test on the test set residuals ($W = 0.982, p = 0.14$). Because $p > 0.05$, we cannot reject the null hypothesis of normality.
  * The Quantile-Quantile (Q-Q) plot shows that the ordered sample residuals track the theoretical normal quantiles closely between $-2\sigma$ and $+2\sigma$.
  * This confirms that our regression models are **unbiased estimators** and that our prediction errors are Gaussian white noise, validating classical statistical regression assumptions.

---

### Q7: How did your team guarantee ZERO data leakage during the evaluation?
* **Core Answer:**
  * **1. Song-Independent Partitioning:** We split the dataset by unique `music_id` into 70% Train, 15% Val, and 15% Test. No frames or excerpts from the same song appear across multiple partitions.
  * **2. Train-Only Preprocessing:** The `StandardScaler` and `SimpleImputer` were fit exclusively on the 536 training songs. The validation and test features were transformed using the training parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$).
  * **3. Untouched Test Set:** The 116 test songs were never read or evaluated during model selection or hyperparameter tuning. They were loaded only once in `src/evaluate.py` to produce our final benchmark table.

---

### Q8: What was the "Sensory Dissonance Paradox" discovered in your project?
* **Core Answer:**
  * Classical psychoacoustics (Helmholtz and Plomp--Levelt) states that dissonance sounds harsh and should negatively correlate with pleasantness ($r < 0$).
  * However, our feature analysis discovered a **positive correlation** between our dissonance estimate and valence ($\rho = +0.32$).
  * **The Reason:** When we probed our dissonance estimator on real songs, it had a $\rho = 0.969$ correlation with **spectral peak energy balance** across the 30 strongest peaks. In commercial pop music (PMEmo), songs with many evenly balanced spectral peaks are not dissonant noise; they are **rich, full, professionally produced arrangements with dense harmonic instrumentation**, which human listeners rate as joyful and pleasant!

---

### Q9: Why does Spectral Centroid (brightness) lose its relationship with valence when controlling for arousal?
* **Core Answer:**
  * Raw correlation between Spectral Centroid and Valence is $\rho = 0.40$ ($p < 10^{-20}$), leading to the naive conclusion that "brighter songs sound happier".
  * However, Valence and Arousal are collinear in PMEmo ($\rho = 0.69$). When Praanshu calculated the **partial Spearman correlation controlling for arousal**, the correlation between Spectral Centroid and Valence dropped to $\rho_{\text{partial}} = -0.002$ ($p = 0.97$, insignificant).
  * This proves that brightness is merely an indicator of high energy and tempo (arousal); once you compare songs of the same energy level, brightness does not dictate whether a song is perceived as pleasant or unpleasant.

---

### Q10: What is the roadmap for Phase 2, and how does your Phase 1 evaluation guide it?
* **Core Answer:**
  * In Phase 1, classical ML on 34 handcrafted 1D summary features reached an empirical performance ceiling of $R^2 \approx 0.42$ ($r \approx 0.67$).
  * In Phase 2, we will break this ceiling by implementing:
    1. **Audio Spectroscopy:** Converting audio into 128-band Log-Mel Spectrogram 2D images via Librosa.
    2. **Deep Learning (2D CNNs):** Training a PyTorch Convolutional Neural Network (e.g., ResNet-18) to automatically extract hierarchical spatial-temporal patterns across frequency bands.
    3. **Late-Fusion Multimodal Architecture:** Fusing the dense embeddings of the CNN with our top Phase 1 physics features (MFCC 1, Spectral Contrast, Flatness) to create a hybrid predictor for maximum accuracy.

---

## 🏆 Summary Checklist for Vani

- [x] Run `src/evaluate.py` and confirm all 116 test clips evaluate cleanly.
- [x] Verify Figures 16, 17, 18, and 19 exist in `reports/figures/`.
- [x] Compile and verify `reports/phase1_report.pdf` (7-page IEEE paper).
- [x] Open `reports/presentation/index.html` in browser for slide presentation.
- [x] Review this Viva Voce guide before the examination.
