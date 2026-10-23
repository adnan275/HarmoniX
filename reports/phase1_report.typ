#set page(
  paper: "us-letter",
  margin: (x: 1.8cm, top: 2.0cm, bottom: 2.2cm),
  columns: 1,
)

#set text(
  font: "Liberation Serif",
  size: 10pt,
)

#align(center)[
  #v(0.5cm)
  #text(17pt, weight: "bold")[The Science of Musical Perception: Predicting the Pleasantness of Music Using Wave Physics, Acoustic Signal Processing, and Classical Machine Learning]
  
  #v(0.3cm)
  #text(10pt, style: "italic")[HarmoniX Phase 1 Research Report — Model Evaluation & Empirical Benchmark]
  #v(0.4cm)

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    align(center)[
      #text(9.5pt, weight: "bold")[Adnan Rizvi]\
      #text(8pt)[Lead & Data Ingestion]\
      #text(7.5pt, fill: rgb("#475569"))[adnan\@harmoni-x.org]
    ],
    align(center)[
      #text(9.5pt, weight: "bold")[Aditya Garariya]\
      #text(8pt)[DSP & Physics Features]\
      #text(7.5pt, fill: rgb("#475569"))[aditya\@harmoni-x.org]
    ],
    align(center)[
      #text(9.5pt, weight: "bold")[Bhoomi]\
      #text(8pt)[ML Modeling & Tuning]\
      #text(7.5pt, fill: rgb("#475569"))[bhoomi\@harmoni-x.org]
    ],
    align(center)[
      #text(9.5pt, weight: "bold")[Praanshu Ranjan]\
      #text(8pt)[Feature Explainability]\
      #text(7.5pt, fill: rgb("#475569"))[praanshu\@harmoni-x.org]
    ],
    align(center)[
      #text(9.5pt, weight: "bold")[Vani Vashishtha]\
      #text(8pt)[Test Eval & Publication]\
      #text(7.5pt, fill: rgb("#475569"))[vani\@harmoni-x.org]
    ]
  )
  
  #v(0.2cm)
  #text(9pt, style: "italic", fill: rgb("#334155"))[Department of Computer Science and Artificial Intelligence]
  #v(0.4cm)
]

#rect(
  fill: rgb("#f8fafc"),
  stroke: rgb("#cbd5e1"),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  #text(9pt, weight: "bold")[Abstract] --- 
  #text(8.5pt)[
    Music evokes profound emotional and affective responses across diverse cultures and languages, often operating independently of semantic or lyrical comprehension. This study investigates the degree to which objective, physical, and acoustic characteristics of sound waves can predict human-perceived musical pleasantness (quantified as continuous emotional valence) without relying on lyrics, cultural metadata, or end-to-end black-box embeddings. Grounded in acoustic wave physics, Digital Signal Processing (DSP), and psychoacoustic consonance theory (Helmholtz and Plomp--Levelt), we extract a 34-dimensional feature vector comprising Mel-Frequency Cepstral Coefficients (MFCCs), spectral moments (centroid, bandwidth, contrast, flatness, and roll-off), waveform energy metrics (RMS, zero-crossing rate), harmonic pitch distributions (chroma STFT), Harmonic-to-Noise Ratio (HNR), and sensory dissonance estimates. Utilizing the PMEmo dataset ($N=767$ verified musical excerpts), we enforce a strict, leakage-free song-independent 70% training ($n=536$), 15% validation ($n=115$), and 15% held-out test split ($n=116$), ensuring that feature scalers and hyperparameter selections never observe the test partition. We evaluate linear regularized benchmark models (Ridge Regression), non-linear bagging ensembles (Random Forest Regressor), and gradient tree boosting (XGBoost Regressor). On the strictly untouched 116-song test set, the Random Forest model achieves superior generalization, attaining an $R^2$ of 0.4183, a Pearson correlation $r = 0.6709$ ($p < 10^(-15)$, 95% CI: [0.557, 0.760]), and an RMSE of 0.1334, outperforming XGBoost ($R^2 = 0.3765$) which exhibited slight overfitting during validation tuning. Feature explainability via SHAP and permutation importance reveals that timbral texture (MFCC 1) and spectral contrast dominate model decisions, while sensory dissonance captures arrangement density rather than micro-tonal roughness. Comprehensive residual diagnostics demonstrate near-Gaussian error profiles alongside classic regression-to-the-mean tendencies at valence extremes. These findings establish a rigorous classical baseline for Phase 2, which will incorporate continuous spectrogram representations and 2D Convolutional Neural Networks.
  ]
  
  #v(0.15cm)
  #text(8.5pt, weight: "bold")[Index Terms] --- 
  #text(8pt, style: "italic")[Music Information Retrieval (MIR), Musical Valence, Psychoacoustics, Plomp--Levelt Consonance, Spectral Moments, Random Forest, XGBoost, Residual Diagnostics, Zero-Leakage Pipeline.]
]

#v(0.4cm)

#set page(columns: 2)

= I. Introduction

Music is an intrinsic aspect of human culture, transcending linguistic, geographical, and cognitive boundaries. Listeners routinely experience emotional resonance---feeling uplifted, calm, melancholic, or agitated---when listening to instrumental compositions or songs performed in languages they do not comprehend. This universal affective phenomenon prompts a foundational scientific inquiry: _To what extent can human emotional pleasantness be predicted purely from the physical mechanics and mathematical properties of acoustic pressure waves?_

Acoustic waves fundamentally represent longitudinal oscillations in atmospheric pressure, governed by classical wave mechanics ($v = f lambda$), frequency decomposition via the Fourier Transform, and psychoacoustic perception. In Music Information Retrieval (MIR) and affective computing, emotional perception is classically operationalized using Russell's Circumplex Model of Affect, which situates emotions along two continuous, orthogonal axes: *Valence* (the hedonic pleasantness or positivity vs. negativity of the stimulus) and *Arousal* (the physiological activation or energy level). While arousal is strongly associated with macroscopic wave energy and tempo, predicting *valence* from audio signals is notoriously challenging due to the complex interplay of harmony, consonance, timbre, and subjective listener psychology.

Phase 1 of Project *HarmoniX* examines the predictive capacity of classical Machine Learning when supplied strictly with physics-informed and acoustic digital signal processing (DSP) features. To maintain scientific rigor and prevent spurious data leakage, we enforce:
1. *Acoustic-only isolation:* All lyrical transcripts, semantic tags, and artist metadata are discarded.
2. *Song-independent partitioning:* A 70% Train / 15% Validation / 15% Held-out Test split is generated such that all audio frames or excerpts belonging to any single musical piece reside exclusively in one partition.
3. *Untouched evaluation integrity:* All feature scalers, imputers, and cross-validation hyperparameter grids are fit exclusively on training data, preserving the test partition untouched until final evaluation.

This paper documents the complete Phase 1 methodology, statistical feature analysis, test set benchmark evaluation, and residual diagnostics, providing the empirical foundation for deep learning extensions in Phase 2.

= II. Theoretical Foundations & Related Work

== A. Helmholtz and Plomp--Levelt Consonance Theory
The physical basis of musical pleasantness has historical roots in acoustic physics. Hermann von Helmholtz posited that perceived dissonance arises from the phenomenon of "beating" between adjacent partial frequencies that fall within the ear's critical auditory bandwidth. When two sinusoidal components lie closer than a critical frequency interval $Delta f_c$, the human basilar membrane cannot resolve them as distinct pitches, producing amplitude modulation perceived as sensory roughness or dissonance.

Plomp and Levelt formalized this relationship empirically. For two pure sinusoidal tones at frequencies $f_1$ and $f_2$ ($f_1 < f_2$), the sensory dissonance index $d(f_1, f_2)$ can be modeled analytically as:
$ d(f_1, f_2) = e^(-a s (f_2 - f_1)) - e^(-b s (f_2 - f_1)) $
where $s = c / (f_("min") + d)$, and $a, b, c, d$ are psychoacoustically calibrated constants ($a approx 3.5, b approx 5.75, c approx 0.24, d approx 230$). Dissonance peaks when the frequency difference equals approximately 25% of the critical bandwidth and decays toward zero as the interval widens toward musical consonance (e.g., perfect octaves, fifths).

== B. Fourier Analysis and Spectral Moments
Real-world musical audio comprises non-stationary superposition of harmonics. By applying the Short-Time Fourier Transform (STFT):
$ X(m, omega) = sum_(n=-infinity)^(infinity) x[n] w[n - m] e^(-j omega n) $
where $w[n]$ is a window function (e.g., Hann), the frequency spectrum is obtained across time frames $m$. From the magnitude spectrogram $|X(m, omega)|$, higher-order spectral moments can be derived:
- *Spectral Centroid:* Represents the "center of mass" of the spectrum, correlated psychoacoustically with brightness:
  $ C(m) = (sum_k f_k |X(m, k)|) / (sum_k |X(m, k)|) $
- *Spectral Flatness:* The ratio of the geometric mean to the arithmetic mean of the power spectrum, quantifying whether a sound resembles white noise (flatness $arrow 1$) or tonal resonance (flatness $arrow 0$).
- *Spectral Contrast:* Measures the energy difference between spectral peaks and spectral valleys across octave sub-bands.

== C. Affective Music Datasets: The PMEmo Benchmark
The PMEmo dataset (Zhang et al.) is an established benchmark designed specifically for continuous and static music emotion recognition. It comprises 794 popular song excerpts annotated by multiple human listeners for static valence and arousal on a normalized continuous scale $[0, 1]$. Unlike datasets with categorical labels, PMEmo provides continuous ground truth, enabling precise regression modeling and residual error analysis.

#figure(
  image("figures/01_valence_distribution.png", width: 95%),
  caption: [Continuous valence distribution in the PMEmo dataset showing near-Gaussian density centered around $mu = 0.56$ with slight positive skew.],
)

#figure(
  image("figures/02_valence_vs_arousal_space.png", width: 95%),
  caption: [Distribution of songs in Russell's 2D Valence--Arousal affect space, exhibiting strong positive collinearity ($rho = 0.69$).],
)

= III. Dataset Architecture & Leakage-Free Pipeline

== A. Data Ingestion & Verification
Data ingestion was architected in `src/data_loader.py`. Audio files were decoded into monophonic floating-point PCM signals resampled to a standardized sample rate of $f_s = 22,050" Hz"$. Songs were cross-referenced against static annotation metadata. A total of 767 audio clips matched annotation rows with complete valence and arousal scores; 27 unannotated audio files were pruned.

== B. Song-Independent Partitioning
To prevent data leakage, partitioning was executed strictly at the song level using a pseudo-random seed ($"seed" = 42$):
- *Training Partition:* 70% ($n = 536$ songs).
- *Validation Partition:* 15% ($n = 115$ songs).
- *Held-Out Test Partition:* 15% ($n = 116$ songs).

Zero song overlap was verified across all partitions ($"Train" inter "Val" = diameter$, $"Train" inter "Test" = diameter$, $"Val" inter "Test" = diameter$). Figure 3 confirms that the target valence distribution is homogeneous across the three splits, ensuring zero distributional shift.

#figure(
  image("figures/03_split_consistency_boxplot.png", width: 90%),
  caption: [Boxplot of valence distributions across Train, Validation, and Test splits, confirming identical medians and interquartile ranges.],
)

= IV. Physics-Informed Feature Engineering

Acoustic feature extraction was implemented in `src/feature_extraction.py`, converting frame-level measurements into a fixed-length 34-dimensional feature vector $bold(x) in RR^(34)$ for each song:
1. *Timbral Descriptors (13 Features):* 13 Mel-Frequency Cepstral Coefficients ($"MFCC"_1$ to $"MFCC"_(13)$), computed using a 2048-sample FFT window and 512-sample hop size. $"MFCC"_1$ captures macroscopic spectral power, while higher coefficients capture spectral envelope curvature.
2. *Spectral Moments (5 Features):* Spectral Centroid (perceived brightness), Spectral Bandwidth (spectral spread around centroid), Spectral Roll-off (frequency below which 85% of spectral energy is concentrated), Spectral Contrast, and Spectral Flatness.
3. *Wave Dynamics & Energy (2 Features):* Root-Mean-Square (RMS) amplitude representing physical energy density $cal(E) prop A^2$, and Zero-Crossing Rate (ZCR) capturing waveform polarity shifts per second.
4. *Harmonic Pitch Classes (12 Features):* 12 Chroma STFT vectors projecting energy onto the 12 semitone pitch classes (C, C\#, D, ..., B) to represent harmonic content.
5. *Harmonic-to-Noise Ratio (HNR) (1 Feature):* Estimated via the maximum autocorrelation peak of the time-domain signal:
  $ "HNR"_("dB") = 10 log_10 ((R_(x x)[tau_("max")]) / (R_(x x)[0] - R_(x x)[tau_("max")])) $
6. *Sensory Dissonance Index (1 Feature):* Implemented via pairwise Plomp--Levelt roughness summation across the 30 most prominent spectral peaks per STFT frame.

= V. Machine Learning Methodology

== A. Preprocessing & Leakage Safeguards
To prevent test set contamination, all data transformations were encapsulated within `scikit-learn` pipelines:
- *Imputation:* Median imputation on training features was applied to handle undefined HNR values occurring in near-silent clips.
- *Standardization:* Feature vectors were transformed via $bold(z) = (bold(x) - bold(mu)_("train")) / bold(sigma)_("train")$, fitted strictly on training data.

== B. Model Architectures & Hyperparameter Tuning
Three distinct regression architectures were formulated:
- *Ridge Regression (L2 Baseline):* Minimizes penalized residual sum of squares:
  $ cal(L)_("Ridge") = sum_(i=1)^n (y_i - bold(w)^T bold(z)_i)^2 + alpha ||bold(w)||_2^2 $
  Hyperparameter $alpha$ was tuned via 5-fold cross-validation across $alpha in [0.01, 100.0]$, selecting $alpha^* = 0.1$.
- *Random Forest Regressor (Bagging Ensemble):* A non-parametric ensemble of $B$ decorrelated decision trees. Each tree split evaluates a random subset of $m = sqrt(p)$ features. Tuning yielded $B = 50$ estimators, minimum samples per leaf $= 2$, and unconstrained tree depth.
- *XGBoost Regressor (Gradient Boosting):* Sequentially builds shallow trees minimizing regularized gradient loss:
  $ cal(L)^((t)) = sum_(i=1)^n [g_i f_t (bold(z)_i) + 1/2 h_i f_t^2 (bold(z)_i)] + gamma T + 1/2 lambda ||bold(w)||_2^2 $
  Optimal hyperparameters identified were: learning rate $eta = 0.1$, max depth $= 3$, estimators $= 150$, subsample $= 0.7$, colsample_bytree $= 0.85$, $lambda = 10.0$, and $alpha = 1.0$.

#figure(
  image("figures/04_feature_valence_correlations.png", width: 95%),
  caption: [Spearman and Pearson correlations between acoustic features and valence with Benjamini--Hochberg FDR significance.],
)

= VI. Feature Analysis & Explainability

Prior to test set evaluation, Praanshu executed statistical and SHAP analyses on training and validation sets (`src/feature_analysis.py`):

#figure(
  image("figures/12_shap_summary_xgboost.png", width: 95%),
  caption: [SHAP beeswarm summary plot for XGBoost on validation data, identifying MFCC 1, Spectral Contrast, and Spectral Flatness as dominant predictors.],
)

#figure(
  image("figures/13_shap_feature_groups.png", width: 90%),
  caption: [Mean absolute SHAP impact grouped by acoustic feature families, highlighting spectral shape and timbral descriptors.],
)

Key explainability findings include:
1. *Arousal Confounding:* Valence and arousal exhibit strong positive collinearity ($rho = 0.69, p < 10^(-70)$). While 24 of 34 features correlate with valence under Benjamini--Hochberg FDR control ($q < 0.05$), controlling for arousal via partial Spearman correlation reveals that apparent "valence" predictors (e.g., Spectral Centroid, raw $rho = 0.40$) drop to zero ($rho_("partial") = -0.002, q = 0.97$). Thus, brighter songs are rated as more positive largely because they are more energetic.
2. *Dominant Acoustic Predictors:* $"MFCC"_1$ (spectral level, $rho = 0.61$), Spectral Flatness ($rho = 0.50$), and Spectral Contrast ($rho = -0.45$) constitute the primary drivers in both tree models.
3. *Sensory Dissonance Paradox:* Dissonance estimates exhibited a surprising positive correlation with valence ($rho = 0.32$). Physical inspection on pure-tone dyads confirmed that while the estimator accurately mirrors Plomp--Levelt curves at A4 (440 Hz), its finite 2048-point STFT resolution cannot resolve sub-20 Hz beats in low registers. On real polyphonic audio, it acts as a proxy for arrangement density (spectral peak fullness) rather than harmonic clash.

#figure(
  image("figures/16_actual_vs_predicted.png", width: 100%),
  caption: [Figure 16: True vs. predicted musical valence on the held-out test set ($N=116$ songs). Red dashed line indicates parity diagonal ($y = hat(y)$). Random Forest achieves the highest fit ($R^2 = 0.4183, r = 0.6709$).],
)

= VII. Experimental Evaluation & Test Benchmark

== A. Held-Out Test Set Results
The final model benchmark was executed by Vani on the untouched test partition ($n = 116$ songs) via `src/evaluate.py`. Evaluation metrics comprise:
- Mean Absolute Error ($"MAE" = 1/n sum |y_i - hat(y)_i|$)
- Root Mean Squared Error ($"RMSE" = sqrt(1/n sum (y_i - hat(y)_i)^2)$)
- Coefficient of Determination ($R^2 = 1 - (sum (y_i - hat(y)_i)^2) / (sum (y_i - macron(y))^2)$)
- Pearson Correlation ($r$) with 95% Fisher-$z$ confidence intervals
- Spearman Rank Correlation ($rho$)

#align(center)[
  #table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1.2fr, 1fr),
    inset: 4pt,
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + rgb("#cbd5e1"),
    fill: (x, y) => if y == 0 { rgb("#e2e8f0") } else if y == 2 { rgb("#ecfdf5") } else { none },
    [*Model*], [*Test MAE*], [*Test RMSE*], [*Test $R^2$*], [*Test $r$ [95% CI]*], [*Test $rho$*],
    [Ridge], [0.1075], [0.1356], [0.3994], [0.639 [0.52, 0.74]], [0.6514],
    [*Random Forest*], [*0.1096*], [*0.1334*], [*0.4183*], [*0.6709 [0.56, 0.76]*], [*0.6934*],
    [XGBoost], [0.1107], [0.1382], [0.3765], [0.620 [0.49, 0.72]], [0.6424],
  )
]

== B. Model Performance Analysis
On the held-out test set, *Random Forest Regressor* emerged as the top-performing model:
- $R^2 = 0.4183$, explaining nearly 42% of human valence variance purely from 34 acoustic measurements.
- Pearson $r = 0.6709$ ($p = 6.2 times 10^(-16)$) and Spearman $rho = 0.6934$.
- Test RMSE of 0.1334, representing an error margin of approximately 13.3% across the continuous valence scale.

Crucially, while XGBoost attained the highest validation score ($R^2 = 0.4464$), its test performance degraded to $R^2 = 0.3765$ ($Delta R^2 = -0.0699$). In contrast, Random Forest exhibited minimal performance decay ($"Val " R^2 = 0.4320 arrow "Test " R^2 = 0.4183, Delta R^2 = -0.0137$). This demonstrates that bagging over decorrelated trees with sub-feature sampling ($sqrt(p)$) provides superior regularization against noise in subjective human affective annotations compared to gradient boosting.

Remarkably, the linear regularized *Ridge baseline* achieved $R^2 = 0.3994$ and $r = 0.6390$ on the test set, outperforming XGBoost. This confirms that low-order acoustic energy and spectral moments exhibit substantial linear alignment with perceived pleasantness.

#figure(
  image("figures/19_model_generalization_comparison.png", width: 95%),
  caption: [Figure 19: Multi-split error comparison. Random Forest demonstrates superior stability across validation and held-out test splits.],
)

= VIII. Residual Diagnostics & Error Analysis

#figure(
  image("figures/17_residual_analysis.png", width: 100%),
  caption: [Figure 17: Residual error diagnostics ($y - hat(y)$ vs. $hat(y)$) on the test set. Models exhibit homoscedastic variance alongside regression to the mean.],
)

#figure(
  image("figures/18_residual_distribution_qq.png", width: 100%),
  caption: [Figure 18: Residual normality diagnostics. Top row: Error histograms and KDE vs. normal curve. Bottom row: Quantile-Quantile (Q-Q) plots confirming normality across $[-2 sigma, +2 sigma]$.],
)

== A. Residual vs. Fitted Diagnostics
Figure 17 plots prediction residuals $e_i = y_i - hat(y)_i$ against fitted values $hat(y)_i$:
- *Homoscedasticity:* Residuals remain uniformly bounded within the $plus.minus 2 sigma$ error band across the entire prediction range, confirming absence of severe heteroscedasticity.
- *Regression to the Mean:* The fitted polynomial trend line exhibits a slight downward inclination. Specifically, songs with true valence $> 0.80$ yield positive residuals (models predict $approx 0.68$), while songs with valence $< 0.30$ yield negative residuals (models predict $approx 0.40$). This shrinkage toward the mean is an expected property of mean-squared-error minimization when predicting subjective human ratings with inherent perceptual noise.

== B. Normality of Residuals
Figure 18 examines the statistical distribution of test residuals:
- Residual mean is zero ($mu_e = -0.0051$ for Random Forest), indicating unbiased estimators.
- Q-Q plots display linear alignment across theoretical quantiles between $-2$ and $+2$, confirming that residual errors follow an approximately Gaussian distribution.

= IX. Limitations & Critical Discussion
Several scientific caveats must be highlighted:
1. *Valence as a Proxy:* Valence reflects perceived positivity/hedonic tone; it does not capture the full multidimensional aesthetic appreciation of music (e.g., poignant melancholy in classical or blues genres).
2. *Chorus Excerpt Bias:* PMEmo analyzes 30-second chorus clips from modern popular songs, which feature high production compression and strong spectral saturation.
3. *Psychoacoustic Estimation Limits:* Fourier-based sensory dissonance approximations do not fully model auditory nerve firing or critical band non-linearities.

= X. Phase 2 Deep Learning Roadmap
Phase 1 established that classical ML on 34 handcrafted acoustic features achieves a robust ceiling of $R^2 approx 0.42$ ($r approx 0.67$). In Phase 2, the team will implement:
1. *Audio Spectroscopy:* Computation of high-resolution 128-band Log-Mel Spectrograms representing fine-grained spectro-temporal timbre.
2. *Deep Convolutional Networks:* PyTorch-based 2D CNNs (e.g., ResNet-18, MobileNetV3) fine-tuned on spectrogram patches.
3. *Late-Fusion Hybrid Model:* Concatenating classical physics feature embeddings with latent CNN dense layers to achieve state-of-the-art multimodal valence prediction.

= XI. Conclusion
Project HarmoniX Phase 1 successfully demonstrated that human musical valence can be predicted with substantial statistical significance ($R^2 = 0.4183, r = 0.6709$) purely from physical wave mechanics and digital signal processing. By upholding strict song-independent partitioning, zero data leakage, and rigorous residual analysis, this study provides an empirical benchmark that bridges physical acoustics, affective psychology, and machine learning.

== References
+ J. A. Russell, "A circumplex model of affect," _Journal of Personality and Social Psychology_, vol. 39, no. 6, pp. 1161--1178, 1980.
+ Y.-H. Yang and H. H. Chen, _Music Emotion Recognition_. CRC Press, 2011.
+ H. von Helmholtz, _On the Sensations of Tone as a Physiological Basis for the Theory of Music_. Longmans, Green, and Co., 1885.
+ R. Plomp and W. J. M. Levelt, "Tonal consonance and critical bandwidth," _The Journal of the Acoustical Society of America_, vol. 38, no. 4, pp. 548--560, 1965.
+ K. Zhang, H. Zhang, S. Li, C. Yao, and M. Guo, "The PMEmo dataset for music emotion recognition," in _Proc. ACM ICMR_, 2018, pp. 481--488.
+ B. McFee et al., "librosa: Audio and music signal analysis in Python," in _Proc. 14th Python in Science Conf._, 2015, pp. 18--25.
+ T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in _Proc. ACM SIGKDD_, 2016, pp. 785--794.
+ S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in _NeurIPS_, 2017, pp. 4765--4774.
