# Model Interpretability and Validation of the Explanations (Phase 2)

Author: Praanshu Ranjan. Source code: `src/deep_explainability.py`. Figures: `reports/figures_phase2/`.
Every number below is read from `reports/phase2_deep_explainability_summary.json` and the CSV tables beside it.

## 1. Scope and models explained

We explain the two Phase 2 models, a spectrogram 2D-CNN ("CNN-only") and the hybrid late-fusion model, both
trained by the unmodified `src/train_cnn.py` (run with `--lambda-grid 0.05`) on the 767-song PMEmo manifest
(536 / 115 / 116 songs, split by song). The checkpoints were retrained locally for this analysis: `*.pt` files are
git-ignored, so the team's checkpoints are not in the repository, and the Phase 2 metrics committed in
`models/phase2_model_metrics.json` come from a 60-song run (42 / 9 / 9 songs, scaler fitted on 126 windows), the
same size as the repository's synthetic sample set. The explanations below therefore refer to these retrained
models; they should be re-run on the final shared checkpoint with the same command. The retrained checkpoints are
kept in the git-ignored folder `models/retrained_real/`; their SHA-256 hashes, the window-cache signature, the seed
and the training command are recorded in `reports/phase2_deep_explainability_summary.json`. All analyses use the **validation split only** (115 songs, 3,110 three-second windows); the test split is never
loaded. On this split the models reach a song-level R² of 0.565 (CNN-only, r = 0.756) and 0.523 (hybrid,
r = 0.735); window-level R² is 0.480 and 0.447. The models are therefore informative but imperfect, and they
shrink extreme ratings towards the middle (for example, song 520: human valence 0.15, model 0.41; song 734:
human 0.82, model 0.47; Figure 01). An explanation describes what the *model* does, including when it is wrong.

**Target.** Only valence is predicted, so the explanations concern valence. Arousal is not a model output.

## 2. Methods

| Component | Method | Implementation choices |
| --- | --- | --- |
| Spectrogram heatmaps | Grad-CAM (Selvaraju et al., 2017) | Last conv block activation (16 x 15), gradient of the **pre-sigmoid** output (the sigmoid saturates near 0 and 1), bilinear upsampling to 128 x 126, eval mode on CPU |
| Neural attribution | Integrated Gradients (Sundararajan et al., 2017) | Silence baseline (-80 dB, which the model maps to 0), 256 Gauss-Legendre steps, attribution in valence units |
| Neural attribution | DeepLIFT (Shrikumar et al., 2017) | Same baseline; completeness checked before use (Section 3.2) |
| Tabular branch | Exact Shapley values (Lundberg and Lee, 2017) | 8 Phase 1 features, all 256 coalitions, interventional baseline from 32 training windows, spectrogram held fixed |
| Modality split | Integrated Gradients over (spectrogram, acoustic vector) | Net attribution summed per modality |

**Regression has no class to explain.** A standard Grad-CAM map is built for one class score. Valence is a
number, so each window gets one *signed* map: positive regions push the valence score up ("raising"), negative
regions push it down ("lowering"). Figures show raising regions in red and lowering regions in blue.

## 3. Validation of the explanations

An attribution method can produce plausible-looking pictures without reflecting the model. We tested this.

### 3.1 Randomisation test (Adebayo et al., 2018)

Layers were re-initialised from the output towards the input (60 windows). A map that depends on what the model
learned must change. For both models the rank correlation with the trained-model map falls to about zero once the
convolutional blocks are randomised (CNN-only, absolute map: 1.00 -> 0.60 -> 0.25 -> 0.02; hybrid: 1.00 -> 0.49 ->
0.25 -> 0.05; Figure 05). The test is passed. One caution: the CNN-only *signed* map recovers to 0.34 when every
layer is random, so signed maps carry some structure from the input itself.

### 3.2 Completeness of Integrated Gradients and DeepLIFT

A complete attribution satisfies: sum of attributions = f(x) - f(baseline).

- **Integrated Gradients:** median gap 1.1% (CNN-only) and 1.0% (hybrid); 89% and 95% of windows within 5%
  (Figure 04). With 32 steps the gap was about 10%, so the step count matters and was raised to 256.
- **DeepLIFT: the assumptions are not satisfied by these networks.** The median gap is 100% (CNN-only) and 59%
  (hybrid). Captum applies DeepLIFT's rescale rules to ReLU, Sigmoid and MaxPool2d but treats LayerNorm and
  AdaptiveMaxPool2d as linear. Replacing those layers (diagnostic copies only) cuts the gap to 1.4% and 5.3%, while
  replacing either one alone does not (LayerNorm only: 18% / 14%; max-pool only: 63% / 64%). We therefore do **not**
  use DeepLIFT for conclusions, show it only for comparison, and rely on Integrated Gradients and Grad-CAM.

### 3.3 Deletion faithfulness

For 150 windows (from 84 distinct songs) we masked the top 5-50% of pixels that the map calls "raising" (or "lowering") and measured the
change in predicted valence. The control is the *same mask shifted to random positions*, which keeps its area and
shape. A faithful map should make "raising" masks lower the prediction more than the control, and "lowering" masks
raise it (Figure 06). Masked pixels were set to silence or to the window mean, because either choice moves the
input off the training distribution. Windows of one song are correlated, so the paired Wilcoxon tests are run on
per-song means (n = 84), not on windows.

- **Hybrid: faithful.** Removing 50% "raising" pixels changes valence by -0.048 (silence) or -0.051 (mean), against
  -0.008 and -0.012 for shifted controls (song-level p < 1e-5 at every mask size and fill). Removing "lowering"
  pixels raises it by +0.025 (silence, p < 1e-6 at every size) and +0.018 (mean fill, 50% mask). With mean fill the
  lowering effect is weaker and not significant at 10-30% masks (p = 0.09-0.20).
- **CNN-only: mixed.** With mean fill the ordering is as expected at every mask size (50%: raising -> +0.005,
  random +0.050, lowering +0.080; song-level p < 0.02 for raising and < 1e-8 for lowering). With silence fill it
  is not: "raising" masks beat the control only at 10-30% (p <= 0.018) and not at 5% or 50% (p = 0.23, 0.66), and
  "lowering" masks do not raise the prediction above the control at 20-50% (at 50% they *reduce* it, -0.068 vs
  -0.035). The CNN-only maps are faithful under in-distribution (mean) masking but not reliably under silence.

### 3.4 Consistency and input dependence

Neighbouring windows of one song (50% overlap) have much more similar frequency profiles than windows of different
songs (CNN-only: r = 0.81 vs 0.30; hybrid: 0.73 vs 0.28; Figure 07; the pairs are not independent, so this is
descriptive), so the maps respond to the audio.
A large share is nonetheless an input-independent pattern: the Spearman correlation between a window's map and the
dataset-average map is 0.47 (CNN-only) and 0.35 (hybrid). Grad-CAM taken at the post-ReLU activation and at the raw
convolution output agree moderately (rho = 0.65 and 0.82).

### 3.5 Disagreement between methods

Integrated Gradients and Grad-CAM, pooled to the same 16 x 15 grid, agree very weakly (rho = 0.03 CNN-only, 0.10
hybrid). The two methods answer different questions (gradient-weighted activation vs path-integrated attribution
from silence), and Integrated Gradients is noisy at pixel level (Figure 03). **No single heatmap should be read as
"the" explanation.**

## 4. Results

**Tabular branch of the hybrid model (Figure 08).** Mean |Shapley value| in valence units: spectral bandwidth
0.018, MFCC 1 0.017, spectral contrast 0.016, spectral centroid 0.014, spectral flatness 0.013, zero-crossing rate
0.009, dissonance estimate 0.006, roll-off 0.003. Efficiency holds exactly (max error 6e-17). The Shapley values of
centroid and bandwidth are negatively correlated across windows (r = -0.58), as are MFCC 1 and roll-off (-0.42).
These features are strongly correlated inputs (Phase 1), so the network appears to use some of them jointly and
credit is split between them; a single feature's value should not be read alone.

**Link to Phase 1.** Phase 1 found that spectral centroid has no association with valence once arousal is
controlled (partial rho = -0.00) and that MFCC 1, spectral contrast and flatness are the top predictors. In the
hybrid's tabular branch MFCC 1, contrast and flatness are again among the five largest contributions, but centroid
and bandwidth also contribute substantially. We did not test whether this reflects a contrast between the two
features or a use of arousal-related information; it is a hypothesis for Phase 2 ablations, not a finding.

**Modality split (Figure 09).** The mean net attribution is 0.076 for the spectrogram and 0.060 for the eight
descriptors (valence units). Both track the prediction's deviation from the silence baseline (r = 0.83 and 0.61).
The two modalities contribute at comparable scale; the hybrid does not beat the CNN-only model on validation
(R² 0.523 vs 0.565), so the descriptors add little that the spectrogram does not already carry.

**Where the models find evidence (Figures 01, 02).** In the CNN-only model about half of the "raising" mass lies in
the 0.5-2 kHz band for both the lowest- and highest-rated thirds of songs (50% and 54%); in the hybrid model it is
24-31%, with more in the bass band. The band profiles differ between the two models, so we draw no general
conclusion about frequency ranges and valence.

## 5. How to read the heatmaps

1. A red region means: if the activations there were stronger, this model's valence score would be higher. It is a
   statement about the model's sensitivity, in time and frequency.
2. Describe regions only by time and frequency ("0.5-2 kHz, 1-2 s"). Do not name instruments, voices, lyrics or
   genres: the maps contain no information about sound sources. Dense harmonics, a vocal, or a synth can occupy the
   same region.
3. Do not read a map as the cause of how listeners felt. It explains the model, which is itself imperfect.
4. Compare methods before concluding anything (Section 3.5), and check that the analysis passed the checks in
   Section 3 for the model in question.
5. Maps are computed per three-second window; a song-level statement needs many windows (Figure 02).

## 6. Limitations

- Faithfulness depends on the masking baseline for the CNN-only model (Section 3.3).
- Integrated Gradients is specified relative to a silence baseline, which is far from real music; different
  baselines give different attributions.
- Grad-CAM resolution is 16 x 15 cells upsampled to 128 x 126, so fine detail in the heatmap is interpolation.
- Windows within a song are correlated; the 3,110 windows are 115 independent songs. Significance tests are
  therefore run on per-song means (n = 84 songs for the deletion test); consistency statistics are descriptive.
- The explained checkpoints are one training run (seed 42, MPS); another run will give different maps.
- Models are trained on one dataset of popular-music choruses; cross-cultural statements need the separate
  evaluation set described in the Phase 2 plan.

## 7. References

- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
- Sundararajan, Taly and Yan (2017). Axiomatic Attribution for Deep Networks. ICML.
- Shrikumar, Greenside and Kundaje (2017). Learning Important Features Through Propagating Activation Differences. ICML.
- Lundberg and Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Adebayo et al. (2018). Sanity Checks for Saliency Maps. NeurIPS.

## 8. Presentation notes (slides and speaking script)

**Slide title:** "What does the model listen to? Explanations we tested before trusting."

**Figures to show:** Figure 01 (Grad-CAM examples, hybrid), Figure 06 (deletion faithfulness, hybrid) and Figure 08
(tabular Shapley values). Keep Figures 04 and 05 as backup slides for questions.

**Three bullets:**

- Grad-CAM shows where the model is sensitive in time and frequency: red raises predicted valence, blue lowers it.
- We tested the explanations: the maps change when the model is randomised, and masking the "raising" regions lowers
  the hybrid model's prediction several times more than masking random regions of the same shape (about 4-6 times at a 50% mask).
- Methods disagree and DeepLIFT failed its assumptions here, so heatmaps describe the model, not instruments or causes.

**45-second script:**

> "To explain the models I used Grad-CAM, which highlights the time-frequency regions that push the predicted valence
> up in red or down in blue. Pictures can be misleading, so we tested them. If I randomise the network, the maps
> disappear, which is what we want. If I mask the regions the map calls important, the hybrid model's prediction drops
> several times more, about four to six times at half the spectrogram, than when I mask random regions of the same shape. DeepLIFT failed its own completeness check
> on our architecture, so we used Integrated Gradients instead, and the two methods agree only weakly. So we read the
> heatmaps as where the model is sensitive, never as instruments or reasons people felt a certain way."

**Likely questions:** why is the CNN-only result mixed? (its maps are faithful when masked areas are filled with
in-distribution values, not with silence); why not DeepLIFT? (LayerNorm and max-pooling break its rescale rules; the
gap fell to 1.4% when both were replaced); why does the hybrid not beat the CNN-only model? (validation song-level
R² 0.523 vs 0.565; the eight descriptors add little beyond the spectrogram).
