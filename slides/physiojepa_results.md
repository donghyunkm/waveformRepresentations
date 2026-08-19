---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 21px;
  }
  table {
    font-size: 18px;
  }
  h1 {
    font-size: 34px;
  }
  h2 {
    font-size: 28px;
  }
  section.small {
    font-size: 19px;
  }
  section.tiny {
    font-size: 17px;
  }
  section.xtiny {
    font-size: 14px;
  }
  section.xtiny table {
    font-size: 13px;
  }
  section.xtiny h2 {
    font-size: 22px;
  }
---

# Physiological Waveform Representation Learning, built on top of PhysioJEPA (Fox et al., ML4H 2024)

**Models**: JEPA · PatchTST · Supervised PatchTST · FCN Baseline
**Data**: MIMIC-III ICU waveforms (ABP, ECG II, PLETH) at 125 Hz
**Task**: Hypotension prediction from 30-minute windows

---

## JEPA (Joint Embedding Predictive Architecture)

<!-- _class: small -->

**Core idea**: Learn representations by predicting in **embedding space**, not pixel/signal space.

**Architecture**:
1. **Context Encoder** — PatchTST transformer (3 layers, 8 heads, dim 512); processes a random subset of input patches → context embeddings
2. **Predictor** — Smaller transformer (2 layers, 4 heads, dim 256); takes context embeddings + mask tokens → predicts target embeddings
3. **Target Encoder** — EMA copy of context encoder (τ=0.996); processes the **full** input → ground-truth targets

**Input & Masking**: 30-min × 3-channel waveform at 125 Hz → 1-second patches (1,800/channel)
- Target mask: 10–30% of patches as prediction targets
- Context mask: 10–40% of remaining patches as visible context

**Training**: MSE loss in embedding space; target encoder updated via EMA only (no gradients); AdamW, one-cycle LR, 100 epochs

**Why embedding-space?** Noisy waveforms make pixel-level reconstruction hard; predicting embeddings learns higher-level temporal and cross-channel structure.

**After pretraining**: Freeze context encoder, attach lightweight attentive classifier for downstream tasks.

---

## PatchTST (Masked Patch Reconstruction)

<!-- _class: tiny -->

**Core idea**: Learn representations by **reconstructing masked signal patches** in the input space.

**Architecture**:
- PatchTST transformer encoder (3 layers, 8 heads, dim 512, FFN 2048) — same backbone as JEPA
- **Channel independence** (key distinction from standard transformers): each channel is tokenized and processed independently — batch and channel dims are flattened together, so the transformer sees single-channel sequences. No cross-channel attention; channels only interact at the downstream classification head.
- Depthwise convolution tokenizer (one kernel per channel)
- Rotary positional embeddings (RoPE)
- Pretrain head: linear projection back to patch dimension for reconstruction

**Input & Masking**: 30-min × 3-channel waveform at 125 Hz → 1-second patches (1,800/channel)
- Randomly mask 10–40% of patches
- Encoder sees only unmasked patches; must reconstruct masked patch values

**Training**: MSE reconstruction loss on masked patches; AdamW, one-cycle LR, 100 epochs

**Key difference from JEPA**: PatchTST predicts raw signal values (pixel-space) rather than latent embeddings. This forces the encoder to preserve low-level waveform details but may over-focus on noise reconstruction.

**After pretraining**: Same downstream protocol — freeze encoder, attach attentive classifier.

---

## Models Overview

| Model | Type | Architecture | Training |
|-------|------|-------------|----------|
| **PhysioJEPA** | Self-supervised | JEPA with context encoder + EMA target encoder | Predict masked patch representations |
| **PatchTST** | Self-supervised | Masked patch reconstruction | Reconstruct masked waveform patches |
| **Supervised PatchTST** | Supervised | PatchTST encoder + attentive classifier | End-to-end training on hypotension |
| **FCN** | Supervised | Fully convolutional network | Direct waveform → label |

All self-supervised models use frozen encoders + lightweight attentive probe for downstream evaluation.

---

## Data & Patient-Level Split

| Split | Subjects | ICU Stays | Samples | Positive Events | Prevalence |
|-------|----------|-----------|---------|-----------------|------------|
| Train (folds 2–9) | 2,013 | 3,195 | 1,022,563 | 43,950 | 4.3% |
| Val (fold 1) | 255 | 399 | 127,831 | 5,493 | 4.3% |
| Test (fold 0) | 256 | 407 | 127,811 | 5,494 | 4.3% |

- Corrected stratified group 10-fold (seed=16)
- **Subject-level disjointness**: no patient appears in more than one split
- All models share the same split for direct comparability
- Self-supervised pretraining excludes downstream val/test subjects

---

## Hypotension Prediction: Task Definition

**Label**: Hypotension event (5 consecutive minutes with MAP < 65) begins within a **5-minute forecast horizon** after the 30-minute window ends.

**Input**: 3-channel waveform window (ABP, ECG II, PLETH), 30 min at 125 Hz. During this window, the patient is hemodynamically stable (MAP ≥ 65) — the model must detect subtle precursors of impending deterioration from apparently normal physiology.

**Challenge**: 4.3% prevalence — highly imbalanced

**Evaluation**: AUROC and Average Precision (AP) on natural-prevalence test set

---

## Hypotension Prediction: Results

| Model | Test AUROC | Test AP | Type |
|-------|-----------|---------|------|
| **Supervised PatchTST** | **0.8688** | **0.2766** | Fully supervised |
| JEPA attentive probe | 0.8431 | 0.2653 | Frozen encoder + probe |
| PatchTST attentive probe | 0.8296 | 0.2344 | Frozen encoder + probe |
| Multi-scale PatchTST | 0.8414 | 0.2340 | Fully supervised |
| FCN baseline | 0.7903 | 0.1911 | Fully supervised |

---

## Frozen-Encoder Probes: Detailed Metrics

<!-- _class: small -->

Checkpoint: best AUPRC at epoch 3 (val AUPRC = 0.2833)

| Split | Samples | AUROC | AP | F1 | Precision | Recall |
|-------|---------|-------|----|----|-----------|--------|
| Val | 127,831 | 0.854 | 0.283 | 0.265 | 0.163 | 0.715 |
| Test | 127,811 | 0.843 | 0.265 | 0.240 | 0.145 | 0.697 |

PatchTST attentive probe (best checkpoint: epoch 3):

| Split | Samples | AUROC | AP |
|-------|---------|-------|----|
| Val | 127,831 | 0.853 | 0.268 |
| Test | 127,811 | 0.830 | 0.234 |

**Key finding**: A frozen JEPA encoder with only a lightweight attentive head achieves **97%** of fully-supervised performance (0.843 vs 0.869 AUROC), demonstrating that the self-supervised representation captures clinically relevant hemodynamic information. The frozen PatchTST probe is lower (0.830 AUROC, 0.234 AP).

---

## Hypotension Prediction: Key Takeaways

<!-- _class: small -->

1. **Self-supervised JEPA nearly matches fully-supervised models**
   - Only 0.026 AUROC gap with frozen encoder (no fine-tuning)
   - The encoder was never trained on hypotension labels

2. **Supervised PatchTST (not in the original paper) outperforms self-supervised models in this label-abundant regime.**
   - With ~1M labeled training samples, end-to-end training (0.869 AUROC) surpasses frozen JEPA (0.843) and PatchTST (0.830) probes
   - Self-supervised pretraining's advantage may emerge in low-label or transfer settings where large labeled datasets are unavailable

3. **Multi-scale tokenization didn't help**
   - Adding 200ms fine-grained patches (+275K params) underperformed standard 1s patches
   - Best at epoch 4, then declined — overfitting with extra parameters

---

## Medical Feature Probing: Overview

<!-- _class: small -->

**Question**: What physiological information do the encoders learn to represent?

**Method**: Ridge regression on **mean-pooled embeddings** (512-d)
- 10,000 randomly sampled windows, patient-level 80/20 split
- 15 physiological features (HR, SBP, DBP, PP, MAP, etc.)
- α (regularization) tuned per feature via cross-validation

**Raw signal statistics baseline**:
- Same Ridge regression pipeline and patient split
- Input: 30 hand-crafted features (10 statistics × 3 channels)
- Statistics: mean, std, min, max, median, IQR, skewness, kurtosis, zero-crossing rate, peak count (scipy `find_peaks`, min distance 0.4s, normalized by window length)
- Captures signal amplitude and distribution but **no temporal structure**
- Serves as a floor: any feature well-predicted by raw stats doesn't need a learned encoder

---

## Medical Feature Probing: Results (R²)

| Feature | JEPA | PatchTST | Raw Stats | Winner |
|---------|------|----------|-----------|--------|
| PLETH_amp | 0.957 | 0.965 | 0.963 | PatchTST |
| PLETH_ACDC | **0.924** | 0.853 | 0.714 | JEPA |
| HR | **0.734** | 0.697 | 0.092 | JEPA |
| dPdt_max | 0.602 | **0.734** | 0.432 | PatchTST |
| HRV_RMSSD | **0.532** | 0.469 | 0.185 | JEPA |
| DBP | 0.390 | 0.496 | **0.927** | Raw Stats |
| SBP | 0.270 | 0.380 | **0.825** | Raw Stats |
| MAP | 0.136 | 0.282 | **0.963** | Raw Stats |
| PTT | -0.058 | -0.015 | 0.071 | Raw Stats |

---

## Medical Feature Probing: Interpretation

### Encoders excel at temporal/morphological features
- **HR** (+0.64 over raw stats): requires R-peak detection + interval computation
- **HRV_RMSSD** (+0.35): requires beat-to-beat variability
- **dPdt_max** (+0.30): requires systolic upstroke identification
- **PLETH_ACDC** (+0.21): requires peak/trough detection in PPG

### Raw stats dominate for absolute signal levels
- MAP ≈ mean ABP by definition → R²=0.96 from channel mean alone
- IQR normalization during pretraining strips absolute values from encoder

---

## Representation Analysis: Clustering Metrics

<!-- _class: small -->

| Metric | What it measures | Range | Interpretation |
|--------|-----------------|-------|----------------|
| **Silhouette** | Cluster compactness and separation | [-1, 1] | 1 = compact & separated; 0 = overlapping; <0 = misassigned |
| **Homogeneity (H)** | Cluster purity: each cluster has one class? | [0, 1] | H=1: no cluster mixes classes (but one class may span multiple clusters) |
| **Completeness (C)** | Class preservation: all samples from one class in one cluster? | [0, 1] | C=1: each class in a single cluster (that cluster may contain others) |
| **ARI** | Pairwise agreement with true labels, chance-corrected | [-1, 1] | 1 = perfect; 0 = chance; penalizes both splitting and merging |

- **H increases trivially with k** (more clusters → smaller → purer) — not informative alone
- **C and ARI** are the key metrics: does the clustering recover the reference partition?

---

## Representation Analysis: K-Means Clustering

<!-- _class: small -->

**Setup**: 1000 windows (20 patients × 50), JEPA embeddings reduced via temporal subsampling (1800→20 patches) + PCA (512→121 dims) + StandardScaler → 7,260-d vectors (3 ch × 20 patches × 121 dims)

**Question**: Do embedding clusters correspond to physiological states or patient identity?

| k | Silhouette | H(patient) | C(patient) | ARI(patient) | ARI(hemo) |
|---|-----------|-----------|-----------|-------------|-----------|
| 2 | 0.027 | 0.14 | 0.79 | 0.04 | -0.09 |
| 5 | 0.051 | 0.41 | 0.79 | 0.22 | 0.03 |
| 10 | 0.070 | 0.57 | 0.77 | 0.37 | -0.01 |
| **20** | **0.078** | **0.78** | **0.79** | **0.61** | **0.00** |
| 30 | 0.079 | 0.83 | 0.76 | 0.61 | 0.01 |
| 50 | 0.058 | 0.88 | 0.70 | 0.51 | 0.00 |

- **Patient identity dominates**: ARI(patient) peaks at k=20 (matching 20 patients)
- **Hemodynamic state not captured**: ARI(hemo) ≈ 0 at all k
- **Low silhouette** (0.03–0.09): no discrete cluster structure

---

## Representation Analysis: Key Findings

1. **Embeddings correspond to patient identity**
   - K-Means at k=20 recovers patient identity (ARI=0.61)
   - Hemodynamic state alignment is at chance (ARI≈0)

2. **Low silhouette (0.03–0.09) indicates weak discrete structure**

---
## Limitations

1. **Embeddings encode patient identity, not hemodynamic state**
   - K-Means at k=20 recovers patient identity (ARI=0.61) but hemodynamic state alignment is at chance (ARI≈0)
   - Low silhouette (0.03–0.09) suggests no discrete physiological structure in the embedding space

2. **Self-supervised models underperform supervised in this label-abundant regime**
   - 0.026 AUROC gap between frozen JEPA probe and fully-supervised PatchTST
   - The value proposition of SSL pretraining is not yet demonstrated for this dataset size

---

## Next Steps

<!-- _class: small -->
- **Scale to longer context windows (e.g., 1–4 hours) for clinical tasks that require them.**
  - Predictions like sepsis onset, ventilator weaning readiness, or decompensation may depend on slower physiological trends (fluid responsiveness, vasopressor effects) missed in 30-min segments
  - Longer windows yield fewer labeled samples — a regime where self-supervised pretraining should provide a larger advantage over supervised baselines
  - Evaluate both frozen-encoder probes and supervised models to test this hypothesis
- **Contrastive objectives to learn hemodynamic state rather than patient identity.**
  - Current embeddings cluster by patient (ARI=0.61) not physiology (ARI≈0)
  - Contrastive loss that pulls together windows with similar hemodynamic states (e.g., same MAP range, HR range) across different patients, and pushes apart windows from the same patient with different states

---

## Literature Search

<!-- _class: xtiny -->

| Paper | Model/Objective | Waveform Input |
|---|---|---|
| **PhysioJEPA** | PhysioJEPA uses PatchTST-style context and target encoders with a predictor trained to infer the latent representations of masked waveform patches from visible patches, producing frozen representations for downstream estimation of near-term hypotension and elevated shock index. | It processes synchronized 30-minute segments of arterial blood pressure, ECG lead II and PPG sampled at 125 Hz, with each channel divided into 1-second patches. |
| **CSFM** | The Cardiac Sensing Foundation Model is a multimodal ViT-style Transformer trained with generative masked modelling—masking 75% of signal tokens and 50% of associated text tokens and reconstructing them—to learn transferable representations across cardiac devices, modalities and clinical tasks. | CSFM processes 10-second recordings containing any available combination of 12-lead or single-lead ECG and PPG, dividing each channel into non-overlapping 0.1-second patches before Transformer encoding. |
| **QualityFM** | QualityFM uses paired teacher–student windowed-sparse Transformers, where a high-quality ECG–PPG segment guides the representation of a nearby low-quality segment through self-distillation, supplemented by reconstruction losses on the signals' amplitude and phase spectra. | Each input contains paired ECG and PPG waveforms of 30 seconds, represented as two channels of 9,000 samples after resampling to 300 Hz; high- and low-quality segments from the same patient may be separated by up to five minutes. |
| **SNUPHY** | SNUPHY-M is a multimodal masked-autoencoder based on a ViT encoder, multiple inter-, intra- and whole-signal masking strategies, and signal-specific cross-attention decoders trained with RMSE and correlation losses to reconstruct masked ECG, PPG and ABP signals. | It processes synchronized 10-second ECG, PPG and ABP segments resampled to 100 Hz, dividing each signal into non-overlapping 0.5-second patches before masking and encoding. |
| **PaPaGei** | PaPaGei uses CNN encoders with morphology-aware contrastive objectives that group PPG signals according to subject identity or physiological waveform characteristics such as blood-volume change and dicrotic-notch morphology. | The model processes 10-second PPG windows resampled to 125 Hz and was pretrained on approximately 20.8 million segments from MIMIC-III, VitalDB and MESA. |