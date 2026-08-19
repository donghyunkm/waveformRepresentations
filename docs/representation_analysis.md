# Representation Analysis

## PCA of JEPA Token Embeddings

**Date:** 2026-08-12  
**Encoder:** Native JEPA (epoch 13, val_loss=0.215)  
**Data:** 1000 windows (50 windows × 20 test patients, seed=42)  
**Embedding shape:** `[1000, 3, 512, 1800]` — 3 channels, 512 d_model, 1800 patches (patch_size=125)  
**Script:** `jobs/probing/pca_embeddings.py`

### Method

PCA was fit on 500,000 randomly subsampled tokens from the full set of 5.4M
tokens (1000 windows × 3 channels × 1800 patches). Each token is a 512-dim
vector from the encoder's output.

### Results: All Channels Combined

| Variance Threshold | Dimensions Needed |
|-------------------|-------------------|
| 50% | 29 |
| 75% | 70 |
| 80% | 84 |
| 85% | 100 |
| **90%** | **121** |
| 95% | 158 |
| 99% | 270 |

**Key finding:** 121 out of 512 dimensions explain 90% of the variance. The
encoder uses its capacity broadly — no single dominant direction (PC1 = 4.85%).

### Top 20 Principal Components

| PC | Variance Explained | Cumulative |
|----|-------------------|------------|
| 1 | 4.85% | 4.85% |
| 2 | 4.48% | 9.34% |
| 3 | 3.55% | 12.88% |
| 4 | 3.21% | 16.09% |
| 5 | 2.82% | 18.91% |
| 6 | 2.30% | 21.22% |
| 7 | 2.01% | 23.22% |
| 8 | 1.95% | 25.17% |
| 9 | 1.81% | 26.98% |
| 10 | 1.65% | 28.63% |
| 11 | 1.56% | 30.19% |
| 12 | 1.47% | 31.67% |
| 13 | 1.43% | 33.09% |
| 14 | 1.37% | 34.46% |
| 15 | 1.34% | 35.80% |
| 16 | 1.32% | 37.12% |
| 17 | 1.23% | 38.35% |
| 18 | 1.20% | 39.54% |
| 19 | 1.16% | 40.70% |
| 20 | 1.14% | 41.84% |

The variance spectrum decays gradually without a sharp elbow, indicating a
distributed representation rather than a low-rank one.

### Per-Channel Analysis

| Channel | 90% Variance | 95% Variance |
|---------|-------------|-------------|
| ABP | 87 dims | 117 dims |
| II (ECG) | 89 dims | 118 dims |
| PLETH | 87 dims | 118 dims |

All three channels have remarkably similar intrinsic dimensionality (~87–89 dims
for 90%), suggesting the encoder learns representations of comparable complexity
for each physiological signal despite their different morphologies.

### Interpretation

- The JEPA encoder distributes information broadly across its 512 dimensions,
  with no major redundancy.
- ~24% of dimensions (121/512) suffice for 90% reconstruction — the remaining
  dimensions carry finer-grained or rarer variations.
- The similar per-channel dimensionality suggests shared architectural patterns
  in how the encoder represents each signal type, possibly driven by the shared
  transformer blocks.
- For downstream tasks that don't require full fidelity (e.g., clustering,
  coarse classification), a 120–160 dim PCA projection would retain most
  information while reducing computation significantly.

### Saved Embeddings

Full token-sequence embeddings saved at:
```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/
  patient_token_sequences/token_embeddings_20patients_50windows_seed42.npz
```

Contents:
- `embeddings`: float16 `[1000, 3, 512, 1800]`
- `subject_id`: patient ID (join key for icuDataExtraction)
- `unique_identifier`: ICU stay ID
- `start_idx` / `end_idx`: sample indices in zarr container
- `file_path`: zarr container path
- `test_sample_idx`: row index into test_samples.csv.gz and window_hemo_clusters.npz
- `hemo_cluster`: hemodynamic cluster (0–6, or -1 if unmatched)
- `hypotension_label`: 5-min-ahead hypotension event (0/1)


## Temporal PCA of JEPA Embeddings

**Date:** 2026-08-12  
**Script:** `jobs/probing/pca_temporal.py`

The initial PCA (above) analyzes the **feature dimension** (512-dim): how many
of the 512 embedding dimensions are needed. This section analyzes the
**temporal dimension** (1800 patches): how many temporal modes/basis functions
are needed to describe how the embedding evolves over the 30-minute window.

### Approach 1: Global Temporal PCA

**Question:** How many shared temporal basis functions explain the variation
across all windows?

Each sample is one (window, channel, feature_dim) viewed as a 1800-dim temporal
signal. 100k subsampled from 1.5M total vectors.

| Threshold | Temporal Modes Needed |
|-----------|-----------------------|
| 50% | 2 |
| 75% | 32 |
| 80% | 88 |
| 85% | 254 |
| 90% | 501 |

Top 10 temporal components:

| TC | Variance Explained | Cumulative |
|----|-------------------|------------|
| 1 | **49.61%** | 49.61% |
| 2 | 4.32% | 53.93% |
| 3 | 3.18% | 57.11% |
| 4 | 2.28% | 59.39% |
| 5 | 1.78% | 61.17% |
| 6 | 1.64% | 62.81% |
| 7 | 1.27% | 64.08% |
| 8 | 1.09% | 65.16% |
| 9 | 0.94% | 66.10% |
| 10 | 0.84% | 66.94% |

**Key finding:** TC1 alone explains ~50% of temporal variance — this is likely
the DC/mean component (constant embedding across time). After that, variance is
distributed across hundreds of modes, indicating rich temporal structure with no
simple low-rank temporal pattern.

Per channel (90% threshold): ABP=501, II=356, PLETH=501 modes. ECG (lead II)
is slightly more temporally compressible, possibly because heartbeat regularity
creates more predictable temporal patterns.

### Approach 2: Per-Window Temporal PCA

**Question:** How many temporal modes does each individual window use?

For each (window, channel) pair: treat the `[512, 1800]` matrix as 512 samples
of 1800-dim temporal vectors and fit PCA. This gives 3000 separate PCA fits
(1000 windows × 3 channels). Each fit yields a "number of modes needed for X%
variance" answer for that specific window-channel. The tables below report
summary statistics (median, mean, IQR) over these 3000 fits (or 1000 per
channel for the per-channel breakdown). This reveals how temporally complex
each individual window is, independent of cross-window variation.

| Threshold | Median | Mean | IQR |
|-----------|--------|------|-----|
| 50% | 2 | 1.7 | [1, 2] |
| 90% | **13** | 13.8 | [10, 17] |
| 95% | 22 | 23.1 | [16, 29] |

Per channel (90% variance):

| Channel | Median | Mean | IQR |
|---------|--------|------|-----|
| ABP | 11 | 11.2 | [8, 14] |
| II (ECG) | 17 | 17.4 | [13, 21] |
| PLETH | 12 | 12.8 | [10, 16] |

### Why the Two Approaches Differ So Much (501 vs 13)

The difference reflects **inter-window diversity** vs **intra-window complexity**.

Global Temporal PCA pools temporal vectors from all windows together. It asks:
"What temporal patterns exist across the entire dataset?" This mixes Patient A's
slowly declining pressure trajectory with Patient B's stable trajectory with
Patient C's oscillating trajectory, etc. These are all *different* temporal
patterns, so the global basis needs hundreds of modes to represent them all. The
501 modes for 90% reflects the space of all possible 30-minute trajectories this
encoder can produce.

Per-Window Temporal PCA fits PCA separately on each (window, channel). It asks:
"Within *this specific* 30-minute window, how complex is the temporal
evolution?" Any single window has smooth, coherent dynamics — a patient's
hemodynamics don't change in 512 independent ways over 30 minutes — so only ~13
modes suffice.

**Analogy:** Consider asking "how many Fourier modes describe human speech?"
Globally across all speakers and utterances: thousands (every person says
different things). Within a single 30-second utterance: far fewer (one voice,
one sentence, smooth prosody). The 501 vs 13 gap is the same phenomenon — the
diversity of the population vs the complexity of an individual trajectory.

### Interpretation

- **Globally, temporal variation is extremely high-dimensional** (501 modes for
  90%). This is because different windows have different temporal patterns — the
  global pool mixes patient-specific dynamics, creating apparent high complexity.

- **Within a single window, only ~13 temporal modes are needed for 90%.** Each
  individual 30-minute trajectory is much simpler than the global pool. The
  encoder produces smooth temporal dynamics that can be compressed to ~13
  temporal basis functions.

- **The first temporal component captures ~50% globally** — this is the
  time-averaged embedding (the "DC component"). The temporal *variation* around
  this mean requires many more modes to capture.

- **ECG (lead II) uses more temporal modes per window (17 vs 11–12)** than ABP
  or PLETH. This likely reflects the more complex morphological variation in
  ECG (P-QRS-T wave evolution, HRV, rhythm changes) compared to the smoother
  hemodynamic signals.

- **Contrast with feature-dim PCA:** The 512-dim feature space needs 121
  components for 90% variance. The temporal space within a single window needs
  only 13. This means the encoder's temporal dynamics are much lower-rank than
  its feature-space utilization — the embedding moves through a
  low-dimensional temporal trajectory within the high-dimensional feature space.

## K-Means Clustering of Full Token Embeddings

**Date:** 2026-08-12  
**Script:** `jobs/probing/cluster_embeddings_kmeans.py`

### Method

1. Temporal subsample: 1800 → 20 evenly spaced patches (each represents ~1.5 min)
2. PCA on 512-dim token space → 121 dims (90.2% variance explained)
3. Flatten per window: `[3 channels × 20 patches × 121 pca] = 7,260` dims
4. Standardize (no second PCA — cluster directly on temporal-aware vectors)
5. K-Means sweep k=2–15 (n_init=10, seed=42)

### K-Means Sweep Results

| k | Silhouette | ARI vs Hemo | NMI vs Hemo |
|---|-----------|-------------|-------------|
| 2 | 0.0271 | 0.0076 | 0.0089 |
| 3 | 0.0395 | 0.0152 | 0.0339 |
| 4 | 0.0453 | 0.0210 | 0.0559 |
| 5 | 0.0508 | 0.0303 | 0.0646 |
| 6 | 0.0550 | 0.0282 | 0.0659 |
| **7** | **0.0621** | **0.0352** | **0.0970** |
| 8 | 0.0656 | 0.0303 | 0.0821 |
| 9 | 0.0735 | 0.0192 | 0.1029 |
| 10 | 0.0700 | 0.0255 | 0.0883 |
| 11 | 0.0762 | 0.0302 | 0.1293 |
| 12 | 0.0746 | 0.0330 | 0.1274 |
| 13 | 0.0713 | 0.0242 | 0.1261 |
| 14 | 0.0763 | 0.0247 | 0.1269 |
| 15 | 0.0755 | 0.0297 | 0.1482 |

Silhouette scores are low across all k (0.03–0.08) with no clear elbow,
indicating no well-separated cluster structure in the temporal embedding space.

### Detailed Analysis at k=7

**vs Hemo clusters:** ARI=0.035, NMI=0.097  
**vs Patient ID:** Homogeneity=0.475, Completeness=0.770  
**vs Hypotension label:** ARI=0.021, NMI=0.031

Hypotension prevalence per cluster:

| Cluster | Windows | Hypo Prevalence |
|---------|---------|-----------------|
| 0 | 153 | 21.6% |
| 1 | 255 | 9.0% |
| 2 | 108 | 20.4% |
| 3 | 198 | 21.7% |
| 4 | 144 | 41.7% |
| 5 | 54 | 3.7% |
| 6 | 88 | 25.0% |

### Interpretation

- **The temporal embedding structure is dominated by patient identity.** Clusters
  group windows from the same patient together (homogeneity=0.475) rather than
  recovering hemodynamic states (ARI=0.035 vs hemo clusters).

- **Poor alignment with hemo clusters.** The hemodynamic clusters were derived
  from mean-pooled ICU features that abstract away patient morphology. The full
  temporal embeddings retain patient-specific waveform shape, which overwhelms
  the hemodynamic state signal.

- **Meaningful clinical signal despite patient dominance.** Cluster 4 (41.7%
  hypo) vs Cluster 5 (3.7%) shows an 11× difference in hypotension prevalence.
  The embeddings encode clinical risk even though clusters primarily separate
  patients.

- **No natural cluster count.** Low silhouette scores with no elbow confirm a
  continuous manifold rather than discrete clusters. The structure is dominated
  by inter-patient variation (20 patients → clusters tend toward patient count).

### Homogeneity Analysis Across k

Full sweep of K-Means k=2–50, measuring how clusters align with patient identity,
hemodynamic state, and hypotension labels.

**Script:** `jobs/probing/cluster_homogeneity_sweep.py`

#### Metric Definitions

- **Silhouette (Sil):** Measures geometric cluster quality. For each point,
  computes `(b - a) / max(a, b)` where `a` = mean distance to same-cluster
  points and `b` = mean distance to nearest other cluster. Ranges [-1, 1];
  higher = tighter, better-separated clusters. Values near 0 indicate
  overlapping clusters with no clear geometric structure.

- **Homogeneity (H):** Given ground-truth classes and predicted clusters, H
  measures whether each cluster contains only members of a single class.
  H=1.0 means every cluster is "pure" (all its members share the same class
  label). H=0 means clusters are maximally mixed. Formally:
  `H = 1 - H(C|K) / H(C)` where H(C|K) is the conditional entropy of the
  class distribution given the cluster assignment.

- **Completeness (C):** The dual of homogeneity — measures whether all members
  of a given class are assigned to the same cluster. C=1.0 means every class
  is fully contained within a single cluster. C=0 means class members are
  scattered across all clusters. Formally: `C = 1 - H(K|C) / H(K)`.

- **V-measure (V):** Harmonic mean of homogeneity and completeness:
  `V = 2 * H * C / (H + C)`. Analogous to F1-score but for clustering.
  Summarizes the tradeoff: you can get high H by having many small pure clusters
  (but low C), or high C by having few large clusters (but low H).

- **Adjusted Rand Index (ARI):** Measures agreement between two partitions,
  corrected for chance. Counts pairs of points that are (same-cluster,
  same-class) or (different-cluster, different-class), adjusted so random
  clustering scores ~0. ARI=1.0 means perfect agreement; ARI≈0 means no
  better than random; ARI<0 means anti-correlated.

**How to read the table:** For each reference label (patient, hemo cluster,
hypotension), high H means clusters are pure w.r.t. that label, high C means
that label's groups are not split across clusters, and high ARI means the
clustering recovers that partition specifically. Since H always increases with
k (more clusters → smaller → purer), ARI and C are more informative for
assessing whether a partition is truly recovered.

| k | Sil | H(patient) | C(patient) | V(patient) | H(hemo) | C(hemo) | H(hypo) | C(hypo) | ARI(pat) | ARI(hemo) |
|---|-----|-----------|-----------|-----------|---------|---------|---------|---------|----------|-----------|
| 2 | 0.027 | 0.140 | 0.787 | 0.238 | 0.019 | 0.031 | 0.000 | 0.000 | 0.043 | -0.090 |
| 3 | 0.040 | 0.307 | 0.858 | 0.452 | 0.035 | 0.028 | 0.010 | 0.005 | 0.149 | -0.022 |
| 4 | 0.045 | 0.350 | 0.784 | 0.484 | 0.072 | 0.047 | 0.048 | 0.018 | 0.180 | 0.028 |
| 5 | 0.051 | 0.407 | 0.792 | 0.537 | 0.079 | 0.045 | 0.048 | 0.016 | 0.217 | 0.029 |
| 7 | 0.062 | 0.475 | 0.770 | 0.587 | 0.091 | 0.043 | 0.071 | 0.020 | 0.273 | 0.002 |
| 10 | 0.070 | 0.567 | 0.766 | 0.652 | 0.118 | 0.046 | 0.181 | 0.042 | 0.365 | -0.006 |
| 15 | 0.076 | 0.713 | 0.805 | 0.756 | 0.173 | 0.056 | 0.325 | 0.062 | 0.544 | 0.001 |
| **20** | **0.078** | **0.776** | **0.794** | **0.785** | 0.229 | 0.068 | 0.330 | 0.057 | **0.608** | 0.003 |
| 25 | 0.087 | 0.810 | 0.778 | 0.793 | 0.279 | 0.078 | 0.493 | 0.080 | 0.622 | 0.007 |
| 30 | 0.079 | 0.830 | 0.756 | 0.791 | 0.307 | 0.081 | 0.523 | 0.081 | 0.612 | 0.008 |
| 40 | 0.062 | 0.868 | 0.732 | 0.794 | 0.324 | 0.079 | 0.553 | 0.079 | 0.575 | 0.007 |
| 50 | 0.058 | 0.883 | 0.701 | 0.781 | 0.349 | 0.080 | 0.694 | 0.093 | 0.508 | 0.005 |

**Key observations:**

1. **Patient identity dominates at all k.** Homogeneity w.r.t. patient climbs
   steadily (0.14 → 0.88) as k increases, while completeness stays high
   (0.70–0.86). At k=20 (matching the 20 patients), ARI with patient ID hits
   0.608 — the clusters are essentially recovering individual patients.

2. **Hemodynamic state is barely captured.** ARI with hemo clusters stays near 0
   at all k (-0.09 to +0.008). Hemo homogeneity grows slowly but this is a
   trivial artifact of increasing k. The critical metric — completeness for hemo
   — stays at 0.03–0.08, meaning members of each hemo cluster are scattered
   across many K-Means clusters rather than grouping together.

3. **Hypotension signal appears only at high k (as a byproduct of patient
   separation).** Hypotension homogeneity jumps from ~0.05 (k≤7) to 0.49–0.69
   (k≥25), but completeness remains <0.1. This means some clusters become nearly
   pure hypotension at high k, but most hypotensive windows remain mixed with
   normotensive ones. Since hypotension is concentrated in specific patients,
   separating patients naturally increases hypo homogeneity.

4. **The "natural" cluster count matches patients, not physiology.** The
   V-measure for patient ID peaks and plateaus around k=20–30. There is no k
   where hemodynamic state alignment overtakes patient identity. The embedding
   space is organized primarily by "who" rather than "what state."

5. **Silhouette peaks at k=25 (0.087) then declines.** The best geometric
   separation occurs when clusters approximate patient-level groupings with
   some within-patient subdivision. Beyond k=30, clusters fragment without
   improving separation.

### JEPA d64 Homogeneity Sweep — Running

Same methodology as above applied to the d64 encoder (d_model=64 vs 512).
Measures whether the narrower bottleneck changes the balance between patient
identity and hemodynamic state in the embedding structure.

- **Script**: `jobs/probing/cluster_homogeneity_sweep_d64.py`
- **Sbatch**: `jobs/probing/slurm/cluster_homogeneity_sweep_d64_gl40s_short.sbatch`
- **Job**: 26487661, running on `gl40s_short` (node gl40s-8006, started 2026-08-16)
- **Output**: `embeddings/patient_token_sequences_d64/`
  - `token_embeddings_20patients_50windows_seed42.npz` — full token embeddings
  - `homogeneity_sweep_d64.csv` — results table
- **PCA adaptation**: min(121, d_model=64) = 64 components (vs 121 for d512)
- **Status**: Running; currently in setup/embedding extraction phase (~17 min elapsed).
- **Hypothesis**: With only 64 dimensions, the encoder may be forced to allocate
  more capacity to clinically-relevant features vs patient fingerprint, potentially
  shifting the homogeneity balance toward hemodynamic state.

## Single-Patient Clustering: p072908

**Data:** 1,000 sampled windows from patient `p072908`, spanning 10 ICU stays.
The embeddings have shape `[1000, 3, 512, 1800]`; K-Means used 20 temporal
positions, 121 PCA feature components, and standardized flattened vectors.

The table reports every reference variable available for this patient: ICU-stay
identity, hypotension label, and within-stay temporal tercile. `H`, `C`, and `V`
are homogeneity, completeness, and V-measure; `ARI` is adjusted Rand index.

| k | Sil | H(stay) | C(stay) | V(stay) | ARI(stay) | H(hypo) | C(hypo) | V(hypo) | H(time) | C(time) | V(time) | ARI(time) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.057 | 0.206 | 0.903 | 0.335 | 0.110 | — | — | — | 0.000 | 0.000 | 0.000 | -0.001 |
| 3 | 0.053 | 0.284 | 0.603 | 0.386 | 0.228 | — | — | — | 0.007 | 0.007 | 0.007 | 0.007 |
| 4 | 0.050 | 0.304 | 0.496 | 0.377 | 0.229 | — | — | — | 0.015 | 0.012 | 0.014 | 0.010 |
| **5** | **0.046** | **0.323** | **0.457** | **0.378** | **0.247** | — | — | — | 0.012 | 0.009 | 0.010 | 0.005 |
| 7 | 0.038 | 0.340 | 0.403 | 0.369 | 0.196 | — | — | — | 0.027 | 0.015 | 0.019 | 0.013 |
| 10 | 0.033 | 0.359 | 0.361 | 0.360 | 0.172 | — | — | — | 0.040 | 0.020 | 0.027 | 0.011 |
| 15 | 0.031 | 0.432 | 0.374 | 0.401 | 0.190 | — | — | — | 0.052 | 0.022 | 0.031 | 0.009 |
| 20 | 0.030 | 0.479 | 0.366 | 0.415 | 0.194 | — | — | — | 0.068 | 0.025 | 0.037 | 0.009 |
| 25 | 0.023 | 0.498 | 0.357 | 0.415 | 0.161 | — | — | — | 0.066 | 0.023 | 0.034 | 0.006 |
| 30 | 0.024 | 0.573 | 0.384 | 0.460 | 0.180 | — | — | — | 0.090 | 0.029 | 0.044 | 0.007 |
| 40 | 0.026 | 0.573 | 0.363 | 0.444 | 0.153 | — | — | — | 0.113 | 0.035 | 0.053 | 0.007 |
| 50 | 0.023 | 0.595 | 0.353 | 0.443 | 0.125 | — | — | — | 0.125 | 0.036 | 0.056 | 0.006 |

**Interpretation:** The strongest stay-identity agreement is modest (ARI=0.247
at k=5), and all silhouette scores are low, indicating overlapping clusters.
Temporal-tercile alignment is effectively chance (ARI≤0.013). Hypotension metrics
are undefined because all 1,000 sampled windows are negative; this patient is a
stay/time control, not a test of clinical-state separation. The raw results are
saved in `embeddings/single_patient/clustering_results_p072908_1000windows_seed42.csv`.



### Implications

For downstream analysis requiring state-level (rather than patient-level)
clustering from temporal embeddings:
- Consider **per-patient normalization** or **patient-identity removal** (e.g.,
  regress out patient means) before clustering.
- Alternatively, cluster **within** each patient's trajectory to find temporal
  state transitions.
- The mean-pooled approach used in the existing hemo clustering pipeline is
  more appropriate for cross-patient state discovery.

## JEPA d64 Homogeneity Sweep

**Date:** 2026-08-16  
**Job:** 26487661 (`physiojepa-homog-d64-sweep`, gl40s_short, L40S)  
**Script:** `jobs/probing/cluster_homogeneity_sweep_d64.py`  
**Checkpoint:** JEPA d64 (epoch 40, val_loss=0.18978)

### Setup

- 20 patients × 50 windows = 1,000 embeddings
- Embedding shape: `(1000, 3, 64, 1800)` — 3 channels, 64-dim, 1800 patches
- PCA: 64 → 64 components (explained variance = 1.0), flattened to 3,840 features
- Hemo cluster distribution: 798 unclustered (−1), 202 labeled across 7 clusters
- Hypotension prevalence: 20.5%

### Results

| k | Sil | H(pat) | C(pat) | V(pat) | H(hemo) | C(hemo) | V(hemo) | H(hypo) | C(hypo) | V(hypo) | ARI(pat) | ARI(hemo) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.042 | 0.113 | 0.488 | 0.183 | 0.010 | 0.012 | 0.011 | 0.012 | 0.009 | 0.010 | 0.052 | 0.009 |
| 3 | 0.040 | 0.176 | 0.486 | 0.259 | 0.019 | 0.015 | 0.017 | 0.096 | 0.045 | 0.061 | 0.087 | 0.022 |
| 4 | 0.044 | 0.236 | 0.521 | 0.325 | 0.027 | 0.017 | 0.021 | 0.079 | 0.030 | 0.043 | 0.118 | -0.008 |
| 5 | 0.046 | 0.290 | 0.577 | 0.386 | 0.038 | 0.022 | 0.028 | 0.088 | 0.030 | 0.044 | 0.148 | -0.014 |
| 7 | 0.040 | 0.385 | 0.639 | 0.480 | 0.046 | 0.022 | 0.030 | 0.115 | 0.032 | 0.050 | 0.212 | -0.017 |
| 10 | 0.042 | 0.471 | 0.651 | 0.547 | 0.113 | 0.045 | 0.065 | 0.168 | 0.039 | 0.064 | 0.275 | -0.005 |
| 15 | 0.031 | 0.547 | 0.637 | 0.589 | 0.147 | 0.049 | 0.074 | 0.289 | 0.057 | 0.095 | 0.339 | 0.002 |
| 20 | 0.040 | 0.627 | 0.650 | 0.638 | 0.183 | 0.055 | 0.085 | 0.289 | 0.051 | 0.086 | 0.420 | 0.000 |
| 25 | 0.037 | 0.712 | 0.686 | 0.699 | 0.235 | 0.065 | 0.102 | 0.418 | 0.068 | 0.117 | 0.464 | 0.002 |
| 30 | 0.041 | 0.703 | 0.644 | 0.672 | 0.236 | 0.063 | 0.099 | 0.356 | 0.055 | 0.096 | 0.421 | 0.000 |
| 40 | 0.037 | 0.717 | 0.609 | 0.659 | 0.282 | 0.069 | 0.111 | 0.421 | 0.061 | 0.106 | 0.387 | 0.002 |
| 50 | 0.028 | 0.752 | 0.612 | 0.675 | 0.337 | 0.079 | 0.128 | 0.472 | 0.065 | 0.114 | 0.370 | 0.004 |

### Interpretation

The d64 embeddings show the same dominant patient-identity structure as d512:
- **Patient ARI** peaks at 0.464 (k=25), comparable to d512 results
- **Hemo cluster ARI** is effectively zero across all k — no hemodynamic state
  separation in clustering space
- **Hypotension homogeneity** reaches 0.47 at k=50, but completeness stays very
  low (0.065), meaning hypotensive windows are concentrated in a few clusters
  but not cleanly separated
- Low silhouette scores (0.028–0.046) confirm overlapping, poorly separated clusters

The 8× compression from d512→d64 preserves the patient fingerprint structure
but does not improve (or degrade) clinical state separation in unsupervised
clustering. Results saved to
`embeddings/patient_token_sequences_d64/homogeneity_sweep_d64.csv`.

## Intrinsic Dimension Estimation

**Date:** 2026-08-12  
**Script:** `jobs/probing/intrinsic_dimension.py`

### Methods

- **PCA Participation Ratio (PR):** `(Σλ_i)² / Σ(λ_i²)` — effective number of
  linear dimensions, accounting for eigenvalue distribution shape. A uniform
  spectrum (all eigenvalues equal) gives PR = d (full rank); a single dominant
  eigenvalue gives PR ≈ 1. This is a linear measure — it overestimates
  intrinsic dimension when data lies on a curved manifold because PCA needs
  multiple linear directions to approximate a curve.

- **Two-NN** (Facco et al. 2017): For each point, compute the ratio μ = r₂/r₁
  of distances to the 2nd and 1st nearest neighbors. On a d-dimensional
  manifold, these ratios follow a Pareto distribution with shape parameter d.
  The MLE estimate is: `d = N / Σ log(μᵢ)`. Key advantages: no hyperparameters
  (uses only 2 neighbors), model-free, captures nonlinear manifold structure,
  and is robust to curvature because it operates at the smallest local scale.

- **Levina–Bickel MLE** (2005): For each point x, estimates local dimension
  from k nearest neighbor distances T₁(x) < T₂(x) < ... < Tₖ(x):
  ```
  m̂ₖ(x) = (1/(k-1)) Σⱼ₌₁ᵏ⁻¹ log(Tₖ(x) / Tⱼ(x))
  d̂(x) = 1 / m̂ₖ(x)
  ```
  This gives a per-point dimension estimate, revealing heterogeneity across the
  manifold (some regions may be lower-dimensional than others). The global
  estimate is the average over all points. Unlike Two-NN, it uses k neighbors,
  making it tunable: small k captures very local structure, large k averages
  over larger neighborhoods. The estimate should stabilize as k increases; if
  it decreases with k, this indicates the manifold has varying local dimension
  or the data has multi-scale structure.

All computed on 20k subsampled points.

PCA PR was fit on 50k subsampled tokens. Two-NN and Levina–Bickel used 20k
subsampled tokens. Window-level analyses used all 1000 windows (no subsampling).

**Neighbor definition:** For both Two-NN and Levina–Bickel, "nearest neighbors"
are defined by Euclidean (L2) distance. What constitutes a "point" and a
"neighbor" differs across the three analysis levels:

- **Token-level:** Each point is a single 512-dim vector — one patch position
  from one channel of one window. Its neighbors are other tokens (from any
  patient, any channel, any time position) that are closest in 512-dim L2
  distance. We subsample 20k tokens from the full 5.4M and measure the local
  geometry of that token cloud. This answers: "How many degrees of freedom does
  a single patch embedding have?"

- **Window-level (mean-pooled):** Each point is one 30-min window, represented
  as a 1,536-dim vector (mean across all 1800 patches, concatenated across 3
  channels: 3 × 512 = 1,536). Its neighbors are other windows closest in
  1,536-dim L2 distance. We use all 1000 windows. This answers: "How many
  factors distinguish one window from another when temporal information is
  discarded?"

- **Window-level (temporal-aware):** Each point is one 30-min window, represented
  as a 7,260-dim vector (20 evenly spaced patches × 121 PCA dims × 3 channels,
  flattened and standardized). Its neighbors are other windows closest in
  7,260-dim L2 distance. We use all 1000 windows. This answers: "How many
  factors distinguish one window from another when temporal trajectory is
  preserved?" Two windows with the same average embedding but different temporal
  evolution (e.g., pressure dropping early vs. late) will be far apart in this
  space.

### Results

#### Token-level

Each point is a single 512-dim embedding at one patch position (the full saved
representation, no pooling). 5.4M vectors total (1000 windows × 3 channels ×
1800 patches); subsampled to 20–50k for estimation.

**Summary (all channels combined):**

| Method | Estimate |
|--------|----------|
| PCA Participation Ratio | 70.8 |
| Two-NN | **6.8** |
| Levina–Bickel (k=10, median) | **8.2** |

**Per channel (Two-NN / LB median at k=10):**

| Channel | PCA PR | Two-NN | LB Median | LB IQR |
|---------|--------|--------|-----------|--------|
| ABP | 43.7 | 6.0 | 8.4 | [5.9, 12.0] |
| II (ECG) | 53.2 | 6.0 | 7.2 | [5.1, 10.4] |
| PLETH | 46.6 | 6.5 | 8.3 | [6.0, 11.6] |

**Levina–Bickel sensitivity to k (all channels):**

How the estimate changes with neighborhood size. Small k = very local geometry;
large k = larger-scale structure. Stable estimates across k indicate consistent
dimensionality at all scales.

| k | Mean | Median | IQR |
|---|------|--------|-----|
| 5 | 11.2 | 8.6 | [5.5, 13.6] |
| 10 | 9.6 | 8.2 | [5.8, 11.9] |
| 20 | 9.2 | 8.3 | [6.1, 11.3] |
| 50 | 9.2 | 8.5 | [6.5, 11.2] |

Stabilizes around k=10–50 (median ~8.3). Wide IQR reveals local dimension
heterogeneity — some manifold regions are lower-dimensional (patient-specific
clusters) while others are higher (transition states).

#### Window-level (mean-pooled)

Each point is one 30-min window, mean-pooled across all 1800 patches (one
512-dim vector per channel, concatenated: 3 × 512 = 1,536 dims). All 1000
windows used.

| Level | PCA PR | Two-NN | LB Mean | LB Median | LB IQR |
|-------|--------|--------|---------|-----------|--------|
| All channels (1,536d) | 16.7 | **2.4** | 4.5 | **3.1** | [1.9, 5.2] |
| ABP (512d) | 15.7 | 2.2 | 4.1 | 3.0 | [1.9, 5.2] |
| II (512d) | 12.7 | 3.0 | 4.0 | 2.8 | [1.9, 4.4] |
| PLETH (512d) | 14.2 | 2.6 | 4.0 | 2.9 | [1.8, 4.8] |

#### Window-level (temporal-aware)

Each point is one 30-min window with temporal structure preserved: 20 evenly
spaced patches (one every ~1.5 min) × 121 PCA dims × 3 channels = 7,260 dims,
standardized. All 1000 windows used.

| Method | Estimate |
|--------|----------|
| PCA Participation Ratio | 111.2 |
| Two-NN | **32.5** |
| Levina–Bickel (k=10, median) | **26.0** |

**Levina–Bickel sensitivity to k:**

Decreasing estimates with larger k indicate the manifold looks higher-dimensional
locally (fine-grained temporal dynamics) but lower-dimensional globally (simpler
large-scale patient/state structure).

| k | Mean | Median | IQR |
|---|------|--------|-----|
| 5 | 39.6 | 31.1 | [19.7, 48.0] |
| 10 | 29.9 | 26.0 | [17.6, 37.8] |
| 20 | 24.4 | 21.2 | [15.3, 29.5] |
| 50 | 18.4 | 15.6 | [9.1, 24.3] |

**Sensitivity to temporal resolution:**

How the intrinsic dimension changes with the number of evenly-spaced time points
included. More patches = finer temporal detail. Plateau indicates the scale at
which the encoder's temporal information becomes smooth (no new degrees of
freedom from finer sampling).

| Patches | Temporal spacing | Total dims | Two-NN | LB median |
|---------|-----------------|-----------|--------|-----------|
| 5 | ~6 min | 1,815 | 22.8 | 20.2 |
| 10 | ~3 min | 3,630 | 26.5 | 22.4 |
| 20 | ~1.5 min | 7,260 | 32.5 | 26.0 |
| 50 | ~36 sec | 18,150 | 36.0 | 27.2 |
| 100 | ~18 sec | 36,300 | 36.8 | 27.7 |

Estimates stabilize around 50 patches (~36 sec spacing), indicating that the
encoder's temporal information is smooth below that scale — finer sampling adds
no new degrees of freedom.

### Interpretation

- **The token manifold is ~7–9 dimensional** (Two-NN: 6.8, Levina–Bickel median:
  8.2–8.5) despite occupying ~71 linear directions (PCA PR). This 8–10× gap
  indicates a strongly nonlinear/curved manifold embedded in 512-dim space.

- **Mean-pooled windows collapse to ~2.4–3.1 dimensions.** Averaging across 1800
  patches destroys most temporal variation, leaving an extremely low-dimensional
  manifold dominated by patient identity (20 patients ≈ 2–3 effective dims).

- **Temporal-aware windows have ~27–33 intrinsic dimensions.** Preserving
  temporal ordering reveals ~25–30 additional degrees of freedom encoding how
  the physiological state evolves over the 30-minute window. The representation
  decomposes roughly as:
  - ~3 dims of "which patient/state" (mean-pooled)
  - ~25 dims of "how the state evolves over 30 minutes" (temporal structure)

- **Per-channel intrinsic dimension is remarkably consistent** (~6.0–8.4 across
  ABP, ECG, PLETH). The encoder learns representations of similar geometric
  complexity for all three signal types.

- **The ~7–9D token manifold likely encodes:**
  - Patient identity (~2–3 dims, from window-level analysis)
  - Temporal position within the window (~1–2 dims)
  - Physiological state / hemodynamic variation (~2–3 dims)
  - Signal morphology / local waveform shape (~1–2 dims)

### Implications

- **For compression/projection:** Nonlinear methods (UMAP, autoencoders) should
  target ~7–10 dims rather than PCA's 121. Linear projections cannot faithfully
  represent this manifold at low dimension.
- **For clustering:** The low intrinsic dimension explains why K-Means in high-dim
  PCA space performs poorly — the data lives on a thin curved surface that
  Euclidean K-Means cannot partition well.
- **For interpretability:** Only ~7 independent factors of variation govern the
  token representations. Disentangling these (patient, time, state) is a
  tractable research direction.
