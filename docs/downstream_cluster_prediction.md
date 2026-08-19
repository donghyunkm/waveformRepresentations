# Downstream Hemodynamic Cluster Prediction

## Overview

Attentive probe that predicts the 7-class hemodynamic cluster from the full token
sequence of frozen self-supervised encoders. Tests whether the attentive head can
decode temporal hemodynamic state cross-patient.

---

## Task Definition

**Label**: Which of 7 hemodynamic clusters is this 30-min window in?

Clusters are defined by icuDataExtraction's pipeline:
1. Compute 19 physiological features over 109 sub-windows (120s each, 10s stride)
   within a 20-min context
2. Compute 7 focused pairwise Pearson correlations between feature trajectories
   (temporal dynamics, not static values)
3. KMeans (k=7) on the 7 correlation features

### Clinical Cluster Meanings

| Cluster | Name | Description |
|---------|------|-------------|
| C0 | Failing Vasoconstriction | |
| C1 | Hemodynamic Quiescence | |
| C2 | High-Output Compensated | |
| C3 | Normal Baroreflex | Largest cluster (~25%) |
| C4 | Tachycardia + Vasoconstriction | |
| C5 | Catecholamine-Driven | Smallest cluster (~4%) |
| C6 | PPG Dissociation | |

### Why This Matters

These clusters capture *temporal correlation dynamics* between physiological signals
— not static vital sign values. A patient's cluster reflects how their cardiovascular
system is *behaving* (e.g., whether ABP area correlates with shock index over time),
which is precisely the kind of temporal information that mean-pooled embeddings discard.

---

## Methodology

- **Encoders**: JEPA (epoch 13) and PatchTST (epoch 3) — same as hypotension probes
- **Architecture**: Frozen encoder → AttentiveClassifier (4 heads, depth=1, mlp_ratio=4)
  with `num_classes=7` and CrossEntropyLoss
- **Input**: Full token sequence (3 channels × n_patches × 512 dims)
- **Split**: Same patient-level train/val/test as hypotension
  (`hypotension_subject_split_fixed_v1.csv`)
- **Class weighting**: Inverse-frequency weights in both sampler and loss (handles
  C5's 4% prevalence)
- **Mixup**: Enabled (α=0.2, 7-class soft targets via manual log-softmax CE)
- **Metrics**: Macro-averaged balanced accuracy + multiclass AUROC
- **Training**: 20 epochs, OneCycle LR (max_lr=0.01), bs=8, accumulate=16
  (effective bs=128)
- **Hardware**: gpu4_short (V100, 12h wall time)

### Label Alignment

Window-level cluster labels are aligned from icuDataExtraction to PhysioJEPA samples
via timestamp matching (`align_hemo_to_split` in `train_hemo_cluster_fixed.py`):

**Source data (icuDataExtraction):**
- 666,492 windows across 1,643 patients
- Anchored every 150 seconds (2.5 min) within each ICU stay
- Each window: 19 features × 109 sub-windows → 7 pairwise temporal correlations → KMeans(k=7)
- Times stored as seconds since 2000-01-01

**Alignment procedure:**
1. Parse segment start time from Zarr filename (e.g., `p000188-2149-04-17-22-52.zarr.zip`) → POSIX timestamp
2. Compute PhysioJEPA window center: `seg_start + (start_idx + end_idx) / 2 / 125`
3. Convert to icuDataExtraction reference frame: subtract epoch offset (946,684,800 = seconds from 1970-01-01 to 2000-01-01)
4. For each PhysioJEPA window, binary-search the nearest icuDataExtraction window for the **same patient**
5. If |temporal offset| ≤ 150s → assign that cluster label; otherwise → -1 (unmatched)

**Match quality (verified):**
- Test split: 22,323 / 127,811 windows matched (17.5%)
- Temporal offset: median **38.2s**, mean 37.7s, p75=55.3s, p95=72.0s, max=149.6s
- The median 38s offset is negligible relative to the cluster's 20-minute temporal context
- Low match rate is a **coverage** issue (icuDataExtraction covers 1,643 patients; only ~131 overlap with the 256 test-split patients), not an alignment quality issue

**Positional alignment verification:**
- The per-split `.npy` label files (`{split}_hemo_clusters.npy`) are positionally aligned with the sample CSVs (`{split}_samples.csv.gz`) — row i in the CSV = label i in the .npy
- The pre-extracted embeddings (`embeddings_nfull_seed42.npz`) are also positionally aligned with the test CSV: patient IDs match 100% at every position (127,811/127,811)
- Therefore `embeddings[i]`, `hemo_labels[i]`, and `test_samples.csv.gz` row i all refer to the same PhysioJEPA window

~17% of windows get valid labels (limited by icuDataExtraction coverage of 1,643
patients and 2.5-min anchor spacing).

---

## Scripts and Configs

```
# Attentive probe (full training)
jobs/jepa/scripts/train_hemo_cluster_fixed.py
jobs/jepa/configs/train_hemo_cluster_fixed.yaml
jobs/jepa/slurm/train_hemo_cluster_fixed_gpu4.sbatch

jobs/patchtst/scripts/train_hemo_cluster_fixed.py
jobs/patchtst/configs/train_hemo_cluster_fixed.yaml
jobs/patchtst/slurm/train_hemo_cluster_fixed_gpu4.sbatch

# Mean-pooled linear probe (Logistic Regression)
jobs/probing/probe_hemo_clusters.py                          # full pipeline (Zarr + encoder + raw stats)
jobs/probing/probe_hemo_clusters_precomputed.py              # uses pre-extracted embeddings (fast)
jobs/probing/slurm/probe_hemo_clusters_gl40s_dev.sbatch      # sbatch for full pipeline
```

---

## Jobs

- `26340214` — JEPA hemo cluster probe (a100_dev) — ran 13 epochs, hit 4h wall time
- `26340215–26340218` — chain links, **cancelled** (no learning after 13 epochs)
- PatchTST hemo cluster probe — not submitted (JEPA result shows task is not feasible
  with frozen probing)

---

## Results

### JEPA Attentive Probe (frozen encoder, epoch 13)

| Epoch | Train Loss | Val Loss | Val AUROC | Val Balanced Acc |
|-------|-----------|----------|-----------|-----------------|
| 0 | 1.870 | 2.060 | 0.501 | 0.137 |
| 1 | 1.870 | 2.110 | 0.470 | 0.128 |
| 2 | 1.900 | 2.070 | 0.486 | 0.134 |
| 3 | 1.970 | 2.100 | 0.484 | 0.130 |
| 4 | 1.930 | 9.450 | 0.482 | 0.143 |
| 5 | 1.940 | 2.140 | 0.503 | 0.148 |
| 6 | 1.990 | 2.070 | **0.506** | 0.148 |
| 7 | 1.960 | 2.050 | 0.499 | 0.148 |
| 8 | 3.450 | 2.140 | 0.482 | 0.128 |
| 9 | 1.940 | 2.100 | 0.496 | 0.147 |
| 10 | 1.930 | 2.080 | 0.505 | 0.147 |
| 11 | 1.900 | 2.110 | 0.503 | 0.133 |
| 12 | 1.890 | 2.090 | 0.502 | 0.149 |

**Verdict: No learning.** After 13 epochs the model is at chance level:
- Val AUROC ≈ 0.50 (random for 7-class one-vs-rest)
- Val balanced accuracy ≈ 0.14 (random for 7 classes = 1/7 = 0.143)
- Train loss barely decreasing (1.87 → 1.89), not even overfitting
- Two instability spikes (epoch 4 val_loss=9.45, epoch 8 train_loss=3.45)

The attentive probe cannot decode hemodynamic cluster identity from frozen JEPA
token sequences cross-patient. This is consistent with the representation analysis
findings:

1. **Patient identity dominates** — K-Means clustering of full token embeddings
   groups by patient (homogeneity=0.475) not hemodynamic state (ARI=0.035 vs
   hemo clusters).
2. **The hemodynamic clusters are defined by temporal correlations** between
   features (e.g., how ABP area covaries with shock index over time). The
   attentive pooler reduces 1800 patches to a single query vector, which may
   be insufficient to capture pairwise temporal correlation patterns.

### Mean-Pooled Linear Probe (Logistic Regression, test split)

To rule out architecture-specific failure, we also evaluated a simple Logistic
Regression (L2-regularized, `class_weight="balanced"`) on **mean-pooled**
embeddings from both encoders. This uses 22,323 windows from the test split
with valid hemo cluster labels (131 patients, patient-level 80/20 split).

| Model | Macro AUROC | Balanced Acc | N_train | N_test |
|-------|-------------|--------------|---------|--------|
| JEPA (512-d) | 0.524 | 0.155 | 17,341 (104 patients) | 4,982 (27 patients) |
| PatchTST (512-d) | 0.532 | 0.161 | 17,341 (104 patients) | 4,982 (27 patients) |
| *Random baseline* | *0.500* | *0.143* | — | — |

Per-class F1 scores:

| Cluster | JEPA F1 | PatchTST F1 |
|---------|---------|-------------|
| C0 (Failing Vasoconstriction) | 0.052 | 0.187 |
| C1 (Hemodynamic Quiescence) | 0.166 | 0.144 |
| C2 (High-Output Compensated) | 0.113 | 0.162 |
| C3 (Normal Baroreflex) | 0.222 | 0.128 |
| C4 (Tachycardia + Vasoconstriction) | 0.091 | 0.161 |
| C5 (Catecholamine-Driven) | 0.090 | 0.084 |
| C6 (PPG Dissociation) | 0.232 | 0.181 |

**Verdict: Also at chance.** Both models hover near the random baseline (AUROC 0.50,
balanced accuracy 1/7 ≈ 0.143). The failure is not specific to the attentive
pooling architecture — mean-pooled representations also cannot decode cluster
identity cross-patient via a linear classifier.

Scripts:
```
jobs/probing/probe_hemo_clusters_precomputed.py  (uses pre-extracted embeddings)
jobs/probing/probe_hemo_clusters.py              (full pipeline with Zarr reads + encoder)
```

---

### Implications

The frozen JEPA encoder does not represent hemodynamic *state dynamics* (temporal
inter-feature correlations) in a way that is decodable cross-patient. This holds
for both attentive probing (nonlinear, trained 13 epochs) and mean-pooled linear
probing. Possible paths forward:

- **Per-patient normalization** before probing (remove patient identity to expose
  state variation)
- **Fine-tuning** the encoder end-to-end on cluster labels
- **Different architecture** — the cluster targets may require explicit temporal
  correlation computation (e.g., temporal attention over feature pairs) rather
  than generic attentive pooling
- **Re-examine labels** — the 7 clusters were fit on a different feature set
  (icuDataExtraction) that may not align with what the JEPA encoder learns
- **Raw signal statistics baseline** — pending; will show whether the clusters
  are even decodable from the signal at all, or if the label alignment itself
  is too noisy
