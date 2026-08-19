# Representation Analysis

## Overview

Note, this uses mean-pooled embeddings so analysis needs to be redone

This document consolidates all representation analysis experiments conducted on
frozen JEPA and PatchTST self-supervised encoders. The central question: **what
do these embeddings actually encode?**

**Answer: Patient identity (waveform morphology), not universal physiology.**

The masked-prediction pretraining objective learns to reconstruct each patient's
unique waveform shapes. This produces representations that are excellent patient
fingerprints but do not capture shared physiological features that generalize
across patients. Within a patient, the encoder does track hemodynamic transitions
(including pre-hypotensive deterioration), but these transitions are encoded
along patient-specific directions that do not align cross-patient.

---

## Methodology

All analyses use embeddings from the full test set (127,811 windows from 256
patients) extracted from:

- **JEPA encoder**: best checkpoint at epoch 13 (val_loss=0.215)
- **PatchTST encoder**: best checkpoint at epoch 3 (val_loss=0.003)

Embeddings are 512-dimensional mean-pooled transformer outputs. Distance metric
is cosine distance throughout.

Hemodynamic cluster labels come from icuDataExtraction's 7-cluster model
(KMeans on 19 physiological features extracted from 20-min windows). Window-level
alignment uses epoch offset correction: subtract 946,684,800s from PhysioJEPA
POSIX timestamps to convert to icuDataExtraction's seconds-since-2000-01-01
reference frame. Windows matched within 2.5-min tolerance.

### icuDataExtraction Alignment

- Only v2 output available (requires RESP signal); v1 was deleted for disk quota.
- v2 covers 1,643 patients → 62.9% overlap with PhysioJEPA test set (161/256
  patients). Re-running without RESP would recover ~71 additional patients.
- Epoch offset: subtract 946,684,800s (seconds between 1970-01-01 and
  2000-01-01) from PhysioJEPA POSIX timestamps to align with
  icuDataExtraction's reference frame.

### Scripts (all in `jobs/probing/`)

```
cluster_embeddings.py              # Extract embeddings + KMeans + UMAP plots
knn_analysis.py                    # Standard + cross-patient kNN
embedding_distances.py             # Pairwise cosine distance by group
within_patient_distances.py        # Hypotension within-patient distances
within_patient_hemo_distances.py   # Hemodynamic within-patient (with time alignment)
cka_analysis.py                    # Centered Kernel Alignment (12 analyses)
cluster_single_patient.py          # Per-patient sub-cluster analysis
characterize_subclusters.py        # Physiological feature characterization
cluster_pooled_patients.py         # Pooled multi-patient clustering
characterize_clusters_icu_features.py  # 19-feature cluster characterization
linear_probe_generalizability.py   # Cross-patient linear probing
```

### Embedding Visualization Methodology

- **Scatter plots (UMAP/t-SNE):** 4 independent random draws (seeds 42–45) of
  2500 pos + 2500 neg test samples each, producing 8 plots per encoder.
- **Trajectory plots:** Use negative-stays-only because positive stays have
  irregular sampling (only event-adjacent windows), making continuous trajectory
  visualization misleading. Negative stays have dense ~60s continuous sampling.
- **Time-to-event visualization:** Attempted but abandoned — positive stays all
  contain windows exactly 5 min from an event (due to sample generation), making
  a time-to-event gradient meaningless.

Scripts: `jobs/visualization/scripts/visualize_scatter.py`,
`jobs/visualization/scripts/visualize_trajectories.py`

### Results location

```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/probing/clustering/
```

---

## 1. k-NN Neighborhood Analysis

For each window, find its 10 nearest neighbors in 512-dim cosine space, then
measure what those neighbors have in common.

### Standard k-NN (20K subsample)

| Metric | JEPA (k=10) | PatchTST (k=10) | Chance |
|--------|-------------|-----------------|--------|
| kNN AUROC (hypotension) | **0.909** | **0.898** | 0.50 |
| Same hemodynamic cluster | **0.825** | **0.801** | 0.213 |
| Same patient | **0.785** | **0.762** | ~0.004 |

### Cross-patient k-NN (same-patient neighbors excluded)

| Metric | JEPA (k=10) | PatchTST (k=10) | Chance |
|--------|-------------|-----------------|--------|
| kNN AUROC (hypotension) | 0.569 | 0.550 | 0.50 |
| Same hemodynamic cluster | 0.140 | 0.119 | 0.213 |

### Interpretation

The strong standard k-NN results (AUROC 0.91) collapse to near-chance (0.57)
when same-patient neighbors are excluded. The embeddings place a patient's
windows near each other regardless of their label, so within one patient's
cluster you find both pre-hypotensive and normal windows grouped together —
giving high AUROC from patient-level risk correlation. But cross-patient, there
is no shared pre-hypotensive signature.

The hemodynamic cluster agreement drops *below chance* (0.14 vs 0.21) —
cross-patient neighbors actually have different phenotypes more often than
expected. The embeddings organize patients along axes orthogonal to hemodynamic
phenotype.

Results: `knn_analysis.csv`, `knn_cross_patient.csv`

---

## 2. Pairwise Embedding Distances

Ratio = mean distance within group / mean distance between groups.
Ratio < 1 → same-group windows are closer (signal present).
Ratio = 1 → no grouping.

| Comparison | JEPA ratio | PatchTST ratio | Signal? |
|-----------|-----------|---------------|---------|
| Same patient / diff patient | **0.799** | **0.690** | ✓ Strong |
| Same hemo / diff hemo | 0.967 | 0.952 | Weak (patient-driven) |
| Same hemo, diff patient / diff hemo, diff patient | **0.995** | **0.997** | ✗ None |
| Same hypo label / diff hypo label | 1.001 | 1.055 | ✗ None |

### Interpretation

The only strong signal is patient identity. Once you control for patient (row 3:
same hemo but different patient vs different hemo different patient), the
hemodynamic phenotype ratio becomes 1.0 — no signal remains. Hypotension labels
show zero signal at the population level.

Results: `embedding_distances.csv`

---

## 3. Patient Centroid Distances by Hemodynamic Cluster

Compute each patient's centroid (mean embedding), then check if patients with
the same dominant hemodynamic phenotype have closer centroids.

| Metric | JEPA ratio | PatchTST ratio |
|--------|-----------|---------------|
| Same hemo / diff hemo centroid distance | **1.043** | **1.073** |

Ratio > 1 means patients with the same phenotype are actually *farther apart*
in embedding space than patients with different phenotypes. The embeddings
**do not** organize patients by hemodynamic state — not even at the coarsest
level.

Results: `patient_centroid_hemo_distances.csv`

---

## 4. Within-Patient Distance Analysis

While cross-patient analyses show no shared physiological signal, within-patient
analyses reveal that the encoder **does** track physiological state changes over
time within a single patient's trajectory.

### 4a. Hypotension (28 patients with both labels)

| Distance type | JEPA ratio | PatchTST ratio | Interpretation |
|---------------|-----------|---------------|----------------|
| Pos-Pos / Pos-Neg | **0.779** | **0.750** | Pre-hypo windows 22–25% closer to each other than to normal |
| Neg-Neg / Pos-Neg | **0.424** | **0.425** | Normal windows 58% closer to each other than to pre-hypo |

**Geometric picture:** Within each patient's embedding subspace, normal windows
occupy a compact region ("home base"). Pre-hypotensive windows form a looser
cloud displaced from the normal cluster. The encoder detects that waveform
morphology changes before the hypotension event.

**Temporal confound note:** Pos-Pos pairs come from windows separated by
hours/days (sparse event-adjacent sampling), while Neg-Neg pairs come from
temporally adjacent windows (often <1 min apart, dense continuous sampling).
The neg-neg tightness (ratio 0.42) is therefore partially driven by temporal
proximity, not just label similarity.

### 4b. Hemodynamic Transitions (68 patients, time-aligned)

| Ratio | JEPA | PatchTST |
|-------|------|----------|
| Same cluster / diff cluster | **0.917** | **0.947** |

Windows belonging to the same hemodynamic cluster are 5–8% closer within a
patient's trajectory. Weaker than hypotension because hemodynamic cluster
transitions are gradual, while hypotension events involve abrupt ABP drops.

### 4c. Combined Summary

| Signal | JEPA ratio | PatchTST ratio | Effect size | N patients |
|--------|-----------|---------------|-------------|------------|
| Hypotension: Pos-Pos / Pos-Neg | 0.779 | 0.750 | 22–25% closer | 28 |
| Hypotension: Neg-Neg / Pos-Neg | 0.424 | 0.425 | 58% closer | 28 |
| Hemodynamic: same / diff cluster | 0.917 | 0.947 | 5–8% closer | 68 |

Results: `within_patient_distances.csv`, `within_patient_hemo_distances.csv`

---

## 5. CKA Analysis (Centered Kernel Alignment)

CKA quantifies whether two sets of embeddings have the same *relational
structure* — whether the pattern of pairwise similarities among samples in
set A mirrors the pattern in set B. Unlike cosine distance (which asks "are
these points close?"), CKA asks "are these points arranged the same way
relative to each other?"

### Summary Table

| Analysis | What we compared | Result | Meaning |
|----------|-----------------|--------|---------|
| 7a. Same vs diff phenotype | CKA between patients grouped by hemo cluster | delta ≈ 0 | Shared phenotype doesn't mean shared structure |
| 7b. JEPA vs PatchTST | CKA between models on same windows | **0.81** | Both models learn the same patterns |
| 7c. States within patient | CKA(pre-hypo, normal) vs CKA(normal, normal) | PatchTST ratio 0.75 | PatchTST arranges hypo windows differently |
| 7d. Pooled clusters | Split-half CKA per phenotype pool | All near 0 | Multi-patient pools have no coherent structure |
| 7e. Same vs diff patient | Split-half CKA vs cross-patient CKA | Ratio 1.17–1.20x | Patient fingerprint is location, not structure |
| 7f. Temporal thirds | CKA(early, late) for hypo vs stable patients | No difference | Deterioration doesn't cause structural drift |
| 7g. Clinical similarity | CKA grouped by vital sign similarity | r ≈ 0 | Similar vitals ≠ similar structure |
| 7h. Pre-hypo vs normal (uncontrolled) | CKA of pre-hypo vs normal pools | 1.36–1.62x | Pre-hypo windows are more generically structured |
| **7i. Within-patient hypotension** | Split-half CKA per state + between-state CKA | **Between/within = 0.40–0.60** | **Hypo states structurally distinct within patient** |
| **7k. Same vs diff condition (controlled)** | Same-cond CKA vs diff-cond CKA | **Ratio 0.79** | **No condition-specific cross-patient structure** |
| **7l. Same vs diff cluster (controlled)** | Same-cluster CKA vs diff-cluster CKA | **Ratio 0.96** | **No cluster-specific cross-patient structure** |
| **7m. Within-patient hemo clusters** | Within-cluster vs between-cluster CKA | **Ratio 0.71–0.78, d=0.44–0.84** | **Hemo states structurally distinct within patient** |

**7h→7k correction:** Analysis 7h initially showed promising cross-patient
pre-hypo CKA (1.36–1.62× higher than normal), suggesting a shared
pre-hypotensive motif. However, the controlled test 7k revealed this was an
artifact of uncontrolled patient-level confounding. When controlling for patient
identity (comparing same-condition different-patient vs different-condition
different-patient CKA), the ratio is 0.79 — same-condition is actually *less*
similar. Pre-hypo windows have high CKA with *everything* from other patients,
not specifically with other pre-hypo windows. This definitively shows no
condition-specific cross-patient representational structure.

### Key CKA Findings

1. **No patient-invariant condition-specific structure.** The controlled tests
   (7k, 7l) definitively show that same-condition windows from different
   patients do NOT share more relational structure than different-condition
   windows. No universal "pre-hypotensive motif" or "cluster-specific motif"
   transfers across patients.

2. **The patient fingerprint is a location effect, not a structural effect.**
   Same-patient CKA is only 17–20% higher than different-patient CKA (7e),
   meaning relational structure is largely shared across patients. Patient
   identity lives in the mean embedding (centroid), not in the
   covariance/relational structure.

3. **Strong within-patient state separation for both conditions:**
   - Hypotension (7i): between/within ratio 0.40–0.60 (very strong)
   - Hemodynamic clusters (7m): between/within ratio 0.71–0.78, Cohen's d
     0.44–0.84, 79–86% of patients show separation

4. **Both models learn the same patterns** (7b, CKA=0.81). Switching from
   JEPA to PatchTST does not change what is learned, only minor details.

### CKA Results Location

```
probing/clustering/
├── cka_phenotype_pairs.csv              # 7a
├── cka_model_comparison.csv             # 7b
├── cka_within_patient_states.csv        # 7c
├── cka_group_level.csv                  # 7d
├── cka_same_vs_diff_patient.csv         # 7e
├── cka_temporal_segments.csv            # 7f
├── cka_clinical_similarity.csv          # 7g
├── cka_cross_patient_hypo.csv           # 7h
├── cka_within_patient_hypo_geometry.csv # 7i
├── cka_controlled_hypo.csv             # 7k
├── cka_controlled_hemo.csv             # 7l
└── cka_within_patient_hemo.csv         # 7m
```

---

## Linear Probe Generalizability

Directly tests whether condition/cluster information is *linearly decodable*
from embeddings across patients (complementing the structural CKA tests).

### Hypotension (AUROC)

- **Within-patient:** High (upper bound — leaks patient identity)
- **Cross-patient:** ~Chance (same conclusion as kNN/distance/CKA analyses)

### Hemodynamic Clusters — JEPA patient_level (Balanced Accuracy)

| Evaluation | Result |
|------------|--------|
| Within-patient | 0.8965 |
| Cross-patient (raw) | 0.1760 ± 0.0123 |
| LOPO (raw) | 0.1419 ± 0.1962 |
| Permutation baseline | 0.1381 |
| Chance (1/6 classes) | 0.1667 |

### Interpretation

Confirms no cross-patient hemodynamic cluster signal in the embeddings:
cross-patient balanced accuracy (0.176) ≈ permutation baseline (0.138) ≈
chance (0.167). Within-patient classification is strong (0.90), consistent
with the within-patient distance and CKA findings. The linear probe cannot
find a hyperplane that separates hemodynamic clusters across patients — the
clusters are encoded along patient-specific directions.

### Status

- PatchTST + window-level results pending (job 26303167)
- Script: `jobs/probing/linear_probe_generalizability.py`

---

## 6. KMeans Clustering

### 6a. Global KMeans (127,811 windows, 256 patients)

Purpose: test whether embeddings form discrete phenotype clusters at population
scale.

Silhouette scores 0.13–0.24 for k=2..30. Near-zero agreement with external
labels (ARI ≈ 0 for hypotension, 0.02–0.04 for hemodynamic clusters). KMeans
is inappropriate at this scale because patient identity dominates — each patient
occupies a distinct region, so KMeans recovers "patient groups" rather than
physiological states.

### 6b. Pooled Multi-Patient KMeans (6 patients, 13,404 windows)

Purpose: at a smaller scale where patient-identity dominance is reduced,
understand what the clusters that form represent physiologically.

#### Agreement Between Clusters and Ground-Truth Groupings

| Grouping | JEPA ARI (k=17) | JEPA NMI | PatchTST ARI (k=11) | PatchTST NMI |
|----------|-----------------|----------|---------------------|--------------|
| Recording segment | **0.396** | **0.617** | **0.377** | **0.570** |
| Patient identity | 0.307 | 0.527 | 0.289 | 0.468 |
| Hemodynamic cluster | 0.113 | 0.188 | 0.156 | 0.239 |
| Hypotension label | 0.002 | 0.018 | 0.002 | 0.010 |

#### Cluster Driver Ranking

1. **Recording segment** (ARI ~0.4) — strongest. Individual recording sessions
   are the primary axis.
2. **Patient identity** (ARI ~0.3) — second. Same patient's different recordings
   cluster near but not identically.
3. **Hemodynamic phenotype** (ARI ~0.1–0.15) — third. Some cross-patient grouping
   by physiological state exists, stronger in PatchTST.
4. **Hypotension label** (ARI ≈ 0) — not a driver at the global level.

#### Cluster Profiles (JEPA, k=17, selected clusters)

| Cluster | Size | Pos% | #Patients | HR | SBP | MAP | PP | PLETH_amp | ShockIdx |
|---------|------|------|-----------|-----|-----|-----|----|-----------|---------| 
| **C2** | **510** | **14.3%** | **5** | 96 | 111 | 72 | 58 | 1.85 | 0.85 |
| **C11** | **221** | **31.2%** | **4** | 96 | 125 | 92 | 50 | 1.88 | 0.77 |
| C0 | 1,536 | 1.2% | 2 | 114 | 127 | 90 | 56 | 0.34 | 0.93 |
| C4 | 1,278 | 3.1% | 5 | 92 | 129 | 85 | 71 | 1.85 | 0.69 |
| C7 | 644 | 0.0% | 4 | 92 | 172 | 102 | 106 | 1.90 | 0.46 |

**C11 (31.2% positive):** The closest thing to a "shared pre-hypotensive state."
Hemodynamics appear normal (MAP 92, SBP 125) — these may be windows just before
the MAP drops, captured while BP is still acceptable but waveform morphology
already shows deterioration.

**C2 (14.3% positive):** The "low blood pressure" cluster. MAP 72 (clinical
concern threshold is 65), SBP 111. Windows in an already-low hemodynamic state
that frequently progresses to frank hypotension.

Results: `probing/clustering/pooled/`

---

## 7. Per-Patient Sub-Clustering

Clustered each patient's embeddings individually (KMeans with silhouette-based k
selection) and characterized what distinguishes sub-clusters.

### Results (JEPA, 15 patients)

| Patient | N | k | Sil | Contiguity | C/Random | ARI hypo | ARI hemo |
|---------|------|---|------|-----------|---------|---------|---------|
| p064538 | 4,084 | 4 | 0.34 | 0.401 | 1.3× | 0.000 | -0.028 |
| p052529 | 3,603 | 8 | 0.31 | 0.303 | 2.2× | 0.001 | — |
| p011342 | 3,374 | 5 | 0.45 | 0.378 | 1.4× | 0.023 | -0.039 |
| p078342 | 2,601 | 9 | 0.42 | 0.483 | 3.3× | 0.000 | 0.045 |
| p046034 | 2,256 | 8 | 0.36 | 0.620 | 4.5× | -0.000 | — |
| p057886 | 1,852 | 2 | 0.59 | 0.970 | 1.3× | -0.001 | — |
| p098276 | 1,716 | 5 | 0.38 | 0.775 | 2.9× | -0.009 | 0.016 |
| p087049 | 1,710 | 6 | 0.38 | 0.829 | 3.7× | 0.002 | 0.018 |
| p044827 | 1,689 | 3 | 0.36 | 0.851 | 2.1× | 0.015 | 0.000 |
| p093560 | 1,656 | 6 | 0.36 | 0.880 | 3.4× | -0.006 | 0.090 |
| p097008 | 1,641 | 8 | 0.37 | 0.814 | 4.2× | 0.004 | — |
| p097441 | 1,203 | 2 | 0.32 | 0.930 | 1.3× | **0.190** | — |

Mean silhouette: 0.386. Mean contiguity ratio: 2.5× random.

### Sub-Cluster Drivers

| Patient | Primary driver | Key distinguishing features |
|---------|---------------|---------------------------|
| p011342 | Recording boundaries | Different hospital admissions |
| p052529 | Hemodynamic regime | SBP 61 mmHg range, HR 31 bpm range across clusters |
| p057886 | Waveform morphology | PLETH amplitude 45% difference |
| p093560 | Temporal drift | HR, MAP, DBP vary 10–15% within one recording |

### Key Conclusions

1. **Sub-clusters are temporally contiguous** (1.3–6.3× random) — they represent
   genuine temporal phases of the ICU stay, not noise.
2. **The encoder organizes each patient's space by hemodynamic regime** — blood
   pressure level and heart rate are the primary axes of variation.
3. **Pre-hypotensive windows concentrate in the lowest-MAP/SBP cluster** (e.g.,
   13.2% positive rate in C4 of p052529 vs <1% elsewhere).
4. **PatchTST finds simpler structure** — often selects k=2–3 with higher
   silhouette (0.46 vs 0.39), suggesting a more compressed representation.

Results: `probing/clustering/per_patient/`

---

## 8. Cluster Characterization with icuDataExtraction Features (19 features)

### Feature Variation Across Pooled Clusters (range of cluster medians / grand mean)

| Feature | Variation | Interpretation |
|---------|-----------|----------------|
| ECG_Ramp | **207%** | Completely different ECG morphology |
| PPV | **166%** | Respiratory-coupled pulse pressure variation |
| HRV_RMSSD | **153%** | Autonomic tone — 5× difference between clusters |
| ABP_tau | **151%** | Vascular compliance (diastolic decay) |
| RESP_amp | **125%** | Respiratory effort/depth |
| ABP_area | **99%** | Beat area (stroke volume proxy) |
| dPdt_max | **80%** | Cardiac contractility |
| PP | **79%** | Pulse pressure regime |
| PLETH_ACDC | **78%** | Perfusion index |
| HR_range | **65%** | Heart rate variability range |
| PLETH_amp | **63%** | PPG amplitude |
| DBP | **42%** | Diastolic pressure |
| ShockIdx | **32%** | Hemodynamic compromise indicator |
| PTT | **26%** | Pulse transit time |
| HR | **24%** | Heart rate |
| SBP | **19%** | Systolic pressure |
| MAP | **17%** | Mean arterial pressure |

### Key Insight

The encoder discriminates far more on **waveform morphology features**
(ECG_Ramp, HRV, ABP_tau, dPdt_max, PLETH_ACDC) than on simple pressure levels
(MAP 17%, SBP 19%). It captures the shape and dynamics of waveforms, not just
their DC level.

### Pre-Hypotensive Cluster Profiles

Pre-hypotensive clusters (C2: 16.7% pos, C11: 23.8% pos) are characterized by:
- **Very low HRV** (RMSSD 42–48 ms vs 150+ in healthy clusters)
- **Reduced HR_range** (60 vs 80+ in safe clusters)
- **Lower dPdt_max** (weaker cardiac contractility)
- **Otherwise normal-appearing BP** (MAP 84–86, SBP 128–133)

**Clinical interpretation:** Pre-hypotensive windows are characterized by
**reduced autonomic variability** and moderately reduced contractility — the
cardiovascular system is losing its compensatory flexibility before the pressure
actually drops. Loss of HRV is a known precursor to hemodynamic decompensation.

Results: `probing/clustering/pooled/cluster_characterization_icu_features_jepa.csv`

---

## 9. Summary: What Do the Embeddings Encode?

| Property | Cross-patient | Within-patient |
|----------|--------------|----------------|
| Patient identity | ✓✓✓ (ratio 0.69–0.80) | N/A |
| Hypotension transitions | ✗ (ratio ≈ 1.0) | ✓✓ (ratio 0.75–0.78) |
| Hemodynamic phenotype | ✗ (ratio 1.04–1.07) | ✓ (ratio 0.92–0.95) |

### Implications for Downstream Classification

The supervised probe's high AUROC (0.844) likely works by:
1. Learning which patients are high-risk during training (patient fingerprint →
   risk level).
2. Within high-risk patients, detecting the trajectory shift toward
   pre-hypotensive windows.

It does **not** work by detecting a universal pre-hypotensive waveform pattern
that generalizes across patients.

### Implications for Pretraining Objectives

The masked-prediction objective naturally encourages learning patient-specific
waveform morphology — that's what makes reconstruction possible. To learn
cross-patient physiological features, the objective would need to:
- Augment with physiological transforms that preserve state but alter morphology
- Use contrastive objectives that pull same-state windows from different patients
  together (→ see [Contrastive Learning](contrastive_learning.md))
- Incorporate physiological feature prediction as an auxiliary loss

### Data Artifacts

- **Embedding scale**: PatchTST embeddings have ~350× smaller cosine distances
  than JEPA (0.001 vs 0.4), indicating PatchTST representations are more
  concentrated on a smaller region of the unit sphere.
- **Sample coverage**: Only 17.5% of PhysioJEPA test windows match to
  icuDataExtraction (limited by patient overlap between systems).
- **Cluster assignment**: Dominant cluster (mode) is a coarse per-patient label.
  Patients spanning multiple clusters over time have their minority states
  ignored.
