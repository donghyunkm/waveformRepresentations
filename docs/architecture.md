# Architecture and Data Flow

## Signal pipeline

The standard experiments use 30-minute windows of ABP, ECG lead II, and PLETH
at 125 Hz. Signals are read lazily from per-record Zarr ZipStore containers,
interpolated, optionally filtered, IQR-normalized, and divided into patches.

## Self-supervised model families

### Native JEPA

A context encoder predicts masked latent target-encoder representations; the
target encoder is updated by EMA. See `nbs/12_jepa.ipynb`.

### PatchTST

Masked waveform patches are reconstructed from transformer representations.
See `nbs/17_patchtst.ipynb`.

### ECG-JEPA

Channel/time patch tokens use joint positional embeddings and a masked latent
predictor. See `nbs/12_jepa.ipynb`.

## Downstream evaluation

Downstream jobs load a pretrained Lightning checkpoint, normally freeze its
encoder, apply an attentive pooling classifier, and predict one label per
forecast horizon.

FCN jobs train directly from waveforms as supervised baselines.

## Data splits

Hypotension and shock-index jobs split by subject with `StratifiedGroupKFold`;
self-supervised jobs use `GroupShuffleSplit`.

All supervised and downstream probe experiments use the same fixed subject-level
split (`manifests/hypotension_subject_split_fixed_v1.csv`):

| Split | Subjects | ICU Stays | Samples | Positive Events | Prevalence |
|-------|----------|-----------|---------|-----------------|------------|
| Train (folds 2–9) | 2,013 | 3,195 | 1,022,563 | 43,950 | 4.3% |
| Val (fold 1) | 255 | 399 | 127,831 | 5,493 | 4.3% |
| Test (fold 0) | 256 | 407 | 127,811 | 5,494 | 4.3% |

Algorithm: corrected stratified group 10-fold (seed=16), enforcing subject-level
disjointness — no patient appears in more than one split.

### Vasopressor-free cohort (stay-level exclusion)

A filtered sub-cohort excludes any waveform stay whose recording overlaps (±1h)
an ICU stay where vasopressors were administered (7 MetaVision + 12 CareVue
item IDs covering norepinephrine, epinephrine, phenylephrine, dopamine,
dobutamine, milrinone, vasopressin).

Exclusion is **stay-level**: a subject can retain some stays while losing others.
Subjects that lose all stays are dropped entirely.

Manifest: `manifests/hypotension_subject_split_vasopressor_free_stays_v1.csv`
Labels:   `labels/hypotension_labels_vasopressor_free_stays_v1.csv.gz`

| Split | Subjects | ICU Stays | Samples | Positive Events | Prevalence |
|-------|----------|-----------|---------|-----------------|------------|
| Train | 1,161 | 1,768 | 785,783 | 15,242 | 1.94% |
| Val | 149 | 222 | 98,161 | 1,905 | 1.94% |
| Test | 150 | 230 | 98,188 | 1,905 | 1.94% |
| **Total** | **1,460** | **2,220** | **982,132** | **19,052** | **1.94%** |

Re-split with the same corrected stratified group 10-fold algorithm (seed=16)
to ensure balanced prevalence across splits. 2,831 waveform stays excluded;
1,064 subjects fully removed.

Scripts: `jobs/data_processing/scripts/exclude_vasopressor_stays.py`,
         `jobs/data_processing/scripts/resplit_vasopressor_free.py`
Summary: `manifests/vasopressor_exclusion_summary.json`,
         `manifests/hypotension_subject_split_vasopressor_free_stays_v1.json`

Self-supervised pretraining (JEPA, PatchTST) excludes the downstream val and
test subjects before building its own 95/5 train/val split. Downstream probes
train the classification head on the same 2,013 train subjects the encoder saw
during pretraining, but evaluate on the 255 val + 256 test subjects that were
never seen during pretraining.

---

*Last updated: 2026-08-14*
