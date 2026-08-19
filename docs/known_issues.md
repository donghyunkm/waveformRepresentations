# Known Issues

Implementation bugs and caveats in the current source tree. Do not silently
work around these when interpreting results. If an issue is fixed, update this
file and add a focused regression check as part of the change.

---

## Library code (physiojepa / nbs)

### Native JEPA masked training path (`apply_masks`)

The standard unshared-channel masked training path currently fails:
`apply_masks` returns from inside its first loop iteration, collapsing the
effective batch and causing a predictor shape mismatch. The encoder/inference
path runs correctly.

**File:** `nbs/12_jepa.ipynb` → `physiojepa/jepa.py`

### `loss.mape` — misspelled `clamp` call

`loss.mape` raises `AttributeError` because the `clamp` call is misspelled.

**File:** `nbs/07_loss.ipynb` → `physiojepa/loss.py`

### `MultiHeadAttention.forward` ignores attention mask

`MultiHeadAttention.forward` accepts a `mask` argument but does not pass it to
scaled dot-product attention. Code that constructs an attention mask (including
the ECG-JEPA channel/time mask) therefore does not currently enforce it.

**File:** `nbs/10_layers.ipynb` → `physiojepa/layers.py`

### `GeneralTimeSupervised.configure_optimizers` — case-sensitive optimizer check

Compares `optimizer_type` with lowercase `"adamw"` without normalizing. The
baseline YAML value `"AdamW"` consequently selects plain `Adam` instead of
`AdamW`.

**File:** `nbs/13_baselines.ipynb` → `physiojepa/baselines.py`

---

## Training scripts (jobs/)

### Supervised scripts ignore `perform_cv`

The supervised scripts read `training.perform_cv` but do not use it to run
multiple folds. They train only the first nested stratified group split.

### Medical feature probe logs only aggregate R²

`jobs/jepa/scripts/train_medical_features_fixed.py` pools all 15 features
before computing R², producing a misleading aggregate score. Evaluate per
feature offline from saved prediction tensors instead.

### PatchTST inference speed

PatchTST inference is ~0.22 it/s versus JEPA ~1.4 it/s (~7× slower) even with
local SSD caching. Root cause unresolved.

---

## Data and labels

### Single-class patient samples

Hypotension metrics (AUROC, AP) are invalid/undefined for single-patient
samples containing only one class (e.g., p072908 is all-negative).

### Sample-index cache compatibility

Sample-index DataFrames are cached by `dataset_filename` only. If data, split
seeds, channels, sampling rules, or label logic change, use a new descriptive
`dataset_filename` or explicitly confirm the existing cache is compatible.
Do not delete an existing cache without user approval.

---

*Last updated: 2026-08-14*
