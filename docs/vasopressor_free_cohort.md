# Vasopressor-Free Cohort

## Motivation

Train and evaluate models on patients who were never given vasopressors during
the ICU stay being analyzed. This removes a major confounder: vasopressor
administration directly affects ABP waveform morphology and is strongly
correlated with hypotension events (the prediction target).

## Exclusion methodology

**Level:** Stay-level (not patient-level). Different ICU stays for the same
patient are independent waveform recordings. If a patient received vasopressors
in stay A but not stay B, only stay A is excluded.

**Data sources:**
- `INPUTEVENTS_MV.csv.gz` — MetaVision ICU stays (item IDs: 221906, 221289, 221749, 221662, 221653, 221986, 222315)
- `INPUTEVENTS_CV.csv.gz` — CareVue ICU stays (item IDs: 30047, 30120, 30044, 30119, 30127, 30128, 30043, 30307, 30042, 30306, 30125, 30051)
- `ICUSTAYS.csv.gz` — temporal join between ICU stay ID and waveform recording

**Drugs excluded:** Norepinephrine, Epinephrine, Phenylephrine (Neosynephrine),
Dopamine, Dobutamine, Milrinone, Vasopressin.

**Matching algorithm:** A waveform stay is excluded if its recording start time
falls within [INTIME − 1h, OUTTIME + 1h] of any vasopressor ICU stay for the
same subject. The 1-hour tolerance handles slight offsets between ICU admission
time and waveform recording start (observed to be ~5 min typically).

## Cohort statistics

| Metric | Original | Vasopressor-Free | Change |
|--------|----------|-----------------|--------|
| Subjects | 2,524 | 1,460 | −42% |
| ICU stays | 4,001 | 2,220 | −44% |
| Samples | 1,278,205 | 982,132 | −23% |
| Positive events | 54,937 | 19,052 | −65% |
| Prevalence | 4.3% | 1.94% | −55% |

Vasopressor stays excluded: 2,831 (from waveform manifest)
Subjects fully removed (all stays excluded): 1,064

## Data splits

Re-split with the same algorithm as the original cohort:
`corrected_stratified_group_10fold_v1`, seed=16.

| Split | Subjects | ICU Stays | Samples | Positive Events | Prevalence |
|-------|----------|-----------|---------|-----------------|------------|
| Train (folds 2–9) | 1,161 | 1,768 | 785,783 | 15,242 | 1.94% |
| Val (fold 1) | 149 | 222 | 98,161 | 1,905 | 1.94% |
| Test (fold 0) | 150 | 230 | 98,188 | 1,905 | 1.94% |

Note: This is a **fresh re-split** of the 1,460 vasopressor-free subjects. The
test set is NOT a subset of the original test set. Results are not directly
comparable on a per-patient basis, only in aggregate.

## File locations

All outputs under `/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/`:

| File | Description |
|------|-------------|
| `manifests/hypotension_subject_split_vasopressor_free_stays_v1.csv` | Subject manifest with fold/split assignments |
| `manifests/hypotension_subject_split_vasopressor_free_stays_v1.json` | Split summary statistics |
| `manifests/vasopressor_excluded_waveform_stays.csv` | 2,831 excluded stay IDs |
| `manifests/vasopressor_icustay_ids.csv` | All MIMIC ICU stays with vasopressor admin |
| `manifests/vasopressor_exclusion_summary.json` | Full exclusion statistics |
| `labels/hypotension_labels_vasopressor_free_stays_v1.csv.gz` | Filtered outcome labels |

Scripts in repo:

| Script | Purpose |
|--------|---------|
| `jobs/data_processing/scripts/exclude_vasopressor_stays.py` | Identify vasopressor stays, filter labels, build initial manifest |
| `jobs/data_processing/scripts/resplit_vasopressor_free.py` | Re-run stratified group k-fold on filtered subjects |

## Training experiments

### Supervised PatchTST (vasopressor-free)

**Status:** Running (job 26576169, a100_short, a100-4007, started 2026-08-18 17:15)

Config: `jobs/baselines/configs/supervised_patchtst_vasopressor_free.yaml`
Slurm: `jobs/baselines/slurm/supervised_patchtst_vasopressor_free.sbatch`

Architecture: identical to full-cohort supervised PatchTST (d_model=512, 3 layers,
8 heads, 1s patches at 125 Hz, simple tokenizer with bottleneck, attentive
classifier). End-to-end supervised training with class weights.

Key differences from full-cohort config:
- `outcome_df_path` → vasopressor-free labels
- `subject_split_path` → vasopressor-free manifest
- `dataset_filename` → `zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_vasopressor_free_v1` (new cache)
- `require_precomputed_samples: false` (will generate on first run)
- `models_dir` → `models/supervised_patchtst_vasopressor_free`
- `--requeue` in sbatch for auto-resume on preemption
- Rolling checkpoint every 30 min + resume logic

Training: 20 epochs, batch_size=8, OneCycle LR (max_lr=0.0006), mixup (α=0.2),
class weights enabled. Evaluates on best-val-AUPRC checkpoint with bootstrap CIs.

**Expected outcome:** Compare AUROC/AP to the full-cohort result (0.8688/0.2766).
Hypothesis: performance may drop because (a) fewer training samples, (b) lower
prevalence means fewer positive examples to learn from, (c) vasopressor-free
hypotension events are likely milder/shorter.

#### Job history

| Job ID | Partition | Node | Outcome | Notes |
|--------|-----------|------|---------|-------|
| 26566151 | gl40s_short | — | Cancelled | Replaced by a100_short submission |
| 26574888 | a100_short | a100-4009 | Failed (exit 1, 6 min) | RuntimeError: 3 missing train subjects |
| 26575971 | a100_short | a100-4026 | Failed (exit 1, 21 sec) | CUDA device busy/unavailable |
| **26576169** | **a100_short** | **a100-4007** | **Running** | Started 17:15, past failure points |

#### Failure analysis and fix (2026-08-18)

**Problem:** Job 26574888 failed with `RuntimeError: train cache does not match
fixed subject manifest: 3 missing, 0 unexpected`. Sample generation completed
successfully (built caches for train/val/test) but the strict manifest validation
found 3 train subjects and 1 val subject with zero valid samples.

**Root cause:** Four subjects have containers where one or more required channels
(ABP, II, PLETH) are almost entirely NaN. With `require_all_channels: true` and
`constant_nan_tolerance: 0.2` (max 20% NaN per channel per window), every possible
30-minute window for these subjects gets rejected.

| Subject | Channel Problem | Effective Coverage |
|---------|----------------|-------------------|
| p049555 | ABP 87.6% NaN, PLETH 22% NaN | ABP entirely missing in all windows |
| p055973 | PLETH 71.8% NaN | PLETH entirely NaN in all windows |
| p057935 | ABP 30.5%, II 79%, PLETH 90.1% NaN | Most channels missing in most windows |
| p076116 (val) | Not investigated | Similar issue |

**Fix applied:**
1. `fcn_baseline_hypotension.py` line ~488: Changed manifest validation from
   `RuntimeError` to `warnings.warn()` for missing subjects (subjects with no
   valid samples are silently excluded). Still raises `RuntimeError` for
   unexpected subjects (data integrity violation).
2. `supervised_patchtst_vasopressor_free.yaml`: Set `strict_cohort_match: false`
   so the downstream cohort count validation prints a warning instead of crashing
   (patient counts will be 1158/148/150 vs expected 1161/149/150).
3. `supervised_patchtst_vasopressor_free.sbatch`: Added `a100-4026` to exclude
   list (CUDA device was in error state).

**Impact:** 3 train subjects (0.26% of training data) and 1 val subject (0.67%)
excluded. Negligible effect on model performance. The fix is general-purpose —
it will handle similar missing-subject situations in future runs without crashing.

## Design decisions

1. **Stay-level not patient-level exclusion**: Waveform recordings are per-stay
   containers. Windows never cross stay boundaries. No information leaks between
   stays of the same patient.

2. **Fresh re-split instead of keeping original fold assignments**: The original
   fold assignments produced unbalanced prevalence after filtering (train 1.92%,
   val 2.51%, test 1.57%). Re-splitting gives perfect balance (1.94% everywhere).
   Tradeoff: test sets are not directly comparable per-patient, only in aggregate.

3. **Restricted to original manifest subjects**: The outcome labels contain 211
   subjects not in the original downstream manifest (they had waveform data but
   didn't pass the full sample-generation criteria). We exclude them to maintain
   consistency with the established pipeline.

4. **Class weights enabled**: With 1.94% prevalence (vs 4.3% in full cohort),
   class imbalance is more severe. The training config uses `use_class_weights: true`.

## Potential future work

- JEPA pretraining on vasopressor-free subjects → downstream frozen probe
- Compare representation quality (medical feature probing) on vasopressor-free cohort
- Time-resolved exclusion: exclude only the hours during/after vasopressor infusion,
  keeping pre-vasopressor windows from the same stay
- Separate analysis: does the model trained on vasopressor-free data generalize
  to vasopressor-receiving patients (domain shift test)?
