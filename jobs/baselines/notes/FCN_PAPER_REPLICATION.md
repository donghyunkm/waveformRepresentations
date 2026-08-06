# FCN hypotension close-replication run

This workflow targets the MIMIC-III held-out hypotension result reported for the
fully supervised convolutional model in the PhysioJEPA paper:

| Metric | Published result (95% CI) |
| --- | --- |
| AUROC | 0.778 (0.771–0.784) |
| Average precision | 0.140 (0.133–0.148) |
| F1 | 0.082 (0.080–0.085) |
| Recall | 0.998 (0.997–0.999) |
| Specificity | 0.073 (0.071–0.074) |
| Sensitivity at 90% specificity | 0.422 (0.406–0.435) |
| Sensitivity at 95% specificity | 0.296 (0.282–0.310) |

Paper: <https://proceedings.mlr.press/v297/fox26a.html>

Released code: <https://github.com/benmfox/PhysioJEPA>

## Fixed subject split

The authors' exact subject split is unavailable. Regenerating the released
seed-16 `StratifiedGroupKFold` split produced a substantially different test
set because its shuffled behavior is sensitive to even small cohort changes.

The replacement is a fixed, corrected, seeded subject split:

- valid samples from the completed full indexing stage are combined once;
- each subject remains indivisible;
- subject identities and their label counts are shuffled together with seed 16;
- the released stratified-group greedy objective assigns ten balanced folds;
- fold 0 is test, fold 1 is validation, and folds 2–9 are training;
- the manifest is saved outside Git at:

  ```text
  /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/hypotension_subject_split_fixed_v1.csv
  ```

The resulting split is:

| Split | Samples | ICU stays | Patients | Positive | Negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,022,563 | 3,195 | 2,013 | 43,950 | 978,613 |
| Validation | 127,831 | 399 | 255 | 5,493 | 122,338 |
| Test | 127,811 | 407 | 256 | 5,494 | 122,317 |

This is effectively 80%/10%/10% by events, with nearly identical class
prevalence across splits and no subject leakage. The manifest, source-cache
SHA-256 hashes, split algorithm, fold mapping, and summary are recorded in:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/hypotension_subject_split_fixed_v1.json
```

The full training entry point refuses to run if the manifest is missing, a
cache includes the wrong subjects, or any fixed-split count differs.

## Model and evaluation decisions

The run retains the released implementation where the paper is silent:

- 30-minute ABP, ECG lead II, and PLETH inputs at 125 Hz;
- 5-minute hypotension forecast horizon;
- three FCN layers with dimensions 128, 256, and 128;
- convolution kernel sizes 7, 5, and 3;
- batch size 16, learning rate `1e-5`, weight decay `1e-3`;
- 20 epochs and a one-cycle scheduler;
- weighted sampling, positive-class loss weighting, random noise, channel
  dropout, and mixup;
- seed 16.

Where the released files conflict with explicit paper statements:

- The paper used one H100 and 16 CPU cores. This cluster has no H100 partition,
  so the job uses one A100 and 16 cores.
- The paper says AdamW. The released optimizer only recognizes lowercase
  `"adamw"` even though its YAML says `"AdamW"`; the fixed config uses lowercase
  to select AdamW.
- The paper does not specify its bootstrap count or unit. The evaluator uses
  1,000 event-level resamples.

This is a reproducible close replication, not an exact reconstruction of the
authors' held-out subjects. Exact metric equality is therefore not expected.

## Completed test results

The completed FCN run was evaluated on all 127,811 fixed-split test windows
(5,494 positive and 122,317 negative). The selected checkpoint was epoch 12,
which had the best validation average precision.

| Metric | Replication test result (95% CI) | Published result (95% CI) |
| --- | ---: | ---: |
| AUROC | 0.7903 (0.7844–0.7961) | 0.778 (0.771–0.784) |
| Average precision | 0.1911 (0.1819–0.2005) | 0.140 (0.133–0.148) |
| F1 | 0.1242 (0.1213–0.1271) | 0.082 (0.080–0.085) |
| Recall | 0.9037 (0.8961–0.9112) | 0.998 (0.997–0.999) |
| Specificity | 0.4320 (0.4291–0.4347) | 0.073 (0.071–0.074) |
| Sensitivity at 90% specificity | 0.4394 (0.4260–0.4530) | 0.422 (0.406–0.435) |
| Sensitivity at 95% specificity | 0.3074 (0.2965–0.3211) | 0.296 (0.282–0.310) |

The F1, recall, and specificity values use the default 0.5 probability
threshold. At that threshold, accuracy was 0.4523 and the confusion matrix was
TN=52,838, FP=69,479, FN=529, TP=4,965. Confidence intervals are percentile
95% intervals from 1,000 event-level bootstrap resamples with seed 16.

The saved metric artifact records the exact prediction and checkpoint paths:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/fcn_hypotension_paper/2026-07-31-fcn-paper-replication-v1/hypotension_label-fcn_baseline_hypotension_paper-paper_metrics.json
```

## Submission

The fixed manifest and all three caches already exist and passed the strict
integration check. Submit full training from the repository root:

```bash
sbatch jobs/baselines/fcn_baseline_hypotension_paper.sbatch
```

The A100 run is allowed up to 14 days. It selects the checkpoint with the best
validation average precision, evaluates the complete validation and test sets,
saves logits and targets, and writes point estimates plus percentile 95%
bootstrap confidence intervals for every paper-reported metric.
