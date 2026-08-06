# Supervised standalone PatchTST baseline

This baseline uses the repository's standalone PatchTST comparison encoder
(`PatchTFTSimple`), not the native PhysioJEPA encoder.

The encoder is initialized from scratch with the architecture in
`jobs/patchtst/configs/train_patchtst.yaml`:

- ABP, ECG lead II, and PLETH;
- 30-minute windows at 125 Hz;
- non-overlapping one-second patches (125 samples, 1,800 patches);
- channel-specific embeddings;
- 512-dimensional tokens;
- 8 attention heads;
- 2,048-dimensional feed-forward layers;
- 3 transformer blocks;
- rotary position encoding;
- dropout 0.1.

`SupervisedPatchTST` forces `pretrain_head=False`, so patch masking and masked
waveform reconstruction are disabled. The reconstruction head is replaced by
the same `AttentiveClassifier` used by the repository's pretrained PatchTST
downstream job. The complete encoder and classifier are optimized jointly from
random initialization with binary cross-entropy.

`SupervisedPatchTST` also supports `classifier_type: "mean"`. This option
mean-pools the contextualized temporal patches separately for each channel,
flattens the three channel representations, and applies one linear
classification layer. Existing configurations default to `"attentive"` and
are unchanged.

## Parameter-matched compact variant

The compact260K experiment controls for the FCN's much smaller parameter
count while preserving full temporal attention:

- `d_model: 96`;
- 4 attention heads;
- `d_ff: 384`;
- 2 transformer blocks;
- temporal mean-pooling classifier;
- 260,281 parameters, versus 265,985 for the FCN;
- batch size 16 and full FP32 precision;
- OneCycle maximum learning rate `1e-4`;
- fixed weight decay `0.001`.

It uses the same protected subject manifests, sample caches, imbalance
handling, mixup, and signal transforms as the corresponding FCN experiment.
Matching parameter count does not match runtime because attention still spans
all 1,800 temporal patches.

A 1,677,089-parameter alternative (`d_model=256`, 8 heads, `d_ff=1024`, two
layers, mean pooling) is documented in `PROGRESS.md` but intentionally has no
configuration or Slurm script yet.

Validate the compact model sequentially:

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_compact260k_hypotension_smoke.sbatch
```

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_compact260k_hypotension_subset10.sbatch
```

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_compact260k_hypotension_full.sbatch
```

Do not submit the subset before the smoke test succeeds, or the full job before
reviewing the subset validation curve.

The full and 10% experiments reuse the exact protected subject manifests and
precomputed sample caches used by the FCN experiments. Outputs are isolated
under:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/supervised_patchtst_hypotension_smoke
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/supervised_patchtst_hypotension_subset10
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/supervised_patchtst_hypotension_full
```

## Validation order

Do not submit the subset or full run before the GPU smoke test succeeds and
establishes memory use and throughput.

Submit the smoke test:

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_hypotension_smoke.sbatch
```

After it completes successfully, submit the 10% experiment:

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_hypotension_subset10.sbatch
```

Only after reviewing the 10% runtime and metrics, submit the full experiment:

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_hypotension_full.sbatch
```

Each Slurm script performs a real CUDA tensor-allocation preflight and uses
offline W&B logging.

## Full-run test results

The completed full supervised PatchTST hypotension run produced the following
metrics on 127,811 test windows (5,494 positive labels), using the default 0.5
probability threshold:

| Metric | Test value |
|---|---:|
| AUROC | 0.8688 |
| Average precision (AP) | 0.2766 |
| Accuracy | 0.6702 |
| F1 | 0.1851 |
| Recall / sensitivity | 0.8713 |
| Specificity | 0.6611 |

The confusion matrix was TN=80,866, FP=41,451, FN=707, TP=4,787. These values
were computed from the saved prediction artifact under
`/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/supervised_patchtst_hypotension_full/`.
