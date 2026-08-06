# PhysioJEPA progress and implementation review

## Current training status

- A native JEPA run and a self-supervised PatchTST run are active on the cluster.
- The PatchTST run is using the one-GPU cluster entry point and the leakage-safe
  manifest/sample-index pipeline.
- The latest observed PatchTST log was healthy (epoch 17 in progress, with no
  traceback). It had approximately 272,068 training samples and 14,540
  validation samples.
- Train/validation subject overlap was zero; downstream-excluded subject overlap
  was zero; there were no duplicate samples; and sample lengths were valid.
- The current cache contains approximately 286,608 samples from 4,689 stays and
  3,035 subjects. The paper reports approximately 356,903 segments from 4,282
  stays and 2,631 patients, so the data inventory is not identical to the paper.
- The observed checkpoint was
  `resume-epoch=17-step=145799.ckpt`; its configuration fingerprint matched the
  active run. GPU memory remained below the requested allocation.

## Paper alignment

The paper uses 30-minute, non-overlapping windows at 125 Hz with ABP, ECG lead
II, and PPG/PLETH. It excludes signals with at least 20% constant or null
values, interpolates nulls, IQR-normalizes the signals, and divides them into
patches.

For native PhysioJEPA, the implementation and YAML match the main architectural
specification: three encoder layers, eight attention heads, model width 512,
feed-forward width 2048, RoPE, a two-layer predictor with width 256 and four
heads, target masking of 10--30%, context masking of 10--40%, EMA target updates,
MSE embedding loss, 100 epochs, AdamW, and OneCycle scheduling.

The current data split is stricter than the paper's stated 95/5 pretraining
split because subjects reserved for downstream validation/test are removed
before pretraining. This is a protocol difference, not an accidental overlap.
There is also a minor boundary difference in the signal-quality check: the
current code accepts exactly 20% constant samples, while the paper says 20% or
more should be excluded.

## PatchTST: paper versus released repository

The paper's Appendix A.1 says that PatchTST uses the same tokenization,
positional embeddings, and encoder dimensions as PhysioJEPA. It specifies
masking before tokenization, 10--30% target masking, reconstruction of the
masked patches with a per-channel linear head, and MSE on the masked patches.

The released [PhysioJEPA repository](https://github.com/benmfox/PhysioJEPA) does
not implement the loss exactly as described in that appendix. In the original
[PatchTST module](https://raw.githubusercontent.com/benmfox/PhysioJEPA/main/physiojepa/patchtst.py):

1. The input is copied to an unmasked `Y_true` target.
2. Patches are masked only when `self.training` is true.
3. The model reconstructs the complete target tensor.
4. The training step computes MSE over the complete reconstruction, with no mask
   selecting only masked positions.
5. Validation runs in evaluation mode, so masking is bypassed, and full-signal
   reconstruction MSE is measured.

Therefore, the released repository performs a denoising-autoencoder-style
objective: visible and masked patches both contribute to the gradient. At a
10--30% mask ratio, roughly 70--90% of the reconstruction terms are visible
patches. This can affect the learned representation and the checkpoint selected
by validation; it should not be described as a masked-only PatchTST objective.

The current cluster PatchTST path uses the same model and loss behavior as the
released code. It is therefore faithful to the GitHub implementation, but not
to the paper's stated masked-only loss. The current cluster YAML uses
`mask_ratio: [0.1, 0.3]`, which is more consistent with the paper. The original
[PatchTST YAML](https://raw.githubusercontent.com/benmfox/PhysioJEPA/main/jobs/patchtst/train_patchtst.yaml)
uses `[0.1, 0.4]`.

Other current-path differences from the original training entry point include
the cluster-safe data backend and manifest, subject-leakage protections, and
resume/checkpoint handling. The original [training script](https://raw.githubusercontent.com/benmfox/PhysioJEPA/main/jobs/patchtst/train_patchtst.py)
uses a 95/5 group split and the original local data-loading setup.

## Interpretation

The answer to "is this how the original code does it?" is yes: the original
repository masks the training input but computes reconstruction loss over all
patches, and validates without masking. The answer to "is this exactly what the
paper says?" is no: the paper describes loss restricted to masked patches.
Matching the repository and matching the paper are therefore different targets.

No source code was changed during this review; this file records the current
status and conclusions only.

---

# Progress Summary — 2026-08-06 (Session 2)

## 1. What Changed This Session

- **Implemented multi-scale PatchTST tokenizer** — a hierarchical dual-path
  tokenizer that adds a fine-grained 200ms local encoder (beat-level) fused
  with the existing 1-second coarse tokenizer via a learnable gate.
- Modified notebooks:
  - `nbs/18_tokenizers.ipynb`: Added `MultiScaleTokenizer` and `_LocalTSTBlock`
    classes; updated imports to include `MultiHeadAttention` and
    `get_activation_fn` from layers.
  - `nbs/17_patchtst.ipynb`: Added `tokenizer_type='multiscale'` option to
    `PatchTFTSimple`; updated import to include `MultiScaleTokenizer`.
- Ran `nbdev_prepare` — regenerated `physiojepa/tokenizers.py` and
  `physiojepa/patchtst.py` with the new classes.
- Created config:
  `jobs/baselines/configs/supervised_patchtst_multiscale_hypotension_full.yaml`
- Created sbatch:
  `jobs/baselines/slurm/supervised_patchtst_multiscale_hypotension_full.sbatch`
  (a100_short, 3 days, 1×A100, 16 CPUs, 128G RAM).
- Submitted job **26165312** (`physiojepa-sptst-multiscale`).

## 2. What Is Working

- **Multi-scale supervised PatchTST** (job 26165312): Just started, running on
  a100-4013. Uses the same fixed subject split, sample caches, and training
  config as the completed full supervised PatchTST. Adds 275K params (+2.9%)
  via the fine-grained local encoder. Smoke-tested: correct output shapes in
  both supervised and self-supervised modes; gradient flow verified through
  both coarse and fine paths.
- **PatchTST self-supervised pretraining** (job 26126590): Running ~25 hours,
  on a100-4004. Previously at epoch 16, val_loss ~0.009.
- **JEPA 1-GPU** (job 26107011): Running ~44 hours on a100-4013. Previously in
  epoch 0, loss trending down.
- **Supervised PatchTST full** (completed earlier): AUROC 0.8688, AP 0.2766.
- **FCN baseline** (completed earlier): AUROC 0.7903, AP 0.1911.

## 3. What Remains Unfinished

- Multi-scale PatchTST training just started — no metrics yet.
- JEPA pretraining still early (was in epoch 0 after ~20 hours).
- PatchTST self-supervised pretraining convergence status unknown (need to
  check current epoch/loss).
- No downstream classification launched with pretrained encoders yet.
- InceptionTime baseline cancelled; needs architectural rework.
- Cross-channel and compact260K PatchTST runs had checkpoints but no completed
  test-metric artifacts.

## 4. Known Bugs

Per AGENTS.md (unchanged this session):

- Native JEPA `apply_masks` returns inside its first loop iteration — collapses
  effective batch (masked training path broken for unshared-channel mode).
- `loss.mape` raises `AttributeError` — misspelled `clamp` call.
- `MultiHeadAttention.forward` accepts `mask` but doesn't pass it to scaled
  dot-product attention.
- `GeneralTimeSupervised.configure_optimizers` compares `optimizer_type` with
  lowercase `"adamw"` without normalizing — YAML value `"AdamW"` selects
  `Adam` instead.
- Supervised scripts don't use `training.perform_cv` to run multiple folds.

## 5. Design Note: Patch Overlap

Both the original supervised PatchTST and the new multi-scale variant use
non-overlapping patches throughout:

- Coarse tokenizer: `kernel_size=125, stride=125` (1-second, no overlap).
- Fine tokenizer (multi-scale only): `kernel_size=25, stride=25` (200ms, no
  overlap).

Overlapping patches were considered but not implemented because:

1. At the coarse level (1s), overlap would expand the sequence from 1,800 to
   ~3,600 tokens — roughly quadrupling attention cost — for minimal benefit,
   since the transformer's job at this scale is modeling multi-minute trends
   where beat-boundary artifacts are irrelevant.
2. At the fine level (200ms), the local transformer only attends over 5 tokens
   per group. Overlap would increase this to 7–9 tokens, which is
   computationally cheap but unnecessary: the dual-path fusion already
   mitigates boundary effects because the coarse Conv1d kernel sees all 125
   samples in a single convolution regardless of where fine sub-patch
   boundaries fall.
3. RoPE positional encoding assumes each position index maps to a distinct
   temporal location. Overlap introduces ambiguity where adjacent tokens share
   most of their temporal extent.
4. Adjacent overlapping patches are largely redundant — the transformer wastes
   capacity re-learning that neighboring tokens are similar.

If a future ablation shows boundary effects in the fine path, a simple change
to `fine_stride=13` (50% overlap) in the `MultiScaleTokenizer` would test this
at negligible compute cost, since local attention remains on fewer than 10
tokens.

## 6. Commands to Continue

### Check running jobs
```bash
squeue -u dk5565
```

### Check multi-scale PatchTST logs
```bash
tail -20 /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/physiojepa-sptst-multiscale-26165312.out
```

### Check PatchTST pretraining
```bash
tail -2 /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/physiojepa-patchtst-1gpu-26126590.out | tr '\r' '\n' | grep -oP 'val_loss_epoch=[\d.]+' | tail -1
```

### Check JEPA pretraining
```bash
tail -2 /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/physiojepa-jepa-1gpu-short-26107011.out | tr '\r' '\n' | grep -oP 'Epoch \d+|train_loss_step=[\d.]+' | tail -2
```

### Activate environment
```bash
source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
cd /gpfs/home/dk5565/PhysioJEPA
```

### Resubmit multi-scale job if preempted/expired
```bash
sbatch jobs/baselines/slurm/supervised_patchtst_multiscale_hypotension_full.sbatch
```

### Cancel a job
```bash
scancel <JOBID>
```
