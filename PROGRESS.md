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
  `jobs/baselines/supervised_patchtst_multiscale_hypotension_full.yaml`
- Created sbatch:
  `jobs/baselines/supervised_patchtst_multiscale_hypotension_full.sbatch`
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
sbatch jobs/baselines/supervised_patchtst_multiscale_hypotension_full.sbatch
```

### Cancel a job
```bash
scancel <JOBID>
```
