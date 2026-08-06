# Native PhysioJEPA — Architecture and Pipeline

This directory contains the native PhysioJEPA pretraining and downstream
probing pipeline. It is separate from the supervised baselines under
`jobs/baselines/`.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Input Representation](#input-representation)
3. [Context Encoder](#context-encoder)
4. [Target Encoder (EMA)](#target-encoder-ema)
5. [Predictor](#predictor)
6. [Masking Strategy](#masking-strategy)
7. [Training Objective](#training-objective)
8. [Optimizer and Scheduler](#optimizer-and-scheduler)
9. [Downstream Probing (Attentive Classifier)](#downstream-probing-attentive-classifier)
10. [ECG-JEPA Variant](#ecg-jepa-variant)
11. [Full Hyperparameter Reference](#full-hyperparameter-reference)
12. [Pipeline Stages](#pipeline-stages)
13. [Data and Leakage Policy](#data-and-leakage-policy)
14. [Known Corrections to Released Code](#known-corrections-to-released-code)

---

## High-Level Overview

PhysioJEPA is a Joint-Embedding Predictive Architecture for physiological
waveforms. It learns representations by predicting masked latent targets
rather than reconstructing raw signals:

```
                    ┌──────────────────────┐
     raw signal ──►│   Target Encoder     │──► full latent [B, 1800, 512]
     (no mask)     │   (EMA copy)         │         │
                    └──────────────────────┘         │ gather target indices
                                                    ▼
                    ┌──────────────────────┐    target representations
     raw signal ──►│   Context Encoder    │──► context latent [B, ctx, 512]
     (masked)      │   (trainable)        │         │
                    └──────────────────────┘         │
                                                    ▼
                    ┌──────────────────────┐    predicted targets
                    │   Predictor          │──► [B, tgt, 512]
                    │   (trainable)        │         │
                    └──────────────────────┘         │
                                                    ▼
                                               MSE Loss
```

The context encoder sees only unmasked (context) patches. The predictor
receives the context representations plus learnable mask tokens at target
positions and predicts the target encoder's latent outputs at those positions.
The target encoder is a momentum-updated (EMA) copy of the context encoder and
is never trained by gradient descent.

---

## Input Representation

**Raw input:** 30-minute windows of 3 physiological channels at 125 Hz.

| Parameter | Value |
|-----------|-------|
| Channels | ABP, ECG Lead II, PPG (PLETH in MIMIC-III) |
| Sampling rate | 125 Hz |
| Window duration | 1800 seconds (30 minutes) |
| Samples per window | 225,000 |
| Patch duration | 1 second |
| Patch size | 125 samples |
| Patch stride | 125 (no overlap) |
| Number of patches | 1800 per channel |
| Signal normalization | IQR per-channel |

**Training data (paper):** 356,903 thirty-minute segments totaling 10,707,090
minutes across 4,282 ICU stays (N=2,631 patients) from the MIMIC-III Waveform
Database Matched Subset. Segments with ≥20% constant or NaN values in any
single channel are excluded (primarily segments with ≥75% missing data).

**Channel handling (unshared embedding):** Each channel is tokenized
independently via a depthwise convolution (one set of D=512 kernels per
channel), equivalent to PatchTST's channel-independent patching (Nie et al.,
2023). The three per-channel sequences are concatenated along the batch
dimension (batch×channel melting), so the transformer processes
`[B×3, 1800, 512]` tokens. This means channels do NOT attend to each other
during encoding — each channel is encoded in isolation. Cross-channel
information is leveraged through (a) channel-specific mask tokens in the
predictor and (b) the downstream classifier's channel concatenation.

**Tokenizer (depthwise convolution):** Per the paper (Eq. 1):

```
h_{p,c,d} = Σ_{i=0}^{k-1} w_{c,d}[i] · x_c[p·k + i] + b_{c,d},  d = 1,...,D
```

where x_c ∈ ℝ^125 is channel c at patch p of size k (kernel size), and each
channel c has D convolutional kernels w_{c,d} ∈ ℝ^k. This is implemented as:

```
Conv1d(in_channels=c_in, out_channels=d_model*c_in, kernel_size=patch_size, stride=patch_size, groups=c_in)
```

producing an embedding Z ∈ ℝ^{c×d×p} where d=512 and p=1800.

With `tokenizer_kwargs`:
- `depth: 1`
- `bottleneck_channels: 112`
- `residual: true`
- `kernel_size: 40`
- `bottleneck: true`

The inception tokenizer variant uses multi-scale convolutions (kernels 40, 20,
10) with a bottleneck, batch normalization, and ReLU, providing richer
multi-resolution patch features.

---

## Context Encoder

**Class:** `Encoder` (from `physiojepa/jepa.py`)

The paper (Section 3.3) describes this as "A PatchTST transformer (Nie et al.,
2023) [that] processes a masked subset of input patches to produce context
encodings h_c." The implementation uses PatchTST-style blocks
(`PositionAwareTSTBlock` with `use_tst_block=true`).

| Parameter | Value |
|-----------|-------|
| Embedding dimension (d_model) | 512 |
| Attention heads | 8 |
| Head dimension | 64 (= 512 / 8) |
| Transformer layers | 3 |
| MLP ratio | 4.0 (hidden dim = 2048) |
| Block type | PositionAwareTSTBlock (post-norm) |
| Positional encoding | Rotary (RoPE) |
| QKV bias | True |
| Dropout (general) | 0.1 |
| Attention dropout | 0.0 |

### Transformer Block (PositionAwareTSTBlock)

Each block uses **post-norm** ordering:

```
x_in ──► MultiHeadAttention ──► Dropout ──► + x_in ──► LayerNorm ──►
     ──► FeedForward        ──► Dropout ──► + prev  ──► LayerNorm ──► x_out
```

**Multi-Head Attention (PositionAwareMultiHeadAttention):**
- Separate Q, K, V linear projections (not fused QKV)
- Scale factor: 1/√d_head = 1/√64 = 0.125
- RoPE applied to Q and K using explicit position indices (critical for
  masked sequences where patch positions are non-contiguous)
- Uses `F.scaled_dot_product_attention` with Flash Attention backend priority
- Supports asymmetric positions (different position sets for Q vs K)

**Rotary Position Embeddings (RoPE):**

Per the paper (Section 3.3.3, Eq. 2): for token at position p with embedding
dimension d, each even-odd pair (h_2j, h_2j+1) is rotated by angle
φ_p,j = p · θ_j where θ_j = 10000^(−2j/d):

```
h̃_p,j = [cos(φ)  −sin(φ)] · h_p,j
         [sin(φ)   cos(φ)]
```

Implementation details:
- Dimension: per-head (64 = 512/8), applied independently per attention head
- Base frequency (θ): 10,000
- Style: language-model frequencies (`freqs_for="lang"`)
- Cache: up to 29,000 positions
- Applied inside attention to Q and K after linear projection
- Position indices track original patch positions through masking, ensuring
  correct relative position encoding even after patch removal

**Feed-Forward Network:**
```
Linear(512, 2048) → GELU → Dropout(0.1) → Linear(2048, 512) → Dropout(0.1)
```

**Weight Initialization:**
- Linear/Conv weights: truncated normal, σ=0.02
- Biases: zeros
- LayerNorm: weight=1, bias=0
- Stability rescaling: attention output projection and MLP fc2 weights are
  divided by √(2×layer_id) per layer

### Forward Pass (Pretraining)

1. Tokenize: `Conv1d` maps raw patches to d_model embeddings
2. Reshape to `[B×c_in, num_patches, d_model]`
3. Create position indices: `arange(1800)` expanded per batch item
4. Apply context mask: gather only context patch embeddings and their positions
5. Pass through 3 transformer blocks (with RoPE using sparse positions)
6. Final LayerNorm (identity for TSTBlock, which already post-norms)
7. Output: `[B×c_in, num_context_patches, 512]`

---

## Target Encoder (EMA)

The target encoder is a **deep copy** of the context encoder with:
- All parameters frozen (`requires_grad_(False)`)
- Permanently in eval mode (BatchNorm/Dropout disabled)
- Updated via exponential moving average after each training batch

**EMA Momentum Schedule:**

```
m(t) = τ_base + progress(t) × (1.0 − τ_base)
```

where `progress(t) = global_step / total_steps` and `τ_base = 0.996`.

This produces a linear warmup from 0.996 → 1.0 over training, meaning the
target encoder changes rapidly early on and stabilizes (becomes nearly frozen)
toward the end of training.

**Update rule (after each training batch):**
```
θ_target = m × θ_target + (1 − m) × θ_online
```

The target encoder processes the FULL input (no masking) and provides the
prediction targets at masked positions.

---

## Predictor

**Class:** `Predictor` (from `physiojepa/jepa.py`)

| Parameter | Value |
|-----------|-------|
| Input dimension | 512 (encoder output) |
| Predictor embedding dim | 256 |
| Attention heads | 4 |
| Head dimension | 64 |
| Transformer layers | 2 |
| MLP ratio | 4.0 (hidden dim = 1024) |
| Block type | PositionAwareTSTBlock |
| Positional encoding | Rotary (RoPE) |
| Mask tokens per channel | 3 (c_in_mask_tokens) |

### Architecture

```
predictor_embed: Linear(512 → 256)     # project from encoder space
mask_token: Parameter [3, 1, 256]       # learnable per-channel tokens
transformer_blocks: 2× PositionAwareTSTBlock(dim=256, heads=4)
predictor_norm: Identity (TSTBlock already post-norms)
predictor_proj: Linear(256 → 512)       # project back to encoder space
```

### Forward Pass

1. **Project context:** `predictor_embed(encoder_output)` → `[B×3, ctx, 256]`
2. **Add positional info to context:** gather positional encoding at context positions, add to projected context
3. **Create target tokens:** repeat `mask_token` to `[B, num_targets, 256]`, add positional encoding at target positions
4. **Concatenate:** `[context_tokens | target_tokens]` along sequence dim
5. **Random shuffle:** permute the concatenated sequence (permutation-equivariance training trick prevents the predictor from using position shortcuts)
6. **Transform:** pass through 2 transformer layers with RoPE
7. **Unshuffle:** restore original order
8. **Extract targets:** take only the target-position outputs (indices after context)
9. **Project back:** `predictor_proj(x)` → `[B×3, num_targets, 512]`

The predictor is intentionally smaller than the encoder (256-dim vs 512-dim)
to encourage the encoder to learn rich representations rather than delegating
to the predictor.

---

## Masking Strategy

Masking is applied in the **patch index space** (not the raw signal space).
One mask ratio pair is sampled per batch (shared across all samples in the
batch).

| Parameter | Value |
|-----------|-------|
| Target mask ratio range | [0.1, 0.3] |
| Context mask ratio range | [0.1, 0.4] |
| Block size range | [1, 1] (individual patches) |

### Mask Generation (`create_masks`)

1. Sample `target_ratio ~ Uniform(0.1, 0.3)`
2. Compute `num_target = floor(1800 × target_ratio)` → 180–540 patches
3. Sample `context_ratio ~ Uniform(0.1, 0.4)`
4. Compute `num_context = floor((1800 − num_target) × context_ratio)` → variable
5. Generate random permutation of patch indices per sample
6. First `num_target` indices → target mask
7. Next `num_context` from remaining → context mask (what the encoder sees)

**Result:** The encoder sees only a subset of non-target patches (not all
remaining patches). This adds an additional information bottleneck beyond
simply removing targets.

**Channel melting:** When `shared_embedding=False` with simple/inception
tokenizers, effective batch = B × c_in. Masks are generated for this melted
batch, meaning each channel independently receives its own random mask pattern.

---

## Training Objective

**Primary loss:** Mean Squared Error (MSE) between predictor outputs and
target encoder representations at masked positions.

```python
loss = mean([MSE(pred_i, target_i) for pred_i, target_i in zip(preds, targets)])
```

Both `pred` and `target` are in the encoder's 512-dimensional latent space.
Targets are L2-normalized (LayerNorm over feature dimension) before loss
computation.

**Optional variance regularization** (`mse_variance_loss`):
```
loss = MSE(pred, target) + α × ReLU(1 − std(representations, dim=-1)).mean()
```
with α=0.2. This penalizes representation collapse (std < 1 triggers the
penalty). Not used in the paper configuration (pure MSE).

**Huber delta:** The config specifies `huber_delta: 1` but the training script
uses `loss_pred` (MSE), not SmoothL1. The ECG-JEPA variant uses SmoothL1.

---

## Optimizer and Scheduler

### Optimizer: AdamW

| Parameter Group | Weight Decay |
|----------------|--------------|
| Encoder weights (dim > 1) | 0.04 → 0.4 (cosine schedule) |
| Predictor weights (dim > 1) | 0.04 → 0.4 (cosine schedule) |
| Encoder bias/norm (dim ≤ 1) | 0.0 |
| Predictor bias/norm (dim ≤ 1) | 0.0 |

Base learning rate: 1e-5 (initial param to optimizer, overridden by scheduler)

**Weight Decay Cosine Schedule:**
```
wd(t) = final_wd + 0.5 × (initial_wd − final_wd) × (1 + cos(π × progress))
```
Anneals from 0.04 → 0.4 over training (increasing regularization).

### Scheduler: OneCycleLR

| Parameter | Value |
|-----------|-------|
| Max learning rate | 6e-4 |
| Warmup fraction | 30% of training (pct_start=0.3) |
| Div factor | 25 (initial LR = 6e-4 / 25 = 2.4e-5) |
| Final div factor | 100 (final LR = 2.4e-5 / 100 = 2.4e-7) |
| Anneal strategy | Cosine |
| Step interval | Per training step (not per epoch) |

**Gradient Clipping:**
- Algorithm: norm
- Max norm: 1.0

---

## Downstream Probing (Attentive Classifier)

After pretraining, the encoder is frozen and an attentive classifier head is
trained for supervised prediction (e.g., hypotension forecasting). Per the
paper: "Training was stopped after 100 epochs, and the last model was used
for linear probing."

**Downstream data split (paper):** 80% train / 10% validation / 10% test,
using a proportional subject-wise data splitter. A weighted sampler is used
during training to account for class imbalance.

### Architecture: AttentiveClassifier (with batch/channel melting)

Per the paper (Section 3.4): "A single learned query vector q ∈ ℝ^d is
utilized to extract relevant encoding information from the context encoder's
representations Z. Queries, keys, and values are calculated Q = qW_q,
K = ZW_k, and V = ZW_v for each attention head."

```
Encoder output: [B, c_in, d_model, num_patches] = [B, 3, 512, 1800]
    │
    ▼ transpose to [B, 3, 1800, 512]
    │
    ▼ reshape to [B×3, 1800, 512]  (melt channels into batch)
    │
    ▼ AttentivePooler
    │   ├── query_token: [1, 1, 512] (learned, trunc_normal σ=0.02)
    │   ├── CrossAttentionBlock(dim=512, heads=4)
    │   │   ├── LayerNorm
    │   │   ├── CrossAttention(Q=query, K/V=encoder output)
    │   │   ├── Residual + LayerNorm
    │   │   ├── MLP(512→2048→512) + Residual
    │   │   └── LayerNorm
    │   └── Output: [B×3, 1, 512]
    │
    ▼ reshape to [B, 3, 512]
    │
    ▼ flatten to [B, 1536]
    │
    ▼ Linear(1536 → 1)  (binary classification)
    │
    ▼ BCEWithLogitsLoss
```

**Probing hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Pooler heads | 4 |
| Pooler depth | 1 |
| MLP ratio | 4.0 |
| Complete block | True (cross-attn + MLP + residuals) |
| Affine per-channel | False |
| Num queries | 1 |
| Fine-tune encoder | False (frozen) |
| Encoder mode | eval() (dropout/BN disabled) |
| Epochs | 20 |
| Batch size | 128 |
| Max LR | 0.01 |
| Optimizer | AdamW (WD=1e-4) |
| Scheduler | OneCycleLR |
| Precision | bf16-mixed |
| Mixup | α=0.2 |
| Augmentations | Jitter (5%), channel masking (p=0.1) |

---

## ECG-JEPA Variant

An alternative architecture based on [ECG-JEPA (arxiv:2410.08559)](https://arxiv.org/abs/2410.08559)
that treats the input as a 2D channel×time grid with joint positional
embeddings.

### Key Differences from Native PhysioJEPA

| Aspect | Native JEPA | ECG-JEPA |
|--------|-------------|----------|
| Channel handling | Unshared, batch-melted (no cross-channel attention) | Joint 2D grid (channels attend to each other) |
| Positional encoding | 1D RoPE per-channel | 2D sincos (channel + time) |
| Masking | Per-channel random patches | Time-dimension mask shared across channels |
| Attention mask | None | Block-diagonal + stripe (within-channel + within-time) |
| Loss | MSE | SmoothL1 (Huber) |
| Tokenizer | Conv1d per-channel | Linear projection of raw patches |
| Downstream head | AttentiveClassifier (with melting) | AttentiveClassifierNoMelt |

### ECG-JEPA Encoder (MaskTransformer)

Per the paper (Appendix A.1): "Encoder size and number of heads were identical
to PhysioJEPA with sinusoidal positional encodings."

Architecture as configured in this repo:
- d_model: 512 (matched to PhysioJEPA)
- Layers: 3 (matched to PhysioJEPA)
- Heads: 8 (matched to PhysioJEPA)
- Patch size: 125 samples (1 second, same as native JEPA)
- Input: `[B, c_in×num_time_patches, d_model]`
- 2D sincos positional embedding encodes both channel identity and time position
- Cross-attention mask constrains attention to within-channel AND within-time-position neighbors

### ECG-JEPA Predictor (MaskTransformerPredictor)

- predictor_embed_dim: 128 (as configured; paper's public YAML)
- Layers: 2
- Heads: 4
- Mask token: single shared token `[1, 1, 128]`
- Positional encoding: 2D sincos for time positions only (shared across channels)

---

## Full Hyperparameter Reference

The paper (Section 3.3) explicitly specifies: encoder dimension 512, 3 layers,
8 heads, feedforward 2048, predictor dimension 256, 2 layers, 4 heads, RoPE,
depthwise convolutions, target mask 10–30%, context mask 10–40%, 100 epochs,
OneCycleLR, AdamW. The paper does NOT specify: EMA decay, batch size, learning
rate, weight decay, or gradient clipping. Those values below come from the
implementation config.

### Pretraining (train_patch_jepa.yaml)

```yaml
# === Run ===
model_type: native_jepa
precision: "32-true"
deterministic: true
random_state: 12

# === Data ===
channels: [ABP, II, PLETH]
frequency: 125
patch_seconds: 1
overlap: 0.0
sample_seq_len_seconds: 1800
sample_stride_seconds: 1800
constant_nan_tolerance: 0.2
normalize_signals: true
require_all_channels: true

# === Encoder ===
d_model: 512
nhead: 8
num_layers: 3
pe_type: rotary
mlp_ratio: 4.0
use_tst_block: true
shared_embedding: false
qkv_bias: true
drop_rate: 0.1
attn_drop_rate: 0.0
jepa: true
tokenizer_type: simple
tokenizer_kwargs:
  depth: 1
  bottleneck_channels: 112
  residual: true
  kernel_size: 40
  bottleneck: true

# === Predictor ===
predictor_embed_dim: 256
nhead: 4
num_layers: 2
use_tst_block: true
pe_type: rotary
mlp_ratio: 4.0
qkv_bias: true
drop_rate: 0.1
attn_drop_rate: 0.0
c_in_mask_tokens: 3

# === Training ===
n_gpus: 2
batch_size: 128              # per GPU
epochs: 100
accumulate_grad_batches: 1
loss_fxn: mse
huber_delta: 1
target_mask_range: [0.1, 0.3]
context_mask_range: [0.1, 0.4]
mask_block_range: [1, 1]
ema_decay: 0.996
use_gradient_clipping: true
gradient_clip_val: 1.0
gradient_clip_algorithm: norm

# === Optimizer ===
learning_rate: 0.00001
weight_decay: 0.04
use_weight_decay_scheduler: true
final_weight_decay: 0.4
optimizer_type: adamw

# === Scheduler (OneCycleLR) ===
scheduler_type: onecycle
max_lr: 0.0006
div_factor: 25
final_div_factor: 100
pct_start: 0.3
anneal_strategy: cos
```

### Derived Values

| Quantity | Calculation | Result |
|----------|-------------|--------|
| Sequence length (samples) | 1800s × 125Hz | 225,000 |
| Patch size (samples) | 1s × 125Hz | 125 |
| Number of patches | 225,000 / 125 | 1,800 |
| Effective batch size | 128 × 2 GPUs | 256 |
| Encoder hidden dim | 512 × 4.0 | 2,048 |
| Predictor hidden dim | 256 × 4.0 | 1,024 |
| Initial LR | max_lr / div_factor | 2.4e-5 |
| Final LR | initial / final_div_factor | 2.4e-7 |
| Target patches (range) | 1800 × [0.1, 0.3] | 180–540 |
| Context patches (range) | remaining × [0.1, 0.4] | ~126–648 |

---

## Pipeline Stages

### 1. Prepare Pretraining Samples

```bash
sbatch prepare_pretraining_samples.sbatch
```

Generates leakage-safe record manifests and train/validation sample tables.
Validates against the downstream subject split to prevent data leakage.

### 2. Smoke Test (Memory/Correctness)

```bash
sbatch train_patch_jepa_smoke.sbatch
```

Uses the same per-GPU batch size and precision as full training. If this fits
in memory, the full run will too. Runs 2 train batches and 1 val batch for 1
epoch with 512 train / 256 val samples.

### 3. Full Pretraining (100 epochs)

```bash
sbatch train_patch_jepa_full.sbatch
```

Two H100 NVLink GPUs, DDP strategy, synchronized batch normalization.

**Checkpointing (fault-tolerant):**
- Rolling checkpoint every 30 minutes
- Best 2 checkpoints by validation loss
- Best 2 checkpoints by training loss (+ save_last)
- Config fingerprint validation on resume

### 4. Downstream Probing

```bash
sbatch train_hypotension_fixed_smoke.sbatch   # smoke first
sbatch train_hypotension_fixed_full.sbatch    # then full
```

Loads the pretrained encoder checkpoint, freezes it in eval mode, trains
the attentive classifier for 20 epochs. Saves validation and test predictions
as `.pt` files for offline evaluation.

---

## Data and Leakage Policy

Pretraining reads the 5,661 Zarr ZipStore containers listed by
`manifests/waveform_manifest.csv`. Subjects assigned to the fixed downstream
validation or test sets are excluded before a deterministic subject-level
95/5 JEPA train/validation split is made. Unlabelled subjects and downstream
training subjects remain eligible for self-supervised pretraining.

The preparation stage reads the conversion-time `quality/valid_window` mask
stored in each container. That mask was computed for 30-minute windows using
ABP, ECG II, and PLETH with a maximum constant-or-NaN fraction of 0.2. It does
not rescan full waveforms. Preparation records a data-only fingerprint covering
the input paths, split policy, channels, and sampling rules. Training rejects a
manifest or sample cache whose fingerprint differs from the active config.

All artifacts are written below:
```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA
```

W&B runs are offline.

**Compute resources (paper, Appendix B):** Self-supervised models were trained
with two Nvidia H100 NVLink GPUs and 16 CPU cores. Classifier models were
trained with one Nvidia H100 NVLink GPU and 16 CPU cores.

---

## Known Corrections to Released Code

The native implementation corrects released-code defects that otherwise break
or change the paper method:

1. **Masking preserves effective batch/channel dimension:** The original
   `apply_masks` returns from inside its first loop iteration, collapsing the
   effective batch. The corrected version samples one ratio pair per batch and
   gathers indices for all samples.

2. **Original patch indices retained for RoPE:** Through encoder masking and
   predictor shuffling, original position indices are tracked and passed to
   RoPE, ensuring correct relative position encoding for non-contiguous patch
   sequences.

3. **EMA target stays in eval mode:** The target encoder is never switched to
   training mode, preventing dropout and batch-norm statistics from corrupting
   the targets.

4. **Frozen downstream encoder in eval mode:** The attentive probe explicitly
   holds the encoder in eval mode and uses 4 attention heads (matching the
   paper specification rather than silently inheriting a default).

5. **Predictor dimension:** The paper specifies predictor dimension 256; the
   authors' public YAML uses 128. This implementation follows the paper.

The smoke test intentionally uses the same per-GPU batch size and precision as
the full job so that a successful smoke validates the full memory configuration.
