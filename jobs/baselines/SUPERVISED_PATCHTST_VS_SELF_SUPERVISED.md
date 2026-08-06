# Supervised PatchTST vs. Original Self-Supervised PatchTST

The two models share the same core PatchTST encoder configuration in the current
full runs, but they optimize different objectives and attach different heads.

## Comparison

| Component | Original self-supervised PatchTST | Supervised PatchTST |
|---|---|---|
| Data | Unlabeled waveform windows sampled every 1,800 seconds | Outcome-indexed windows from fixed train/validation/test subject splits |
| Input | ABP, ECG lead II, PLETH; 1,800 seconds at 125 Hz | Same: 3 channels × 1,800 seconds at 125 Hz |
| Patching | 1-second patches: 1,800 patches × 125 samples | Same |
| Tokenizer | `simple` → `TS_Tokenizer`, a channel-specific grouped `Conv1d` patch projection | Same |
| Transformer encoder | `d_model=512`, 8 heads, width 2,048, 3 blocks, rotary positions | Same in the full configuration |
| Training-time masking | Patch masking is applied before tokenization | The config lists `patch_mask`, but the supervised model sets `pretrain_head=False`, so the self-supervised masking/reconstruction path is not used |
| Head | `MaskedAutogressionFeedForward` reconstructs waveform patches | Attentive classifier pools contextualized patch embeddings and emits one logit |
| Target | Original waveform patches (`Y_true`) | Hypotension label for the 300-second forecast horizon |
| Loss | Reconstruction loss, configured as MSE | Binary classification through `BCEWithLogitsLoss` in the one-logit path |
| Training schedule | 100 epochs, batch size 32, 2 GPUs in the original config | 20 epochs, batch size 8, 1 GPU in the full config |

## Tokenizer equivalence

For the full runs, the tokenizer is identical in both models:

| Setting | Original self-supervised | Supervised full |
|---|---:|---:|
| tokenizer_type | simple | simple |
| Implementation | TS_Tokenizer grouped Conv1d | TS_Tokenizer grouped Conv1d |
| Input channels (c_in) | 3 | 3 |
| Patch size | 125 samples (1 second) | 125 samples (1 second) |
| Patch stride | 125 samples | 125 samples |
| Embedding width (d_model) | 512 | 512 |
| Channel sharing | shared_embedding=False | shared_embedding=False |
| Tokenizer depth | 1 | 1 |
| Bottleneck channels | 112 | 112 |
| Residual block | enabled | enabled |
| Inception kernel setting | 40 | 40 |
| Bottleneck option | enabled | enabled |

Thus, the full supervised model does not change the tokenizer dimensions or
configuration; the architectural difference begins after the shared encoder,
where self-supervised training reconstructs waveform patches and supervised
training applies a classification head. The compact supervised variant is
different: it uses d_model=96 and an empty tokenizer_kwargs mapping.

## Effective training-set size

The tokenizer and encoder dimensions match, but the number and construction of
training examples do not:

| Quantity | Original self-supervised | Supervised full |
|---|---:|---:|
| Window construction | Non-overlapping 1,800-second windows from unlabeled records | Outcome-indexed 1,800-second windows tied to 300-second labels |
| Train examples | Not fixed in the config; equals the number of eligible waveform windows | 1,022,563 cached train windows |
| Validation examples | Determined by the unsupervised record split | 127,831 cached validation windows |
| Test examples | Determined by the unsupervised record split | 127,811 cached test windows |
| Batch size / GPUs | 32 / 2 GPUs | 8 / 1 GPU |
| Epochs | 100 | 20 |

The supervised loader uses a weighted sampler with
num_samples equal to the full training-set size each epoch, so it presents
about 1,022,563 sampled windows per epoch (with replacement). Across 20 epochs
that is about 20.45 million supervised window presentations. The original
self-supervised run presents approximately 100 times its available unlabeled
window count, with a per-device batch size of 32 across two GPUs. Therefore,
matching encoder dimensions does not make the optimization problems equivalent:
the supervised run has a known, much larger event-indexed training table, while
the self-supervised effective size depends on the number and duration of
eligible waveform records.

## Self-supervised flow

```text
[B, 3, 225000]
        │
        ├─ split into 1800 one-second patches
        ├─ randomly mask patches during training
        ├─ simple grouped Conv1d tokenizer
        ├─ 3-layer rotary PatchTST encoder
        ├─ reconstruction head predicts patch waveforms
        └─ compare predicted patches with original patches using MSE
```

The encoder learns representations without outcome labels. The model returns
the contextualized embeddings, reconstructed patches, and original patch target
(`z`, `x_hat`, `Y_true`). The reconstruction head is part of this pretraining
model and is not a classification head.

## Supervised flow

```text
[B, 3, 225000]
        │
        ├─ split into 1800 one-second patches
        ├─ simple grouped Conv1d tokenizer
        ├─ 3-layer rotary PatchTST encoder
        ├─ attentive pooling over contextualized patches per channel
        └─ linear classifier → [B, 1] hypotension logit
```

The supervised model reuses the encoder but constructs it with
`pretrain_head=False`. It therefore returns embeddings to the supervised
classifier rather than reconstructing waveform patches. The label is associated
with the configured 300-second forecast window following the 30-minute input.

## What is—and is not—transfer learning here

Architecturally, the supervised model is compatible with the self-supervised
encoder because the input channels, patching, tokenizer, and transformer widths
match. However, the current supervised training configuration constructs a new
model; it does not automatically load a self-supervised checkpoint. Loading a
pretrained encoder and optionally freezing or fine-tuning it would be a separate
transfer-learning workflow.

The compact supervised comparison variant changes the encoder to `d_model=96`,
two transformer blocks, four heads, and a mean-pooling classifier to match the
FCN parameter budget. That compact variant is not architecturally identical to
the original self-supervised 512-dimensional model.

## Full supervised test results

The completed full supervised PatchTST hypotension run was evaluated on 127,811
test windows (5,494 positives) with a 0.5 probability threshold:

| Metric | Test value |
|---|---:|
| AUROC | 0.8688 |
| Average precision (AP) | 0.2766 |
| Accuracy | 0.6702 |
| F1 | 0.1851 |
| Recall / sensitivity | 0.8713 |
| Specificity | 0.6611 |

The confusion matrix was TN=80,866, FP=41,451, FN=707, TP=4,787.
