# Supervised PatchTST vs. FCN

Both baselines consume the same three-channel waveform window: ABP, ECG lead II,
and PLETH. The configured input is 1,800 seconds at 125 Hz (225,000 samples per
channel). The 300-second forecast horizon is the prediction target after the
input window; it is not appended to the model input.

## Architecture at a glance

| Component | Supervised PatchTST | FCN baseline |
|---|---|---|
| Input | `[batch, channels, time]` = `[B, 3, 225000]` | Same |
| Temporal representation | 1-second non-overlapping patches: 1,800 patches × 125 samples | Continuous sample-level convolution |
| Front end | `simple` tokenizer: grouped/channel-specific `Conv1d` patch projection (`kernel_size=stride=125`); `shared_embedding=False` | Three `Conv1d` blocks |
| Sequence model | Transformer self-attention over patch tokens, rotary positional encoding | No attention or explicit token sequence |
| Hidden architecture (full run) | `d_model=512`, 8 heads, feed-forward width 2,048, 3 transformer blocks | Channels 128 → 256 → 128; kernels 7 → 5 → 3 |
| Temporal aggregation | Attentive pooling over patch tokens, then a linear classifier | Adaptive average pooling over time, then a linear classifier |
| Output | One logit for the hypotension label | One logit for the hypotension label |

## Temporal resolution

The supervised PatchTST makes its encoder decisions on one-second tokens: each
125-sample patch at 125 Hz becomes one temporal embedding. The FCN does not
tokenize or downsample the input; all three convolutional blocks operate at the
native 125 Hz sample resolution and preserve the full time axis until the final
global average pool. Its kernel spans are therefore:

- kernel 7: 56 ms
- kernel 5: 40 ms
- kernel 3: 24 ms

Thus, PatchTST models long-range relationships between 1-second patch tokens,
whereas the FCN extracts features from the original sample-level signal and
only removes temporal resolution at global average pooling.

The practical tradeoff is therefore:

- **FCN:** much finer temporal detail, with local features formed directly from individual samples and 24–56 ms convolutional windows.
- **PatchTST:** coarser 1-second temporal tokens, but explicit attention over all 1,800 tokens to model long-range relationships before classification.
- **InceptionTime:** also operates at the native 125 Hz (8 ms) sample resolution. Each Inception module uses parallel 39-, 19-, and 9-sample convolutions plus a 3-sample max-pooling branch—approximately 312, 152, 72, and 24 ms—while stacked modules expand the effective receptive field.

## Supervised PatchTST flow

```text
[B, 3, 225000]
        │
        ├─ split into 1800 × 125-sample patches
        ├─ tokenize each channel into d_model-dimensional patch embeddings
        ├─ add rotary position information
        ├─ transformer self-attention blocks over the patch sequence
        ├─ attentive classifier pools the contextualized patches per channel
        └─ linear projection → [B, 1] logit
```

The current supervised runs use `tokenizer_type: "simple"`. This is the
`TS_Tokenizer` implementation: a `Conv1d` whose 125-sample kernel and 125-sample
stride create one embedding per non-overlapping one-second patch. Because
`shared_embedding=False`, the convolution is grouped by input channel, so ABP,
ECG lead II, and PLETH receive separate channel-specific projections before the
transformer processes their patch tokens.

The supervised model reuses the PatchTST encoder but disables its reconstruction
head (`pretrain_head=False`). It supports two classifier choices:

- `attentive` (the full configuration): learned attentive pooling followed by a
  linear projection.
- `mean`: temporal mean pooling followed by a linear projection.

The compact comparison run uses `d_model=96`, two transformer blocks, four heads,
and the `mean` classifier. It has 260,281 trainable parameters, approximately
matching the 265,985-parameter FCN configuration while retaining temporal
self-attention.

## FCN flow

```text
[B, 3, 225000]
        │
        ├─ Conv1d(3 → 128, kernel 7) + BatchNorm + ReLU
        ├─ Conv1d(128 → 256, kernel 5) + BatchNorm + ReLU
        ├─ Conv1d(256 → 128, kernel 3) + BatchNorm + ReLU
        ├─ adaptive average pool over the full time axis
        └─ linear projection → [B, 1] logit
```

All convolutions use same padding, so the temporal length is preserved until
global average pooling. The FCN therefore builds local features with a small
stack of convolutional receptive fields and discards temporal order at the
final average-pooling step.

## Practical distinction

PatchTST first compresses the long waveform into one-second tokens and models
relationships between those tokens with attention. FCN processes every sample
with local convolutions and uses global average pooling for classification. The
compact PatchTST/FCN comparison controls approximately for parameter count, but
the models still differ in inductive bias: attention over patch tokens versus
local convolution followed by order-invariant pooling.

## Supervised PatchTST full-run test results

For reference, the completed full supervised PatchTST hypotension run achieved
the following on 127,811 test windows (threshold 0.5):

| Metric | Test value |
|---|---:|
| AUROC | 0.8688 |
| Average precision (AP) | 0.2766 |
| Accuracy | 0.6702 |
| F1 | 0.1851 |
| Recall / sensitivity | 0.8713 |
| Specificity | 0.6611 |

The confusion matrix was TN=80,866, FP=41,451, FN=707, TP=4,787.

## FCN full-run test results

The completed FCN hypotension run used the same 127,811-window test split and
the epoch-12 checkpoint selected by validation average precision. At the
default 0.5 probability threshold, its results were:

| Metric | Test value |
| --- | ---: |
| AUROC | 0.7903 |
| Average precision (AP) | 0.1911 |
| Accuracy | 0.4523 |
| F1 | 0.1242 |
| Recall / sensitivity | 0.9037 |
| Specificity | 0.4320 |

The confusion matrix was TN=52,838, FP=69,479, FN=529, TP=4,965. The
[FCN replication report](FCN_PAPER_REPLICATION.md) records the 95% bootstrap
confidence intervals and comparison with the published result.
