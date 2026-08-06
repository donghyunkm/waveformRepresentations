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

The answer to “is this how the original code does it?” is yes: the original
repository masks the training input but computes reconstruction loss over all
patches, and validates without masking. The answer to “is this exactly what the
paper says?” is no: the paper describes loss restricted to masked patches.
Matching the repository and matching the paper are therefore different targets.

No source code was changed during this review; this file records the current
status and conclusions only.
