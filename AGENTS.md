# AGENTS.md

## Scope

These instructions apply to the entire PhysioJEPA repository.

This checkout is for local experimentation only. Do not commit, push, open pull
requests, or modify remotes unless the user explicitly asks.

## Local environment

Use the existing Conda environment named `physiojepa`. It is located at:

```text
/gpfs/home/dk5565/.conda/envs/physiojepa
```

Initialize Conda and activate the environment with:

```bash
source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
cd /gpfs/home/dk5565/PhysioJEPA
```

The environment uses Python 3.10 and contains an editable installation of this
checkout. It includes the dependencies declared in `settings.ini`, PyTorch
2.6.0 with CUDA 12.4 support, and `wfdb`, which is required by
`jobs/convert_to_zarr.py` but is not declared by the upstream package.

For non-interactive commands, either activate the environment first or invoke:

```text
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python
```

The registered Jupyter kernel is named `physiojepa` and is displayed as
`Python (PhysioJEPA)`.

The committed notebooks currently record the generic `python3` kernelspec.
When working interactively, select `Python (PhysioJEPA)` and avoid notebook
metadata or output churn unrelated to the requested change.

Do not create another virtual environment or reinstall the full dependency set
unless the user explicitly requests it.

## Repository model

PhysioJEPA is an nbdev repository:

- `nbs/` contains the source-of-truth notebooks.
- `physiojepa/` contains Python modules generated from those notebooks.
- `jobs/` contains data-processing and training entry points with YAML configs.
- `_docs/` contains generated documentation.
- `_proc/` contains intermediate documentation/notebook processing output.
- `settings.ini` contains package metadata and dependency declarations.

The notebook-to-module map is:

- `00_bedside.ipynb` -> `bedside.py`: Zarr-backed forecasting and
  self-supervised datasets.
- `03_signal.ipynb` -> `signal.py`: filtering, resampling, normalization, ABP
  beat detection, feature extraction, and signal-quality processing.
- `05_utils.ipynb` -> `utils.py`: truncated-normal initialization.
- `06_train.ipynb` -> `train.py`: downstream linear-probing/fine-tuning
  Lightning wrapper.
- `07_loss.ipynb` -> `loss.py`: reconstruction, variance, cross-entropy, and
  focal losses.
- `08_data_preprocessing.ipynb` -> `data_preprocessing.py`: sample indexing,
  signal validity checks, interpolation, and multiprocessing helpers.
- `09_augmentations.ipynb` -> `augmentations.py`: patch/value/channel
  augmentations and training callbacks.
- `10_layers.ipynb`, `11_heads.ipynb`, and `18_tokenizers.ipynb` -> reusable
  transformer layers, attentive classifiers, and waveform tokenizers.
- `12_jepa.ipynb` -> `jepa.py`: native PhysioJEPA and ECG-JEPA encoders,
  predictors, masking, EMA target encoders, and Lightning modules.
- `13_baselines.ipynb` -> `baselines.py`: FCN baseline and generic supervised
  Lightning wrapper.
- `17_patchtst.ipynb` -> `patchtst.py`: PatchTST-style masked reconstruction
  encoder and Lightning module.

When changing library behavior, make the primary change in the corresponding
notebook under `nbs/`, then regenerate the package with:

```bash
nbdev_prepare
```

Avoid editing generated files in `physiojepa/` alone because a later nbdev
export can overwrite those edits. If a generated module must be patched for a
quick experiment, clearly identify it as temporary.

## Verification

Use checks proportional to the change. Useful baseline checks are:

```bash
python -m pip check
python -c "import physiojepa, torch; print(torch.__version__)"
```

For library changes, import the affected module and run the relevant nbdev
notebook tests. `nbdev_prepare` may regenerate tracked files, so inspect
`git status --short` afterward and do not discard unrelated user changes.

There is no conventional `tests/` suite. The notebook suite mostly checks that
definitions execute rather than asserting detailed behavior. Run it with:

```bash
MPLCONFIGDIR=/tmp/physiojepa-mpl \
nbdev_test --path nbs --n_workers 0 --pause 0
```

Serial execution is important in restricted environments where nbdev's
multiprocessing manager cannot create a socket. Add focused tensor-shape and
numerical checks for changed behavior instead of relying only on notebook
execution. To check notebook/export drift without modifying the working tree,
export a temporary copy of the repository and compare its `physiojepa/`
directory with this checkout.

PyTorch reports CUDA availability only when running on a node with a visible
GPU. A `False` result from `torch.cuda.is_available()` on a login or CPU node
does not by itself indicate a broken installation.

## Architecture and data flow

The standard experiments use 30-minute windows of ABP, ECG lead II, and PLETH
at 125 Hz. Signals are read lazily from per-record Zarr stores, interpolated,
optionally filtered, IQR-normalized, and divided into patches.

The three self-supervised families are:

- Native JEPA: a context encoder predicts masked latent target-encoder
  representations; the target encoder is updated by EMA.
- PatchTST: masked waveform patches are reconstructed from transformer
  representations.
- ECG-JEPA: channel/time patch tokens use joint positional embeddings and a
  masked latent predictor.

Downstream jobs load a pretrained Lightning checkpoint, normally freeze its
encoder, apply an attentive pooling classifier, and predict one label per
forecast horizon. FCN jobs train directly from waveforms as supervised
baselines. Hypotension and shock-index jobs split by subject with
`StratifiedGroupKFold`; self-supervised jobs use `GroupShuffleSplit`.

## Known current-tree caveats

Do not silently work around these when interpreting results. Fix them in the
source notebooks and add focused regression checks if the user requests
corrective work.

- The native JEPA inference/encoder path runs, but the standard unshared-channel
  masked training path currently fails: `apply_masks` returns from inside its
  first loop iteration, collapsing the effective batch and causing a predictor
  shape mismatch.
- `loss.mape` currently raises `AttributeError` because its `clamp` call is
  misspelled.
- `MultiHeadAttention.forward` accepts `mask` but does not pass it to scaled
  dot-product attention. Code that constructs an attention mask, including the
  ECG-JEPA channel/time mask, therefore does not currently enforce it.
- `GeneralTimeSupervised.configure_optimizers` compares `optimizer_type` with
  lowercase `"adamw"` without normalizing it. The baseline YAML value
  `"AdamW"` consequently selects `Adam`.
- The supervised scripts read `training.perform_cv`, but do not use it to run
  multiple folds. They train only the first nested stratified group split.

## Jobs, data, and paths

All derived data and run artifacts for this checkout must be written under:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA
```

This includes converted Zarr stores, labels, sample-index caches, manifests,
checkpoints, prediction tensors, logs, and W&B files. Do not place these
artifacts in the repository or alongside the read-only source waveforms at
`/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched`. Prefer descriptive
subdirectories such as `zarr/`, `labels/`, `sample_indices/`, `models/`,
`predictions/`, `logs/`, and `manifests/` beneath the required output root.

The containing GPFS fileset has a tight shared file-count quota. Processed
waveforms must therefore use a containerized layout, normally one chunked
container per ICU stay, rather than a Zarr `DirectoryStore` that creates one
filesystem entry per chunk. Suitable implementations include Zarr `ZipStore`
or chunked HDF5; choose and document the exact backend before conversion and
update readers accordingly. Write containers atomically and keep a resumable
manifest. Do not start conversion, labeling, sample generation, or training
jobs without a separate explicit user request.

Training scripts load YAML files by filename relative to the current working
directory. Run them from their own directory, for example:

```bash
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa
python train_patch_jepa.py
```

Review the associated YAML before launching a job. Its data, model, label, and
checkpoint paths are relative and must point to locally available resources.

The training scripts execute substantial setup at import time, including
reading their YAML, discovering data, loading checkpoints, creating output
directories, and constructing an online W&B logger. Treat them as entry points,
not import-safe modules. Most trainers hard-code GPU acceleration, DDP,
synchronized batch normalization, persistent data-loader workers, and online
W&B logging.

Sample-index dataframes are cached as gzip CSV files under `models_dir` and
reused based only on `dataset_filename`. If data, split seeds, channels,
sampling rules, or label logic changes, use a new descriptive
`dataset_filename` or explicitly confirm that an existing cache is compatible.
Do not delete an existing cache without user approval. Check each script rather
than assuming every YAML key is honored; some configuration fields are
currently unused or overridden in Python.

The cluster-ready data-preparation pipeline is documented in
`jobs/data_processing/README.md`. It uses separately submitted Slurm stages to
build a manifest, convert array-task slices to one Zarr `ZipStore` per stay,
validate conversion reports, extract minute-level labels in an array, and
merge label shards. The submission helper never chains stages. The older
`jobs/convert_to_zarr.py` and
`jobs/label_processing/create_hypotension_outcome_df.py` use directory stores
or workstation-style multiprocessing and are not suitable for this fileset.

After minute-label extraction,
`create_hypotension_shock_labels.ipynb` requires records at least two hours
long and creates event labels from five consecutive positive minutes.
Supervised jobs then turn those events into preceding waveform/forecast
samples, cache the sample tables, train, checkpoint, and save validation/test
prediction tensors. Current training readers still assume directory-backed
Zarr paths and must be updated to open `ZipStore` containers before training
against the cluster-ready output.

The repository does not include MIMIC waveform data, derived Zarr stores,
outcome-label files, or pretrained checkpoints. Do not launch training or data
conversion unless the required inputs are present. Do not download restricted
clinical data or authenticate to Weights & Biases on the user's behalf without
an explicit request.

Training is GPU- and resource-intensive. Do not run full training on a login
node; use the cluster's scheduled GPU resources when the user requests a run.

## Working-tree safety

- Preserve existing user changes and keep unrelated files untouched.
- Check `git status --short` before and after material changes.
- Do not commit or push for this local experimentation checkout.
- Keep datasets, cached sample tables, checkpoints, W&B runs, prediction
  tensors, model outputs, and credentials out of Git.
- Do not edit shell startup files merely to activate this environment.
