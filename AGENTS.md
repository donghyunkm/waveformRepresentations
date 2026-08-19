# AGENTS.md

## Scope and core safety

These instructions apply to the entire PhysioJEPA repository.

This checkout is for local experimentation only. Do not commit, push, open pull
requests, or modify remotes unless the user explicitly asks.

## Session workflow

At the beginning of every session:

1. Read `PROGRESS.md` to understand current priorities, active jobs, and
   immediate next steps.
2. Read relevant `docs/*.md` files for the workstream you will be working on.

Sources of truth:

```text
PROGRESS.md = source of truth for current project state (priorities, jobs, progress)
docs/*.md   = source of truth for detailed technical/project knowledge
```

## Before modifying code

1. Read `PROGRESS.md`.
2. Read the relevant `docs/*.md` workstream document.
3. Check `git status --short`.
4. Inspect the existing implementation before proposing a replacement.
5. For library code, identify the source notebook under `nbs/`; do not treat
   generated `physiojepa/*.py` files as the primary source.
6. Make the smallest change that solves the requested problem.
7. Run focused verification before broader tests.
8. Update documentation if the work materially changes project state or knowledge.

## Local environment

Use the existing Conda environment named `physiojepa`:

```text
/gpfs/home/dk5565/.conda/envs/physiojepa
```

Initialize and activate with:

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

The registered Jupyter kernel is named `physiojepa` (displayed as
`Python (PhysioJEPA)`). The committed notebooks record the generic `python3`
kernelspec. When working interactively, select `Python (PhysioJEPA)` and avoid
notebook metadata or output churn unrelated to the requested change.

Do not create another virtual environment or reinstall the full dependency set
unless the user explicitly requests it.

## Repository structure (nbdev)

PhysioJEPA is an nbdev repository:

- `nbs/` — source-of-truth notebooks.
- `physiojepa/` — Python modules generated from those notebooks.
- `jobs/` — data-processing and training entry points with YAML configs.
- `_docs/` — generated documentation.
- `_proc/` — intermediate documentation/notebook processing output.
- `settings.ini` — package metadata and dependency declarations.

The notebook-to-module map is:

| Notebook | Module | Purpose |
|----------|--------|---------|
| `00_bedside.ipynb` | `bedside.py` | Zarr-backed forecasting and SSL datasets |
| `03_signal.ipynb` | `signal.py` | Filtering, resampling, normalization, ABP beat detection |
| `05_utils.ipynb` | `utils.py` | Truncated-normal initialization |
| `06_train.ipynb` | `train.py` | Downstream linear-probing/fine-tuning Lightning wrapper |
| `07_loss.ipynb` | `loss.py` | Reconstruction, variance, CE, and focal losses |
| `08_data_preprocessing.ipynb` | `data_preprocessing.py` | Sample indexing, validity checks, interpolation |
| `09_augmentations.ipynb` | `augmentations.py` | Patch/value/channel augmentations and callbacks |
| `10_layers.ipynb` | `layers.py` | Reusable transformer layers |
| `11_heads.ipynb` | `heads.py` | Attentive classifiers |
| `12_jepa.ipynb` | `jepa.py` | PhysioJEPA and ECG-JEPA encoders, predictors, masking, EMA |
| `13_baselines.ipynb` | `baselines.py` | FCN baseline and supervised Lightning wrapper |
| `17_patchtst.ipynb` | `patchtst.py` | PatchTST masked reconstruction encoder |
| `18_tokenizers.ipynb` | `tokenizers.py` | Waveform tokenizers |

When changing library behavior, edit the corresponding notebook under `nbs/`,
then regenerate the package:

```bash
nbdev_prepare
```

Do not edit generated files in `physiojepa/` alone — a later `nbdev_prepare`
will overwrite those edits. If a generated module must be patched for a quick
experiment, clearly identify it as temporary.

## Slide generation

Presentation slides are created from Markdown using
[Marp CLI](https://github.com/marp-team/marp-cli). The source lives in
`slides/` as `.md` files with Marp frontmatter.

```bash
module load nodejs/22.9.0
npx @marp-team/marp-cli slides/physiojepa_results.md -o slides/physiojepa_results.html
```

Do not commit `node_modules/` or Marp cache directories.

## Verification

Use checks proportional to the change. Useful baseline checks:

```bash
python -m pip check
python -c "import physiojepa, torch; print(torch.__version__)"
```

For library changes, import the affected module and run the relevant nbdev
notebook tests:

```bash
MPLCONFIGDIR=/tmp/physiojepa-mpl \
nbdev_test --path nbs --n_workers 0 --pause 0
```

Serial execution (`--n_workers 0`) is important in restricted environments
where nbdev's multiprocessing manager cannot create a socket.

Add focused tensor-shape and numerical checks for changed behavior instead of
relying only on notebook execution.

After `nbdev_prepare`, inspect `git status --short` and do not discard
unrelated user changes.

PyTorch reports CUDA availability only when running on a GPU node. A `False`
result from `torch.cuda.is_available()` on a login/CPU node does not indicate
a broken installation.

## Architecture

The project trains self-supervised waveform encoders (JEPA, PatchTST, ECG-JEPA)
and evaluates them on downstream clinical prediction tasks.

For detailed architecture, data flow, model families, and data splits, see
[`docs/architecture.md`](docs/architecture.md).

## Known issues

Before modifying or interpreting model behavior, read
[`docs/known_issues.md`](docs/known_issues.md) for currently known
implementation issues.

Do not silently work around documented issues. If an issue is fixed, update
`docs/known_issues.md` as part of the change.

## Data, outputs, and paths

All derived data and run artifacts must be written under:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA
```

This includes converted Zarr stores, labels, sample-index caches, manifests,
checkpoints, prediction tensors, logs, and W&B files. Prefer descriptive
subdirectories (`zarr/`, `labels/`, `sample_indices/`, `models/`,
`predictions/`, `logs/`, `manifests/`).

Do not place artifacts in the repository or alongside the read-only source
waveforms at `/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched`.

### GPFS file-count constraints

The containing GPFS fileset has a tight shared file-count quota. Processed
waveforms must use a containerized layout (one chunked container per ICU stay)
rather than a Zarr `DirectoryStore`. Suitable implementations include Zarr
`ZipStore` or chunked HDF5. Write containers atomically and keep a resumable
manifest.

### Training entry points

Training scripts load YAML files by filename relative to the current working
directory. Run them from their own directory:

```bash
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa
python train_patch_jepa.py
```

Review the associated YAML before launching a job. These scripts execute
substantial setup at import time (reading YAML, discovering data, loading
checkpoints, creating output directories, constructing an online W&B logger).
Treat them as entry points, not import-safe modules. Most trainers hard-code
GPU acceleration, DDP, synchronized batch normalization, persistent data-loader
workers, and online W&B logging.

### Sample-cache compatibility

Sample-index DataFrames are cached as gzip CSV files under `models_dir` and
reused based only on `dataset_filename`. If data, split seeds, channels,
sampling rules, or label logic change, use a new descriptive
`dataset_filename` or explicitly confirm the cache is compatible. Do not
delete an existing cache without user approval. Check each script rather than
assuming every YAML key is honored; some configuration fields are currently
unused or overridden in Python.

### Data pipeline

The cluster-ready data-preparation pipeline is documented in
`jobs/data_processing/README.md`. Current training readers still assume
directory-backed Zarr paths and must be updated to open `ZipStore` containers
before training against the cluster-ready output.

The repository does not include MIMIC waveform data, derived Zarr stores,
outcome-label files, or pretrained checkpoints. Do not launch training or data
conversion unless the required inputs are present. Do not download restricted
clinical data or authenticate to Weights & Biases on the user's behalf without
an explicit request.

## Slurm and expensive operations

Training is GPU- and resource-intensive. Do not run full training on a login
node; use the cluster's scheduled GPU resources.

You may create or modify Slurm scripts when requested, but **submitting**
(`sbatch`), **cancelling** (`scancel`), requeuing, or otherwise changing
running or queued jobs requires explicit user authorization.

Do not start data conversion, labeling, sample generation, or training jobs
without a separate explicit user request.

## Working-tree safety

- Preserve existing user changes and keep unrelated files untouched.
- Check `git status --short` before and after material changes.
- Do not commit or push for this local experimentation checkout.
- Keep datasets, cached sample tables, checkpoints, W&B runs, prediction
  tensors, model outputs, and credentials out of Git.
- Do not edit shell startup files merely to activate this environment.

## Experimental integrity

- Never report a metric, result, completed job, or successful experiment unless
  it was actually observed in an output, log, artifact, or existing documentation.
- Clearly distinguish proposed experiments, implemented experiments, submitted
  jobs, running jobs, completed jobs, failed jobs, and validated results.
- Record the checkpoint, configuration, and data split associated with reported
  results when known.
- Do not overwrite prior experimental outputs unless explicitly requested.

## Documentation and progress policy

```text
PROGRESS.md = concise chronological project log / dashboard
docs/*.md   = detailed topic-specific project knowledge
```

`PROGRESS.md` should be short: what changed, what's running, what's next,
with links to `docs/` for details. Do not put methodology, results tables,
full commands, or long analyses there.

`docs/` files group related work by workstream (`snake_case` filenames).
Update existing documents rather than creating duplicates.

At the top of `PROGRESS.md`, maintain a small project-status section with
current priorities and active jobs. Keep it short and current. Remove completed
or obsolete items rather than allowing it to grow indefinitely.

### Update documentation as part of meaningful work

At the end of a material work session:

1. Update the relevant `docs/*.md` file with important technical details.
2. Add a concise entry to today's section in `PROGRESS.md`.
3. Update `Current priorities` if priorities changed.
4. Update `Active jobs` if jobs were launched, completed, failed, or replaced.
5. Add clear next-step checkboxes.

Update documentation only for work that materially changes project state,
implementation, experimental results, understanding, or next steps. Do not log
routine inspection, trivial commands, formatting-only changes, or dead-end
exploration unless it reveals something important.
