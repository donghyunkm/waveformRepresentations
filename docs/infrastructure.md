# Infrastructure & Cluster Operations

## Overview

All training and data processing runs on the BigPurple HPC cluster (NYU Langone).
This document covers scheduling strategies, SSD caching, data paths, known
problem nodes, and operational patterns.

---

## Data Paths

### Source Data (read-only)

```
/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched
```

### All Derived Outputs

```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/
├── zarr/                    # Converted ZipStore containers (~510 GB, 4001 files)
├── labels/                  # Minute-level and event-level labels
├── sample_indices/          # Cached sample tables (gzip CSV)
├── manifests/               # Data manifests and subject splits
├── models/                  # Training checkpoints
├── predictions/             # Saved prediction tensors
├── logs/slurm/              # Slurm job output logs
├── figures/                 # Visualization outputs
└── probing/                 # Representation analysis results
    ├── clustering/          # KMeans, kNN, CKA, distance analyses
    └── medical_features/    # Ridge regression probe results
```

---

## Cluster Partitions and Scheduling

### Available Partitions

| Partition | GPU | VRAM | Max Wall Time | QOS | Status |
|-----------|-----|------|---------------|-----|--------|
| `gl40s_dev` | L40S | 48 GB | 4 hours | normal | ✅ Primary (fast start) |
| `gl40s_short` | L40S | 48 GB | 3 days | normal | ✅ Available (often full) |
| `gl40s_long` | L40S | 48 GB | 28 days | normal | ✅ Available (often full) |
| `a100_dev` | A100 | 80 GB | 4 hours | qos_a100_dev | ❌ QOS restricted |
| `a100_short` | A100 | 80 GB | 3 days | qos_a100_short | ❌ QOS restricted |
| `a100_long` | A100 | 80 GB | 28 days | qos_a100_long | ❌ QOS restricted |
| `gpu4_dev` | V100 | 32 GB | 4 hours | qos_gpu4_dev | ❌ QOS restricted |
| `gpu4_short` | V100 | 32 GB | 12 hours | qos_gpu4_short | ❌ QOS restricted |
| `gpu4_medium` | V100 | 32 GB | 3 days | qos_gpu4_short | ❌ QOS restricted |
| `cpu_short` | L40S | 48 GB | varies | normal | QOS GPU limit shared |
| `cpu_dev` | L40S | 48 GB | 4 hours | normal | QOS GPU limit shared |

### QOS Policy Change (2026-08-16)

**Critical**: As of ~Aug 14–16, all `a100_*`, `gpu4_*`, and `gpu8_*` partitions
enforce partition-specific QOS requirements. User account `dk5565`
(account=`system`) only has QOS `normal`, which is now restricted to `gl40s_*`
and `cpu_*` partitions.

The JEPA d64 pretraining (job 26381948, completed Aug 15) was the last job to
successfully run on `a100_short` before the restriction. All subsequent
submissions to `a100_short` fail with:
```
Job's QOS not permitted to use this partition (a100_short allows qos_a100_short not normal)
```

**To restore A100 access**, a cluster admin must run:
```bash
sacctmgr modify user dk5565 set qos+=qos_a100_short
```

Until resolved, all GPU jobs must use `gl40s_*` partitions (L40S, 48 GB VRAM).

### QOS `normal` Limitations

- Account `dk5565` (account=`system`) only has access to QOS `normal`
- QOS `normal` can use `cpu_*` and `gl40s_*` partitions with GPUs
- `cpu_*` partitions share a **single-GPU limit per user** under QOS `normal` —
  an interactive session (`cpu_dev`) blocks all batch GPU jobs on
  cpu_short/cpu_medium/cpu_long
- `gl40s_dev` is the most reliable path for GPU access with fast job starts

---

## SSD Caching (8.5× Speedup)

### Problem

Reading ZipStore containers from GPFS is I/O-limited. Training at ~0.14 it/s
on GPFS vs ~1.2 it/s with local SSD.

### Solution

Copy the ~510 GB of required ZipStore containers (4001 files) to the compute
node's local `/tmp` before training begins.

### Implementation

The `*_local.sbatch` variants:
1. Copy containers from GPFS to `/tmp/physiojepa_containers/` at job start
2. Set `PHYSIOJEPA_CONTAINERS_OVERRIDE=/tmp/physiojepa_containers`
3. Training scripts honor this env var to remap `file_path` entries in the
   sample cache to the local directory
4. Cleanup happens automatically on job exit

### Performance

| Metric | GPFS | Local SSD |
|--------|------|-----------|
| Training speed | 0.14 it/s | 1.2 it/s |
| **Speedup** | — | **8.5×** |
| Copy time | — | ~55s (A100), ~725s (L40S) |
| Storage per job | — | ~510 GB in `/tmp` |

### Environment Variable

```bash
PHYSIOJEPA_CONTAINERS_OVERRIDE=/tmp/physiojepa_containers
```

When set, training scripts remap sample cache file paths to this directory.
When unset, scripts work unchanged (read from GPFS).

### Caveats

- Copy must repeat on every resubmission (data is ephemeral in `/tmp`)
- If two jobs land on the same node and both copy 510 GB, they use ~1 TB of
  local storage (nodes have 12 TB, so typically fine)
- The overhead is acceptable: 55s copy vs hours of training

---

## Dependency Chain Pattern

When `gl40s_dev` has free GPUs but only allows 4-hour jobs, use `afterany`
dependency chains to automatically resubmit after wall-time expiry.

### Pattern

```bash
# Start the first job
FIRST=$(sbatch --parsable --partition=gl40s_dev --time=4:00:00 --gres=gpu:l40s:1 \
  /path/to/your.sbatch)

# Chain N follow-ups (each starts after previous finishes for any reason)
PREV=$FIRST
for i in $(seq 1 5); do
  PREV=$(sbatch --parsable --dependency=afterany:$PREV \
    --partition=gl40s_dev --time=4:00:00 --gres=gpu:l40s:1 \
    /path/to/your.sbatch)
done
```

### Key Details

- `afterany` = start regardless of whether predecessor succeeded, failed, or
  timed out. Correct for checkpoint-resume workflows.
- The sbatch script must auto-detect and resume from the latest checkpoint
  (e.g., `find_resume_checkpoint`). No manual path updates needed.
- 6 × 4h = 24h total capacity per chain.
- To extend a chain that's about to exhaust, find the last job ID and append.

### Active Chain Examples (Session 14)

```
JEPA probe:         26285628 → 26286512 → 13 → 14 → 15 → 16  (gl40s_dev)
PatchTST probe:     26285629 → 26286517 → 18 → 19 → 20 → 21  (gl40s_dev)
Physio-Contrastive: 26294860 → 61 → 62 → 63 → 64 → 65        (gl40s_dev)
```

### Caveats

- If a job crashes (not timeout), the chain continues — check logs afterward
- `gl40s_dev` nodes: gl40s-8003 (fast, often idle), gl40s-8002 (mixed, can
  have slow I/O)
- L40S has 48 GB VRAM — batch sizes that fit on A100 (80 GB) may need halving
  with 2× gradient accumulation

---

## Known Bad Nodes

| Node | Partition | Issue |
|------|-----------|-------|
| gn-0015 | gpu4_short | Launch fails with no error output |
| gn-0019 | gpu4_short | Launch fails with no error output |
| a100-4003 | a100_short | GPUs show free but "CUDA device busy" at runtime |
| a100-4029 | a100_short | Previously excluded (intermittent issues) |
| All gn-* | gpu4_* | **Cluster-wide broken** — immediate requeue with 0s runtime |

### V100 Node Class Failure

All V100 nodes (gpu4_short, gpu4_medium, gpu4_long) are completely broken as
of 2026-08-10. Every attempt is immediately requeued with 0-second runtime.
This is a cluster-wide GPU prolog/driver issue, not fixable from user side.
Not usable for any jobs.

### Exclude Lists

For jobs on functional partitions, exclude known bad nodes:

```bash
# A100 partitions:
--exclude=a100-4003,a100-4029

# GPU4 partitions (all broken, but if attempting):
--exclude=gn-0015,gn-0019
```

---

## Typical Training Speeds by Hardware

| Task | L40S (gl40s) | A100 | Notes |
|------|-------------|------|-------|
| JEPA pretraining | 1.45 it/s | 0.29 it/s | L40S faster with SSD cache |
| PatchTST pretraining | — | 1.80 it/s | — |
| Attentive probe (JEPA) | 7.72 it/s | — | With SSD cache, ~22 min/epoch |
| Physio-contrastive JEPA | 1.46 it/s | — | With SSD, ~1.6h/epoch |
| Multi-scale PatchTST | — | — | ~2h/epoch on A100 |

---

## Job Monitoring Commands

```bash
# Check all running jobs
squeue -u dk5565 --format="%.10i %.12P %.40j %.2t %.10M %.20R" --sort=i

# Check specific job progress (example for probe)
tail -1 /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/<job_name>-<job_id>.out | tr '\r' '\n' | tail -1

# Check latest checkpoints
ls -lt /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/<model_dir>/*.ckpt | head -3

# Cancel jobs
scancel <job_id>

# Cancel chain
scancel <id1> <id2> <id3> <id4> <id5> <id6>
```

---

## Environment Setup

```bash
source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
cd /gpfs/home/dk5565/PhysioJEPA
```

Python 3.10, PyTorch 2.6.0, CUDA 12.4.

---

## Data Processing Pipeline

The cluster-ready data-preparation pipeline (`jobs/data_processing/`) uses
separately submitted Slurm stages:

1. **Manifest** — build file listing from source waveforms
2. **Zarr conversion** — array-task slices converting to one ZipStore per stay
3. **Validation** — verify conversion reports
4. **Label extraction** — minute-level labels in an array task
5. **Label merging** — merge label shards into final output

The submission helper never chains stages. Each must be verified before the next.

### Container Format

Due to tight GPFS file-count quota, waveforms use **ZipStore** containers (one
chunked container per ICU stay) rather than DirectoryStore (which creates one
filesystem entry per chunk).

### File Counts

- ZipStore containers: ~4,001 files
- Total container storage: ~510 GB

---

## Slurm Script Locations

```
jobs/jepa/slurm/
├── train_hypotension_fixed_local.sbatch       # Attentive probe (SSD cache)
├── train_physio_contrastive_jepa.sbatch        # Physio-contrastive pretraining
├── train_physio_contrastive_jepa_chain.sbatch  # Chain variant with --signal
└── train_patch_jepa_1gpu_debug.sbatch         # JEPA pretraining

jobs/patchtst/slurm/
├── train_hypotension_fixed_local.sbatch       # PatchTST probe (SSD cache)
└── train_patchtst_1gpu.sbatch                 # PatchTST pretraining

jobs/baselines/slurm/
├── supervised_patchtst_multiscale_hypotension_full.sbatch
└── (other baseline variants)

jobs/probing/
├── submit_probe.sbatch                        # Medical feature probing
├── submit_cluster.sbatch                      # Embedding clustering
└── submit_linear_probe_gen.sbatch             # Linear probe generalizability
```
