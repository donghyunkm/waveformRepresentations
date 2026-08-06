# MIMIC-III waveform processing on Slurm

This directory contains a staged, resumable processing pipeline for:

```text
/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched
```

All generated artifacts go below:

```text
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA
```

Nothing in this directory submits work automatically. The submission helper
submits exactly the stage named on its command line and never chains later
stages.

## Layout

- `manifests/waveform_manifest.csv`: deterministic list of records advertising
  ABP, II, and PLETH and having a matching numeric record.
- `containers/<stay>.zarr.zip`: one Zarr v2 `ZipStore` per stay. Each container
  has continuous float32 ABP, II, and PLETH arrays plus 30-minute quality
  arrays. A valid window has no more than 20% identical-or-NaN samples in
  every channel.
- `manifests/conversion_status/task_*.csv`: resumable array-task reports.
- `manifests/conversion_summary.json`: aggregate conversion validation.
- `labels/minute_shards/task_*.csv.gz`: minute-level ABP label shards.
- `labels/hypotension_si_labels_mimic.csv.gz`: merged minute labels.
- `logs/slurm/`: Slurm stdout and stderr.

ZipStore was selected because the shared GPFS fileset is close to its inode
quota. This layout uses roughly one filesystem entry per stay rather than one
entry per Zarr chunk. Containers are built as hidden partial files and renamed
only after metadata consolidation succeeds.

## Resources and array sizing

The default array unit is 10 records per task with at most 32 concurrent tasks.
For an expected manifest of 5,661 records, that is 567 tasks. The cluster
allows arrays of up to 5,001 tasks. Each worker requests one CPU, 8 GiB RAM,
and at most 12 hours on `cpu_short`.

Change the batching or filesystem pressure cap at submission time:

```bash
RECORDS_PER_TASK=5 MAX_CONCURRENT=48 \
  bash jobs/data_processing/slurm/submit_stage.sh convert
```

Using fewer records per task improves load balancing. Increasing concurrency
can reduce wall time but also increases simultaneous reads and writes on GPFS.

## Stages

Run these from the repository root, waiting for and checking each stage before
submitting the next:

```bash
bash jobs/data_processing/slurm/submit_stage.sh manifest
bash jobs/data_processing/slurm/submit_stage.sh convert
bash jobs/data_processing/slurm/submit_stage.sh summarize
bash jobs/data_processing/slurm/submit_stage.sh labels
bash jobs/data_processing/slurm/submit_stage.sh merge
```

The `summarize` stage exits nonzero if reports or manifest rows are missing,
duplicated, unexpected, or failed. Conversion tasks skip a recognized,
complete container, so failed or preempted array indices can be resubmitted.
An incomplete hidden `.partial` file is removed by the next attempt.

Useful checks after a submission are:

```bash
squeue --me
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

To rerun one failed array index, use the same exported settings:

```bash
sbatch --array=ARRAY_INDEX \
  --export=ALL,RECORDS_PER_TASK=10 \
  jobs/data_processing/slurm/convert_array.sbatch
```

The minute-label stage intentionally starts only after conversion validation;
it fails if an assigned container is missing. The merge stage verifies that
the exact shard set implied by the manifest and `RECORDS_PER_TASK` exists
before producing the combined file.

## Important downstream compatibility

These jobs implement conversion and minute-level outcome extraction. The
repository's current training data readers call `zarr.open()` on directory
stores and do not yet open `.zarr.zip` files explicitly. They must be updated
to use `zarr.storage.ZipStore` before training against these containers.
The event-label notebook is also still a separate step after the merged
minute-level file. Do not launch training until both pieces are addressed.
