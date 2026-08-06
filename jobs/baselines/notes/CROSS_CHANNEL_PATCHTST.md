# Supervised PatchTST with cross-channel attention

This experiment is a controlled architectural ablation of the standalone
supervised PatchTST baseline. Its data, tokenizer, temporal transformer,
attentive classifier, loss, optimizer, scheduler, augmentations, fixed subject
splits, and sample caches are unchanged.

The encoder adds one residual cross-channel self-attention layer after each
existing temporal transformer layer:

```text
[batch, channels, patches, embedding]
  -> temporal attention as [batch * channels, patches, embedding]
  -> channel attention as [batch * patches, channels, embedding]
  -> repeat for each temporal layer
```

Channel attention has no feed-forward sublayer, channel embedding, or rotary
position encoding. It reuses the temporal model's head count, attention
dropout, residual dropout, query/key/value bias, and normalization convention.

The base supervised PatchTST has 12,805,217 parameters. The cross-channel model
has 15,960,161 parameters; all 3,154,944 added parameters belong to the three
cross-channel attention layers. Every original encoder and classifier parameter
name and shape remains unchanged.

## Staged runs

Smoke test:

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_cross_channel_hypotension_smoke.sbatch
```

Fixed 10% subject subset:

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_cross_channel_hypotension_subset10.sbatch
```

Full fixed cohort:

```bash
sbatch jobs/baselines/slurm/supervised_patchtst_cross_channel_hypotension_full.sbatch
```

The scripts are pinned to `a100-4041`, the A100 node that passed the prior CUDA
smoke test. A submitted job may remain pending while that node is reserved.
