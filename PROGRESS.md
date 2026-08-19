# Progress

Concise daily project log. Detailed notes live in [`docs/`](docs/).

## Current priorities

- [ ] Monitor supervised PatchTST vasopressor-free training (job 26576169, a100_short)
- [ ] Monitor JEPA physio-contrastive pretraining (job 26497397)
- [ ] Analyze vasopressor-free PatchTST results when job completes
- [ ] Bootstrap CIs for JEPA vs PatchTST probe metrics

## Active jobs

| Job ID | Name | Partition | Status | Purpose |
|--------|------|-----------|--------|---------|
| 26576169 | physiojepa-sptst-nopressor | a100_short | Running (a100-4007) | Supervised PatchTST on vasopressor-free cohort |
| 26497397 | physiojepa-physio-contrastive | gl40s_short | Running (gl40s-8008) | Contrastive JEPA pretraining |

---

## 2026-08-18

- **Vasopressor-free cohort created** (stay-level exclusion): Queried MIMIC INPUTEVENTS_MV/CV for 19 vasopressor item IDs, matched 2,831 waveform stays to vasopressor ICU stays via temporal overlap (±1h tolerance). Remaining cohort: 1,460 subjects, 982k samples, 1.94% prevalence (vs original 4.3%). Re-split with corrected stratified group 10-fold (seed=16) for balanced prevalence across train/val/test. [Details](docs/architecture.md#vasopressor-free-cohort-stay-level-exclusion)
  - Scripts: `jobs/data_processing/scripts/exclude_vasopressor_stays.py`, `resplit_vasopressor_free.py`
  - Outputs: `manifests/hypotension_subject_split_vasopressor_free_stays_v1.csv`, `labels/hypotension_labels_vasopressor_free_stays_v1.csv.gz`
- **Supervised PatchTST vasopressor-free training**: Debugged and fixed manifest validation failure. 4 subjects (3 train, 1 val) have no valid 30-min windows because required channels are entirely NaN. Relaxed strict check to warning; training now running (job 26576169, a100-4007). [Details](docs/vasopressor_free_cohort.md#failure-analysis-and-fix-2026-08-18)
  - Failed jobs: 26574888 (manifest RuntimeError), 26575971 (bad GPU node a100-4026)
  - Fix: `fcn_baseline_hypotension.py` (warning instead of error for missing subjects), YAML `strict_cohort_match: false`, sbatch exclude list updated

### Next steps

- [ ] Monitor job 26576169; confirm it produces first checkpoint (~30 min)
- [ ] Compare vasopressor-free PatchTST AUROC/AP to full-cohort result (0.8688/0.2766)
- [ ] If successful, create vasopressor-free JEPA pretraining + downstream probe configs
- [ ] Consider whether lower prevalence (1.94% vs 4.3%) warrants adjusted class weights or focal loss gamma

---

## 2026-08-17

- **Slide deck improvements** (`slides/physiojepa_results.md`): Combined overlapping JEPA slides into one; added PatchTST architecture slide (emphasizing channel independence, no cross-channel attention); improved Key Takeaways, Limitations, and Next Steps with specific metrics and framing; clarified K-Means PCA pipeline; added font-size classes for overflow control. [Details](docs/slides.md)
- Created `docs/slides.md` documenting slide structure, generation commands, and content decisions.

### Next steps

- [ ] Monitor JEPA d64 downstream evaluation jobs (2 running on gl40s_short)
- [ ] Request `qos_a100_short` from cluster admins
- [ ] Analyze d64 downstream results when remaining jobs complete
- [ ] Bootstrap CIs for JEPA vs PatchTST probe metrics
- [ ] PatchTST slide still slightly overflows at `tiny` (17px) — may need content trim or split

---

## 2026-08-16

- **JEPA d64 medfeat probe completed** (job 26487639, 17 min on L40S): Mean R²=**0.26** (vs d512=0.51). PLETH morphology preserved (0.94); ABP hemodynamics severely degraded by 8× compression. [Details](docs/downstream_feature_prediction.md)
- **JEPA d64 hypotension probe running** (job 26487623, gl40s-8005): Epoch 0 val_auroc=**0.812**, val_auprc=0.205. Training epoch 1 at ~4.3 it/s. [Details](docs/downstream_hypotension_prediction.md)
- **JEPA d64 homogeneity sweep running** (job 26487661, gl40s-8006): Embedding extraction phase. [Details](docs/representation_analysis.md)
- **JEPA d64 pretraining completed** (job 26381948): 100/100 epochs in 2d 12h on A100. Best val_loss=**0.18978** at epoch 40. Unlike full JEPA (diverged at epoch 13), d64 trained stably through all epochs. [Details](docs/pretraining.md)
- **Cluster QOS policy change**: All `a100_*`, `gpu4_*`, `gpu8_*` partitions now require their own QOS (e.g. `qos_a100_short`). User account only has `normal` QOS, restricting GPU access to `gl40s_*` (L40S) partitions only. Previous jobs ran before this restriction was enforced. [Details](docs/infrastructure.md)
- Created and submitted 3 downstream evaluation jobs for JEPA d64:
  1. **Attentive hypotension probe** — same config as full JEPA probe, d64 encoder. [Details](docs/downstream_hypotension_prediction.md)
  2. **Mean-pooled Ridge medical feature probe** — 15 physiological features, 10k windows. [Details](docs/downstream_feature_prediction.md)
  3. **K-Means homogeneity sweep** — k=2–50, patient/hemo/hypo alignment. [Details](docs/representation_analysis.md)
- All 3 jobs queued on `gl40s_short` (0 idle nodes currently; waiting on priority).

### Next steps

- [ ] Request QOS access for A100 partition from cluster admins.
- [ ] When d64 probe completes, compare final AUROC/AP to full JEPA (0.843/0.265).
- [ ] When d64 homogeneity sweep completes, compare patient vs hemo vs hypo alignment to d512.
- [ ] If d64 approaches full JEPA downstream → strong efficiency argument for the paper.
- [ ] If d64 underperforms → quantify the capacity-performance tradeoff across all 3 evaluation axes.
- [ ] Bootstrap CIs on existing JEPA/PatchTST hypotension AUROC/AP.

---

## 2026-08-14

- Hemo cluster mean-pooled linear probe (precomputed embeddings, 22k test windows): JEPA AUROC **0.524**, PatchTST **0.532** — both at chance. Confirms attentive probe failure is not architecture-specific. [Details](docs/downstream_cluster_prediction.md)
- Verified label alignment chain: embeddings ↔ test CSV ↔ hemo labels all positionally aligned (100% patient ID match). Median temporal offset to icuDataExtraction: 38s.
- Created `probe_hemo_clusters.py` (full pipeline) and `probe_hemo_clusters_precomputed.py` (uses cached embeddings, runs in seconds on CPU).
- JEPA d64 pretraining healthy: epoch 32/100, val_loss 0.203 (down from 0.51 at epoch 8). On track to finish ~Aug 15 evening.
- Refactored `AGENTS.md`: moved volatile knowledge to `docs/known_issues.md` and `docs/architecture.md`, added experimental integrity and workflow sections.
- Trimmed `PROGRESS.md` from ~230 to ~90 lines.

- Job `26417288` (hemo cluster full probe) **timed out** after 4h on `gl40s_dev`. Waveform cache written (8.4 GB, 10k windows). Did not reach raw stats or encoder inference. Resubmit with `--skip-extraction` + longer wall-time if raw stats baseline is needed.

### Next steps

- [ ] Identify mixed-label patients for within-patient clustering.
- [ ] Bootstrap CIs on JEPA/PatchTST hypotension AUROC/AP.
- [ ] Monitor JEPA d64 completion (~Aug 15 evening).
- [ ] (Optional) Resubmit raw stats baseline with longer wall-time or vectorized computation.

---

## 2026-08-13

- PatchTST frozen attentive hypotension inference complete: test AUROC/AP **0.8296/0.2344**. [Details](docs/downstream_hypotension_prediction.md)
- JEPA frozen attentive hypotension confirmed: test AUROC/AP **0.8431/0.2653**.
- Single-patient clustering (p072908): ARI 0.247 at k=5; all-negative labels made hypotension metrics undefined. [Details](docs/representation_analysis.md)
- JEPA d64 pretraining started (job 26381948). Updated slides.

### Known bugs/caveats

- `train_medical_features_fixed.py` logs only aggregate R², not per-feature.
- PatchTST inference ~0.22 it/s vs JEPA ~1.4 it/s; cause unresolved.
- Upstream code bugs documented in [`docs/known_issues.md`](docs/known_issues.md).

---

## 2026-08-12

- JEPA attentive probe reached val_auroc=**0.865** at epoch 15. [Details](docs/downstream_hypotension_prediction.md)
- Created hemo cluster probe (7-class) and medical feature probe (15 features). [Details](docs/downstream_cluster_prediction.md), [Details](docs/downstream_feature_prediction.md)
- Representation analysis: intrinsic dim ~7–9D, K-Means dominated by patient identity. [Details](docs/representation_analysis.md)
- JEPA medfeat probe epoch 0: aggregate val_R²=0.639 (surpasses Ridge baseline 0.51). [Details](docs/downstream_feature_prediction.md)
- Fixed timestamp alignment bug, checkpoint fingerprint mismatches, OOM resubmissions.

---

## 2026-08-11

- Implemented physio-contrastive JEPA. [Details](docs/contrastive_learning.md)
- CKA analysis confirms patient fingerprint is location, not structure. [Details](docs/representation_analysis.md)
- Linear probe generalizability: all cross-patient results at chance (mean-pooled).

---

## 2026-08-10

- Local SSD caching pipeline (8.5x speedup). [Details](docs/infrastructure.md)
- Representation analysis suite. [Details](docs/representation_analysis.md)
- Medical feature probing (Ridge, mean-pooled). [Details](docs/medical_feature_probing.md)
- Key finding: embeddings are patient fingerprints; within-patient signal exists.

---

## 2026-08-07

- Pretraining loss analysis. [Details](docs/pretraining.md)
- Downstream attentive probe configs created.
- Embedding visualization (UMAP, t-SNE). [Details](docs/representation_analysis.md)

---

## 2026-08-06

- Multi-scale PatchTST tokenizer. [Details](docs/downstream_hypotension_prediction.md)
- JEPA and PatchTST pretraining monitored. [Details](docs/pretraining.md)
