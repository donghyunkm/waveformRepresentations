# Slides / Presentations

## Overview

Presentation slides summarizing PhysioJEPA project results, architecture, and
findings. Built with [Marp CLI](https://github.com/marp-team/marp-cli) from
Markdown source.

## Files

| File | Purpose |
|------|---------|
| `slides/physiojepa_results.md` | Marp Markdown source |
| `slides/physiojepa_results.html` | Generated HTML (do not edit directly) |

## Generation

```bash
module load nodejs/22.9.0
npx @marp-team/marp-cli slides/physiojepa_results.md -o slides/physiojepa_results.html
```

Run from the repo root. The HTML must be regenerated after any Markdown changes.

## Slide Deck Structure (as of 2025-08-17)

1. **Title** — project overview, models, data, task
2. **JEPA architecture** — core idea, 3 components, masking, training, rationale (class: small)
3. **PatchTST architecture** — masked reconstruction, channel independence, key difference from JEPA (class: tiny)
4. **Models Overview** — comparison table of all 4 models
5. **Data & Patient-Level Split** — split table, subject disjointness
6. **Hypotension Prediction: Task Definition** — label definition, input, challenge, evaluation
7. **Hypotension Prediction: Results** — main results table
8. **Frozen-Encoder Probes: Detailed Metrics** — JEPA and PatchTST probe metrics with F1/precision/recall (class: small)
9. **Hypotension Prediction: Key Takeaways** — 3 points: SSL nearly matches supervised, supervised wins in label-abundant regime, multi-scale didn't help (class: small)
10. **Medical Feature Probing: Overview** — method, raw stats baseline (class: small)
11. **Medical Feature Probing: Results** — R² table with JEPA vs PatchTST vs Raw Stats
12. **Medical Feature Probing: Interpretation** — temporal/morphological vs absolute levels
13. **Representation Analysis: Clustering Metrics** — metric definitions table (class: small)
14. **Representation Analysis: K-Means Clustering** — setup (PCA pipeline), k-sweep table, findings (class: small)
15. **Representation Analysis: Key Findings** — patient fingerprints, low silhouette
16. **Limitations** — patient identity encoding, SSL underperformance in label-abundant regime
17. **Next Steps** — longer context windows, contrastive objectives (class: small)

## Style Configuration

- Default font: 21px
- Tables: 18px
- `section.small`: 19px (for dense slides)
- `section.tiny`: 17px (for the densest slides — currently PatchTST)
- 16:9 aspect ratio, paginated

## Content Decisions (2025-08-17 session)

- **Combined two JEPA slides** (architecture + training) into one — they had
  significant overlap.
- **Added PatchTST architecture slide** emphasizing channel independence as the
  key architectural distinction from standard transformers (no cross-channel
  attention; channels only interact at the classification head).
- **Key Takeaway #2 reframed**: Supervised PatchTST outperforms SSL in this
  label-abundant regime (~1M labeled samples). SSL advantage expected in
  low-label/transfer settings.
- **Limitations slide** (renamed from "Rooms for improvement"): Two specific,
  evidence-backed points with cited metrics (ARI, silhouette, AUROC gap).
- **Next Steps improved**: 
  - Longer context windows (1–4 hrs) tied to specific clinical tasks (sepsis,
    ventilator weaning, decompensation) and the label-scarcity hypothesis.
  - Contrastive objectives to disentangle patient identity from hemodynamic
    state, directly addressing the clustering findings.
- **Peak count clarification**: Added parenthetical noting scipy `find_peaks`
  with 0.4s min distance, normalized by window length.
- **K-Means setup clarified**: Now explicitly states the PCA reduction pipeline
  (temporal subsampling 1800→20, PCA 512→121, StandardScaler → 7,260-d).
