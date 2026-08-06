"""Paper-style binary event metrics and bootstrap confidence intervals."""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _sensitivity_at_specificity(targets, probabilities, specificity):
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, probabilities)
    eligible = false_positive_rate <= (1.0 - specificity)
    if not np.any(eligible):
        return 0.0
    return float(np.max(true_positive_rate[eligible]))


def binary_event_metrics(targets, logits):
    """Calculate the event-level metrics reported in the PhysioJEPA paper."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = _sigmoid(logits)
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    return {
        'auroc': float(roc_auc_score(targets, probabilities)),
        'average_precision': float(average_precision_score(targets, probabilities)),
        'f1': float(f1_score(targets, predictions, zero_division=0)),
        'recall': float(recall_score(targets, predictions, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) else float('nan'),
        'sensitivity_at_90_specificity': _sensitivity_at_specificity(
            targets, probabilities, 0.90
        ),
        'sensitivity_at_95_specificity': _sensitivity_at_specificity(
            targets, probabilities, 0.95
        ),
    }


def bootstrap_binary_event_metrics(targets, logits, n_resamples=1000, seed=16):
    """Return percentile 95% CIs from event-level bootstrap resampling."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(targets) != len(logits):
        raise ValueError("targets and logits must have the same length")
    if len(np.unique(targets)) != 2:
        raise ValueError("binary metrics require both target classes")

    point = binary_event_metrics(targets, logits)
    values = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    completed = 0
    attempts = 0
    max_attempts = max(n_resamples * 2, n_resamples + 10)
    while completed < n_resamples and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, len(targets), size=len(targets))
        resampled_targets = targets[indices]
        if len(np.unique(resampled_targets)) != 2:
            continue
        metrics = binary_event_metrics(resampled_targets, logits[indices])
        for name, value in metrics.items():
            values[name].append(value)
        completed += 1
    if completed != n_resamples:
        raise RuntimeError(
            f"Only completed {completed} of {n_resamples} bootstrap resamples"
        )

    result = {
        'n_events': int(len(targets)),
        'positive_events': int(targets.sum()),
        'negative_events': int((targets == 0).sum()),
        'bootstrap_unit': 'event',
        'bootstrap_resamples': int(n_resamples),
        'bootstrap_seed': int(seed),
        'metrics': {},
    }
    for name, point_value in point.items():
        lower, upper = np.percentile(values[name], [2.5, 97.5])
        result['metrics'][name] = {
            'point': float(point_value),
            'ci_95_lower': float(lower),
            'ci_95_upper': float(upper),
        }
    return result


def save_metrics_json(result, destination):
    """Atomically save a metrics result."""
    destination = Path(destination)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(partial, destination)
