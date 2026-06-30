"""evaluate.py — Topic model evaluation and metric persistence for TrendScope."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path


def save_metrics(metrics, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def _clean(v):
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, dict): return {k: _clean(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_clean(x) for x in v]
        return v
    with open(path, "w") as f: json.dump(_clean(metrics), f, indent=2)
    return metrics


def print_report(metrics):
    print("=" * 50)
    print("  TrendScope Topic Model Evaluation")
    print("=" * 50)
    print(f"  Documents: {metrics['n_documents']}")
    print(f"  Topics: {metrics['n_topics']}")
    print(f"  LDA coherence: {metrics['lda_coherence']:.4f}")
    print(f"  LDA diversity: {metrics['lda_diversity']:.4f}")
    print(f"  NMF coherence: {metrics['nmf_coherence']:.4f}")
    print(f"  NMF diversity: {metrics['nmf_diversity']:.4f}")
    for t in metrics.get("trends", []):
        print(f"  Topic {t['topic_idx']}: {t['trend_direction']} "
              f"(slope={t['trend_slope']:.2f}, articles={t['total_articles']})")
