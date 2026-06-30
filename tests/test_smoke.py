"""Smoke tests for TrendScope — topic modeling and trend detection."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.model import fit_lda, fit_nmf, detect_trends, evaluate_topics, fit_and_evaluate
from src.core import topic_coherence_umass, topic_diversity, trend_slope


def test_data():
    """Synthetic corpus has articles, timestamps, and topic labels."""
    d = make_synthetic(n_articles=100, seed=42)
    assert d["n_documents"] == 100
    assert len(d["documents"]) == 100
    assert d["timestamps"].shape == (100,)
    assert d["n_topics"] == 5


def test_lda():
    """LDA discovers topics and produces document-topic distributions."""
    d = make_synthetic(n_articles=100, seed=42)
    result = fit_lda(d["documents"], n_topics=5, seed=42)
    assert result["doc_topic_dist"].shape == (100, 5)
    assert len(result["topics"]) == 5
    assert all(len(t) > 0 for t in result["topics"])


def test_nmf():
    """NMF discovers topics and produces document-topic distributions."""
    d = make_synthetic(n_articles=100, seed=42)
    result = fit_nmf(d["documents"], n_topics=5, seed=42)
    assert result["doc_topic_dist"].shape == (100, 5)
    assert len(result["topics"]) == 5


def test_trend_detection():
    """Trend detection produces monthly counts and slopes per topic."""
    d = make_synthetic(n_articles=100, seed=42)
    lda_result = fit_lda(d["documents"], n_topics=5, seed=42)
    trends = detect_trends(lda_result["doc_topic_dist"], d["timestamps"], d["n_months"])
    assert len(trends) == 5
    for t in trends:
        assert len(t["monthly_counts"]) == 12
        assert t["trend_direction"] in ("rising", "declining", "stable")


def test_coherence():
    """Topic coherence computes for a simple example."""
    docs_tokens = [["ai", "model", "neural", "ai"], ["market", "trading", "stocks"]]
    coh = topic_coherence_umass(["ai", "model", "neural"], docs_tokens)
    assert isinstance(coh, float)


def test_diversity():
    """Topic diversity computes correctly."""
    div = topic_diversity([["a", "b", "c"], ["b", "c", "d"]])
    assert 0.0 <= div <= 1.0


def test_trend_slope():
    """Trend slope is positive for increasing, negative for decreasing."""
    assert trend_slope(np.array([1, 2, 3, 4, 5])) > 0
    assert trend_slope(np.array([5, 4, 3, 2, 1])) < 0


def test_fit_and_evaluate():
    """Full pipeline returns model and metrics."""
    d = make_synthetic(n_articles=150, seed=42)
    model, metrics = fit_and_evaluate(d, n_topics=5, seed=42)
    assert "lda" in model
    assert "trends" in metrics
    assert "lda_coherence" in metrics
    assert len(metrics["trends"]) == 5


if __name__ == "__main__":
    test_data()
    test_lda()
    test_nmf()
    test_trend_detection()
    test_coherence()
    test_diversity()
    test_trend_slope()
    test_fit_and_evaluate()
    print("All TrendScope smoke tests passed!")
