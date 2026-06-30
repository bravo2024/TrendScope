"""core.py — Topic coherence and trend analysis metrics for TrendScope.

These are NLP/topic-modeling metrics, NOT generic classification metrics:
  * **Topic coherence (u_mass)** — log-conditional pairwise agreement
    between top words in a topic (Mimno et al. 2011).
  * **Topic diversity** — fraction of unique top words across all topics.
  * **Trend slope** — linear regression slope of article counts over time.
  * **Silhouette** — cluster separation on document-topic distributions.

References
----------
Mimno et al. (2011), "Optimizing Semantic Coherence in Topic Models."
Röder et al. (2015), "Exploring the Space of Topic Coherence Measures."
"""
from __future__ import annotations
import numpy as np
from collections import Counter


def topic_coherence_umass(top_words: list[str], docs_tokens: list[list[str]], top_n: int = 10) -> float:
    """UMass coherence for a single topic.

    C_umass = sum_{i<j} log( (D(w_i, w_j) + 1) / D(w_j) )
    where D(w) = document frequency of word w, D(w_i, w_j) = co-document frequency.
    """
    words = top_words[:top_n]
    if len(words) < 2:
        return 0.0
    doc_freq = Counter()
    co_doc_freq = Counter()
    for doc_tokens in docs_tokens:
        doc_set = set(doc_tokens)
        for w in words:
            if w in doc_set:
                doc_freq[w] += 1
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if words[i] in doc_set and words[j] in doc_set:
                    co_doc_freq[(words[i], words[j])] += 1
    coherence = 0.0
    for j in range(1, len(words)):
        for i in range(j):
            denom = doc_freq.get(words[j], 0)
            if denom == 0:
                continue
            numer = co_doc_freq.get((words[i], words[j]), 0) + 1
            coherence += np.log(numer / denom)
    return float(coherence)


def topic_diversity(all_top_words: list[list[str]], top_n: int = 10) -> float:
    """Fraction of unique words across all topics' top-N lists."""
    all_words = []
    for words in all_top_words:
        all_words.extend(words[:top_n])
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def trend_slope(counts: np.ndarray, window: int = None) -> float:
    """Linear regression slope of a time series (trend direction).

    Positive slope = increasing trend, negative = declining.
    """
    c = np.asarray(counts, dtype=float)
    if c.size < 2:
        return 0.0
    x = np.arange(c.size, dtype=float)
    x_mean, c_mean = x.mean(), c.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (c - c_mean)) / denom)


def silhouette_score_simple(doc_topic_dist: np.ndarray, labels: np.ndarray) -> float:
    """Simplified silhouette score for document-topic clustering.

    Measures how similar a document is to its assigned topic cluster
    vs other clusters. Range [-1, 1], higher = better separated.
    """
    D = np.asarray(doc_topic_dist, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    if n < 2 or len(set(labels)) < 2:
        return 0.0
    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            a_i = 0.0
        else:
            a_i = np.mean(np.linalg.norm(D[i] - D[same], axis=1))
        b_i = float('inf')
        for other_label in set(labels) - {labels[i]}:
            other = labels == other_label
            if other.sum() > 0:
                b_i = min(b_i, np.mean(np.linalg.norm(D[i] - D[other], axis=1)))
        if b_i == float('inf'):
            scores.append(0.0)
        else:
            s = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
            scores.append(s)
    return float(np.mean(scores))