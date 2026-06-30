"""model.py — Topic modeling and trend detection for TrendScope (Gartner).

Uses Latent Dirichlet Allocation (LDA) via sklearn and Non-negative Matrix
Factorization (NMF) to discover latent topics from a corpus of articles.
Trend detection counts articles per topic per month and computes trend slopes.

This is NLP/topic-modeling, NOT classification — completely different from
the generic sklearn classification template it replaces.

References
----------
Blei et al. (2003), "Latent Dirichlet Allocation." JMLR.
Lee & Seung (1999), "Learning the parts of objects by non-negative matrix
factorization." Nature.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from src.core import topic_coherence_umass, topic_diversity, trend_slope


def fit_lda(documents: list[str], n_topics: int = 5, max_features: int = 200, seed: int = 42):
    """Fit LDA topic model and return topics, document-topic distributions, and vocabulary."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english", token_pattern=r"[a-z]{3,}")
    doc_term = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=20, random_state=seed,
        learning_method="batch", verbose=0,
    )
    doc_topic = lda.fit_transform(doc_term)
    topics = _extract_top_words(lda.components_, vocab, top_n=10)
    return {
        "model": lda, "vectorizer": vectorizer, "vocab": vocab,
        "doc_topic_dist": doc_topic, "topics": topics,
        "doc_term_matrix": doc_term, "method": "LDA",
    }


def fit_nmf(documents: list[str], n_topics: int = 5, max_features: int = 200, seed: int = 42):
    """Fit NMF topic model and return topics, document-topic distributions, and vocabulary."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", token_pattern=r"[a-z]{3,}")
    doc_term = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()
    nmf = NMF(n_components=n_topics, max_iter=300, random_state=seed, init="nndsvd")
    doc_topic = nmf.fit_transform(doc_term)
    topics = _extract_top_words(nmf.components_, vocab, top_n=10)
    return {
        "model": nmf, "vectorizer": vectorizer, "vocab": vocab,
        "doc_topic_dist": doc_topic, "topics": topics,
        "doc_term_matrix": doc_term, "method": "NMF",
    }


def _extract_top_words(components, vocab, top_n=10) -> list[list[str]]:
    """Extract top-N words for each topic from the components matrix."""
    topics = []
    for comp in components:
        top_idx = comp.argsort()[-top_n:][::-1]
        topics.append([vocab[i] for i in top_idx])
    return topics


def detect_trends(doc_topic_dist: np.ndarray, timestamps: np.ndarray,
                  n_months: int = 12) -> list[dict]:
    """Count articles per topic per month and compute trend slopes.

    Returns a list of dicts, one per topic, with monthly counts and slope.
    """
    n_topics = doc_topic_dist.shape[1]
    dominant_topics = doc_topic_dist.argmax(axis=1)
    trends = []
    for topic_idx in range(n_topics):
        monthly_counts = np.zeros(n_months)
        for month in range(n_months):
            mask = (timestamps == month) & (dominant_topics == topic_idx)
            monthly_counts[month] = mask.sum()
        slope = trend_slope(monthly_counts)
        trends.append({
            "topic_idx": topic_idx,
            "monthly_counts": monthly_counts.tolist(),
            "trend_slope": slope,
            "trend_direction": "rising" if slope > 0.5 else "declining" if slope < -0.5 else "stable",
            "total_articles": int(monthly_counts.sum()),
        })
    return trends


def evaluate_topics(topic_result: dict, documents: list[str], n_topics: int = 5) -> dict:
    """Evaluate topic quality using coherence and diversity metrics."""
    topics = topic_result["topics"]
    # Tokenize documents for coherence computation
    docs_tokens = [doc.split() for doc in documents]
    coherences = [topic_coherence_umass(t, docs_tokens) for t in topics]
    diversity = topic_diversity(topics)
    return {
        "method": topic_result["method"],
        "n_topics": n_topics,
        "coherences": coherences,
        "mean_coherence": float(np.mean(coherences)),
        "topic_diversity": diversity,
        "topics": topics,
    }


def fit_and_evaluate(data: dict, n_topics: int = 5, seed: int = 42) -> tuple:
    """Convenience: fit LDA, evaluate topics, detect trends. Returns (model, metrics)."""
    docs = data["documents"]
    lda_result = fit_lda(docs, n_topics=n_topics, seed=seed)
    nmf_result = fit_nmf(docs, n_topics=n_topics, seed=seed)
    lda_eval = evaluate_topics(lda_result, docs, n_topics)
    nmf_eval = evaluate_topics(nmf_result, docs, n_topics)
    trends = detect_trends(lda_result["doc_topic_dist"], data["timestamps"], data["n_months"])
    model = {
        "lda": lda_result, "nmf": nmf_result,
        "trends": trends, "n_topics": n_topics,
    }
    metrics = {
        "n_documents": len(docs),
        "n_topics": n_topics,
        "lda_coherence": lda_eval["mean_coherence"],
        "lda_diversity": lda_eval["topic_diversity"],
        "nmf_coherence": nmf_eval["mean_coherence"],
        "nmf_diversity": nmf_eval["topic_diversity"],
        "trends": trends,
    }
    return model, metrics