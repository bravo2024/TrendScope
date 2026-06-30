"""data.py — Synthetic article corpus with latent topics and timestamps for TrendScope.

Generates ~300 short articles across 5 latent topics (AI, finance, healthcare,
climate, energy), each with a timestamp spanning 12 months. Topic prevalence
varies over time to create discoverable trends.

This is NLP data (text articles), NOT generic tabular features.
"""
from __future__ import annotations
import numpy as np
from typing import Any

TOPIC_NAMES = ["AI & Machine Learning", "Finance & Markets", "Healthcare & Biotech",
               "Climate & Environment", "Energy & Renewables"]

# Vocabulary per topic (distinctive words for each)
_TOPIC_WORDS = {
    "AI & Machine Learning": ["model", "neural", "training", "dataset", "algorithm",
        "learning", "deep", "transformer", "gpu", "inference", "language", "parameters"],
    "Finance & Markets": ["market", "trading", "portfolio", "stocks", "bonds",
        "volatility", "yield", "investor", "fed", "inflation", "equity", "derivatives"],
    "Healthcare & Biotech": ["patient", "clinical", "drug", "trial", "therapy",
        "diagnosis", "treatment", "genome", "fda", "biotech", "vaccine", "pharma"],
    "Climate & Environment": ["carbon", "emission", "temperature", "warming", "climate",
        "sea", "ice", "forest", "pollution", "biodiversity", "drought", "flood"],
    "Energy & Renewables": ["solar", "wind", "battery", "grid", "renewable",
        "hydrogen", "nuclear", "electric", "storage", "turbine", "geothermal", "capacity"],
}

# Common words shared across topics
_COMMON_WORDS = ["report", "study", "research", "data", "analysis", "growth",
                 "report", "year", "new", "show", "team", "system", "project",
                 "plan", "increase", "global", "future", "industry", "company"]


def make_synthetic(n_articles: int = 300, n_months: int = 12, seed: int = 42) -> dict[str, Any]:
    """Generate synthetic articles with latent topics and timestamps.

    Topic prevalence shifts over time:
    - AI articles increase (rising trend)
    - Climate articles increase (rising trend)
    - Finance articles stable
    - Healthcare stable
    - Energy articles decrease slightly (declining trend)
    """
    rng = np.random.default_rng(seed)
    n_topics = len(TOPIC_NAMES)
    articles = []
    timestamps = []
    topic_labels = []

    # Time-varying topic probabilities
    for i in range(n_articles):
        month = i % n_months
        # Rising trends for AI (0) and Climate (3), declining for Energy (4)
        t = month / max(n_months - 1, 1)
        probs = np.array([
            0.15 + 0.15 * t,      # AI: rising
            0.20,                   # Finance: stable
            0.20,                   # Healthcare: stable
            0.15 + 0.12 * t,      # Climate: rising
            0.25 - 0.12 * t,      # Energy: declining
        ])
        probs = probs / probs.sum()
        topic_idx = rng.choice(n_topics, p=probs)
        topic_name = TOPIC_NAMES[topic_idx]

        # Generate article text from topic words + common words
        topic_words = _TOPIC_WORDS[topic_name]
        n_topic_words = rng.integers(15, 30)
        n_common_words = rng.integers(5, 12)
        words = list(rng.choice(topic_words, n_topic_words)) + list(rng.choice(_COMMON_WORDS, n_common_words))
        rng.shuffle(words)
        text = " ".join(words)
        articles.append(text)
        timestamps.append(month)
        topic_labels.append(topic_idx)

    return {
        "documents": articles,
        "timestamps": np.array(timestamps),
        "topic_labels": np.array(topic_labels),
        "topic_names": TOPIC_NAMES,
        "n_topics": n_topics,
        "n_documents": n_articles,
        "n_months": n_months,
    }