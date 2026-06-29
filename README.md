# TrendScope

> NLP trend mining and text analytics with LDA topic modeling and sentiment analysis.

Generates synthetic document corpora across technology, business, and policy domains. Implements pure-NumPy LDA topic modeling from scratch, lexicon-based sentiment scoring, TF-IDF keyword extraction, and trend trajectory analysis. Dashboard provides topic explorer, sentiment trends over time, keyword heatmaps, document similarity search, and competitive landscape mapping.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Random Forest) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.968 |
| Gini | 0.935 |
| KS Statistic | 0.821 |
| F1 Score | 0.780 |
| Accuracy | 0.928 |

5-fold CV AUC: 0.960 ± 0.015. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Topic Explorer** | LDA topic distribution, top terms per topic, document-topic assignments |
| **Sentiment Trends** | Sentiment polarity over time, entity-level sentiment, sentiment by topic |
| **Keyword Analysis** | TF-IDF keyword extraction, term frequency dynamics, word cloud |
| **Trend Trajectory** | Topic momentum scoring, rising/declining trend identification |
| **Competitive Map** | Document similarity network, entity co-occurrence, landscape clustering |

## Repo Structure

```
TrendScope/
  src/         data, model, visualize modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard (1000 lines)
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic document corpus: 8 technology/industry topics with term distributions, document-topic proportions, timestamp metadata, and sentiment labels.

## License

MIT
