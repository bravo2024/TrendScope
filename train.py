"""train.py — Run topic modeling and trend detection pipeline."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import make_synthetic
from src.model import fit_and_evaluate
from src.evaluate import save_metrics, print_report
from src.persist import save_model


def main():
    data = make_synthetic(n_articles=300, seed=42)
    print(f"Generated {data['n_documents']} articles across {data['n_topics']} topics")
    model, metrics = fit_and_evaluate(data, n_topics=5, seed=42)
    print_report(metrics)
    save_model(model)
    save_metrics(metrics)
    print("\nSaved model -> models/model.pkl and metrics -> models/metrics.json")


if __name__ == "__main__":
    main()
