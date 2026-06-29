from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["post_frequency","engagement_rate","sentiment_volatility","topic_diversity","influencer_score","content_originality","geo_spread","temporal_recency","audience_growth","competitor_overlap"]
CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = FEATURE_NAMES
TARGET_NAME = "trend_viral"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "post_frequency": rng.poisson(lam=5,size=n).clip(0,30),
        "engagement_rate": rng.beta(2,8,size=n).round(4),
        "sentiment_volatility": rng.uniform(0,1,size=n).round(3),
        "topic_diversity": rng.uniform(0,1,size=n).round(3),
        "influencer_score": rng.uniform(0,100,size=n).round(1),
        "content_originality": rng.beta(4,3,size=n).round(3),
        "geo_spread": rng.uniform(0,1,size=n).round(3),
        "temporal_recency": rng.beta(5,2,size=n).round(3),
        "audience_growth": rng.normal(0.05,0.02,size=n).clip(-0.05,0.20).round(4),
        "competitor_overlap": rng.uniform(0,1,size=n).round(3),
    })
    freq=np.clip(df["post_frequency"]/30,0,1); eng=df["engagement_rate"]*10; vol=df["sentiment_volatility"]
    div=df["topic_diversity"]; inf=df["influencer_score"]/100; orig=df["content_originality"]
    geo=df["geo_spread"]; rec=df["temporal_recency"]; growth=np.clip((df["audience_growth"]+0.05)/0.25,0,1)
    comp=df["competitor_overlap"]
    log_odds = -3.0 + 0.5*freq + 1.5*eng + 0.3*vol + 0.4*div + 1.0*inf + 0.6*orig + 0.5*geo + 0.4*rec + 0.8*growth - 0.3*comp + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,85)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(trend_viral=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
