"""
TrendScope — NLP Trend Mining & Text Analytics Dashboard
Production-grade Streamlit app: 5 tabs, pure-Python NLP, NumPy LDA, scratch sentiment.
"""
from __future__ import annotations
import math, re, string, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrendScope | NLP Trend Mining",
    layout="wide",
    page_icon="📊",
)

DARK_BG   = "#0f172a"
CARD_BG   = "#1e293b"
ACCENT    = "#38bdf8"
GREEN     = "#22c55e"
RED       = "#f43f5e"
ORANGE    = "#fb923c"
PURPLE    = "#a78bfa"
YELLOW    = "#facc15"

PALETTE = [ACCENT, GREEN, ORANGE, PURPLE, YELLOW, RED, "#34d399", "#f472b6"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "axes.edgecolor": "#334155", "grid.color": "#1e293b",
    "axes.titlecolor": "white", "legend.facecolor": CARD_BG,
    "legend.edgecolor": "#334155",
})

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — CATEGORIES & TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIES = [
    "AI/ML", "Cryptocurrency", "Climate Tech",
    "Electric Vehicles", "Biotech", "Cybersecurity", "Fintech", "Gaming",
]
SOURCES = ["Twitter", "Reddit", "Bloomberg", "TechCrunch", "HackerNews"]

CAT_KEYWORDS: dict[str, list[str]] = {
    "AI/ML":           ["GPT", "neural", "transformer", "LLM", "diffusion", "PyTorch", "inference",
                        "training", "model", "dataset", "benchmark", "fine-tuning", "RLHF", "AGI",
                        "embedding", "attention", "multimodal", "agents", "reasoning", "alignment"],
    "Cryptocurrency":  ["Bitcoin", "Ethereum", "DeFi", "NFT", "blockchain", "wallet", "staking",
                        "altcoin", "Web3", "Solana", "Cardano", "mining", "exchange", "liquidity",
                        "yield", "protocol", "bridge", "token", "DAO", "consensus"],
    "Climate Tech":    ["solar", "wind", "carbon", "emissions", "renewable", "grid", "storage",
                        "hydrogen", "offset", "sustainability", "net-zero", "ESG", "methane",
                        "capture", "geothermal", "battery", "electrification", "IPCC", "climate", "green"],
    "Electric Vehicles": ["EV", "Tesla", "battery", "charging", "range", "autonomous", "Rivian",
                          "BYD", "Lucid", "powertrain", "regenerative", "NACS", "fleet", "OTA",
                          "Autopilot", "range", "lithium", "cathode", "fast-charging", "grid"],
    "Biotech":         ["CRISPR", "mRNA", "clinical", "trial", "FDA", "genome", "protein",
                        "antibody", "sequencing", "drug", "immunotherapy", "oncology", "biomarker",
                        "gene", "cell", "therapy", "vaccine", "pharma", "diagnostic", "phenotype"],
    "Cybersecurity":   ["ransomware", "breach", "phishing", "zero-day", "CVE", "firewall", "SIEM",
                        "endpoint", "threat", "malware", "patch", "vulnerability", "SOC", "XDR",
                        "identity", "MFA", "encryption", "CISO", "APT", "botnet"],
    "Fintech":         ["payments", "neobank", "BNPL", "API", "open-banking", "KYC", "AML",
                        "insurtech", "regtech", "stablecoin", "CBDC", "embedded", "lending",
                        "credit", "robo-advisor", "treasury", "ledger", "SaaS", "interchange", "yield"],
    "Gaming":          ["AAA", "indie", "metaverse", "esports", "VR", "AR", "Unity", "Unreal",
                        "streaming", "subscription", "DLC", "multiplayer", "NFT", "GPU", "raytracing",
                        "mobile", "PC", "console", "LiveOps", "monetisation"],
}

VERBS   = ["launches", "announces", "reports", "reveals", "accelerates", "disrupts",
           "advances", "achieves", "integrates", "scales", "secures", "optimizes"]
NOUNS   = ["breakthrough", "milestone", "partnership", "investment", "roadmap",
           "platform", "framework", "protocol", "solution", "strategy", "upgrade", "deal"]
APPS    = ["enterprise", "consumer", "healthcare", "finance", "education", "logistics",
           "government", "manufacturing", "retail", "research", "defense", "media"]
TECHS   = ["machine learning", "distributed ledger", "edge computing", "quantum computing",
           "5G", "cloud infrastructure", "open-source stack", "API gateway", "data pipeline"]

ENTITIES = ["Tesla", "OpenAI", "Google", "Bitcoin", "Meta", "Apple", "Nvidia",
            "Ethereum", "Microsoft", "Amazon", "Samsung", "Pfizer", "SpaceX",
            "Anthropic", "AMD", "Intel", "Rivian", "BYD", "Coinbase", "Stripe"]

# ─────────────────────────────────────────────────────────────────────────────
# STOPWORDS (150 common English words)
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at be
because been before being below between both but by can't cannot could couldn't
did didn't do does doesn't doing don't down during each few for from further get
got had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't
it it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd she'll
she's should shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've
were weren't what what's when when's where where's which while who who's whom
why why's will with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves just been also really very much many more most such
even still well back first last next new old high low big small large great good
best better now just already yet still both each every some any all few more
most other another one two three many however therefore thus hence accordingly
moreover furthermore additionally consequently nevertheless nonetheless meanwhile
""".split())

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT LEXICON
# ─────────────────────────────────────────────────────────────────────────────
POS_WORDS = set("""
excellent surge breakthrough innovative bullish adoption growth record milestone
launch leading strong profit gain success revolutionary promising robust advance
outperform rally boost soar skyrocket accelerate achieve win expand thrive
flourish recover rebound upgrade pioneer disruptive transformative efficient
reliable secure scalable profitable dominant superior outstanding remarkable
exceptional impressive phenomenal stellar lucrative booming thriving prosperous
""".split())

NEG_WORDS = set("""
crash failure hack breach loss decline disappointing bearish collapse vulnerability
risk crisis fraud lawsuit ban recall delay plunge drop slump tumble sink struggle
underperform disappoint miss shortfall deficit problem issue concern threat warning
damage destroy corrupt exploit attack infiltrate compromise leak expose weak poor
terrible awful horrible dreadful catastrophic devastating disastrous alarming
volatile uncertain unstable risky dangerous harmful toxic flawed broken failing
""".split())

NEGATORS = {"not", "no", "never", "nor", "cannot", "isn't", "aren't", "wasn't",
            "weren't", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't"}

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS GENERATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating 5 000-document corpus …")
def generate_corpus(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = 5_000
    now = pd.Timestamp("2026-06-29")
    rows = []
    cats_arr = rng.choice(CATEGORIES, size=n)

    for i in range(n):
        cat  = cats_arr[i]
        kws  = CAT_KEYWORDS[cat]
        # build text from template
        kw1  = kws[int(rng.integers(0, len(kws)))]
        kw2  = kws[int(rng.integers(0, len(kws)))]
        verb = VERBS[int(rng.integers(0, len(VERBS)))]
        noun = NOUNS[int(rng.integers(0, len(NOUNS)))]
        app  = APPS[int(rng.integers(0, len(APPS)))]
        tech = TECHS[int(rng.integers(0, len(TECHS)))]
        ent  = ENTITIES[int(rng.integers(0, len(ENTITIES)))]

        # occasionally inject sentiment words & entity
        pos = list(POS_WORDS)[int(rng.integers(0, len(POS_WORDS)))]
        neg = list(NEG_WORDS)[int(rng.integers(0, len(NEG_WORDS)))]
        use_pos = rng.random() > 0.35
        use_neg = (not use_pos) and rng.random() > 0.45

        sentiment_word = pos if use_pos else (neg if use_neg else "")
        text = (
            f"{ent} {verb} {noun} for {app} using {tech}. "
            f"The {kw1} and {kw2} segment shows {sentiment_word} momentum "
            f"as developers adopt {kw1}-driven {app} pipelines."
        )

        days_ago = int(rng.integers(0, 365))
        ts  = now - pd.Timedelta(days=int(days_ago))
        src = SOURCES[int(rng.integers(0, len(SOURCES)))]
        eng = int(rng.integers(10, 10_001))
        flw = int(rng.integers(100, 1_000_001))

        rows.append({
            "id":              i,
            "text":            text,
            "category":        cat,
            "source":          src,
            "timestamp":       ts,
            "engagement":      eng,
            "author_followers": flw,
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    df["week"]      = df["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# NLP PIPELINE (pure Python + NumPy)
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]

SUFFIXES = ["ing", "tion", "ness", "ment", "ed", "ly"]

def stem(word: str) -> str:
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)]
    return word

def preprocess(text: str) -> list[str]:
    return [stem(t) for t in remove_stopwords(tokenize(text))]

@st.cache_data(show_spinner="Running NLP pipeline …")
def run_pipeline(df: pd.DataFrame):
    tokens_list  = [preprocess(t) for t in df["text"]]
    vocab_set: set[str] = set()
    for toks in tokens_list:
        vocab_set.update(toks)
    vocab = sorted(vocab_set)
    w2i  = {w: i for i, w in enumerate(vocab)}
    V    = len(vocab)
    N    = len(tokens_list)

    # TF-IDF
    tf_matrix  = np.zeros((N, V), dtype=np.float32)
    for d, toks in enumerate(tokens_list):
        if not toks:
            continue
        for t in toks:
            tf_matrix[d, w2i[t]] += 1
        tf_matrix[d] /= len(toks)

    df_counts = (tf_matrix > 0).sum(axis=0)  # doc frequency per term
    idf = np.log(N / (1 + df_counts)) + 1
    tfidf = tf_matrix * idf

    # word frequency overall
    all_tokens = [t for toks in tokens_list for t in toks]
    freq: dict[str, int] = {}
    for t in all_tokens:
        freq[t] = freq.get(t, 0) + 1

    return {
        "tokens_list": tokens_list,
        "vocab":       vocab,
        "w2i":         w2i,
        "tfidf":       tfidf,
        "idf":         idf,
        "freq":        freq,
        "doc_lengths": [len(toks) for toks in tokens_list],
    }

# ─────────────────────────────────────────────────────────────────────────────
# LDA — COLLAPSED GIBBS SAMPLING (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training LDA (Gibbs sampling) …")
def run_lda(tokens_list: list[list[str]], vocab: list[str], w2i: dict[str, int],
            K: int = 8, n_iter: int = 50, alpha: float = 0.1, beta: float = 0.01):
    rng = np.random.default_rng(42)
    V   = len(vocab)
    D   = len(tokens_list)

    # encode docs
    docs = [[w2i[t] for t in toks] for toks in tokens_list]

    n_dk = np.zeros((D, K), dtype=np.int32)   # doc-topic counts
    n_kw = np.zeros((K, V), dtype=np.int32)   # topic-word counts
    n_k  = np.zeros(K,      dtype=np.int32)   # topic totals

    z_dn: list[list[int]] = []
    for d, doc in enumerate(docs):
        z = []
        for w in doc:
            k = int(rng.integers(0, K))
            z.append(k)
            n_dk[d, k] += 1
            n_kw[k, w] += 1
            n_k[k]     += 1
        z_dn.append(z)

    for _ in range(n_iter):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                k_old = z_dn[d][i]
                n_dk[d, k_old] -= 1
                n_kw[k_old, w] -= 1
                n_k[k_old]     -= 1

                num   = (n_dk[d] + alpha) * (n_kw[:, w] + beta)
                denom = n_k + V * beta
                probs = num / denom
                probs /= probs.sum()
                k_new = int(rng.choice(K, p=probs))

                z_dn[d][i]     = k_new
                n_dk[d, k_new] += 1
                n_kw[k_new, w] += 1
                n_k[k_new]     += 1

    # compute distributions
    theta = (n_dk + alpha) / (n_dk + alpha).sum(axis=1, keepdims=True)
    phi   = (n_kw + beta)  / (n_kw + beta).sum(axis=1,  keepdims=True)

    return {"theta": theta, "phi": phi, "n_dk": n_dk, "n_kw": n_kw, "vocab": vocab}

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT (lexicon, from scratch)
# ─────────────────────────────────────────────────────────────────────────────
def sentiment_score(text: str) -> float:
    tokens = tokenize(text)
    score  = 0.0
    for i, tok in enumerate(tokens):
        window = tokens[max(0, i - 3): i]
        negated = any(w in NEGATORS for w in window)
        if tok in POS_WORDS:
            score += -1 if negated else 1
        elif tok in NEG_WORDS:
            score += 1 if negated else -1
    return score / max(len(tokens), 1)

def sentiment_label(s: float) -> str:
    if s > 0.05:
        return "Positive"
    if s < -0.05:
        return "Negative"
    return "Neutral"

@st.cache_data(show_spinner="Computing sentiment …")
def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    scores  = [sentiment_score(t) for t in df["text"]]
    labels  = [sentiment_label(s) for s in scores]
    df2     = df.copy()
    df2["sentiment_score"] = scores
    df2["sentiment"]       = labels
    return df2

# ─────────────────────────────────────────────────────────────────────────────
# MANN-KENDALL TREND TEST (from scratch)
# ─────────────────────────────────────────────────────────────────────────────
def mann_kendall(x: np.ndarray):
    n = len(x)
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            S   += int(diff > 0) - int(diff < 0)
    var_S = n * (n - 1) * (2 * n + 5) / 18
    if S > 0:
        Z = (S - 1) / math.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / math.sqrt(var_S)
    else:
        Z = 0.0
    trend = "↑ Rising" if Z > 1.96 else ("↓ Falling" if Z < -1.96 else "→ Stable")
    return {"S": S, "Z": Z, "trend": trend, "significant": abs(Z) > 1.96}

# ─────────────────────────────────────────────────────────────────────────────
# AR(3) FORECAST via OLS (NumPy lstsq)
# ─────────────────────────────────────────────────────────────────────────────
def ar3_forecast(series: np.ndarray, steps: int = 14):
    p = 3
    n = len(series)
    if n < p + 5:
        return None, None, None
    X, y = [], []
    for i in range(p, n):
        X.append(series[i - p: i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = resid.std()
    preds = []
    buf   = list(series[-p:])
    for _ in range(steps):
        val = float(np.array(buf[-p:]) @ coef)
        preds.append(val)
        buf.append(val)
    preds = np.array(preds)
    return preds, sigma, coef

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    st.markdown("---")
    sel_cats = st.multiselect(
        "Categories", CATEGORIES, default=CATEGORIES, key="sel_cats"
    )
    sel_srcs = st.multiselect(
        "Sources", SOURCES, default=SOURCES, key="sel_srcs"
    )
    date_range = st.slider(
        "Date range (days ago)", 0, 365, (0, 365), key="date_range"
    )
    K_topics = st.slider("Number of LDA Topics (K)", 5, 12, 8, key="K_topics")
    sent_thresh = st.slider(
        "Sentiment threshold", 0.01, 0.20, 0.05, step=0.01, key="sent_thresh"
    )
    st.markdown("---")
    st.caption("TrendScope v2.0 · NLP Analytics Platform")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD + FILTER
# ─────────────────────────────────────────────────────────────────────────────
corpus = generate_corpus()
now    = pd.Timestamp("2026-06-29")
mask = (
    corpus["category"].isin(sel_cats if sel_cats else CATEGORIES) &
    corpus["source"].isin(sel_srcs if sel_srcs else SOURCES) &
    (corpus["timestamp"] >= now - pd.Timedelta(days=date_range[1])) &
    (corpus["timestamp"] <= now - pd.Timedelta(days=date_range[0]))
)
df_filt = corpus[mask].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER KPIs
# ─────────────────────────────────────────────────────────────────────────────
pipeline_data = run_pipeline(df_filt)
senti_df      = compute_sentiment(df_filt)
vocab_size    = len(pipeline_data["vocab"])
avg_sent      = senti_df["sentiment_score"].mean()

# top trending topic (Mann-Kendall highest Z)
cat_mk = {}
for cat in (sel_cats or CATEGORIES):
    sub = df_filt[df_filt["category"] == cat]
    if len(sub) >= 10:
        daily = sub.groupby("date").size().sort_index()
        mk    = mann_kendall(daily.values[-30:] if len(daily) >= 30 else daily.values)
        cat_mk[cat] = mk["Z"]
top_trend = max(cat_mk, key=cat_mk.get) if cat_mk else "N/A"

st.markdown(
    """
    <h1 style='text-align:center;color:#38bdf8;margin-bottom:4px'>
    📊 TrendScope — NLP Trend Mining & Text Analytics
    </h1>
    <p style='text-align:center;color:#94a3b8;margin-top:0'>
    Real-time social media & news analytics · 5 000-document corpus · Pure-Python NLP
    </p>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Documents",  f"{len(df_filt):,}")
k2.metric("Categories",       str(len(sel_cats or CATEGORIES)))
k3.metric("Vocabulary Size",  f"{vocab_size:,}")
k4.metric("Top Trending",     top_trend)
k5.metric("Avg Sentiment",    f"{avg_sent:+.4f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📰 Text Data Explorer",
    "🔤 NLP Pipeline",
    "🤖 Topic Modeling (LDA)",
    "😊 Sentiment Analysis",
    "📈 Trend Detection & Forecasting",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.subheader("📰 Text Data Explorer")
    st.dataframe(df_filt[["id","category","source","timestamp","engagement",
                            "author_followers","text"]].head(40),
                 use_container_width=True, height=220)

    c1, c2 = st.columns(2)

    # Pie — doc count by category
    with c1:
        cat_counts = df_filt["category"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            cat_counts.values, labels=cat_counts.index,
            colors=PALETTE[:len(cat_counts)], autopct="%1.1f%%",
            startangle=90, textprops={"color": "white", "fontsize": 7},
        )
        for at in autotexts:
            at.set_fontsize(7)
        ax.set_title("Documents by Category", fontsize=12, pad=8)
        fig.tight_layout()
        st.pyplot(fig)

    # Source bar chart
    with c2:
        src_counts = df_filt["source"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(src_counts.index, src_counts.values,
                      color=PALETTE[:len(src_counts)])
        ax.set_title("Posts by Source", fontsize=12)
        ax.set_xlabel("Source"); ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, src_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(v), ha="center", va="bottom", fontsize=8, color="white")
        fig.tight_layout()
        st.pyplot(fig)

    # Time series — daily volume + 7-day rolling
    daily_vol = df_filt.groupby("date").size().reset_index(name="count")
    daily_vol["date"] = pd.to_datetime(daily_vol["date"])
    daily_vol = daily_vol.sort_values("date")
    daily_vol["rolling7"] = daily_vol["count"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(daily_vol["date"], daily_vol["count"], alpha=0.25, color=ACCENT)
    ax.plot(daily_vol["date"], daily_vol["count"], color=ACCENT, linewidth=0.8, label="Daily")
    ax.plot(daily_vol["date"], daily_vol["rolling7"], color=ORANGE, linewidth=2,
            label="7-day rolling avg")
    ax.set_title("Posting Volume Over Time", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Posts")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

    c3, c4 = st.columns(2)

    # Avg engagement by category
    with c3:
        avg_eng = df_filt.groupby("category")["engagement"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(avg_eng.index, avg_eng.values,
                       color=[PALETTE[i % len(PALETTE)] for i in range(len(avg_eng))])
        ax.set_title("Avg Engagement by Category", fontsize=11)
        ax.set_xlabel("Avg Engagement")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    # Top 20 words for selected category
    with c4:
        sel_cat_wf = st.selectbox("Category for word frequency", CATEGORIES, key="wf_cat")
        sub_df = df_filt[df_filt["category"] == sel_cat_wf]
        cat_tokens = [t for text in sub_df["text"] for t in preprocess(text)]
        wf: dict[str, int] = {}
        for t in cat_tokens:
            wf[t] = wf.get(t, 0) + 1
        top20 = sorted(wf.items(), key=lambda x: x[1], reverse=True)[:20]
        words20, cnts20 = zip(*top20) if top20 else ([], [])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(list(words20)[::-1], list(cnts20)[::-1], color=GREEN)
        ax.set_title(f"Top 20 Words — {sel_cat_wf}", fontsize=10)
        ax.set_xlabel("Frequency")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NLP PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.subheader("🔤 NLP Pipeline — From Scratch")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Vocabulary Size", f"{vocab_size:,}")
    col_b.metric("Total Tokens",    f"{sum(pipeline_data['doc_lengths']):,}")
    col_c.metric("Avg Doc Length",  f"{np.mean(pipeline_data['doc_lengths']):.1f} tokens")

    st.markdown("#### TF-IDF Formulas")
    st.latex(r"\text{TF}(t,d) = \frac{\text{count}(t,d)}{|d|}")
    st.latex(r"\text{IDF}(t) = \ln\!\left(\frac{N}{1 + \text{df}(t)}\right) + 1")
    st.latex(r"\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)")

    st.markdown("#### TF-IDF Top Terms per Category (2×4 Grid)")
    cats_show = (sel_cats or CATEGORIES)[:8]
    ncols = 4
    nrows = math.ceil(len(cats_show) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    tfidf_mat  = pipeline_data["tfidf"]
    vocab_list = pipeline_data["vocab"]
    w2i        = pipeline_data["w2i"]

    for idx, cat in enumerate(cats_show):
        ax = axes_flat[idx]
        cat_mask = df_filt["category"] == cat
        cat_indices = np.where(cat_mask.values[:len(tfidf_mat)])[0]
        if len(cat_indices) == 0:
            ax.axis("off"); continue
        cat_tfidf = tfidf_mat[cat_indices].mean(axis=0)
        top_idx   = np.argsort(cat_tfidf)[-10:][::-1]
        top_words = [vocab_list[i] for i in top_idx]
        top_vals  = cat_tfidf[top_idx]
        ax.barh(top_words[::-1], top_vals[::-1], color=PALETTE[idx % len(PALETTE)])
        ax.set_title(cat, fontsize=9); ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)

    for j in range(len(cats_show), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("TF-IDF Top Terms by Category", fontsize=13, color="white")
    fig.tight_layout()
    st.pyplot(fig)

    c_left, c_right = st.columns(2)

    # Document length histogram
    with c_left:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(pipeline_data["doc_lengths"], bins=30, color=ACCENT, alpha=0.8, edgecolor="#334155")
        ax.set_title("Document Length Distribution", fontsize=11)
        ax.set_xlabel("Tokens per document"); ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    # Zipf's law
    with c_right:
        st.markdown("##### Zipf's Law")
        st.latex(r"f(r) \propto r^{-\alpha}")
        freq_vals = sorted(pipeline_data["freq"].values(), reverse=True)[:500]
        ranks     = np.arange(1, len(freq_vals) + 1, dtype=float)
        log_r     = np.log(ranks)
        log_f     = np.log(np.array(freq_vals, dtype=float) + 1)
        # fit power law via linear regression on log-log
        slope, intercept, r_val, *_ = stats.linregress(log_r, log_f)
        alpha_zipf = -slope
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(log_r, log_f, s=4, color=ACCENT, alpha=0.6, label="Observed")
        ax.plot(log_r, intercept + slope * log_r, color=ORANGE, linewidth=2,
                label=f"Power law α={alpha_zipf:.2f}")
        ax.set_title("Zipf's Law (log-log)", fontsize=11)
        ax.set_xlabel("log(rank)"); ax.set_ylabel("log(frequency)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.info(f"Fitted Zipf exponent α = **{alpha_zipf:.3f}** (R² = {r_val**2:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LDA TOPIC MODELING
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("🤖 Topic Modeling — LDA with Collapsed Gibbs Sampling")

    st.markdown("#### Gibbs Sampling Equation")
    st.latex(
        r"p(z_{ij}=k \mid \mathbf{z}_{-ij}, \mathbf{w}) \;"
        r"\propto\; \frac{n_{dk}^{-ij}+\alpha}{\sum_k (n_{dk}^{-ij}+\alpha)}"
        r"\cdot \frac{n_{kw}^{-ij}+\beta}{n_k^{-ij}+V\beta}"
    )

    K = K_topics
    with st.spinner(f"Running LDA (K={K}, 50 iterations) …"):
        lda_res = run_lda(
            pipeline_data["tokens_list"],
            pipeline_data["vocab"],
            pipeline_data["w2i"],
            K=K,
        )

    phi   = lda_res["phi"]     # K × V
    theta = lda_res["theta"]   # D × K
    vocab_list = lda_res["vocab"]

    # Auto-label topics from top word
    topic_labels = []
    for k in range(K):
        top_w = vocab_list[np.argsort(phi[k])[-1]]
        topic_labels.append(f"T{k+1}:{top_w}")

    # Top 10 words per topic — 2×4 grid (up to K panels)
    st.markdown("#### Top-10 Words per Topic")
    ncols2 = 4
    nrows2 = math.ceil(K / ncols2)
    fig, axes2 = plt.subplots(nrows2, ncols2, figsize=(16, 4 * nrows2))
    axes2_flat = axes2.flatten() if hasattr(axes2, "flatten") else [axes2]
    for k in range(K):
        ax = axes2_flat[k]
        top_idx  = np.argsort(phi[k])[-10:]
        top_words = [vocab_list[i] for i in top_idx]
        top_probs = phi[k][top_idx]
        ax.barh(top_words, top_probs, color=PALETTE[k % len(PALETTE)])
        ax.set_title(topic_labels[k], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    for j in range(K, len(axes2_flat)):
        axes2_flat[j].axis("off")
    fig.suptitle("LDA Topic — Top-10 Words", fontsize=13, color="white")
    fig.tight_layout()
    st.pyplot(fig)

    c_heat, c_vol = st.columns(2)

    # Document-topic heatmap (50 docs)
    with c_heat:
        n_sample = min(50, len(theta))
        sample_theta = theta[:n_sample]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sample_theta, aspect="auto", cmap="plasma")
        ax.set_xlabel("Topic"); ax.set_ylabel("Document")
        ax.set_title(f"Doc-Topic Heatmap (first {n_sample} docs)", fontsize=11)
        ax.set_xticks(range(K)); ax.set_xticklabels(topic_labels, rotation=45, fontsize=6)
        plt.colorbar(im, ax=ax, label="θ")
        fig.tight_layout()
        st.pyplot(fig)

    # Topic volume over time (weekly)
    with c_vol:
        n_docs = min(len(theta), len(df_filt))
        df_topic = df_filt.iloc[:n_docs].copy()
        df_topic["dominant_topic"] = np.argmax(theta[:n_docs], axis=1)
        weekly_topic = df_topic.groupby(["week", "dominant_topic"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        for k in range(K):
            if k in weekly_topic.columns:
                ax.plot(weekly_topic.index, weekly_topic[k],
                        label=topic_labels[k], linewidth=1.5,
                        color=PALETTE[k % len(PALETTE)])
        ax.set_title("Topic Volume Over Time (weekly)", fontsize=11)
        ax.set_xlabel("Week"); ax.set_ylabel("Post Count")
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.grid(alpha=0.25)
        plt.xticks(rotation=30, fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.subheader("😊 Sentiment Analysis — Lexicon-Based (from Scratch)")

    st.markdown(
        "**Method:** each word scored ±1 (positive/negative lexicon). "
        "Negation within 3-word window flips polarity. "
        f"Class boundaries: Positive > {sent_thresh:.2f}, Negative < {-sent_thresh:.2f}."
    )

    def relabel(score: float) -> str:
        return "Positive" if score > sent_thresh else ("Negative" if score < -sent_thresh else "Neutral")

    senti_df2        = senti_df.copy()
    senti_df2["sentiment"] = senti_df2["sentiment_score"].apply(relabel)

    c1, c2, c3 = st.columns(3)
    vc = senti_df2["sentiment"].value_counts()
    c1.metric("Positive",  str(vc.get("Positive", 0)))
    c2.metric("Neutral",   str(vc.get("Neutral",  0)))
    c3.metric("Negative",  str(vc.get("Negative", 0)))

    r1c1, r1c2 = st.columns(2)

    # Stacked bar — sentiment by category
    with r1c1:
        cats_s = sel_cats or CATEGORIES
        pos_c, neu_c, neg_c = [], [], []
        for cat in cats_s:
            sub = senti_df2[senti_df2["category"] == cat]
            tot = max(len(sub), 1)
            pos_c.append(sub[sub["sentiment"] == "Positive"].shape[0] / tot)
            neu_c.append(sub[sub["sentiment"] == "Neutral"].shape[0]  / tot)
            neg_c.append(sub[sub["sentiment"] == "Negative"].shape[0] / tot)
        x = np.arange(len(cats_s))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x, pos_c, label="Positive", color=GREEN, alpha=0.9)
        ax.bar(x, neu_c, bottom=pos_c, label="Neutral", color=YELLOW, alpha=0.9)
        ax.bar(x, neg_c, bottom=[p+n for p, n in zip(pos_c, neu_c)],
               label="Negative", color=RED, alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(cats_s, rotation=40, ha="right", fontsize=7)
        ax.set_title("Sentiment Distribution by Category", fontsize=11)
        ax.set_ylabel("Proportion"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    # Sentiment vs engagement scatter
    with r1c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_map = {"Positive": GREEN, "Neutral": YELLOW, "Negative": RED}
        for sent, grp in senti_df2.groupby("sentiment"):
            ax.scatter(
                np.log10(grp["engagement"] + 1),
                grp["sentiment_score"],
                c=colors_map.get(sent, ACCENT),
                s=6, alpha=0.4, label=sent,
            )
        ax.set_title("Sentiment vs Engagement (log scale)", fontsize=11)
        ax.set_xlabel("log₁₀(engagement)"); ax.set_ylabel("Sentiment Score")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)

    # Sentiment time series — 7-day rolling avg by category
    st.markdown("#### Sentiment Time Series (7-day Rolling Average)")
    senti_df2["date"] = pd.to_datetime(senti_df2["date"])
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for idx, cat in enumerate(cats_s[:6]):
        sub = (senti_df2[senti_df2["category"] == cat]
               .groupby("date")["sentiment_score"].mean()
               .sort_index()
               .rolling(7, min_periods=1).mean())
        ax.plot(sub.index, sub.values, label=cat, linewidth=1.6,
                color=PALETTE[idx % len(PALETTE)])
    ax.set_title("7-day Rolling Avg Sentiment by Category", fontsize=12)
    ax.set_xlabel("Date"); ax.set_ylabel("Sentiment Score")
    ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, loc="upper left", ncol=3); ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

    # Entity-level sentiment
    st.markdown("#### Entity-Level Sentiment")
    entity_scores: dict[str, list[float]] = {e: [] for e in ENTITIES}
    for _, row in senti_df2.iterrows():
        txt_lower = row["text"].lower()
        for ent in ENTITIES:
            if ent.lower() in txt_lower:
                entity_scores[ent].append(row["sentiment_score"])
    ent_means = {e: np.mean(v) for e, v in entity_scores.items() if v}
    ent_sorted = sorted(ent_means.items(), key=lambda x: x[1])
    ents, means = zip(*ent_sorted) if ent_sorted else ([], [])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_e = [GREEN if m >= 0 else RED for m in means]
    ax.barh(list(ents), list(means), color=colors_e)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_title("Entity-Level Mean Sentiment", fontsize=12)
    ax.set_xlabel("Mean Sentiment Score"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TREND DETECTION & FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.subheader("📈 Trend Detection & Forecasting")

    st.markdown("#### Mann-Kendall Trend Test")
    st.latex(r"S = \sum_{i<j} \operatorname{sgn}(x_j - x_i)")
    st.latex(r"\operatorname{Var}(S) = \frac{n(n-1)(2n+5)}{18}")
    st.latex(
        r"Z = \begin{cases}"
        r"\dfrac{S-1}{\sqrt{\operatorname{Var}(S)}} & S>0 \\"
        r"0 & S=0 \\"
        r"\dfrac{S+1}{\sqrt{\operatorname{Var}(S)}} & S<0"
        r"\end{cases}"
    )

    cats_mk = sel_cats or CATEGORIES
    mk_rows = []
    for cat in cats_mk:
        sub = df_filt[df_filt["category"] == cat]
        daily = sub.groupby("date").size().sort_index()
        vals  = daily.values[-90:] if len(daily) >= 10 else daily.values
        if len(vals) >= 5:
            mk = mann_kendall(vals)
            mk_rows.append({
                "Category":    cat,
                "S":           mk["S"],
                "Z-score":     round(mk["Z"], 3),
                "Trend":       mk["trend"],
                "Significant": "✅" if mk["significant"] else "—",
            })
    st.dataframe(pd.DataFrame(mk_rows), use_container_width=True)

    st.markdown("---")

    # Emerging topics: 30-day growth rate
    st.markdown("#### Emerging & Declining Topics")
    now_ts   = pd.Timestamp("2026-06-29")
    last30   = df_filt[df_filt["timestamp"] >= now_ts - pd.Timedelta(days=30)]
    prior30  = df_filt[(df_filt["timestamp"] >= now_ts - pd.Timedelta(days=60)) &
                       (df_filt["timestamp"] <  now_ts - pd.Timedelta(days=30))]
    emerge_rows = []
    for cat in cats_mk:
        m_last  = last30[last30["category"] == cat].shape[0]
        m_prior = prior30[prior30["category"] == cat].shape[0]
        rate    = (m_last - m_prior) / (m_prior + 1)
        emerge_rows.append({"Topic": cat, "Last 30d": m_last,
                            "Prior 30d": m_prior, "Growth Rate": round(rate, 3)})
    emerge_df = pd.DataFrame(emerge_rows).sort_values("Growth Rate", ascending=False)

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Top 5 Emerging**")
        st.dataframe(emerge_df.head(5), use_container_width=True)
    with ec2:
        st.markdown("**Top 5 Declining**")
        st.dataframe(emerge_df.tail(5).iloc[::-1], use_container_width=True)

    # Burst detection
    st.markdown("#### Burst Detection")
    daily_all = df_filt.groupby("date").size().reset_index(name="count")
    daily_all["date"]     = pd.to_datetime(daily_all["date"])
    daily_all             = daily_all.sort_values("date")
    daily_all["roll_mu"]  = daily_all["count"].rolling(7, min_periods=1).mean()
    daily_all["roll_std"] = daily_all["count"].rolling(7, min_periods=1).std().fillna(0)
    daily_all["burst"]    = daily_all["count"] > daily_all["roll_mu"] + 2 * daily_all["roll_std"]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(daily_all["date"], daily_all["count"], color=ACCENT, linewidth=1, label="Daily posts")
    ax.plot(daily_all["date"], daily_all["roll_mu"] + 2 * daily_all["roll_std"],
            color=ORANGE, linewidth=1.2, linestyle="--", label="Burst threshold")
    burst_pts = daily_all[daily_all["burst"]]
    ax.scatter(burst_pts["date"], burst_pts["count"], color=RED, zorder=5,
               s=30, label="Burst day")
    ax.set_title("Volume Burst Detection (rolling mean + 2σ)", fontsize=12)
    ax.set_xlabel("Date"); ax.set_ylabel("Posts")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    st.info(f"Detected **{burst_pts.shape[0]}** burst days in the selected date range.")

    # AR(3) Forecast — 14 days ahead
    st.markdown("#### AR(3) Topic Forecast — Next 14 Days")
    st.latex(r"x_t = c_1 x_{t-1} + c_2 x_{t-2} + c_3 x_{t-3} + \varepsilon_t")
    st.latex(r"\text{CI: } \hat{x}_t \pm 1.96\,\sigma")

    sel_cat_fc = st.selectbox("Category to forecast", cats_mk, key="fc_cat")
    sub_fc = df_filt[df_filt["category"] == sel_cat_fc]
    daily_fc = sub_fc.groupby("date").size().sort_index()
    if len(daily_fc) >= 10:
        vals_fc    = daily_fc.values[-60:].astype(float)
        idx_fc     = pd.to_datetime(list(daily_fc.index[-60:]))
        preds, sigma_fc, coef_fc = ar3_forecast(vals_fc, steps=14)
        if preds is not None:
            future_idx = pd.date_range(idx_fc[-1] + pd.Timedelta(days=1), periods=14, freq="D")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(idx_fc, vals_fc, color=ACCENT, linewidth=1.5, label="Historical")
            ax.plot(future_idx, preds, color=GREEN, linewidth=2, linestyle="--", label="AR(3) Forecast")
            ax.fill_between(future_idx, preds - 1.96*sigma_fc, preds + 1.96*sigma_fc,
                            color=GREEN, alpha=0.2, label="95% CI")
            ax.set_title(f"AR(3) Forecast — {sel_cat_fc} (next 14 days)", fontsize=12)
            ax.set_xlabel("Date"); ax.set_ylabel("Daily Posts")
            ax.legend(fontsize=9); ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            st.caption(f"AR(3) coefficients: {coef_fc.round(4).tolist()} | σ = {sigma_fc:.2f}")
        else:
            st.warning("Not enough data for AR(3) forecast.")
    else:
        st.warning("Not enough data in selected date range for this category.")

    # Trending keywords table (top 20 bigrams by WoW change)
    st.markdown("#### Trending Keywords — Week-over-Week")
    this_week = df_filt[df_filt["timestamp"] >= now_ts - pd.Timedelta(days=7)]
    last_week = df_filt[(df_filt["timestamp"] >= now_ts - pd.Timedelta(days=14)) &
                        (df_filt["timestamp"] <  now_ts - pd.Timedelta(days=7))]

    def word_freq_from_df(sub: pd.DataFrame) -> dict[str, int]:
        freq_d: dict[str, int] = {}
        for txt in sub["text"]:
            for tok in preprocess(txt):
                freq_d[tok] = freq_d.get(tok, 0) + 1
        return freq_d

    freq_this = word_freq_from_df(this_week)
    freq_last = word_freq_from_df(last_week)
    kw_rows   = []
    all_kws   = set(list(freq_this.keys())[:200])
    for kw in sorted(all_kws, key=lambda k: freq_this.get(k, 0), reverse=True)[:30]:
        cur  = freq_this.get(kw, 0)
        prev = freq_last.get(kw, 1)
        wow  = (cur - prev) / prev
        kw_rows.append({"Keyword": kw, "This Week": cur, "Last Week": prev,
                        "WoW Change": f"{wow:+.1%}"})
    kw_df = pd.DataFrame(kw_rows).sort_values("This Week", ascending=False)
    st.dataframe(kw_df.head(20), use_container_width=True)

    # Cross-topic correlation matrix
    st.markdown("#### Cross-Topic Correlation Matrix")
    n_docs_lda  = min(len(lda_res["theta"]), len(df_filt))
    df_topics_x = df_filt.iloc[:n_docs_lda].copy()
    theta_x     = lda_res["theta"][:n_docs_lda]
    topic_names = [f"T{k+1}" for k in range(K_topics)]
    theta_df    = pd.DataFrame(theta_x, columns=topic_names)
    corr_mat    = theta_df.corr().values

    fig, ax = plt.subplots(figsize=(8, 6))
    im2 = ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(K_topics)); ax.set_xticklabels(topic_names, rotation=45, fontsize=8)
    ax.set_yticks(range(K_topics)); ax.set_yticklabels(topic_names, fontsize=8)
    for i in range(K_topics):
        for j in range(K_topics):
            ax.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr_mat[i,j]) > 0.4 else "black")
    plt.colorbar(im2, ax=ax, label="Pearson r")
    ax.set_title("Cross-Topic Correlation (doc-topic distributions)", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:12px'>"
    "TrendScope v2.0 · NLP Trend Mining & Text Analytics · "
    "Built with Streamlit, NumPy, Pandas, Matplotlib · "
    "All NLP/ML implemented from scratch"
    "</p>",
    unsafe_allow_html=True,
)
