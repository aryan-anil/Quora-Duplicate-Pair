"""
Feature engineering pipeline for Quora duplicate question detection.

Feature groups:
  1. Basic NLP features        (~15 features)
  2. TF-IDF features           (~3 features)
  3. Semantic embedding feats  (~10 features)
  4. Graph / "magic" features  (~8 features)
  5. Question-word features    (~3 features)
"""
import hashlib
import logging
import re
import string
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, cityblock, euclidean, minkowski
from scipy.stats import skew, kurtosis
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Text cleaning
# ══════════════════════════════════════════════════════════════════════════════
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Minimal cleaning: lowercase, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def tokenize_simple(text: str) -> list:
    """Whitespace tokenizer on cleaned text."""
    return clean_text(text).split()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Basic NLP features
# ══════════════════════════════════════════════════════════════════════════════
def basic_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """~15 basic textual similarity features."""
    log.info("  Computing basic NLP features …")
    q1 = df["question1"].fillna("").astype(str)
    q2 = df["question2"].fillna("").astype(str)

    feats = pd.DataFrame(index=df.index)

    # lengths
    feats["q1_word_count"] = q1.apply(lambda x: len(x.split()))
    feats["q2_word_count"] = q2.apply(lambda x: len(x.split()))
    feats["word_count_diff"] = (feats["q1_word_count"] - feats["q2_word_count"]).abs()
    feats["word_count_ratio"] = feats.apply(
        lambda r: min(r["q1_word_count"], r["q2_word_count"]) /
                  max(r["q1_word_count"], r["q2_word_count"])
        if max(r["q1_word_count"], r["q2_word_count"]) > 0 else 0.0, axis=1
    )
    feats["q1_char_count"] = q1.str.len()
    feats["q2_char_count"] = q2.str.len()
    feats["char_count_diff"] = (feats["q1_char_count"] - feats["q2_char_count"]).abs()
    feats["char_count_ratio"] = feats.apply(
        lambda r: min(r["q1_char_count"], r["q2_char_count"]) /
                  max(r["q1_char_count"], r["q2_char_count"])
        if max(r["q1_char_count"], r["q2_char_count"]) > 0 else 0.0, axis=1
    )

    # shared words
    q1_words = q1.apply(lambda x: set(x.lower().split()))
    q2_words = q2.apply(lambda x: set(x.lower().split()))
    common = q1_words.combine(q2_words, func=lambda a, b: a & b)
    union = q1_words.combine(q2_words, func=lambda a, b: a | b)

    feats["common_word_count"] = common.apply(len)
    feats["total_unique_words"] = union.apply(len)
    feats["common_word_ratio"] = feats["common_word_count"] / (
        feats["total_unique_words"].replace(0, 1)
    )

    # Jaccard similarity
    feats["jaccard"] = feats["common_word_count"] / (
        feats["total_unique_words"].replace(0, 1)
    )

    # n-gram overlap
    def _get_ngrams(text, n):
        words = text.lower().split()
        return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()

    for n, label in [(2, "bigram"), (3, "trigram")]:
        q1_ng = q1.apply(lambda x: _get_ngrams(x, n))
        q2_ng = q2.apply(lambda x: _get_ngrams(x, n))
        shared = q1_ng.combine(q2_ng, func=lambda a, b: len(a & b))
        total = q1_ng.combine(q2_ng, func=lambda a, b: len(a | b))
        feats[f"shared_{label}"] = shared
        feats[f"{label}_ratio"] = shared / total.replace(0, 1)

    # punctuation
    feats["q1_punct_count"] = q1.apply(lambda x: sum(c in string.punctuation for c in x))
    feats["q2_punct_count"] = q2.apply(lambda x: sum(c in string.punctuation for c in x))

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 2. TF-IDF features
# ══════════════════════════════════════════════════════════════════════════════
def tfidf_features(df: pd.DataFrame, max_features: int = 50_000,
                   ngram_range=(1, 2),
                   vectorizer=None) -> tuple:
    """
    TF-IDF cosine similarity + LSA/SVD components.
    Returns (features_df, fitted_vectorizer).
    """
    log.info("  Computing TF-IDF features …")
    q1 = df["question1"].fillna("").astype(str)
    q2 = df["question2"].fillna("").astype(str)

    if vectorizer is None:
        # Fit on the union of all questions
        all_questions = pd.concat([q1, q2], ignore_index=True)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            dtype=np.float32,
        )
        vectorizer.fit(all_questions)

    tfidf_q1 = vectorizer.transform(q1)
    tfidf_q2 = vectorizer.transform(q2)

    feats = pd.DataFrame(index=df.index)

    # Row-wise cosine similarity (using sparse dot)
    dot_product = np.array(tfidf_q1.multiply(tfidf_q2).sum(axis=1)).flatten()
    norm_q1 = np.sqrt(np.array(tfidf_q1.multiply(tfidf_q1).sum(axis=1)).flatten())
    norm_q2 = np.sqrt(np.array(tfidf_q2.multiply(tfidf_q2).sum(axis=1)).flatten())
    denom = norm_q1 * norm_q2
    denom[denom == 0] = 1.0
    feats["tfidf_cosine"] = dot_product / denom

    # LSA: project into 20-dim space & compute cosine there
    n_components = min(20, max_features)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    all_tfidf = vectorizer.transform(pd.concat([q1, q2], ignore_index=True))
    svd.fit(all_tfidf)

    lsa_q1 = svd.transform(tfidf_q1)
    lsa_q2 = svd.transform(tfidf_q2)

    # cosine in LSA space
    dot_lsa = np.sum(lsa_q1 * lsa_q2, axis=1)
    norm_lsa1 = np.linalg.norm(lsa_q1, axis=1)
    norm_lsa2 = np.linalg.norm(lsa_q2, axis=1)
    denom_lsa = norm_lsa1 * norm_lsa2
    denom_lsa[denom_lsa == 0] = 1.0
    feats["lsa_cosine"] = dot_lsa / denom_lsa

    # euclidean in LSA space
    feats["lsa_euclidean"] = np.linalg.norm(lsa_q1 - lsa_q2, axis=1)

    return feats, vectorizer


# ══════════════════════════════════════════════════════════════════════════════
# 3. Semantic embedding features (GloVe / Word2Vec via gensim)
# ══════════════════════════════════════════════════════════════════════════════
def _load_embedding_model(model_name: str):
    """Lazy-load gensim word vector model."""
    import gensim.downloader as gensim_api
    log.info(f"  Loading embedding model: {model_name} (first time may download) …")
    return gensim_api.load(model_name)


def _sentence_vector(tokens: list, model, dim: int) -> np.ndarray:
    """Average word vectors for a list of tokens."""
    vecs = []
    for tok in tokens:
        if tok in model:
            vecs.append(model[tok])
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def _safe_distance(fn, a, b):
    """Compute distance, return 0 if vectors are zero."""
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    try:
        return float(fn(a, b))
    except Exception:
        return 0.0


def embedding_features(df: pd.DataFrame, model_name: str = "glove-wiki-gigaword-300",
                       dim: int = 300) -> pd.DataFrame:
    """Distances between averaged word-vector representations of Q1 and Q2."""
    log.info("  Computing embedding features …")
    model = _load_embedding_model(model_name)

    q1_tokens = df["question1"].fillna("").apply(tokenize_simple)
    q2_tokens = df["question2"].fillna("").apply(tokenize_simple)

    q1_vecs = np.stack(q1_tokens.apply(lambda t: _sentence_vector(t, model, dim)).values)
    q2_vecs = np.stack(q2_tokens.apply(lambda t: _sentence_vector(t, model, dim)).values)

    feats = pd.DataFrame(index=df.index)

    # Distance metrics
    feats["emb_cosine"] = [_safe_distance(cosine, q1_vecs[i], q2_vecs[i])
                           for i in range(len(df))]
    feats["emb_manhattan"] = [_safe_distance(cityblock, q1_vecs[i], q2_vecs[i])
                              for i in range(len(df))]
    feats["emb_euclidean"] = [_safe_distance(euclidean, q1_vecs[i], q2_vecs[i])
                              for i in range(len(df))]
    feats["emb_minkowski_p3"] = [
        _safe_distance(lambda a, b: minkowski(a, b, p=3), q1_vecs[i], q2_vecs[i])
        for i in range(len(df))
    ]

    # Difference vector statistics
    diff = q1_vecs - q2_vecs
    feats["emb_diff_mean"] = np.mean(diff, axis=1)
    feats["emb_diff_std"] = np.std(diff, axis=1)
    feats["emb_diff_skew"] = [skew(diff[i]) for i in range(len(df))]
    feats["emb_diff_kurtosis"] = [kurtosis(diff[i]) for i in range(len(df))]

    # Dot product
    feats["emb_dot"] = np.sum(q1_vecs * q2_vecs, axis=1)

    # Norm ratio
    norm_q1 = np.linalg.norm(q1_vecs, axis=1)
    norm_q2 = np.linalg.norm(q2_vecs, axis=1)
    feats["emb_norm_ratio"] = np.minimum(norm_q1, norm_q2) / (
        np.maximum(norm_q1, norm_q2) + 1e-8
    )

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 4. Graph / "Magic" features
# ══════════════════════════════════════════════════════════════════════════════
def graph_features(df: pd.DataFrame, full_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Structural graph features based on the question-pair graph.
    Uses the full training set (or provided df) to build the graph.

    Features:
      - qid frequency: how often each question appears across *all* pairs
      - shared neighbour count: overlap in the neighbourhood of qid1 and qid2
      - k-core number: the k-core of each question in the pair graph
      - question-text hash frequency: frequency of question text hash
    """
    log.info("  Computing graph / magic features …")
    feats = pd.DataFrame(index=df.index)

    # Use full_df for graph construction if provided, otherwise use df
    graph_df = full_df if full_df is not None else df

    # ── QID frequency ─────────────────────────────────────────────────────
    if "qid1" in graph_df.columns and "qid2" in graph_df.columns:
        qid_counts = pd.concat([
            graph_df["qid1"], graph_df["qid2"]
        ]).value_counts().to_dict()

        if "qid1" in df.columns and "qid2" in df.columns:
            feats["qid1_freq"] = df["qid1"].map(qid_counts).fillna(0)
            feats["qid2_freq"] = df["qid2"].map(qid_counts).fillna(0)
            feats["qid_freq_min"] = feats[["qid1_freq", "qid2_freq"]].min(axis=1)
            feats["qid_freq_max"] = feats[["qid1_freq", "qid2_freq"]].max(axis=1)
        else:
            # test set won't have qid columns; we use text-hash frequencies
            feats["qid1_freq"] = 0
            feats["qid2_freq"] = 0
            feats["qid_freq_min"] = 0
            feats["qid_freq_max"] = 0

        # ── Neighbor overlap ──────────────────────────────────────────────
        neighbours = defaultdict(set)
        for _, row in graph_df.iterrows():
            q1, q2 = row.get("qid1"), row.get("qid2")
            if pd.notna(q1) and pd.notna(q2):
                neighbours[q1].add(q2)
                neighbours[q2].add(q1)

        if "qid1" in df.columns and "qid2" in df.columns:
            feats["shared_neighbours"] = df.apply(
                lambda r: len(neighbours.get(r["qid1"], set()) &
                              neighbours.get(r["qid2"], set()))
                if pd.notna(r.get("qid1")) and pd.notna(r.get("qid2")) else 0,
                axis=1,
            )
        else:
            feats["shared_neighbours"] = 0

        # ── K-core (simplified via networkx) ──────────────────────────────
        try:
            import networkx as nx
            G = nx.Graph()
            for _, row in graph_df.iterrows():
                q1, q2 = row.get("qid1"), row.get("qid2")
                if pd.notna(q1) and pd.notna(q2):
                    G.add_edge(int(q1), int(q2))
            core_numbers = nx.core_number(G)

            if "qid1" in df.columns and "qid2" in df.columns:
                feats["q1_kcore"] = df["qid1"].map(core_numbers).fillna(0).astype(int)
                feats["q2_kcore"] = df["qid2"].map(core_numbers).fillna(0).astype(int)
                feats["max_kcore"] = feats[["q1_kcore", "q2_kcore"]].max(axis=1)
            else:
                feats["q1_kcore"] = 0
                feats["q2_kcore"] = 0
                feats["max_kcore"] = 0
        except ImportError:
            log.warning("networkx not installed — skipping k-core features")
            feats["q1_kcore"] = 0
            feats["q2_kcore"] = 0
            feats["max_kcore"] = 0
    else:
        # No qid columns at all
        for col in ["qid1_freq", "qid2_freq", "qid_freq_min", "qid_freq_max",
                     "shared_neighbours", "q1_kcore", "q2_kcore", "max_kcore"]:
            feats[col] = 0

    # ── Question-text hash frequency ──────────────────────────────────────
    def _text_hash(text):
        return hashlib.md5(str(text).lower().strip().encode()).hexdigest()

    # Build hash frequency table from full dataset
    all_q1 = graph_df["question1"].fillna("").apply(_text_hash)
    all_q2 = graph_df["question2"].fillna("").apply(_text_hash)
    hash_freq = pd.concat([all_q1, all_q2]).value_counts().to_dict()

    q1_hash = df["question1"].fillna("").apply(_text_hash)
    q2_hash = df["question2"].fillna("").apply(_text_hash)
    feats["q1_hash_freq"] = q1_hash.map(hash_freq).fillna(0)
    feats["q2_hash_freq"] = q2_hash.map(hash_freq).fillna(0)

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 5. Question-word features
# ══════════════════════════════════════════════════════════════════════════════
WH_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whom",
            "whose", "is", "are", "do", "does", "did", "can", "could",
            "will", "would", "should"}


def question_word_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features based on the first word (interrogative) of each question."""
    log.info("  Computing question-word features …")
    feats = pd.DataFrame(index=df.index)

    def _first_word(text):
        text = str(text).strip().lower()
        match = re.match(r"[a-z]+", text)
        return match.group() if match else ""

    q1_fw = df["question1"].fillna("").apply(_first_word)
    q2_fw = df["question2"].fillna("").apply(_first_word)

    feats["same_first_word"] = (q1_fw == q2_fw).astype(int)
    feats["both_wh_words"] = (
        q1_fw.isin(WH_WORDS) & q2_fw.isin(WH_WORDS)
    ).astype(int)
    feats["wh_mismatch"] = (
        feats["both_wh_words"].astype(bool) & ~feats["same_first_word"].astype(bool)
    ).astype(int)

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: build full feature matrix
# ══════════════════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame,
                   full_train_df: pd.DataFrame = None,
                   embedding_model: str = "glove-wiki-gigaword-300",
                   embedding_dim: int = 300,
                   tfidf_max_features: int = 50_000,
                   tfidf_ngram_range=(1, 2),
                   tfidf_vectorizer=None,
                   cache_dir: str = None) -> tuple:
    """
    Build the complete feature matrix for a DataFrame.

    Returns (feature_df, tfidf_vectorizer) so the vectorizer can be reused
    for the test set.
    """
    log.info(f"Building features for {len(df)} rows …")

    parts = [
        basic_nlp_features(df),
    ]

    tfidf_feats, tfidf_vec = tfidf_features(
        df, max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
        vectorizer=tfidf_vectorizer,
    )
    parts.append(tfidf_feats)

    parts.append(embedding_features(df, model_name=embedding_model,
                                    dim=embedding_dim))
    parts.append(graph_features(df, full_df=full_train_df))
    parts.append(question_word_features(df))

    feature_df = pd.concat(parts, axis=1)

    # Replace inf/nan
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0)

    log.info(f"  Total features: {feature_df.shape[1]}")
    return feature_df, tfidf_vec
