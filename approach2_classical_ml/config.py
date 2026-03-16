"""
Configuration for Approach 2: Classical ML + Feature Engineering.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
TRAIN_CSV = os.path.join(DATA_DIR, "train", "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test", "test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FEATURE_DIR = os.path.join(BASE_DIR, "features_cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)


@dataclass
class ClassicalConfig:
    """All configuration for the classical ML pipeline."""

    # ── Data ──────────────────────────────────────────────────────────────
    seed: int = 42
    n_folds: int = 10

    # ── Embeddings ────────────────────────────────────────────────────────
    # Will download automatically via gensim if not present
    embedding_model: str = "glove-wiki-gigaword-300"
    embedding_dim: int = 300

    # ── TF-IDF ────────────────────────────────────────────────────────────
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: tuple = (1, 2)

    # ── LightGBM ──────────────────────────────────────────────────────────
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    })

    # ── Logistic Regression ───────────────────────────────────────────────
    lr_params: Dict[str, Any] = field(default_factory=lambda: {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    })

    early_stopping_rounds: int = 100

    # ── Calibration ───────────────────────────────────────────────────────
    train_pos_rate: float = 0.3692
    test_pos_rate: float = 0.165

    # ── Paths ─────────────────────────────────────────────────────────────
    output_dir: str = OUTPUT_DIR
    feature_dir: str = FEATURE_DIR
