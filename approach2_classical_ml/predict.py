"""
Inference for the Classical ML approach.

Usage:
    python -m approach2_classical_ml.predict
    python -m approach2_classical_ml.predict --dry-run
"""
import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd

from .config import TRAIN_CSV, TEST_CSV, ClassicalConfig
from .features import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Calibration (same Bayesian rescaling reused across approaches)
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_probabilities(probs: np.ndarray,
                            train_pos: float, test_pos: float) -> np.ndarray:
    """Adjust probabilities for train/test class prior shift."""
    a = test_pos / train_pos
    b = (1.0 - test_pos) / (1.0 - train_pos)
    return (probs * a) / (probs * a + (1.0 - probs) * b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    cfg = ClassicalConfig()

    # ── load test data ────────────────────────────────────────────────────
    log.info("Loading test data …")
    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df.fillna("")
    if args.dry_run:
        test_df = test_df.head(500)
        log.info("🔬  Dry-run: 500 test samples")

    # ── load train data (for graph features) ──────────────────────────────
    log.info("Loading training data (for graph feature construction) …")
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df.dropna(subset=["question1", "question2"]).reset_index(drop=True)

    # ── load saved artifacts ──────────────────────────────────────────────
    tfidf_vec = joblib.load(os.path.join(cfg.output_dir, "tfidf_vectorizer.pkl"))
    feature_names = joblib.load(os.path.join(cfg.output_dir, "feature_names.pkl"))

    # ── build features ────────────────────────────────────────────────────
    log.info("Building test features …")
    test_features, _ = build_features(
        test_df, full_train_df=train_df,
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        tfidf_max_features=cfg.tfidf_max_features,
        tfidf_ngram_range=cfg.tfidf_ngram_range,
        tfidf_vectorizer=tfidf_vec,
    )

    # Ensure column alignment
    for col in feature_names:
        if col not in test_features.columns:
            test_features[col] = 0
    test_features = test_features[feature_names]
    X_test = test_features.values.astype(np.float32)

    # ── predict with each model type ──────────────────────────────────────
    model_types = {
        "lgbm": ("lgbm_fold{}.pkl", X_test),
        "xgb": ("xgb_fold{}.pkl", X_test),
    }

    # LR needs scaled features
    scaler_path = os.path.join(cfg.output_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test_scaled = scaler.transform(X_test)
        model_types["lr"] = ("lr_fold{}.pkl", X_test_scaled)

    all_probs = {}

    for model_name, (pattern, X_input) in model_types.items():
        fold_probs = []
        for fold_idx in range(cfg.n_folds):
            path = os.path.join(cfg.output_dir, pattern.format(fold_idx))
            if not os.path.exists(path):
                continue
            model = joblib.load(path)
            probs = model.predict_proba(X_input)[:, 1]
            fold_probs.append(probs)
        if fold_probs:
            avg = np.mean(fold_probs, axis=0)
            all_probs[model_name] = avg
            log.info(f"  [{model_name}] Averaged {len(fold_probs)} folds, "
                     f"mean prob = {avg.mean():.4f}")

    if not all_probs:
        raise FileNotFoundError("No model files found. Train first.")

    # ── ensemble across model types (simple average) ──────────────────────
    ensemble_prob = np.mean(list(all_probs.values()), axis=0)

    # ── calibration ───────────────────────────────────────────────────────
    if not args.no_calibrate:
        ensemble_prob = calibrate_probabilities(
            ensemble_prob, cfg.train_pos_rate, cfg.test_pos_rate,
        )

    # ── save submission ───────────────────────────────────────────────────
    sub = pd.DataFrame({
        "test_id": test_df["test_id"],
        "is_duplicate": ensemble_prob,
    })
    out_path = os.path.join(cfg.output_dir, "submission_classical.csv")
    sub.to_csv(out_path, index=False)
    log.info(f"Submission saved → {out_path}  ({len(sub)} rows)")

    # ── also save per-model submissions (uncalibrated) ────────────────────
    for model_name, probs in all_probs.items():
        raw_sub = pd.DataFrame({
            "test_id": test_df["test_id"],
            "is_duplicate": probs,
        })
        raw_path = os.path.join(cfg.output_dir, f"submission_{model_name}_raw.csv")
        raw_sub.to_csv(raw_path, index=False)
    log.info("Done!")


if __name__ == "__main__":
    main()
