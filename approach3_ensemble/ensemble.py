"""
Ensemble pipeline: combine predictions from the transformer and classical ML
approaches for a more robust final model.

Three ensembling strategies:
  1. Weighted average of probabilities
  2. Rank-based averaging
  3. Stacking (meta-learner on OOF predictions)

Usage:
    python -m approach3_ensemble.ensemble
    python -m approach3_ensemble.ensemble --method stacking
    python -m approach3_ensemble.ensemble --method rank
    python -m approach3_ensemble.ensemble --dry-run
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .calibration import calibrate_probabilities, find_optimal_threshold
from .config import EnsembleConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _load_predictions(path: str, col: str = "is_duplicate") -> np.ndarray:
    """Load a submission CSV and return the probability column."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            f"Please run the corresponding approach first."
        )
    df = pd.read_csv(path)
    return df[col].values.astype(np.float64)


def _rank_average(*prob_arrays: np.ndarray) -> np.ndarray:
    """
    Rank-based ensemble: convert each set of probabilities to ranks,
    average the ranks, then normalize to [0, 1].
    This is more robust than simple probability averaging when models
    have different calibration scales.
    """
    from scipy.stats import rankdata
    ranks = [rankdata(p) for p in prob_arrays]
    avg_rank = np.mean(ranks, axis=0)
    # Normalize to [0, 1]
    return (avg_rank - avg_rank.min()) / (avg_rank.max() - avg_rank.min() + 1e-15)


# ══════════════════════════════════════════════════════════════════════════════
# Method 1: Weighted average
# ══════════════════════════════════════════════════════════════════════════════
def weighted_average(cfg: EnsembleConfig) -> np.ndarray:
    """
    Simple weighted mean of transformer + classical ML probabilities.
    Uses the pre-calibrated submissions by default.
    """
    log.info("Ensembling via weighted average …")

    probs = {}
    weights = {}

    if os.path.exists(cfg.transformer_submission):
        probs["transformer"] = _load_predictions(cfg.transformer_submission)
        weights["transformer"] = cfg.weights.get("transformer", 0.5)
        log.info(f"  Loaded transformer predictions: {len(probs['transformer'])} rows")

    if os.path.exists(cfg.classical_submission):
        probs["classical"] = _load_predictions(cfg.classical_submission)
        weights["classical"] = cfg.weights.get("classical", 0.5)
        log.info(f"  Loaded classical predictions: {len(probs['classical'])} rows")

    if not probs:
        raise FileNotFoundError("No prediction files found from either approach.")

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Weighted sum
    n = max(len(v) for v in probs.values())
    ensemble = np.zeros(n)
    for name, p in probs.items():
        ensemble += weights[name] * p
        log.info(f"  {name}: weight={weights[name]:.3f}, mean_prob={p.mean():.4f}")

    log.info(f"  Ensemble mean prob = {ensemble.mean():.4f}")
    return ensemble


# ══════════════════════════════════════════════════════════════════════════════
# Method 2: Rank-based average
# ══════════════════════════════════════════════════════════════════════════════
def rank_ensemble(cfg: EnsembleConfig) -> np.ndarray:
    """
    Rank-based ensemble: convert to ranks, average, normalize.
    More robust when models have very different probability scales.
    """
    log.info("Ensembling via rank averaging …")

    arrays = []
    names = []

    # Try to load all raw (uncalibrated) per-model predictions for diversity
    for name, path in [
        ("transformer", cfg.transformer_submission),
        ("lgbm", cfg.lgbm_raw),
        ("xgb", cfg.xgb_raw),
        ("lr", cfg.lr_raw),
    ]:
        if os.path.exists(path):
            arrays.append(_load_predictions(path))
            names.append(name)
            log.info(f"  Loaded {name}: {len(arrays[-1])} rows")

    # Fallback to the calibrated ensemble predictions
    if not arrays:
        if os.path.exists(cfg.transformer_submission):
            arrays.append(_load_predictions(cfg.transformer_submission))
            names.append("transformer")
        if os.path.exists(cfg.classical_submission):
            arrays.append(_load_predictions(cfg.classical_submission))
            names.append("classical")

    if not arrays:
        raise FileNotFoundError("No prediction files found.")

    log.info(f"  Rank-averaging {len(arrays)} models: {names}")
    ensemble = _rank_average(*arrays)

    # Calibrate the rank-averaged scores
    ensemble = calibrate_probabilities(
        ensemble, cfg.train_pos_rate, cfg.test_pos_rate,
    )

    log.info(f"  Ensemble (calibrated) mean = {ensemble.mean():.4f}")
    return ensemble


# ══════════════════════════════════════════════════════════════════════════════
# Method 3: Stacking
# ══════════════════════════════════════════════════════════════════════════════
def stacking_ensemble(cfg: EnsembleConfig) -> np.ndarray:
    """
    Train a meta-learner (logistic regression or LightGBM) on out-of-fold
    predictions from approach 2, then predict on test using all available
    model outputs as features.

    OOF file columns: lgbm, xgb, lr, target
    Test features: predictions from each model (averaged across folds).
    """
    log.info("Ensembling via stacking …")

    # ── load OOF training data ────────────────────────────────────────────
    if not os.path.exists(cfg.oof_predictions):
        raise FileNotFoundError(
            f"OOF predictions not found: {cfg.oof_predictions}\n"
            f"Run approach2_classical_ml.train first."
        )
    oof_df = pd.read_csv(cfg.oof_predictions)
    log.info(f"  OOF data: {oof_df.shape}, columns: {oof_df.columns.tolist()}")

    feature_cols = [c for c in oof_df.columns if c != "target"]
    X_oof = oof_df[feature_cols].values
    y_oof = oof_df["target"].values

    # ── train meta-learner ────────────────────────────────────────────────
    if cfg.stacking_model == "lightgbm":
        try:
            import lightgbm as lgb
            meta = lgb.LGBMClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=cfg.seed, verbose=-1,
            )
        except ImportError:
            log.warning("LightGBM not available, falling back to Logistic Regression")
            meta = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg.seed)
    else:
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg.seed)

    meta.fit(X_oof, y_oof)

    # Evaluate on OOF
    oof_probs = meta.predict_proba(X_oof)[:, 1]
    oof_preds = (oof_probs >= 0.5).astype(int)
    log.info(
        f"  Meta-learner OOF: acc={accuracy_score(y_oof, oof_preds):.4f}  "
        f"f1={f1_score(y_oof, oof_preds):.4f}  "
        f"ll={log_loss(y_oof, oof_probs, labels=[0, 1]):.4f}"
    )

    # Optimal threshold
    best_thr, best_f1 = find_optimal_threshold(y_oof, oof_probs)
    log.info(f"  Optimal F1 threshold: {best_thr:.3f} → F1={best_f1:.4f}")

    # ── predict on test ───────────────────────────────────────────────────
    test_features = []
    for col in feature_cols:
        # Try to load the raw per-model test prediction
        raw_map = {
            "lgbm": cfg.lgbm_raw,
            "xgb": cfg.xgb_raw,
            "lr": cfg.lr_raw,
        }
        path = raw_map.get(col)
        if path and os.path.exists(path):
            test_features.append(_load_predictions(path))
            log.info(f"  Loaded test feature: {col}")
        else:
            log.warning(f"  Missing test predictions for '{col}', "
                        f"using zeros — result may be inaccurate")
            n_test = max(len(f) for f in test_features) if test_features else 0
            test_features.append(np.zeros(n_test))

    X_test = np.column_stack(test_features)
    test_probs = meta.predict_proba(X_test)[:, 1]

    # Calibrate
    test_probs = calibrate_probabilities(
        test_probs, cfg.train_pos_rate, cfg.test_pos_rate,
    )

    log.info(f"  Stacking ensemble mean prob = {test_probs.mean():.4f}")
    return test_probs


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Ensemble predictions from approaches 1 & 2.")
    parser.add_argument(
        "--method", choices=["weighted", "rank", "stacking"], default="weighted",
        help="Ensembling strategy (default: weighted).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip calibration for weighted/rank methods.")
    args = parser.parse_args()

    cfg = EnsembleConfig()

    # ── run selected method ───────────────────────────────────────────────
    if args.method == "weighted":
        probs = weighted_average(cfg)
    elif args.method == "rank":
        probs = rank_ensemble(cfg)
    elif args.method == "stacking":
        probs = stacking_ensemble(cfg)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # ── load test IDs ─────────────────────────────────────────────────────
    test_df = pd.read_csv(cfg.transformer_submission if os.path.exists(
        cfg.transformer_submission) else cfg.classical_submission)
    test_ids = test_df["test_id"]

    if args.dry_run:
        probs = probs[:500]
        test_ids = test_ids[:500]

    # ── save ──────────────────────────────────────────────────────────────
    sub = pd.DataFrame({
        "test_id": test_ids,
        "is_duplicate": probs,
    })
    out_path = os.path.join(cfg.output_dir,
                            f"submission_ensemble_{args.method}.csv")
    sub.to_csv(out_path, index=False)
    log.info(f"Ensemble submission saved → {out_path}  ({len(sub)} rows)")

    # ── summary stats ─────────────────────────────────────────────────────
    log.info(f"  Mean prob:  {probs.mean():.4f}")
    log.info(f"  Std prob:   {probs.std():.4f}")
    log.info(f"  Min / Max:  {probs.min():.4f} / {probs.max():.4f}")
    log.info(f"  % > 0.5:    {(probs > 0.5).mean() * 100:.1f}%")
    log.info("Done!")


if __name__ == "__main__":
    main()
