"""
Training script for the Classical ML approach.

Runs 10-fold stratified CV with LightGBM (primary), XGBoost, and Logistic
Regression.  Saves fold models, OOF predictions, and per-fold metrics.

Usage:
    python -m approach2_classical_ml.train
    python -m approach2_classical_ml.train --dry-run          # 500 samples
    python -m approach2_classical_ml.train --folds 5
"""
import argparse
import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from config import TRAIN_CSV, ClassicalConfig
from features import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Model factories
# ══════════════════════════════════════════════════════════════════════════════
def _make_lgbm(params: dict):
    """Build a LightGBM classifier."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(**params)


def _make_xgb(params: dict):
    """Build an XGBoost classifier."""
    import xgboost as xgb
    return xgb.XGBClassifier(**params)


def _make_lr(params: dict):
    return LogisticRegression(**params)


# ══════════════════════════════════════════════════════════════════════════════
# Per-fold training
# ══════════════════════════════════════════════════════════════════════════════
def train_fold(model_cls, model_params, X_train, y_train, X_val, y_val,
               model_name: str = "model", early_stopping_rounds: int = 100):
    """
    Train a single model on one fold.  Returns (model, val_probs, metrics).
    """
    model = model_cls(model_params)

    fit_kwargs = {}
    if model_name in ("lgbm", "xgb"):
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        if model_name == "lgbm":
            fit_kwargs["callbacks"] = [
                __import__("lightgbm").early_stopping(early_stopping_rounds, verbose=False),
                __import__("lightgbm").log_evaluation(period=0),
            ]
        elif model_name == "xgb":
            # XGBoost sklearn API changed across versions; prefer callbacks for compatibility
            fit_kwargs["verbose"] = False

    if model_name == "xgb":
        try:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            # Some XGBoost versions don't accept early_stopping_rounds or callbacks in sklearn API.
            fit_kwargs.pop("early_stopping_rounds", None)
            fit_kwargs.pop("callbacks", None)
            model.fit(X_train, y_train, **fit_kwargs)
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_val, val_preds),
        "f1": f1_score(y_val, val_preds),
        "log_loss": log_loss(y_val, val_probs, labels=[0, 1]),
    }
    return model, val_probs, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main K-fold training
# ══════════════════════════════════════════════════════════════════════════════
def run_kfold(model_name: str, model_cls, model_params: dict,
              X: np.ndarray, y: np.ndarray, feature_names: list,
              cfg: ClassicalConfig):
    """
    Full K-fold CV for a single model type.
    Returns list of models, OOF predictions, and metrics.
    """
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                          random_state=cfg.seed)
    oof_probs = np.zeros(len(y), dtype=np.float64)
    models = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model, val_probs, metrics = train_fold(
            model_cls, model_params, X_tr, y_tr, X_va, y_va,
            model_name=model_name,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )

        oof_probs[val_idx] = val_probs
        models.append(model)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        elapsed = time.time() - t0
        log.info(
            f"  [{model_name}] Fold {fold_idx}/{cfg.n_folds}  "
            f"acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
            f"ll={metrics['log_loss']:.4f}  ({elapsed:.0f}s)"
        )

    # overall OOF metrics
    oof_preds = (oof_probs >= 0.5).astype(int)
    overall = {
        "model": model_name,
        "oof_accuracy": accuracy_score(y, oof_preds),
        "oof_f1": f1_score(y, oof_preds),
        "oof_log_loss": log_loss(y, oof_probs, labels=[0, 1]),
        "folds": fold_metrics,
    }
    log.info(
        f"  [{model_name}] OOF  acc={overall['oof_accuracy']:.4f}  "
        f"f1={overall['oof_f1']:.4f}  ll={overall['oof_log_loss']:.4f}"
    )
    return models, oof_probs, overall


def main():
    parser = argparse.ArgumentParser(
        description="Classical ML training with feature engineering.")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use only 500 samples for quick testing.")
    parser.add_argument("--skip-xgb", action="store_true",
                        help="Skip XGBoost (saves time for quick iterations).")
    args = parser.parse_args()

    cfg = ClassicalConfig()
    if args.folds:
        cfg.n_folds = args.folds

    np.random.seed(cfg.seed)

    # ── load data ─────────────────────────────────────────────────────────
    log.info("Loading training data …")
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=["question1", "question2"]).reset_index(drop=True)
    if args.dry_run:
        df = df.head(500)
        cfg.n_folds = min(cfg.n_folds, 3)
        log.info("🔬  Dry-run mode: 500 samples, %d folds", cfg.n_folds)

    log.info(f"Training data: {len(df)} rows, dup rate = {df['is_duplicate'].mean():.3f}")

    # ── features ──────────────────────────────────────────────────────────
    feature_df, tfidf_vec = build_features(
        df, full_train_df=df,
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        tfidf_max_features=cfg.tfidf_max_features,
        tfidf_ngram_range=cfg.tfidf_ngram_range,
    )

    X = feature_df.values.astype(np.float32)
    y = df["is_duplicate"].values
    feature_names = feature_df.columns.tolist()
    log.info(f"Feature matrix: {X.shape}")

    # Save TF-IDF vectorizer for test prediction
    joblib.dump(tfidf_vec, os.path.join(cfg.output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(feature_names, os.path.join(cfg.output_dir, "feature_names.pkl"))

    results = {}
    all_oof = {}

    # ── LightGBM ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Training LightGBM …")
    lgbm_models, lgbm_oof, lgbm_res = run_kfold(
        "lgbm", _make_lgbm, cfg.lgbm_params, X, y, feature_names, cfg,
    )
    results["lgbm"] = lgbm_res
    all_oof["lgbm"] = lgbm_oof
    for i, m in enumerate(lgbm_models):
        joblib.dump(m, os.path.join(cfg.output_dir, f"lgbm_fold{i}.pkl"))

    # ── XGBoost ───────────────────────────────────────────────────────────
    if not args.skip_xgb:
        log.info("=" * 60)
        log.info("Training XGBoost …")
        xgb_models, xgb_oof, xgb_res = run_kfold(
            "xgb", _make_xgb, cfg.xgb_params, X, y, feature_names, cfg,
        )
        results["xgb"] = xgb_res
        all_oof["xgb"] = xgb_oof
        for i, m in enumerate(xgb_models):
            joblib.dump(m, os.path.join(cfg.output_dir, f"xgb_fold{i}.pkl"))

    # ── Logistic Regression (for ensemble diversity) ──────────────────────
    log.info("=" * 60)
    log.info("Training Logistic Regression …")
    # Standardise features for LR
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(cfg.output_dir, "scaler.pkl"))

    lr_models, lr_oof, lr_res = run_kfold(
        "lr", _make_lr, cfg.lr_params, X_scaled, y, feature_names, cfg,
    )
    results["lr"] = lr_res
    all_oof["lr"] = lr_oof
    for i, m in enumerate(lr_models):
        joblib.dump(m, os.path.join(cfg.output_dir, f"lr_fold{i}.pkl"))

    # ── Save OOF predictions (for stacking in approach 3) ─────────────────
    oof_df = pd.DataFrame(all_oof)
    oof_df["target"] = y
    oof_path = os.path.join(cfg.output_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    log.info(f"OOF predictions saved → {oof_path}")

    # ── Feature importance (LightGBM) ─────────────────────────────────────
    importances = np.zeros(len(feature_names))
    for m in lgbm_models:
        importances += m.feature_importances_
    importances /= len(lgbm_models)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    imp_path = os.path.join(cfg.output_dir, "feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    log.info(f"Feature importance saved → {imp_path}")
    log.info("Top 15 features:")
    for _, row in imp_df.head(15).iterrows():
        log.info(f"  {row['feature']:30s}  {row['importance']:.0f}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_path = os.path.join(cfg.output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Training summary saved → {summary_path}")


if __name__ == "__main__":
    main()
