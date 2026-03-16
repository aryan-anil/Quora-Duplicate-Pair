"""
Configuration for Approach 3: Ensembling.
"""
import os
from dataclasses import dataclass, field
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
TEST_CSV = os.path.join(DATA_DIR, "test", "test.csv")

# Approach output directories
A1_OUTPUT = os.path.join(PROJECT_DIR, "approach1_transformer", "outputs")
A2_OUTPUT = os.path.join(PROJECT_DIR, "approach2_classical_ml", "outputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class EnsembleConfig:
    """All configuration for the ensembling pipeline."""

    seed: int = 42

    # ── Paths to approach predictions ─────────────────────────────────────
    transformer_submission: str = os.path.join(A1_OUTPUT, "submission_transformer.csv")
    classical_submission: str = os.path.join(A2_OUTPUT, "submission_classical.csv")

    # per-model raw (uncalibrated) submissions for re-calibration
    transformer_raw: str = os.path.join(A1_OUTPUT, "submission_transformer_raw.csv")
    lgbm_raw: str = os.path.join(A2_OUTPUT, "submission_lgbm_raw.csv")
    xgb_raw: str = os.path.join(A2_OUTPUT, "submission_xgb_raw.csv")
    lr_raw: str = os.path.join(A2_OUTPUT, "submission_lr_raw.csv")

    # OOF predictions from approach 2 (for stacking meta-learner)
    oof_predictions: str = os.path.join(A2_OUTPUT, "oof_predictions.csv")

    # ── Ensemble weights ──────────────────────────────────────────────────
    # Weighted average: transformer is typically stronger so gets more weight
    weights: Dict[str, float] = field(default_factory=lambda: {
        "transformer": 0.6,
        "classical": 0.4,
    })

    # ── Calibration ───────────────────────────────────────────────────────
    train_pos_rate: float = 0.3692
    test_pos_rate: float = 0.165

    # ── Stacking meta-learner ─────────────────────────────────────────────
    stacking_model: str = "logistic"  # "logistic" or "lightgbm"

    output_dir: str = OUTPUT_DIR
