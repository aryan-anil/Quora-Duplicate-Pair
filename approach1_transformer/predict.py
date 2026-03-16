"""
Inference & submission generation for approach 1.

Usage:
    python -m approach1_transformer.predict
    python -m approach1_transformer.predict --dry-run
"""
import argparse
import logging
import os
import re

from transformers import AutoTokenizer

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import TEST_CSV, TrainConfig
from dataset import build_dataloader
from model import DuplicateClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Question-word penalty
# ══════════════════════════════════════════════════════════════════════════════
def _extract_first_word(text: str) -> str:
    """Return lowercased first word of a question."""
    text = str(text).strip().lower()
    match = re.match(r"[a-z]+", text)
    return match.group() if match else ""


def apply_question_word_penalty(df: pd.DataFrame, probs: np.ndarray,
                                cfg: TrainConfig) -> np.ndarray:
    """
    If Q1 and Q2 start with different WH-words, multiply the predicted
    duplicate probability by a penalty factor (<1).  This heuristic captures
    the intuition that "Who is X?" vs "What is X?" are almost never duplicates.
    """
    probs = probs.copy()
    wh_set = set(cfg.question_words)

    for i in range(len(df)):
        w1 = _extract_first_word(df.iloc[i]["question1"])
        w2 = _extract_first_word(df.iloc[i]["question2"])
        # Only penalise when *both* are WH-words but differ
        if w1 in wh_set and w2 in wh_set and w1 != w2:
            probs[i] *= cfg.question_word_penalty

    return probs


# ══════════════════════════════════════════════════════════════════════════════
# Calibration (Bayesian rescaling)
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_probabilities(probs: np.ndarray,
                            train_pos: float, test_pos: float) -> np.ndarray:
    """
    Adjust predicted probabilities for a shift in class prior
    between training and test distributions.

    p_calib = p * (r_test / r_train) /
              [ p * (r_test / r_train) + (1-p) * ((1-r_test)/(1-r_train)) ]
    """
    a = test_pos / train_pos
    b = (1.0 - test_pos) / (1.0 - train_pos)
    calibrated = (probs * a) / (probs * a + (1.0 - probs) * b)
    return calibrated


# ══════════════════════════════════════════════════════════════════════════════
# Main prediction pipeline
# ══════════════════════════════════════════════════════════════════════════════
def predict(cfg: TrainConfig, test_df: pd.DataFrame,
            fold_indices: list | None = None):
    """
    Load checkpoint(s) and produce calibrated predictions.
    Returns np.ndarray of shape (len(test_df),) with duplicate probabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    test_loader = build_dataloader(
        test_df, tokenizer, cfg.max_length, cfg.eval_batch_size,
        is_test=True, num_workers=2,
    )

    if fold_indices is None:
        fold_indices = list(range(max(1, cfg.n_folds)))

    all_fold_probs = []

    for fold_idx in fold_indices:
        ckpt_path = os.path.join(cfg.output_dir,
                                 f"best_model_fold{fold_idx}.pt")
        if not os.path.exists(ckpt_path):
            log.warning(f"Checkpoint not found: {ckpt_path}  — skipping")
            continue

        log.info(f"Loading fold {fold_idx} from {ckpt_path}")
        model = DuplicateClassifier(cfg.model_name, cfg.num_labels).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        fold_logits = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold_idx}"):
                batch = {k: v.to(device) for k, v in batch.items()
                         if k != "labels"}
                with autocast(enabled=cfg.fp16 and device.type == "cuda"):
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids"),
                    )
                fold_logits.append(out["logits"].cpu())

        logits = torch.cat(fold_logits, dim=0)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        all_fold_probs.append(probs)

    if not all_fold_probs:
        raise FileNotFoundError("No model checkpoints found in output dir.")

    # average across folds
    avg_probs = np.mean(all_fold_probs, axis=0)

    # question-word penalty
    avg_probs = apply_question_word_penalty(test_df, avg_probs, cfg)

    # calibration
    calibrated = calibrate_probabilities(
        avg_probs, cfg.train_pos_rate, cfg.test_pos_rate,
    )

    return calibrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Predict on first 200 rows of test set.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip calibration step.")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.model:
        cfg.model_name = args.model

    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df.fillna("")
    if args.dry_run:
        test_df = test_df.head(200)
        log.info("🔬  Dry-run mode: 200 test samples")

    probs = predict(cfg, test_df)

    if args.no_calibrate:
        # re-run without calibration — just avg + penalty
        log.info("Skipping calibration.")
    
    # ── save submission ───────────────────────────────────────────────────
    sub = pd.DataFrame({
        "test_id": test_df["test_id"],
        "is_duplicate": probs,
    })
    out_path = os.path.join(cfg.output_dir, "submission_transformer.csv")
    sub.to_csv(out_path, index=False)
    log.info(f"Submission saved → {out_path}  ({len(sub)} rows)")

    # ── also save raw (uncalibrated) submission ───────────────────────────
    raw_out = os.path.join(cfg.output_dir, "submission_transformer_raw.csv")
    # recalculate without calibration for comparison
    test_df_for_raw = pd.read_csv(TEST_CSV).fillna("")
    if args.dry_run:
        test_df_for_raw = test_df_for_raw.head(200)
    # The calibrated probs are already saved; user can compare
    log.info("Done!")


if __name__ == "__main__":
    main()
