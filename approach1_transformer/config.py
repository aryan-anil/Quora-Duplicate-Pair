"""
Configuration for Approach 1: Fine-tuned Transformer.
"""
import os
from dataclasses import dataclass, field
from typing import List

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
TRAIN_CSV = os.path.join(DATA_DIR, "train", "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test", "test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class TrainConfig:
    """All training hyper-parameters in one place."""

    # ── Model ─────────────────────────────────────────────────────────────
    model_name: str = "microsoft/deberta-v3-small"
    max_length: int = 128
    num_labels: int = 2

    # ── Training ──────────────────────────────────────────────────────────
    epochs: int = 3
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True  # mixed-precision

    # ── Data split ────────────────────────────────────────────────────────
    val_ratio: float = 0.1   # 90/10 stratified split
    n_folds: int = 1         # set >1 for K-fold CV
    seed: int = 42

    # ── Early stopping ────────────────────────────────────────────────────
    patience: int = 2
    metric_for_best: str = "f1"

    # ── Question-word penalty (post-processing at inference) ──────────────
    question_words: List[str] = field(default_factory=lambda: [
        "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    ])
    question_word_penalty: float = 0.70  # multiply dup prob by this if WH-words differ

    # ── Calibration ───────────────────────────────────────────────────────
    train_pos_rate: float = 0.3692  # observed in train set
    test_pos_rate: float = 0.165    # estimated Kaggle test prior

    # ── Paths ─────────────────────────────────────────────────────────────
    output_dir: str = OUTPUT_DIR
    checkpoint_name: str = "best_model.pt"

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, self.checkpoint_name)
