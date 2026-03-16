"""
Shared calibration utilities for all approaches.

The key insight: the training set has ~37% duplicates, but the Kaggle test
set has a much lower duplicate rate (~16.5%).  If we don't account for this
shift, our log-loss will suffer.  Bayesian rescaling adjusts the predicted
probabilities to the target prior.
"""
import numpy as np
from sklearn.metrics import f1_score


def calibrate_probabilities(
    probs: np.ndarray,
    train_pos_rate: float = 0.3692,
    test_pos_rate: float = 0.165,
) -> np.ndarray:
    """
    Bayesian rescaling of predicted probabilities from one prior to another.

    Given:
      - p = P(dup | x) estimated under the training distribution
      - r_train = P(dup) in training set
      - r_test  = P(dup) in test set

    Calibrated probability:
      p_test = p · (r_test / r_train)
               ────────────────────────────────────────────────────
               p · (r_test / r_train)  +  (1−p) · ((1−r_test) / (1−r_train))

    Parameters
    ----------
    probs : array-like
        Raw predicted duplicate probabilities (trained on ~37% positive rate).
    train_pos_rate : float
        Positive class rate in the training set.
    test_pos_rate : float
        Estimated positive class rate in the test set.

    Returns
    -------
    np.ndarray
        Calibrated probabilities adjusted for the test-set prior.
    """
    probs = np.asarray(probs, dtype=np.float64)
    # Clip to avoid division by zero
    probs = np.clip(probs, 1e-15, 1.0 - 1e-15)

    a = test_pos_rate / train_pos_rate
    b = (1.0 - test_pos_rate) / (1.0 - train_pos_rate)

    calibrated = (probs * a) / (probs * a + (1.0 - probs) * b)
    return calibrated


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray = None,
) -> tuple:
    """
    Search for the threshold that maximises the given metric on validation data.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_prob : array-like
        Predicted probabilities.
    metric : str
        One of 'f1' (default).
    thresholds : array of floats, optional
        Thresholds to try.  Default: 0.01, 0.02, …, 0.99.

    Returns
    -------
    (best_threshold, best_score)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_thr, best_score = 0.5, 0.0
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if score > best_score:
            best_score = score
            best_thr = thr

    return float(best_thr), float(best_score)
