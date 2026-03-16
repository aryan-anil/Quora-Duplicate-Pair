"""
Training script for the fine-tuned transformer approach.

Usage:
    python -m approach1_transformer.train
    python -m approach1_transformer.train --dry-run          # quick sanity check
    python -m approach1_transformer.train --model bert-base-uncased
"""
import argparse
import json
import logging
import os
import time

from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.cuda.amp import GradScaler, autocast

from config import TRAIN_CSV, TrainConfig
from dataset import build_dataloader, get_kfold_splits, get_train_val_split
from model import DuplicateClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predownload_model(cfg: TrainConfig):
    """Download tokenizer + model weights before training starts."""
    log.info("  Pre-downloading tokenizer and model...")
    AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    AutoModel.from_pretrained(cfg.model_name)
    log.info("  Pre-download complete.")


def evaluate(model, dataloader, device):
    """Run one full evaluation pass; return metrics dict."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="  Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
            )
            total_loss += out["loss"].item()
            n_batches += 1
            all_logits.append(out["logits"].cpu())
            all_labels.append(batch["labels"].cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "log_loss": log_loss(labels, probs, labels=[0, 1]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training loop (single fold)
# ══════════════════════════════════════════════════════════════════════════════
def train_one_fold(cfg: TrainConfig, train_df: pd.DataFrame,
                   val_df: pd.DataFrame, fold_idx: int = 0):
    """Train one fold and return best metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Fold {fold_idx}] Device: {device}")

    # ── tokenizer & dataloaders ───────────────────────────────────────────
    log.info("  Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    except Exception as e:
        log.error(f"Failed to load tokenizer: {e}")
        import traceback
        log.error(traceback.format_exc())
        raise e
    
    log.info("  Building dataloaders...")
    train_loader = build_dataloader(
        train_df, tokenizer, cfg.max_length, cfg.train_batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = build_dataloader(
        val_df, tokenizer, cfg.max_length, cfg.eval_batch_size,
        num_workers=0,
    )

    # ── model ─────────────────────────────────────────────────────────────
    log.info("  Initializing model...")
    model = DuplicateClassifier(cfg.model_name, cfg.num_labels).to(device)

    # ── optimizer & scheduler ─────────────────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight"}
    params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=cfg.fp16 and device.type == "cuda")

    # ── training ──────────────────────────────────────────────────────────
    best_metric = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        t_start = time.time()

        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                    desc=f"  Epoch {epoch}/{cfg.epochs}")
        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=cfg.fp16 and device.type == "cuda"):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    labels=batch["labels"],
                )
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running_loss += loss.item()
            avg_loss = running_loss / step
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # ── validation ────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t_start
        log.info(
            f"[Fold {fold_idx}] Epoch {epoch}/{cfg.epochs}  "
            f"val_loss={metrics['loss']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"logloss={metrics['log_loss']:.4f}  "
            f"({elapsed:.0f}s)"
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / len(train_loader)
        history.append(metrics)

        # ── early stopping & checkpointing ────────────────────────────────
        current = metrics[cfg.metric_for_best]
        if current > best_metric:
            best_metric = current
            patience_counter = 0
            ckpt = os.path.join(cfg.output_dir,
                                f"best_model_fold{fold_idx}.pt")
            torch.save(model.state_dict(), ckpt)
            log.info(f"  ✓ Saved best model ({cfg.metric_for_best}="
                     f"{best_metric:.4f}) → {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    return {"best_metric": best_metric, "history": history}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer on Quora duplicate questions.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name, e.g. bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use only 200 samples for a quick pipeline check")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.model:
        cfg.model_name = args.model
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.batch_size:
        cfg.train_batch_size = args.batch_size
    if args.max_length:
        cfg.max_length = args.max_length
    if args.n_folds:
        cfg.n_folds = args.n_folds

    set_seed(cfg.seed)
    log.info(f"Config: {cfg}")

    # ── load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=["question1", "question2"]).reset_index(drop=True)
    if args.dry_run:
        df = df.head(200)
        cfg.epochs = 1
        log.info("🔬  Dry-run mode: 200 samples, 1 epoch")

    log.info(f"Training data: {len(df)} rows, "
             f"dup rate = {df['is_duplicate'].mean():.3f}")

    # Ensure model assets are downloaded before training begins.
    predownload_model(cfg)

    # ── train ─────────────────────────────────────────────────────────────
    all_results = []

    if cfg.n_folds <= 1:
        train_df, val_df = get_train_val_split(df, cfg.val_ratio, cfg.seed)
        result = train_one_fold(cfg, train_df, val_df, fold_idx=0)
        all_results.append(result)
    else:
        for fold_idx, train_df, val_df in get_kfold_splits(
                df, cfg.n_folds, cfg.seed):
            result = train_one_fold(cfg, train_df, val_df, fold_idx)
            all_results.append(result)

    # ── save summary ──────────────────────────────────────────────────────
    summary_path = os.path.join(cfg.output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Training summary saved → {summary_path}")


if __name__ == "__main__":
    main()
