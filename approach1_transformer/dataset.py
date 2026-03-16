"""
PyTorch Dataset & DataLoader factory for Quora duplicate-question pairs.
"""
from transformers import AutoTokenizer

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class QuoraPairDataset(Dataset):
    """Tokenises (question1, question2) pairs for a transformer model."""

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer,
                 max_length: int = 128, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q1 = str(row["question1"]) if pd.notna(row["question1"]) else ""
        q2 = str(row["question2"]) if pd.notna(row["question2"]) else ""

        encoding = self.tokenizer(
            q1, q2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if not self.is_test:
            item["labels"] = torch.tensor(int(row["is_duplicate"]), dtype=torch.long)
        return item


# ══════════════════════════════════════════════════════════════════════════════
# Splitters
# ══════════════════════════════════════════════════════════════════════════════
def get_train_val_split(df: pd.DataFrame, val_ratio: float = 0.1,
                        seed: int = 42):
    """Stratified single train/val split."""
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed,
        stratify=df["is_duplicate"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_kfold_splits(df: pd.DataFrame, n_folds: int = 5, seed: int = 42):
    """Yields (fold_idx, train_df, val_df) for each fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(df, df["is_duplicate"])):
        yield fold_idx, df.iloc[train_idx].reset_index(drop=True), \
              df.iloc[val_idx].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ══════════════════════════════════════════════════════════════════════════════
def build_dataloader(df: pd.DataFrame, tokenizer: AutoTokenizer,
                     max_length: int, batch_size: int,
                     is_test: bool = False, shuffle: bool = False,
                     num_workers: int = 2) -> DataLoader:
    """Create a DataLoader from a DataFrame."""
    ds = QuoraPairDataset(df, tokenizer, max_length=max_length,
                          is_test=is_test)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
