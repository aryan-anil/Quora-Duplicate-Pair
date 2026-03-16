"""
Transformer-based classifier for duplicate question detection.

Architecture:
    [CLS] Q1 [SEP] Q2 [SEP]  →  Transformer  →  [CLS] embedding
    →  Dropout  →  Linear(hidden, 2)  →  logits
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DuplicateClassifier(nn.Module):
    """Wraps a pre-trained transformer with a 2-way classification head."""

    def __init__(self, model_name: str, num_labels: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(
            model_name, config=self.config,
        )
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_classifier()

    # ─── helpers ──────────────────────────────────────────────────────────
    def _init_classifier(self):
        """Xavier init for the classification head."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                labels=None):
        """
        Returns
        -------
        dict with keys:
            logits   – (B, num_labels) raw scores
            loss     – scalar CE loss (only when labels are provided)
        """
        # Some models (e.g. DeBERTa-v3) do not use token_type_ids
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and hasattr(self.config, "type_vocab_size"):
            if self.config.type_vocab_size > 0:
                kwargs["token_type_ids"] = token_type_ids

        outputs = self.transformer(**kwargs)

        # [CLS] representation — first token of last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result
