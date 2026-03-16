"""
Quick inference script for the Duplicate Question Transformer.

Takes two questions and a model checkpoint, prints the duplicate probability.

Usage:
    python inference.py --q1 "How do I learn Python?" --q2 "What is the best way to learn Python?" --model_path outputs/best_model_fold0.pt
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig

from model import DuplicateClassifier


def run_inference(question1: str, question2: str, model_path: str,
                  model_name: str = "microsoft/deberta-v3-small",
                  max_length: int = 128) -> dict:
    """
    Run inference on a single question pair.

    Parameters
    ----------
    question1 : str
        First question.
    question2 : str
        Second question.
    model_path : str
        Path to the saved model checkpoint (.pt file).
    model_name : str
        HuggingFace model identifier (must match what was used during training).
    max_length : int
        Maximum token length for the tokenizer.

    Returns
    -------
    dict with keys:
        prediction  – 0 (not duplicate) or 1 (duplicate)
        probability – float, probability that the pair is a duplicate
        logits      – list[float], raw logits from the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load tokenizer & model ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    config = AutoConfig.from_pretrained(model_name)
    num_labels = 2
    model = DuplicateClassifier(model_name, num_labels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # Checkpoints saved during training contain a dict with metadata;
    # checkpoints saved with just model.state_dict() are loaded directly.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # ── Tokenize the question pair ────────────────────────────────────────
    encoding = tokenizer(
        question1,
        question2,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    # ── Forward pass ──────────────────────────────────────────────────────
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    logits = output["logits"].cpu()
    probs = torch.softmax(logits, dim=1)
    dup_prob = probs[0, 1].item()          # probability of class 1 (duplicate)
    prediction = int(probs.argmax(dim=1).item())

    return {
        "prediction": prediction,
        "probability": dup_prob,
        "logits": logits.squeeze().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run duplicate-question inference on a single pair."
    )
   
    parser.add_argument("--model_name", type=str,
                        default="microsoft/deberta-v3-small",
                        help="HuggingFace model name (must match training).")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length.")
    args = parser.parse_args()

    # ── Variables ─────────────────────────────────────────────────────────
    question1 = "What is a cat?"
    question2 = "What is a dog?"
    model_path = r"C:\Users\Aryan\Documents\DuplicateQuestions\approach1_transformer\outputs\model_fold0_epoch3.pt"

    result = run_inference(question1, question2, model_path,
                           model_name=args.model_name,
                           max_length=args.max_length)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Duplicate Question Inference")
    print("=" * 60)
    print(f"  Q1 : {question1}")
    print(f"  Q2 : {question2}")
    print("-" * 60)
    label = "DUPLICATE" if result["prediction"] == 1 else "NOT DUPLICATE"
    print(f"  Prediction  : {label}")
    print(f"  Dup Prob    : {result['probability']:.4f}")
    print(f"  Logits      : {result['logits']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
