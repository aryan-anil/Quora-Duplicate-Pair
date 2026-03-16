import torch
from transformers import AutoTokenizer
import sys

model_name = "microsoft/deberta-v3-base"

print("Checking CUDA...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print(f"Attempting to load tokenizer for {model_name} with use_fast=False...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Tokenizer loaded successfully!")
    print(f"Tokenizer class: {type(tokenizer)}")
    test_text = "Is this a duplicate question?"
    tokens = tokenizer(test_text)
    print(f"Test tokenization successful: {tokens['input_ids'][:10]}...")
except Exception as e:
    print(f"Error loading tokenizer: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
