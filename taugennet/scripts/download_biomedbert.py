#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""Download PubMedBERT (BiomedBERT) to a local path for offline use on GPU nodes.

Run this on the LOGIN NODE (which has internet). Adroit compute nodes are offline,
so `from_pretrained("microsoft/...")` fails inside a job — mirror the existing
`clip_model/` pattern and cache the encoder + tokenizer to BIOMEDBERT_MODEL_PATH.

    /home/sz3962/.conda/envs/taugennet/bin/python3 scripts/download_biomedbert.py
"""
import os
import sys

# allow `from src.config import ...` when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModel, AutoTokenizer

from src.config import BIOMEDBERT_MODEL_PATH

HF_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


def main():
    print(f"Downloading {HF_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    model     = AutoModel.from_pretrained(HF_ID)

    os.makedirs(BIOMEDBERT_MODEL_PATH, exist_ok=True)
    tokenizer.save_pretrained(BIOMEDBERT_MODEL_PATH)
    model.save_pretrained(BIOMEDBERT_MODEL_PATH)

    print(f"Saved to: {BIOMEDBERT_MODEL_PATH}")
    print(f"hidden_size = {model.config.hidden_size}  (expected 768)")
    assert model.config.hidden_size == 768, "unexpected hidden size"
    print("Done.")


if __name__ == "__main__":
    main()
