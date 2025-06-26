# === File: src/utils.py ===
import logging
import torch.nn.functional as F
import torch

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_confidence(logits):
    probs = F.softmax(logits, dim=-1).squeeze()
    confidence, label = torch.max(probs, dim=-1)
    return confidence.item(), label.item()

def label_id_to_name(label_id):
    return "Positive" if label_id == 1 else "Negative"
