from langgraph.graph import StateGraph
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from utils import calculate_confidence, label_id_to_name
import logging
from typing import TypedDict

CONFIDENCE_THRESHOLD = 0.75

# Define input and output schema
class InputState(TypedDict):
    text: str

class OutputState(TypedDict):
    label: str

# Load fine-tuned tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("./model")
model = DistilBertForSequenceClassification.from_pretrained("./model")
model.eval()

# === Node 1: Inference ===
class InferenceNode:
    def __call__(self, state):
        inputs = tokenizer(state['text'], return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        confidence, label_id = calculate_confidence(logits)
        label_name = label_id_to_name(label_id)
        logging.info(f"[InferenceNode] Predicted label: {label_name} | Confidence: {confidence * 100:.2f}%")
        return {
            "label": label_name,
            "confidence": confidence,
            "text": state['text']
        }

# === Node 2: Confidence Check ===
class ConfidenceCheckNode:
    def __call__(self, state):
        if state['confidence'] < CONFIDENCE_THRESHOLD:
            logging.info("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
            return "fallback"
        return "accept"

# === Node 3: Fallback Mechanism ===
class FallbackNode:
    def __call__(self, state):
        clarification = input("[FallbackNode] Could you clarify your intent? Was this a negative review?\nUser: ")
        final_label = "Negative" if "neg" in clarification.lower() else "Positive"
        logging.info(f"Final Label: {final_label} (Corrected via user clarification)")
        return {
            "label": final_label,
            "text": state['text'],
            "source": "fallback"
        }

# === DAG Construction ===
def build_dag():
    builder = StateGraph(InputState)

    # Add nodes
    builder.add_node("infer", InferenceNode())
    builder.add_node("check", ConfidenceCheckNode())
    builder.add_node("fallback", FallbackNode())

    # Entry point
    builder.set_entry_point("infer")
    builder.add_edge("infer", "check")

    # Set fallback routing
    def route_condition(state):
        return state  # returns "accept" or "fallback"

    # Set where the DAG finishes if accepted
    builder.set_finish_point("check")

    builder.add_conditional_edges(
        "check",
        route_condition,
        {
            "fallback": "fallback",   # continue to fallback node
            "accept": "check"         # loop back to same node to terminate
        }
    )

    # Finish at fallback too (after clarification)
    builder.set_finish_point("fallback")

    return builder.compile()
