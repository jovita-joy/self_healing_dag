# Self-Healing Classification DAG

This project implements a LangGraph-based DAG (Directed Acyclic Graph) for robust and interactive text classification. It features a fallback mechanism triggered when the model's confidence is low — allowing a human-in-the-loop to improve reliability.

---

## 🧠 Project Objective

Build a classification system with:

- ✅ Fine-tuned DistilBERT model (on SST-2 sentiment dataset)
- ✅ Confidence-based fallback using LangGraph
- ✅ CLI with clarification fallback via user input
- ✅ Modular and interpretable workflow with logging

---

## 🔧 Project Structure

self_healing_dag/
│
├── src/
│ ├── utils.py # Helper functions: confidence, logging
│ ├── train.py # Fine-tune DistilBERT on SST-2
│ ├── dag.py # LangGraph DAG with nodes
│ └── main.py # CLI entry point
│
├── model/ # Saved model and tokenizer (after training)
│
├── sst2_data/ # Exported CSV versions of SST-2
├── sst2_data.zip # Zipped dataset (optional)
│
├── requirements.txt # Required packages
└── README.md # Project documentation


---

## 📦 Installation

1. Clone the repository:

```sh
git clone https://github.com/your-username/self_healing_dag.git
cd self_healing_dag
```
2.Create a virtual environment and install dependencies:
```sh
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```
## Dataset
We use the SST-2 dataset (Stanford Sentiment Treebank v2) via HuggingFace Datasets:

Two classes: Positive, Negative

Preprocessed and exported to CSV
## Training
Fine-tune DistilBERT on a subset of SST-2:
```sh
python src/train.py
```
Model and tokenizer will be saved in the ./model/ directory.

## Run the DAG
Launch the CLI app:
```sh
python src/main.py
```
## Nodes in the LangGraph DAG
Node	             Description
InferenceNode	     Runs model prediction and calculates confidence
ConfidenceCheck	   Compares confidence score to a threshold (default = 0.75)
FallbackNode	     If confidence is low, prompts the user to clarify intent

## Logging
All intermediate steps, predictions, and fallback decisions are logged for traceability:

[InferenceNode]

[ConfidenceCheckNode]

[FallbackNode]

## Requirements
``sh
transformers
datasets
torch
langgraph
peft
accelerate
```
Install via:
```sh
pip install -r requirements.txt
```
## Notes
This DAG structure is ideal for human-in-the-loop ML systems

The fallback mechanism is rule-based but can be extended to include a second model

OUTPUT

