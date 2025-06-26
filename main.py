# === File: src/main.py ===
from dag import build_dag
from utils import setup_logging

setup_logging()

def main():
    dag = build_dag()
    print("Enter a review (or 'exit' to quit):")
    while True:
        text = input("\nReview: ")
        if text.lower() == "exit":
            break
        result = dag.invoke({"text": text})
        print(f"\n[Final Output] Label: {result['label']}")

if __name__ == "__main__":
    main()