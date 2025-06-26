# === File: src/train.py ===
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os, zipfile

def train_model():
    dataset = load_dataset("glue", "sst2")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def preprocess(example):
        return tokenizer(example["sentence"], truncation=True, padding="max_length")

    tokenized = dataset.map(preprocess, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    training_args = TrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        num_train_epochs=2,
        save_strategy="no",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"].shuffle(seed=42).select(range(1000)),
        eval_dataset=tokenized["validation"].select(range(200))
    )
    trainer.train()
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")

    os.makedirs("sst2_data", exist_ok=True)
    dataset["train"].select(range(1000)).to_csv("sst2_data/train.csv")
    dataset["validation"].select(range(200)).to_csv("sst2_data/validation.csv")

    with zipfile.ZipFile("sst2_data.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("sst2_data"):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, "sst2_data"))

