from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load base model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Save to ./model directory
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
