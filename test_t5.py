from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("AventIQ-AI/T5-small-grammar-correction")
print("Tokenizer loaded")
