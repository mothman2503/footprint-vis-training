from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model + tokenizer from your output folder
model_dir = "distilbert_output"  # or "tinybert_output"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Label mapping (you'll need this from your encoder)
label_names = [  # Update these with your actual 24 class labels
    "IAB1 Arts & Entertainment", "IAB2 Automotive", "IAB3 Business", 
    "IAB4 Careers", "IAB5 Education", "IAB6 Family & Parenting", 
    "IAB7 Health & Fitness", "IAB8 Food & Drink", "IAB9 Hobbies & Interests", 
    "IAB10 Home & Garden", "IAB11 Law, Gov't & Politics", 
    "IAB12 News", "IAB13 Personal Finance", "IAB14 Society", 
    "IAB15 Science", "IAB16 Pets", "IAB17 Real Estate", 
    "IAB18 Style & Fashion", "IAB19 Technology & Computing", 
    "IAB20 Travel", "IAB21 Religion & Spirituality", 
    "IAB22 Non-Standard Content", "IAB23 Uncategorized", 
    "IAB24 Illegal Content"
]

# 🔍 Your test phrase
test_text = "cricket"
# Tokenize and predict
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

# Show prediction
print(f"Input: {test_text}")
print(f"Predicted class: {label_names[predicted_class]} (confidence: {confidence:.2f})")
