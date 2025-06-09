from transformers import BertForSequenceClassification, BertTokenizerFast
import torch

# Load your trained model and tokenizer
model_path = "output/final_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# Make sure the model is in eval mode
model.eval()

# Your test query
query = "Biryani with raita and rice"

# Tokenize
inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

# Map index back to label
# Recreate this from the same label2id mapping used during training
id2label = {
    0: "IAB1 Arts & Entertainment",
    1: "IAB2 Automotive",
    2: "IAB3 Business",
    3: "IAB4 Careers",
    4: "IAB5 Education",
    5: "IAB6 Family & Parenting",
    6: "IAB7 Health & Fitness",
    7: "IAB8 Food & Drink",
    8: "IAB9 Hobbies & Interests",
    9: "IAB10 Home & Garden",
    10: "IAB11 Law, Gov’t & Politics",
    11: "IAB12 News",
    12: "IAB13 Personal Finance",
    13: "IAB14 Society",
    14: "IAB15 Science",
    15: "IAB16 Pets",
    16: "IAB17 Sports",
    17: "IAB18 Style & Fashion",
    18: "IAB19 Technology & Computing",
    19: "IAB20 Travel",
    20: "IAB21 Real Estate",
    21: "IAB22 Shopping",
    22: "IAB23 Religion & Spirituality",
    23: "IAB24 Uncategorized"
}

# Print result
print(f"Input: {query}")
print(f"Predicted IAB Category: {id2label[predicted_class_id]}")
