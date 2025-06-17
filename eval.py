from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_dir = "tinybert_noweights"  # or "tinybert_output"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Label list (ensure this matches your training labels)
label_names = [
    "IAB1 Arts & Entertainment", "IAB2 Automotive", "IAB3 Business", "IAB4 Careers",
    "IAB5 Education", "IAB6 Family & Parenting", "IAB7 Health & Fitness", "IAB8 Food & Drink",
    "IAB9 Hobbies & Interests", "IAB10 Home & Garden", "IAB11 Law, Gov’t & Politics",
    "IAB12 News", "IAB13 Personal Finance", "IAB14 Society", "IAB15 Science", "IAB16 Pets",
    "IAB17 Sports", "IAB18 Style & Fashion", "IAB19 Technology & Computing", "IAB20 Travel",
    "IAB21 Real Estate", "IAB22 Shopping", "IAB23 Religion & Spirituality", "IAB24 Uncategorized"
]

# Interactive loop
print("🧠 Enter a search phrase to classify (type 'exit' to quit):")
while True:
    text = input("> ").strip()
    if text.lower() == "exit":
        break
    if not text:
        continue

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze()
        predicted_idx = torch.argmax(probs).item()
        predicted_label = label_names[predicted_idx]
        confidence = probs[predicted_idx].item()

    print(f"🔍 Predicted class: {predicted_label} (confidence: {confidence:.2f})")
    print("-" * 50)
