# zero_shot_labeling.py
from transformers import pipeline
import pandas as pd
import json

# Load your data (this time from train_text_only.json!)
with open("train_text_only.json", "r") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
df = df[["text"]]  # Only keep the "text" column

# Define full IAB labels mapping
iab_labels = [
    "IAB1 Arts & Entertainment", "IAB2 Automotive", "IAB3 Business", "IAB4 Careers",
    "IAB5 Education", "IAB6 Family & Parenting", "IAB7 Health & Fitness", "IAB8 Food & Drink",
    "IAB9 Hobbies & Interests", "IAB10 Home & Garden", "IAB11 Law, Gov’t & Politics",
    "IAB12 News", "IAB13 Personal Finance", "IAB14 Society", "IAB15 Science", "IAB16 Pets",
    "IAB17 Sports", "IAB18 Style & Fashion", "IAB19 Technology & Computing", "IAB20 Travel",
    "IAB21 Real Estate", "IAB22 Shopping", "IAB23 Religion & Spirituality", "IAB24 Uncategorized"
]

# Prepare candidate labels for the model (without IAB number, model understands better this way)
candidate_labels = [label.split(" ", 1)[1] for label in iab_labels]

# Load classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Run zero-shot classification
texts = df["text"].tolist()
results = []

# You can adjust batch size here for RunPod (start small → scale up!)
for i, text in enumerate(texts[:1000]):  # WARNING: test first → then do full set!
    output = classifier(text, candidate_labels)
    top_label = output["labels"][0]
    score = round(output["scores"][0], 2)

    # Map back to full IAB label
    full_label = next(iab for iab in iab_labels if iab.endswith(top_label))

    results.append((text, full_label, score))

    if i % 100 == 0:
        print(f"Processed {i} texts...")

# Save results
result_df = pd.DataFrame(results, columns=["text", "iab_label", "confidence"])
result_df.to_csv("labeled_zero_shot_output.csv", index=False)
print("✅ Saved labeled_zero_shot_output.csv!")
