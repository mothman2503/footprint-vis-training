# zero_shot_labeling.py
from transformers import pipeline
import pandas as pd
import json
import os

# CONFIG — change these safely!
INPUT_FILE = "train_text_only.json"
BATCH_SIZE = 1000
TOTAL_ROWS = 300000  # Or len(texts) if smaller!
OUTPUT_FOLDER = "output_chunks"  # Folder to save batch CSVs

# Load your data
with open(INPUT_FILE, "r") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
df = df[["text"]]  # Only keep the "text" column

# IAB labels
iab_labels = [
    "IAB1 Arts & Entertainment", "IAB2 Automotive", "IAB3 Business", "IAB4 Careers",
    "IAB5 Education", "IAB6 Family & Parenting", "IAB7 Health & Fitness", "IAB8 Food & Drink",
    "IAB9 Hobbies & Interests", "IAB10 Home & Garden", "IAB11 Law, Gov’t & Politics",
    "IAB12 News", "IAB13 Personal Finance", "IAB14 Society", "IAB15 Science", "IAB16 Pets",
    "IAB17 Sports", "IAB18 Style & Fashion", "IAB19 Technology & Computing", "IAB20 Travel",
    "IAB21 Real Estate", "IAB22 Shopping", "IAB23 Religion & Spirituality", "IAB24 Uncategorized"
]

# Candidate labels (cleaned for model)
candidate_labels = [label.split(" ", 1)[1] for label in iab_labels]

# Load classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Prepare output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Prepare texts
texts = df["text"].tolist()
results = []

# Loop in batches
for start_idx in range(0, min(TOTAL_ROWS, len(texts)), BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_texts = texts[start_idx:end_idx]

    batch_results = []
    for i, text in enumerate(batch_texts):
        output = classifier(text, candidate_labels)
        top_label = output["labels"][0]
        score = round(output["scores"][0], 2)

        # Map back to full IAB label
        full_label = next(iab for iab in iab_labels if iab.endswith(top_label))

        batch_results.append((text, full_label, score))

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} texts in current batch...")

    # Save this batch to CSV
    batch_df = pd.DataFrame(batch_results, columns=["text", "iab_label", "confidence"])
    batch_filename = os.path.join(OUTPUT_FOLDER, f"labeled_batch_{start_idx}_{end_idx}.csv")
    batch_df.to_csv(batch_filename, index=False)
    print(f"✅ Saved {batch_filename}!")

print("🎉 All batches processed!")

# Optional: merge all batch CSVs into one final CSV (uncomment if you want)
# import glob
# all_files = glob.glob(os.path.join(OUTPUT_FOLDER, "labeled_batch_*.csv"))
# final_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
# final_df.to_csv("labeled_zero_shot_output.csv", index=False)
# print("✅ Merged all batches into labeled_zero_shot_output.csv!")
