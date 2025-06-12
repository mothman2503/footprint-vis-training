# train.py (robust version with auto-resume + empty string handling + auto JSON/JSONL detect)
from transformers import pipeline
import pandas as pd
import json
import os

# CONFIG — change these safely!
INPUT_FILE = "train_text_only.json"
BATCH_SIZE = 1000  # How many rows to save per CSV
INFER_BATCH_SIZE = 16  # How many texts to send to model at once → T4: 8–16 is good
TOTAL_ROWS = 300000  # Or len(texts) if smaller!
OUTPUT_FOLDER = "output_chunks"

# Load your data — auto-detect JSON or JSONL
with open(INPUT_FILE, "r") as f:
    first_char = f.read(1)
    f.seek(0)  # reset file pointer
    if first_char == "[":
        print("Detected JSON array format.")
        data = json.load(f)
    else:
        print("Detected JSONL format.")
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
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)  # device=0 → use GPU

# Prepare output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Prepare texts
texts = df["text"].tolist()

# Loop in batches (save every BATCH_SIZE rows)
for start_idx in range(0, min(TOTAL_ROWS, len(texts)), BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE

    # Auto-resume → skip batch if already processed
    batch_filename = os.path.join(OUTPUT_FOLDER, f"labeled_batch_{start_idx}_{end_idx}.csv")
    if os.path.exists(batch_filename):
        print(f"⏩ Skipping already processed batch {start_idx} to {end_idx}...")
        continue

    batch_texts = texts[start_idx:end_idx]

    batch_results = []
    print(f"\n🚀 Processing batch {start_idx} to {end_idx}...")

    # Loop in inference sub-batches
    for infer_start in range(0, len(batch_texts), INFER_BATCH_SIZE):
        infer_end = infer_start + INFER_BATCH_SIZE
        infer_texts = batch_texts[infer_start:infer_end]

        # Filter empty strings
        infer_texts = [t for t in infer_texts if t.strip() != ""]
        if len(infer_texts) == 0:
            print(f"⚠️ Skipping empty infer_texts at batch {start_idx}-{end_idx}, sub-batch {infer_start}-{infer_end}")
            continue

        # Run zero-shot
        outputs = classifier(infer_texts, candidate_labels)

        # If single input → outputs is dict; if multiple → list of dicts
        if isinstance(outputs, dict):
            outputs = [outputs]

        # Process outputs
        for text, output in zip(infer_texts, outputs):
            top_label = output["labels"][0]
            score = round(output["scores"][0], 2)

            # Map back to full IAB label
            full_label = next(iab for iab in iab_labels if iab.endswith(top_label))

            batch_results.append((text, full_label, score))

        print(f"Processed {min(infer_end, len(batch_texts))} / {len(batch_texts)} in current batch...")

    # Save this batch to CSV
    batch_df = pd.DataFrame(batch_results, columns=["text", "iab_label", "confidence"])
    batch_df.to_csv(batch_filename, index=False)
    print(f"✅ Saved {batch_filename}!")

print("\n🎉 All batches processed!")
