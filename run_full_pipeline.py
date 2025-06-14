# run_full_pipeline.py (FINAL VERSION — with classes/ and word_frequencies/ folders)

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from collections import Counter

# Paths to your component scripts
merge_script = "merge_batches_from_chunks.py"         # your combine script → this must save labeled_zero_shot_output_combined.csv
balance_split_script = "balance_and_split.py"  # your balance/split script → this must save train/val/test + metadata_balanced.csv

# Step 1 — run combine_batches.py
print("\n=== STEP 1: Merging batches ===")
subprocess.run(["python3", merge_script], check=True)

# Step 2 — run balance_and_split.py
print("\n=== STEP 2: Balancing, splitting, generating metadata ===")
subprocess.run(["python3", balance_split_script], check=True)

# Step 3 — run visualization on metadata_balanced.csv
print("\n=== STEP 3: Visualizing balanced dataset ===")

meta_file = "balanced_split_output/metadata_balanced.csv"

if not os.path.exists(meta_file):
    raise FileNotFoundError(f"ERROR: {meta_file} not found! Run balance_and_split.py first.")

meta = pd.read_csv(meta_file)

# Class distribution
plt.figure(figsize=(12, 6))
meta.sort_values("num_samples", ascending=False).plot.bar(x="iab_label", y="num_samples", legend=False)
plt.title("Class Distribution (Balanced Data)")
plt.ylabel("# Samples")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("balanced_split_output/class_distribution.png")
plt.close()

# Avg confidence per class
plt.figure(figsize=(12, 6))
meta.sort_values("avg_confidence", ascending=False).plot.bar(x="iab_label", y="avg_confidence", legend=False, color="orange")
plt.title("Average Confidence per Class")
plt.ylabel("Avg Confidence")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("balanced_split_output/avg_confidence.png")
plt.close()

# % synthetic vs natural
if {"pct_synthetic", "pct_natural", "pct_manual"}.issubset(meta.columns):
    meta_plot = meta[["iab_label", "pct_synthetic", "pct_natural", "pct_manual"]].set_index("iab_label")
    meta_plot.plot.bar(
        stacked=True,
        figsize=(12,6),
        color=["#ff9999", "#99ccff", "#a0e57c"]
    )
    plt.title("Source Composition (% Synthetic, Natural, Manual)")
    plt.ylabel("%")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("balanced_split_output/source_composition.png")
    plt.close()
else:
    print("\n⚠️ No 'source' column found → skipping source composition plot.")

# Step 4 — Save one CSV per class → balanced_split_output/classes/
print("\n=== STEP 4: Saving per-class CSVs ===")

train_file = "balanced_split_output/train.csv"
train_df = pd.read_csv(train_file)

# Make folder balanced_split_output/classes/ if not exist
classes_folder = "balanced_split_output/classes"
os.makedirs(classes_folder, exist_ok=True)

# Step 5 — Visualizing word variation per class → balanced_split_output/word_frequencies/
print("\n=== STEP 5: Visualizing word variation per class ===")

# Make folder balanced_split_output/word_frequencies/ if not exist
word_freq_folder = "balanced_split_output/word_frequencies"
os.makedirs(word_freq_folder, exist_ok=True)

# Simple tokenizer → split on non-word characters
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

# Process each class
for label in train_df["iab_label"].unique():
    class_df = train_df[train_df["iab_label"] == label]
    
    all_words = []
    for text in class_df["text"]:
        all_words.extend(tokenize(text))
    
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(20)  # top 20 words
    
    words, counts = zip(*most_common) if most_common else ([], [])
    
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color="skyblue")
    plt.title(f"Top 20 Words in Class: {label}")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    label_safe = label.replace(" ", "_").replace("’", "").replace(",", "").replace("–", "-")
    word_freq_filename = os.path.join(word_freq_folder, f"word_freq_{label_safe}.png")
    
    plt.savefig(word_freq_filename)
    plt.close()
    
    print(f"✅ Saved {word_freq_filename}")

print("\n🎉 FULL PIPELINE COMPLETE! Outputs in balanced_split_output/ 🚀")
