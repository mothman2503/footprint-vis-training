import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# CONFIG
INPUT_FILE = "labeled_zero_shot_output_combined.csv"
OUTPUT_FOLDER = "balanced_split_output"
MAX_SAMPLES_PER_CLASS = 2000  # Adjust this → e.g. 1000, 2000, etc.
CONFIDENCE_THRESHOLD = 0.3  # Filter out examples below this confidence

# Load data
df = pd.read_csv(INPUT_FILE)

# Filter out low confidence rows
initial_count = len(df)
df = df[df["confidence"] >= CONFIDENCE_THRESHOLD].reset_index(drop=True)
filtered_count = len(df)

print(f"\n✅ Filtered examples below confidence {CONFIDENCE_THRESHOLD}:")
print(f"→ Initial count: {initial_count}")
print(f"→ After filtering: {filtered_count}")

# Make output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Show initial class distribution
print("\nInitial class distribution after filtering:")
print(df["iab_label"].value_counts())

# Balance dataset → downsample big classes
balanced_df = (
    df.groupby("iab_label")
    .apply(lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_CLASS), random_state=42))
    .reset_index(drop=True)
)

# Show balanced class distribution
class_counts = balanced_df["iab_label"].value_counts()

print("\nBalanced class distribution:")
print(class_counts)

# Check for classes with too few samples
under_sampled_classes = class_counts[class_counts < MAX_SAMPLES_PER_CLASS]

if not under_sampled_classes.empty:
    print("\n⚠️ WARNING: The following classes had fewer than", MAX_SAMPLES_PER_CLASS, "samples:")
    for label, count in under_sampled_classes.items():
        print(f"→ {label}: {count} samples")
else:
    print(f"\n✅ All classes have at least {MAX_SAMPLES_PER_CLASS} samples.")

# Split → first split off test set
train_val_df, test_df = train_test_split(
    balanced_df, test_size=0.1, stratify=balanced_df["iab_label"], random_state=42
)

# Split train/val
train_df, val_df = train_test_split(
    train_val_df, test_size=0.1111, stratify=train_val_df["iab_label"], random_state=42
)
# → 0.1111 so that final split is ~80/10/10

# Save files
train_df.to_csv(os.path.join(OUTPUT_FOLDER, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_FOLDER, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_FOLDER, "test.csv"), index=False)

print(f"\n✅ Saved balanced splits to: {OUTPUT_FOLDER}")
print(f"→ Train: {len(train_df)} rows")
print(f"→ Val:   {len(val_df)} rows")
print(f"→ Test:  {len(test_df)} rows")
