
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

# CONFIG
INPUT_FILE = "labeled_zero_shot_output_combined.csv"
OUTPUT_FOLDER = "balanced_split_output"
CLASS_FOLDER = os.path.join(OUTPUT_FOLDER, "classes")
MISSING_FOLDER = os.path.join(OUTPUT_FOLDER, "missing_classes")
MAX_SAMPLES_PER_CLASS = 1000
CONFIDENCE_THRESHOLD = 0.2

# Load full dataset
df_all = pd.read_csv(INPUT_FILE)

# Split low- and high-confidence sets
df_low_conf = df_all[df_all["confidence"] < CONFIDENCE_THRESHOLD]
df_high_conf = df_all[df_all["confidence"] >= CONFIDENCE_THRESHOLD].reset_index(drop=True)

print(f"\n✅ Filtered examples below confidence {CONFIDENCE_THRESHOLD}:")
print(f"→ Low confidence removed: {len(df_low_conf)} rows")
print(f"→ High confidence retained: {len(df_high_conf)} rows")

# Create output folders
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CLASS_FOLDER, exist_ok=True)
os.makedirs(MISSING_FOLDER, exist_ok=True)

# Balance high-confidence dataset
balanced_df = (
    df_high_conf.groupby("iab_label")
    .apply(lambda x: x.sort_values("confidence", ascending=False).head(MAX_SAMPLES_PER_CLASS))
    .reset_index(drop=True)
)

# Save one CSV per class (high confidence → used)
print(f"\n📦 Saving class-level balanced CSVs to: {CLASS_FOLDER}")
for label in balanced_df["iab_label"].unique():
    df_class = balanced_df[balanced_df["iab_label"] == label].sort_values("confidence", ascending=False)
    label_safe = label.replace(" ", "_").replace("’", "").replace(",", "").replace("–", "-")
    out_path = os.path.join(CLASS_FOLDER, f"class_{label_safe}.csv")
    df_class.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path} ({len(df_class)} rows)")

# Identify underrepresented classes and get missing samples from low-confidence data
print(f"\n🔍 Saving low-confidence 'missing' samples for underrepresented classes to: {MISSING_FOLDER}")
class_counts = balanced_df["iab_label"].value_counts()
for label in class_counts.index:
    current_count = class_counts[label]
    missing_count = (MAX_SAMPLES_PER_CLASS - current_count) * 2
    if missing_count > 0:
        missing_candidates = df_low_conf[df_low_conf["iab_label"] == label]
        if not missing_candidates.empty:
            top_missing = missing_candidates.sort_values("confidence", ascending=False).head(missing_count)
            if not top_missing.empty:
                label_safe = label.replace(" ", "_").replace("’", "").replace(",", "").replace("–", "-")
                out_path = os.path.join(MISSING_FOLDER, f"missing_{label_safe}.csv")
                top_missing.to_csv(out_path, index=False)
                print(f"→ {label}: saved {len(top_missing)} missing samples")

# Remove classes with < 2 samples before splitting
valid_labels = balanced_df["iab_label"].value_counts()
valid_labels = valid_labels[valid_labels >= 2].index.tolist()
balanced_df = balanced_df[balanced_df["iab_label"].isin(valid_labels)]

# Stratified split into train/val/test
train_val_df, test_df = train_test_split(
    balanced_df, test_size=0.1, stratify=balanced_df["iab_label"], random_state=42
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.1111, stratify=train_val_df["iab_label"], random_state=42
)

# Save splits
train_df.to_csv(os.path.join(OUTPUT_FOLDER, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_FOLDER, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_FOLDER, "test.csv"), index=False)
print(f"\n✅ Saved splits: train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)})")

# Generate metadata
def generate_metadata(df, filename):


    source_exists = "source" in df.columns
    if source_exists:
        meta = df.groupby("iab_label").agg(
            num_samples=("text", "count"),
            avg_confidence=("confidence", "mean"),
            num_synthetic=("source", lambda x: (x == "synthetic").sum()),
            num_pseudo_labeled=("source", lambda x: (x == "pseudo labeled").sum()),
            num_scraped=("source", lambda x: (x == "scraped").sum()),
            num_subcategories=("source", lambda x: (x == "subcategories").sum())
        ).reset_index()

        meta["pct_synthetic"] = (meta["num_synthetic"] / meta["num_samples"]).round(3)
        meta["pct_pseudo_labeled"] = (meta["num_pseudo_labeled"] / meta["num_samples"]).round(3)
        meta["pct_scraped"] = (meta["num_scraped"] / meta["num_samples"]).round(3)
        meta["pct_subcategories"] = (meta["num_subcategories"] / meta["num_samples"]).round(3)

        meta["Synthetic"] = meta["num_synthetic"]
        meta["Pseudo Labeled"] = meta["num_pseudo_labeled"]
        meta["Scraped"] = meta["num_scraped"]
        meta["Subcategories"] = meta["num_subcategories"]

    else:
        print("⚠️ No 'source' column found → skipping source breakdown.")
        meta = df.groupby("iab_label").agg(
            num_samples=("text", "count"),
            avg_confidence=("confidence", "mean")
        ).reset_index()

    # Extract the numeric part of the IAB label and create a temporary sort column
    meta["iab_number"] = meta["iab_label"].apply(lambda x: int(re.search(r"IAB(\d+)", x).group(1)))
    # Sort the DataFrame by that number
    meta = meta.sort_values("iab_number").drop(columns="iab_number")

    meta.to_csv(os.path.join(OUTPUT_FOLDER, filename), index=False)
    print(f"📊 Saved: {filename}")
    return meta

meta_df = generate_metadata(balanced_df, "metadata_balanced.csv")
# NEW: Print per-class summary including confidence and missing samples
print("\n📋 Class Summary:")
print(f"\n{'Status':<6} {'IAB Label':<35} {'Count':>6} {'Confidence':>10}\n____________________________________________________________\n")
for _, row in meta_df.iterrows():
    label = row["iab_label"]
    count = int(row["num_samples"])
    conf = round(row["avg_confidence"], 3)
    missing = max(0, MAX_SAMPLES_PER_CLASS - count)
    emoji = "✅" if missing == 0 else "⚠️\u200A"
    print(f"{emoji}     {label:<35} {count:>6} {conf:>10.3f}")


    
