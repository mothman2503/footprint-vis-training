import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# CONFIG
INPUT_FILE = "labeled_zero_shot_output_combined.csv"
OUTPUT_FOLDER = "balanced_split_output"
CLASS_FOLDER = os.path.join(OUTPUT_FOLDER, "classes")
MISSING_FOLDER = os.path.join(OUTPUT_FOLDER, "missing_classes")
MAX_SAMPLES_PER_CLASS = 2000
CONFIDENCE_THRESHOLD = 0.45

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
    missing_count = (MAX_SAMPLES_PER_CLASS - current_count)*2
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
            num_natural=("source", lambda x: (x == "natural").sum()),
            num_manual=("source", lambda x: (x == "manual").sum())
        ).reset_index()

        meta["pct_synthetic"] = (meta["num_synthetic"] / meta["num_samples"]).round(3)
        meta["pct_natural"] = (meta["num_natural"] / meta["num_samples"]).round(3)
        meta["pct_manual"] = (meta["num_manual"] / meta["num_samples"]).round(3)

    else:
        print("⚠️ No 'source' column found → skipping source breakdown.")
        meta = df.groupby("iab_label").agg(
            num_samples=("text", "count"),
            avg_confidence=("confidence", "mean")
        ).reset_index()

    meta.to_csv(os.path.join(OUTPUT_FOLDER, filename), index=False)
    print(f"📊 Saved: {filename}")

generate_metadata(train_df, "metadata_train.csv")
generate_metadata(val_df, "metadata_val.csv")
generate_metadata(test_df, "metadata_test.csv")
generate_metadata(balanced_df, "metadata_balanced.csv")

metadata_path = os.path.join(OUTPUT_FOLDER, "metadata_balanced.csv")
if os.path.exists(metadata_path):
    meta_df = pd.read_csv(metadata_path)
    low_conf_classes = meta_df[meta_df["avg_confidence"] < 0.75]
    if not low_conf_classes.empty:
        print(f"\n⚠️ WARNING: {len(low_conf_classes)} class(es) have average confidence below 0.75!")
        print("Classes affected:")
        print(low_conf_classes[["iab_label", "avg_confidence"]])
    else:
        print("\n✅ All classes have average confidence ≥ 0.75.")