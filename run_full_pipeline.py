# run_full_pipeline.py
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

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
if "pct_synthetic" in meta.columns:
    meta_plot = meta[["iab_label", "pct_synthetic", "pct_natural"]].set_index("iab_label")
    meta_plot.plot.bar(stacked=True, figsize=(12,6), color=["#ff9999", "#99ccff"])
    plt.title("Source Composition (% Synthetic vs Natural)")
    plt.ylabel("%")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("balanced_split_output/source_composition.png")
    plt.close()
else:
    print("\n⚠️ No 'source' column found → skipping source composition plot.")

print("\n🎉 FULL PIPELINE COMPLETE! Outputs in balanced_split_output/ 🚀")

