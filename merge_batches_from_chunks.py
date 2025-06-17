import pandas as pd
import glob
import os

# Find all folders starting with "output_chunks_"
output_folders = glob.glob("output_chunks/output_chunks_*")

# Collect all batch CSV files from all matching folders
batch_files = []
for folder in output_folders:
    files = glob.glob(os.path.join(folder, "labeled_batch_*.csv"))
    for file in files:
        batch_files.append((file, folder))  # store both file and folder info

    files = glob.glob(os.path.join(folder, "missing_*.csv"))
    for file in files:
        batch_files.append((file, folder))  # store both file and folder info
        

# Optional: print how many files found
print(f"Found {len(batch_files)} batch files in {len(output_folders)} folders.")

# Load and concatenate all batches
df_list = []
for file, folder in batch_files:
    df = pd.read_csv(file)
    
    # Determine source based on folder name
    if folder.startswith("output_chunks/output_chunks_synthetic"):
        df["source"] = "synthetic"
    elif folder.startswith("output_chunks/output_chunks_pseudo_labeled"):
        df["source"] = "pseudo labeled"
    elif folder.startswith("output_chunks/output_chunks_scraped"):
        df["source"] = "scraped"
    elif folder.startswith("output_chunks/output_chunks_subcategories"):
        df["source"] = "subcategories"
    else:
        df["source"] = "unknown"  # fallback (safe guard)

    # Determine source based on folder name
    if folder.startswith("output_chunks/output_chunks_synthetic_TOP_QUALITY"):
        df["confidence"] = 1.0
    elif folder.startswith("output_chunks/output_chunks_synthetic_GOOD_QUALITY"):
        df["confidence"] = 0.9
    else:
        df["confidence"] = df["confidence"] * 0.8
    
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

# Remove duplicate rows → based on "text" only
combined_df = combined_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# Save combined CSV
combined_df.to_csv("labeled_zero_shot_output_combined.csv", index=False)

print(f"\n✅ Combined {len(batch_files)} batch files from {len(output_folders)} folders.")
print(f"✅ Combined CSV saved as labeled_zero_shot_output_combined.csv with {len(combined_df)} unique rows.")
