import pandas as pd
import glob
import os

# Path to your output_chunks directory
batch_folder = "output_chunks"

# Find all batch CSVs
batch_files = glob.glob(os.path.join(batch_folder, "labeled_batch_*.csv"))

# Sort files by start index (optional but nice)
batch_files.sort(key=lambda x: int(x.split("_")[-2]))

# Load and concatenate all batches
df_list = [pd.read_csv(file) for file in batch_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Save combined CSV
combined_df.to_csv("labeled_zero_shot_output_combined.csv", index=False)

print(f"✅ Combined {len(batch_files)} batch files.")
print(f"✅ Combined CSV saved as labeled_zero_shot_output_combined.csv with {len(combined_df)} rows.")
