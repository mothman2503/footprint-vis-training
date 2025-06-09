import pandas as pd

df = pd.read_csv("final_balanced_multilingual_dataset.csv")
sports_df = df[df['label'] == "IAB17 Sports"]

print(sports_df.sample(10, random_state=42)[["search_term", "synthetic"]])
