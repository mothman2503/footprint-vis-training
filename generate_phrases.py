import json
import csv
from pathlib import Path

# Load IAB category hierarchy
with open("iab_categories.json", "r") as f:
    iab = json.load(f)

# Output directory
output_dir = Path("output_chunks/output_chunks_subcats")
output_dir.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.95

for parent_code, parent_data in iab.items():
    parent_name = parent_data["name"]
    full_label = f"{parent_code} {parent_name}"
    subcategories = parent_data.get("subcategories", [])
    rows = []

    for sub in subcategories:
        sub_code = sub["code"]
        sub_name = sub["name"]
        query = sub_name.strip()
        rows.append([query, full_label, CONFIDENCE])

    if rows:
        safe_filename = f"labeled_batch_{parent_code}_{parent_name.replace(' ', '_').replace('&', 'and')}.csv"
        output_path = output_dir / safe_filename
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "iab_label", "confidence"])
            writer.writerows(rows)
        print(f"✅ Saved: {safe_filename} ({len(rows)} queries)")
