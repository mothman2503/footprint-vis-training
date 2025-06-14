import os
import csv
import json
import random
import re
from transformers import pipeline
from tqdm import tqdm

# Load structured IAB category data
with open("iab_categories.json", "r") as f:
    category_data = json.load(f)

# Flatten all subcategories
subcategories = []
for parent_code, entry in category_data.items():
    for subcat in entry["subcategories"]:
        subcategories.append({
            "parent": parent_code,
            "subcategory_code": subcat["code"],
            "subcategory_name": subcat["name"]
        })

# Output directory
output_dir = "output_chunks/output_chunks_synthetic_GPT2"
os.makedirs(output_dir, exist_ok=True)

# GPT-2 setup
generator = pipeline("text-generation", model="gpt2", device=0)  # remove device=0 if no GPU

queries_per_category = 10
confidence = 0.8

# Prompt with direct instruction for short, query-only answer
def build_prompt(topic):
    return f"Return only a short (2-5 words) search query about {topic.lower()}:"

# Clean GPT output
def clean_text(text):
    text = text.strip().replace("\n", " ").replace('"', '').replace(":", "").strip()
    if len(text.split()) < 2 or len(text.split()) > 6:
        return None
    if any(x in text.lower() for x in [
        "what would someone", "search query about", "example of", 
        "return only", "<", ">", "::", "youtube", "mailto:", "SELECT", "FROM"
    ]):
        return None
    return text

# Generate queries
for item in tqdm(subcategories):
    topic = item["subcategory_name"]
    parent = item["parent"]
    label = f"{item['subcategory_code']} {item['subcategory_name']}"
    prompt = build_prompt(topic)

    generations = generator(
        [prompt] * queries_per_category,
        max_length=20,
        num_return_sequences=1,
        do_sample=True,
        temperature=1.0,
        top_k=50,
    )

    results = []
    for gen in generations:
        gen_text = gen[0]["generated_text"].replace(prompt, "").strip()
        cleaned = clean_text(gen_text)
        if cleaned:
            results.append({
                "text": cleaned,
                "iab_label": parent,
                "confidence": confidence
            })

    if not results:
        print(f"⚠️ Skipped {label} (no valid results).")
        continue

    safe_label = label.replace(" ", "_").replace("/", "_").replace("&", "and")
    filename = f"labeled_batch_{safe_label}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "iab_label", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Saved {len(results)} entries for {label} → {filepath}")
