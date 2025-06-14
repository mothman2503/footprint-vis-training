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

# GPT-2 (or medium if available)
generator = pipeline("text-generation", model="gpt2-medium", device=0)  # use gpt2 if gpt2-medium is unavailable

queries_per_category = 10
confidence = 0.8

# Example few-shot queries
few_shot_examples = [
    "Search query: best historical novels",
    "Search query: celebrity gossip today",
    "Search query: how to grill ribs",
    "Search query: guitar chords for beginners"
]

# Prompt using examples
def build_prompt(topic):
    examples_text = "\n".join(random.sample(few_shot_examples, 2))
    return f"{examples_text}\nSearch query:"

# Clean and validate output
def clean_text(text):
    text = text.strip().replace("\n", " ").replace('"', '').strip()
    if "Search query:" in text:
        text = text.split("Search query:")[-1].strip()
    if len(text.split()) < 2 or len(text.split()) > 6:
        return None
    if any(x in text.lower() for x in [
        "example of", "<", ">", "::", "youtube", "mailto:", "SELECT", "FROM"
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
        max_length=25,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9,
        top_k=50,
    )

    results = []
    for gen in generations:
        gen_text = gen[0]["generated_text"]
        cleaned = clean_text(gen_text)
        if cleaned:
            results.append({
                "text": cleaned,
                "iab_label": parent,
                "confidence": confidence
            })

    # Retry once if empty
    if not results:
        print(f"⚠️ Retrying {label} with fallback...")
        fallback_query = f"{topic.lower()} tips"
        results = [{
            "text": fallback_query,
            "iab_label": parent,
            "confidence": confidence
        }]

    safe_label = label.replace(" ", "_").replace("/", "_").replace("&", "and")
    filename = f"labeled_batch_{safe_label}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "iab_label", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Saved {len(results)} entries for {label} → {filepath}")
