import os
import csv
import json
import random
import re
from transformers import pipeline
from tqdm import tqdm

# Load structured IAB categories
with open("iab_categories.json", "r") as f:
    category_data = json.load(f)

# Flatten subcategories for processing
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

# Set up GPT-2 generator
generator = pipeline("text-generation", model="gpt2", device=0)  # Remove device=0 if not using GPU

# Parameters
queries_per_category = 10
confidence = 0.8

# Prompt templates
prompt_templates = [
    "What would someone search on Google if they were interested in {}?",
    "Write an example of a search someone might type about {}.",
    "Give a realistic search engine query related to {}.",
    "What is a sample search query for the topic of {}?"
]

def build_prompt(topic):
    template = random.choice(prompt_templates)
    return template.format(topic.lower())

# Aggressive filtering
def clean_text(text):
    text = text.strip().replace("\n", " ").replace('"', '').strip()
    if len(text) < 10:
        return None
    bad_patterns = ["<", ">", "http", "SELECT", "FROM", "game->", "pokemon", "{", "}", "&&", "||", "::", "==", "youtube.com"]
    if any(bad in text for bad in bad_patterns):
        return None
    if re.search(r"[{}<>|\[\]]", text):
        return None
    return text

# Generate and save queries
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
