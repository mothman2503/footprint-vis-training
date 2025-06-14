import os
import csv
import json
import random
import re
from transformers import pipeline
from tqdm import tqdm

# Load IAB subcategories
with open("iab_subcategories.json", "r") as f:
    subcategories = json.load(f)

# Output directory
output_dir = "output_chunks/output_chunks_synthetic_GPT2"
os.makedirs(output_dir, exist_ok=True)

# Set up GPT-2 generator
generator = pipeline("text-generation", model="gpt2", device=0)  # Remove device=0 if not using GPU

# Parameters
queries_per_category = 10
confidence = 0.8

# Extract plain topic name and parent category
def extract_topic(label):
    match = re.search(r"IAB\d+-\d+\s+(.*)", label)
    return match.group(1) if match else label

def extract_parent_category(label):
    match = re.match(r"(IAB\d+)-\d+", label)
    return match.group(1) if match else label.split("-")[0]

# Prompt templates for variation
prompt_templates = [
    "Example of a search query about {}:",
    "What is a realistic search query for someone interested in {}?",
    "Write a common Google search query on the topic of {}.",
    "Give me a sample web search related to {}."
]

def build_prompt(topic):
    template = random.choice(prompt_templates)
    return template.format(topic.lower())

# Filtering function for GPT-2 output
def clean_text(text):
    text = text.strip().replace("\n", " ")
    if len(text) < 10:
        return None
    if any(x in text.lower() for x in ["<query", "</", "xmlns", "mailto:", "select *", "<a href"]):
        return None
    if text.lower().startswith("iab"):
        return None
    return text

# Generate and save queries for each subcategory
for label in tqdm(subcategories):
    topic = extract_topic(label)
    parent = extract_parent_category(label)
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
        print(f"Skipped {label} (no valid results).")
        continue

    safe_label = label.replace(" ", "_").replace("/", "_").replace("&", "and")
    filename = f"labeled_batch_{safe_label}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "iab_label", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Saved {len(results)} entries for {label} to {filepath}")
