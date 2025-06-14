import os
import json
import csv
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

print("🚀 Script started")

# Load environment variables from .env (optional)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY not set in environment variables.")

# OpenAI client (v1+)
client = OpenAI(api_key=api_key)

# Load IAB categories
with open("iab_categories.json", "r") as f:
    iab_categories = json.load(f)

# Output directory
output_dir = Path("output_chunks/output_chunks_synthetic_GPT35")
output_dir.mkdir(parents=True, exist_ok=True)

# Generation config
TEMPERATURE = 0.8
MAX_TOKENS = 60
NUM_QUERIES = 10
CONFIDENCE = 0.85

# Prompt template
PROMPT_TEMPLATE = (
    "Generate a list of short, realistic search queries (2–5 words) someone might type online. "
    "Topic: {label}. Start each query with the actual query only—no bullets or numbering."
)

def generate_queries(iab_code, label):
    try:
        prompt = PROMPT_TEMPLATE.format(label=label)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at writing search queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        text = response.choices[0].message.content.strip().split("\n")
        queries = [line.strip("•-1234567890. ").strip() for line in text if line.strip()]
        return queries
    except Exception as e:
        print(f"❌ Error generating for {iab_code}-{label}: {e}")
        return []

# Loop through IAB hierarchy
for parent_code, parent_data in iab_categories.items():
    parent_label = parent_data.get("name", "")
    for sub_code, sub_label in parent_data.get("children", {}).items():
        print(f"🧠 Generating queries for {sub_code} — {sub_label}")
        queries = generate_queries(sub_code, sub_label)
        if not queries:
            print(f"⚠️ No queries for {sub_code} - {sub_label}")
            continue

        filename = f"labeled_batch_{sub_code.replace('-', '_')}_{sub_label.replace(' ', '_').replace('&', 'and')}.csv"
        output_file = output_dir / filename
        with output_file.open("w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["text", "iab_label", "confidence"])
            for query in queries:
                writer.writerow([query, parent_code, CONFIDENCE])
        print(f"✅ Saved: {output_file.name} with {len(queries)} queries")
