import os
import json
import openai
import csv
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env (optional)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY not set in environment variables.")

# Use new OpenAI client
client = openai.OpenAI(api_key=api_key)

# Load IAB categories
with open("iab_categories.json", "r") as f:
    iab_categories = json.load(f)

# Output directory
output_dir = Path("output_chunks/output_chunks_synthetic_GPT35")
output_dir.mkdir(parents=True, exist_ok=True)

# Generation config
TEMPERATURE = 0.8
MAX_TOKENS = 40
NUM_QUERIES = 10
CONFIDENCE = 0.85  # Adjust if needed

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
            max_tokens=MAX_TOKENS,
            n=NUM_QUERIES,
        )
        texts = [choice.message.content.strip() for choice in response.choices]
        return texts
    except Exception as e:
        print(f"❌ Error generating for {iab_code}-{label}: {e}")
        return []

# Loop over IAB categories and generate queries
for parent_code, parent_data in iab_categories.items():
    for sub_code, sub_label in parent_data.get("children", {}).items():
        queries = generate_queries(sub_code, sub_label)
        if not queries:
            print(f"⚠️ No queries for {sub_code} - {sub_label}")
            continue

        output_file = output_dir / f"labeled_batch_{sub_code.replace('-', '_')}_{sub_label.replace(' ', '_')}.csv"
        with output_file.open("w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["text", "iab_label", "confidence"])
            for query in queries:
                writer.writerow([query, parent_code, CONFIDENCE])
        print(f"✅ Saved: {output_file.name} with {len(queries)} queries")
