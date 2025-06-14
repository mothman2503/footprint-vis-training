import json
import os
import openai
import csv
from pathlib import Path

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set")

# Load IAB categories
with open("iab_categories.json", "r") as f:
    iab_categories = json.load(f)

# Output directory
output_dir = Path("generated_phrases_gpt35")
output_dir.mkdir(parents=True, exist_ok=True)

# Generation settings
NUM_QUERIES = 10
TEMPERATURE = 0.8
MAX_TOKENS = 50

def generate_queries(iab_code, label):
    system_msg = f"You are a helpful assistant that writes example search queries for the topic '{label}'."
    user_msg = f"Generate {NUM_QUERIES} search queries someone might use online about '{label}'. Start each with the answer (2–5 words), then a dash, and then the query."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS * NUM_QUERIES,
        )
        content = response.choices[0].message["content"]
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]
    except Exception as e:
        print(f"❌ Error generating for {iab_code}: {e}")
        return []

# Loop through categories and generate
for iab_code, label in iab_categories.items():
    queries = generate_queries(iab_code, label)
    output_file = output_dir / f"labeled_batch_{iab_code.replace('-', '_')}_{label.replace(' ', '_')}.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "iab_label", "confidence"])
        for q in queries:
            writer.writerow([q, iab_code.split("-")[0], 0.9])
        print(f"✅ {output_file.name} saved with {len(queries)} queries.")
