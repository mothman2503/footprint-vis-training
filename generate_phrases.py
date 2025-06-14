import os
import csv
import openai
import json
from tqdm import tqdm

# Load your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load categories
with open("iab_categories.json") as f:
    categories = json.load(f)

# Output directory
os.makedirs("output_chunks/output_chunks_gpt35", exist_ok=True)

# Generation function using ChatCompletion
def generate_query(topic):
    system = "You are a helpful assistant that generates realistic, short search queries."
    user = f"Give me a single 2-5 word search query someone might type related to: {topic}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=30,
            temperature=0.8
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None

# Generate and save queries
for parent, info in tqdm(categories.items()):
    for subcat in info["subcategories"]:
        topic = subcat["name"]
        code = subcat["code"]
        filename = f"labeled_batch_{code}_{topic.replace(' ', '_').replace('/', '_')}.csv"
        filepath = os.path.join("output_chunks/output_chunks_gpt35", filename)

        queries = []
        for _ in range(10):
            query = generate_query(topic)
            if query:
                queries.append({"text": query, "iab_label": parent, "confidence": 0.95})

        # Save CSV
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "iab_label", "confidence"])
            writer.writeheader()
            writer.writerows(queries)

        print(f"✅ {filename} saved with {len(queries)} queries.")
