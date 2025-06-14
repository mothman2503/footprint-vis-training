import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random

# — CONFIGURATION —
MODEL_NAME = "gpt2-medium"
OUTPUT_FOLDER = "iab_generated_queries"
N_PER_SUBCAT = 100
MAX_TOKENS = 32
TEMPERATURE = 0.9
TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load subcategories from JSON file
with open("iab_subcategories.json", "r") as f:
    IAB_SUBCATEGORIES = json.load(f)


# — MODEL LOADING —
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# — GENERATION —
for subcat in tqdm(IAB_SUBCATEGORIES, desc="Generating queries"):
    prompt_base = f"Search query about {subcat.lower()}:"
    generated = []

    while len(generated) < N_PER_SUBCAT:
        prompt = prompt_base
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        out = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + MAX_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        text = text.split("?")[0].split(".")[0].strip()

        if 2 <= len(text.split()) <= 8 and text.lower() not in [q.lower() for q in generated]:
            generated.append(text)

    # Save to file
    safe_filename = subcat.replace(" ", "_").replace("/", "-")
    out_path = os.path.join(OUTPUT_FOLDER, f"{safe_filename}.json")
    with open(out_path, "w") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)

print("✅ DONE! Queries saved in:", OUTPUT_FOLDER)
