import os
import json
import pandas as pd
import requests
import joblib
from config import api_key  # make sure your Cohere key is in config.py

COHERE_EMBED_URL = "https://api.cohere.ai/v1/embed"

def create_embedding(text_list):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "embed-english-v3.0",  # or embed-multilingual-v3.0
        "texts": text_list,
        "input_type": "search_document"  # ✅ required for v3.0 models
    }
    r = requests.post(COHERE_EMBED_URL, headers=headers, json=data)
    if r.status_code != 200:
        print("❌ Error response from Cohere:")
        print(r.text)
    r.raise_for_status()

    resp_json = r.json()
    return resp_json["embeddings"]




# -----------------------------
# Load all subtitle chunks
# -----------------------------
json_folder = "jsons"
all_chunks = []

for file in os.listdir(json_folder):
    if file.endswith(".json"):
        path = os.path.join(json_folder, file)
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
            title = content.get("title", "Unknown")
            number = content.get("number", "N/A")

            for chunk in content.get("chunks", []):
                all_chunks.append({
                    "title": title,
                    "number": number,
                    "start": chunk.get("start", 0),
                    "end": chunk.get("end", 0),
                    "text": chunk.get("text", "")
                })

# Convert to DataFrame
df = pd.DataFrame(all_chunks)
print(f"Loaded {len(df)} chunks from {len(os.listdir(json_folder))} JSON files.")

# -----------------------------
# Create embeddings in batches
# -----------------------------
print("Creating embeddings with Cohere (this may take a while)...")

batch_size = 50
embeddings = []

for i in range(0, len(df), batch_size):
    batch_texts = df["text"].iloc[i:i+batch_size].tolist()
    emb = create_embedding(batch_texts)
    embeddings.extend(emb)
    print(f"✅ Processed {i + len(batch_texts)} / {len(df)}")

df["embedding"] = embeddings

# -----------------------------
# Save for later use in Flask app
# -----------------------------
joblib.dump(df, "embeddings.joblib")
print("✅ All embeddings created and saved to embeddings.joblib")
