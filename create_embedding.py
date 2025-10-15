import os
import json
import pandas as pd
import requests
import joblib
from config import api_key  

COHERE_EMBED_URL = "https://api.cohere.ai/v1/embed"

def create_embedding(text_list):
    """Create embeddings using Cohere v3."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "embed-english-v3.0",  # or embed-multilingual-v3.0
        "texts": text_list,
        "input_type": "search_document"  # required for v3.0 models
    }
    try:
        r = requests.post(COHERE_EMBED_URL, headers=headers, json=data)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("Error response from Cohere:")
        print(r.text)
        raise e

    resp_json = r.json()
    return resp_json["embeddings"]


# -----------------------------
# Load all subtitle chunks from new folder
# -----------------------------
json_folder = "newjsons"  # updated folder
all_chunks = []

json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

for file in json_files:
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

df = pd.DataFrame(all_chunks)
print(f"Loaded {len(df)} chunks from {len(json_files)} JSON files.")


# -----------------------------
# Create embeddings in batches
# -----------------------------
print("Creating embeddings with Cohere (this may take a while)...")
batch_size = 50
embeddings = []

for i in range(0, len(df), batch_size):
    batch_texts = df["text"].iloc[i:i + batch_size].tolist()
    try:
        batch_emb = create_embedding(batch_texts)
        embeddings.extend(batch_emb)
    except Exception as e:
        print(f" Failed at batch {i}–{i+batch_size}: {e}")
        continue
    print(f"Processed {min(i + batch_size, len(df))} / {len(df)}")

df["embedding"] = embeddings

# -----------------------------
# Save for later use in Flask app
# -----------------------------
joblib.dump(df, "embeddings.joblib")
print(" All embeddings created and saved to embeddings.joblib")
