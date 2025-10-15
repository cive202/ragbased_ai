import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import numpy as np
import joblib
import requests
from config import api_key  # Cohere API key
import os
from io import BytesIO
import logging

# configure basic logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# -----------------------------
# Simple health check for hosting platforms
# -----------------------------
@app.route('/health')
def health():
    return "OK", 200

# -----------------------------
# Cohere API endpoints
# -----------------------------
COHERE_CHAT_URL = "https://api.cohere.com/v2/chat"
COHERE_EMBED_URL = "https://api.cohere.ai/v1/embed"

# -----------------------------
# Embedding creation function
# -----------------------------
def create_embedding(text_list, input_type="search_query"):
    if not api_key:
        return None, "Cohere API key not configured. Set COHERE_API_KEY environment variable."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "embed-english-v3.0",
        "texts": text_list,
        "input_type": input_type
    }
    try:
        r = requests.post(COHERE_EMBED_URL, headers=headers, json=data, timeout=30)
        if r.status_code in (429, 402):
            logger.warning("Cohere embed API returned status %s", r.status_code)
            return None, "API limit reached or billing required. Please try again later."
        r.raise_for_status()
        return r.json()["embeddings"], None
    except Exception as e:
        logger.exception("Error calling Cohere embed API")
        return None, f"Cohere embedding error: {e}"

# -----------------------------
# Cohere Chat API inference
# -----------------------------
def inference_cohere(messages):
    if not api_key:
        return None, "Cohere API key not configured. Set COHERE_API_KEY environment variable."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "command-a-03-2025",
        "messages": messages,
        "temperature": 0.2
    }
    try:
        r = requests.post(COHERE_CHAT_URL, headers=headers, json=data, timeout=30)
        if r.status_code in (429, 402):
            logger.warning("Cohere chat API returned status %s", r.status_code)
            return None, "API limit reached or billing required. Please try again later."
        r.raise_for_status()
        response = r.json()
        return response["message"]["content"][0]["text"].strip(), None
    except Exception as e:
        logger.exception("Error calling Cohere chat API")
        return None, f"Cohere chat error: {e}"

# -----------------------------
# Load precomputed embeddings
# -----------------------------
emb_url = os.environ.get("EMBEDDINGS_URL")
df = None
if emb_url:
    try:
        logger.info("Downloading embeddings from %s", emb_url)
        r = requests.get(emb_url, timeout=60)
        r.raise_for_status()
        df = joblib.load(BytesIO(r.content))
        logger.info("Loaded %d subtitle chunks from EMBEDDINGS_URL.", len(df))
    except Exception as e:
        logger.exception("Failed to download/load embeddings from EMBEDDINGS_URL")
        # Don't crash the app; keep running and show a helpful message on the UI
        df = None
else:
    try:
        df = joblib.load('embeddings.joblib')
        logger.info("Loaded %d subtitle chunks.", len(df))
    except Exception as e:
        logger.exception("Failed to load local embeddings.joblib")
        df = None

# -----------------------------
# Flask route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        incoming_query = request.form.get("queryInput", "").strip()
        if not incoming_query:
            return render_template("index.html", answer="Please enter a question.", query="")

        if df is None:
            msg = "Embeddings not loaded. Please set EMBEDDINGS_URL or add embeddings.joblib to the project."
            logger.warning(msg)
            return render_template("index.html", answer=msg, query=incoming_query)

        # 1️⃣ Create embedding for user query
        question_embedding, error = create_embedding([incoming_query], input_type="search_query")
        if error:
            return render_template("index.html", answer=error)
        question_embedding = question_embedding[0]

        # 2️⃣ Compute cosine similarity (safe checks)
        try:
            if 'embedding' not in df.columns or df['embedding'].isnull().all():
                raise ValueError("No embeddings found in dataframe.")

            embeddings_matrix = np.vstack(df['embedding'].values)
            similarities = cosine_similarity(embeddings_matrix, [question_embedding]).flatten()
        except Exception as e:
            logger.exception("Error computing similarity")
            return render_template("index.html", answer=f"Error computing similarity: {e}", query=incoming_query)

        # 3️⃣ Get top 5 most similar chunks
        top_results = 5
        max_indx = similarities.argsort()[::-1][:top_results]
        new_df = df.loc[max_indx].copy()
        new_df["text"] = new_df["text"].str[:1000]  # truncate long text

        # 4️⃣ Build prompt
        prompt_text = f"""I am teaching OpenGL in my OpenGL course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, and the text at that time:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}

User asked the question: "{incoming_query}"

Answer in a human-friendly way, guiding the user to the specific video(s) and timestamps where the content is taught.
If unrelated to the course, say you can only answer course-related questions.
Answer in points. Each point must contain timestamps in **bold**.
"""

        # Save prompt for debugging
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt_text)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]

        # 5️⃣ Generate response
        response, error = inference_cohere(messages)
        if error:
            return render_template("index.html", answer=error)

        # Save response for debugging
        with open("response.txt", "w", encoding="utf-8") as f:
            f.write(response)

    return render_template("index.html", answer=response, query=incoming_query)

    return render_template("index.html", answer=None, query=None)

# -----------------------------
# Run Flask app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
