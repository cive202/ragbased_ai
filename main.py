import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import numpy as np
import joblib
import requests
from config import api_key  # Cohere API key

app = Flask(__name__)

# -----------------------------
# Cohere API endpoints
# -----------------------------
COHERE_CHAT_URL = "https://api.cohere.com/v2/chat"  # v2 chat endpoint
COHERE_EMBED_URL = "https://api.cohere.ai/v1/embed"  # embedding endpoint

# -----------------------------
# Embedding creation function
# -----------------------------
def create_embedding(text_list, input_type="search_query"):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "embed-english-v3.0",
        "texts": text_list,
        "input_type": input_type
    }
    r = requests.post(COHERE_EMBED_URL, headers=headers, json=data)
    if r.status_code != 200:
        print("❌ Cohere embedding error:", r.text)
        r.raise_for_status()
    return r.json()["embeddings"]

# -----------------------------
# Cohere Chat API inference
# -----------------------------
def inference_cohere(messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "command-a-03-2025",
        "messages": messages,
        "temperature": 0.2
    }
    r = requests.post(COHERE_CHAT_URL, headers=headers, json=data)
    if r.status_code != 200:
        print("❌ Cohere chat error:", r.text)
        r.raise_for_status()
    response = r.json()
    return response["message"]["content"][0]["text"].strip()

# -----------------------------
# Load precomputed embeddings
# -----------------------------
df = joblib.load('embeddings.joblib')
print(f"✅ Loaded {len(df)} subtitle chunks.")

# -----------------------------
# Flask route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        incoming_query = request.form["queryInput"]

        # 1️⃣ Create embedding for user query
        try:
            question_embedding = create_embedding([incoming_query], input_type="search_query")[0]
        except Exception as e:
            return f"Error creating embedding: {e}"

        # 2️⃣ Compute cosine similarity with dataset embeddings
        try:
            similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
        except Exception as e:
            return f"Error computing similarity: {e}"

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

        # 5️⃣ Prepare messages for Cohere Chat API
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]

        # 6️⃣ Generate response
        try:
            response = inference_cohere(messages)
        except Exception as e:
            return f"Error generating response: {e}"

        # Save response for debugging
        with open("response.txt", "w", encoding="utf-8") as f:
            f.write(response)

        return render_template("index.html", answer=response)

    return render_template("index.html", answer=None)

# -----------------------------
# Run Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
