import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask , render_template , request
import numpy as np 
import joblib 
import requests
import os

app = Flask(__name__)

# Get Ollama URL from environment variable, default to localhost for development
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    try:
        # Try bge-m3 first
        r = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        })
        r.raise_for_status()
        embedding = r.json()["embeddings"] 
        return embedding
    except Exception as e:
        print(f"Error with bge-m3, trying nomic-embed-text: {e}")
        try:
            # Try alternative embedding model
            r = requests.post(f"{OLLAMA_URL}/api/embed", json={
                "model": "nomic-embed-text",
                "input": text_list
            })
            r.raise_for_status()
            embedding = r.json()["embeddings"] 
            return embedding
        except Exception as e2:
            print(f"Error creating embedding: {e2}")
            # Return a dummy embedding if Ollama is not available
            return [[0.0] * 1024 for _ in text_list]

def inference(prompt):
    try:
        # Try llama3.2 first
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
        r.raise_for_status()
        response = r.json()
        print(response)
        return response
    except Exception as e:
        print(f"Error with llama3.2, trying tinyllama: {e}")
        try:
            # Try smaller model
            r = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            })
            r.raise_for_status()
            response = r.json()
            print(response)
            return response
        except Exception as e2:
            print(f"Error in inference: {e2}")
            return {"response": "Sorry, the AI model is currently unavailable. Please try again later."}

df = joblib.load('embeddings.joblib')

@app.route("/", methods = ["GET","POST"])
def result():
    if request.method == "POST":
        incoming_query = request.form["queryInput"]
        question_embedding = create_embedding([incoming_query])[0] 

    # Find similarities of question_embedding with other embeddings
    # print(np.vstack(df['embedding'].values))
    # print(np.vstack(df['embedding']).shape)
        similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
        print(similarities)
        top_results = 5
        max_indx = similarities.argsort()[::-1][0:top_results]
    # print(max_indx)main.py
        new_df = df.loc[max_indx] 
    # print(new_df[["title", "number", "text"]])

        prompt = f'''I am teaching OpenGL in my OpenGL course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

        {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
        ---------------------------------
        "{incoming_query}"
        User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course,And dont ask any further questions. give the answer in point. the points must contain the timestamps in bold letters
    '''
        with open("prompt.txt", "w") as f:
            f.write(prompt)

        response = inference(prompt)["response"]
        print(response)
        return render_template("index.html",answer=response)
    return render_template("index.html",answer=None)

    with open("response.txt", "w") as f:
        f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 5000))
    # Only run in debug mode if not in production
    debug_mode = os.getenv('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)