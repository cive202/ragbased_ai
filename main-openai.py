import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask , render_template , request
import numpy as np 
import joblib 
import requests
import os

app = Flask(__name__)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def create_embedding(text_list):
    """Create embeddings using OpenAI API"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": text_list,
            "model": "text-embedding-3-small"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            embeddings = [item['embedding'] for item in response.json()['data']]
            return embeddings
        else:
            print(f"OpenAI API error: {response.status_code}")
            return [[0.0] * 1536 for _ in text_list]  # OpenAI embedding size
            
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return [[0.0] * 1536 for _ in text_list]

def inference(prompt):
    """Generate response using OpenAI API"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return {"response": response.json()['choices'][0]['message']['content']}
        else:
            print(f"OpenAI API error: {response.status_code}")
            return {"response": "Sorry, the AI model is currently unavailable. Please try again later."}
            
    except Exception as e:
        print(f"Error in inference: {e}")
        return {"response": "Sorry, the AI model is currently unavailable. Please try again later."}

# Load embeddings
df = joblib.load('embeddings.joblib')

@app.route("/", methods = ["GET","POST"])
def result():
    if request.method == "POST":
        incoming_query = request.form["queryInput"]
        question_embedding = create_embedding([incoming_query])[0] 

        # Find similarities of question_embedding with other embeddings
        similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
        print(similarities)
        top_results = 5
        max_indx = similarities.argsort()[::-1][0:top_results]
        new_df = df.loc[max_indx] 

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

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 5000))
    # Only run in debug mode if not in production
    debug_mode = os.getenv('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
