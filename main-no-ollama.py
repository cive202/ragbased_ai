import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask , render_template , request
import numpy as np 
import joblib 
import os

app = Flask(__name__)

# Load embeddings
df = joblib.load('embeddings.joblib')

def create_embedding(text_list):
    """Create dummy embeddings for testing without Ollama"""
    # Return random embeddings for testing
    return [np.random.rand(1024).tolist() for _ in text_list]

def inference(prompt):
    """Mock inference for testing without Ollama"""
    return {
        "response": f"Mock response: Based on your question '{prompt}', here are some relevant OpenGL topics from the course videos. This is a test response since Ollama is not available."
    }

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
