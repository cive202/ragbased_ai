import pandas as pd 
from flask import Flask , render_template , request
import numpy as np 
import joblib 
import requests
import os

app = Flask(__name__)

# Load embeddings
df = joblib.load('embeddings.joblib')

def inference(prompt):
    """Generate response using free AI service (Hugging Face)"""
    try:
        # Using Hugging Face's free inference API
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}  # You need to get a free token
        
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        
        output = query({
            "inputs": prompt,
            "parameters": {"max_length": 200}
        })
        
        if isinstance(output, list) and len(output) > 0:
            return {"response": output[0].get("generated_text", "Sorry, I couldn't generate a response.")}
        else:
            return {"response": "Sorry, the AI model is currently unavailable. Please try again later."}
            
    except Exception as e:
        print(f"Error in inference: {e}")
        return {"response": "Sorry, the AI model is currently unavailable. Please try again later."}

@app.route("/", methods = ["GET","POST"])
def result():
    if request.method == "POST":
        incoming_query = request.form["queryInput"]
        
        # Simple text search instead of embeddings for now
        query_terms = incoming_query.lower().split()
        relevant_videos = []
        
        for idx, row in df.iterrows():
            text_lower = row['text'].lower()
            score = sum(1 for term in query_terms if term in text_lower)
            if score > 0:
                relevant_videos.append((score, row))
        
        # Sort by relevance and take top 5
        relevant_videos.sort(key=lambda x: x[0], reverse=True)
        top_videos = [video[1] for video in relevant_videos[:5]]
        
        if not top_videos:
            # If no matches, return a general response
            prompt = f'''You are an OpenGL course assistant. A student asked: "{incoming_query}"
            
            Please provide a helpful response about OpenGL concepts. If the question is not related to OpenGL, politely explain that you can only help with OpenGL course questions.'''
        else:
            # Create prompt with relevant video chunks
            video_info = []
            for video in top_videos:
                video_info.append({
                    "title": video['title'],
                    "number": video['number'],
                    "start": video['start'],
                    "end": video['end'],
                    "text": video['text']
                })
            
            prompt = f'''I am teaching OpenGL in my OpenGL course. Here are relevant video subtitle chunks:

            {video_info}
            ---------------------------------
            Student asked: "{incoming_query}"
            
            Based on the video content above, provide a helpful answer. Mention which video and timestamp contains the relevant information. If the question is not related to OpenGL, politely explain that you can only help with OpenGL course questions.'''

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
