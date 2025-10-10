import pandas as pd 
from flask import Flask , render_template , request
import numpy as np 
import joblib 
import os
import random

app = Flask(__name__)

# Load embeddings
df = joblib.load('embeddings.joblib')

def inference(prompt):
    """Generate smart mock response based on content"""
    
    # Extract key terms from the prompt
    key_terms = []
    if "triangle" in prompt.lower():
        key_terms.append("triangle")
    if "shader" in prompt.lower():
        key_terms.append("shader")
    if "texture" in prompt.lower():
        key_terms.append("texture")
    if "lighting" in prompt.lower():
        key_terms.append("lighting")
    if "3d" in prompt.lower():
        key_terms.append("3D")
    if "window" in prompt.lower():
        key_terms.append("window")
    if "buffer" in prompt.lower():
        key_terms.append("buffer")
    if "model" in prompt.lower():
        key_terms.append("model")
    
    # Generate contextual response
    if key_terms:
        response = f"""Based on your question about {', '.join(key_terms)}, here's what I found in the OpenGL course:

**Relevant Video Content:**
- The course covers {key_terms[0]} concepts in multiple videos
- You can find detailed explanations in the video tutorials
- Look for timestamps where these topics are discussed

**Key Points:**
- This is an important OpenGL concept
- The course provides step-by-step examples
- Practice with the provided code samples

**Next Steps:**
- Watch the relevant video sections
- Try the code examples
- Ask specific questions about implementation

*Note: This is a demo response. The full AI system would provide more detailed, personalized answers based on the exact video content.*"""
    else:
        response = """I can help you with OpenGL course questions! 

**What I can assist with:**
- OpenGL concepts and implementation
- Video content explanations
- Code examples and tutorials
- Step-by-step guidance

**Please ask specific questions about:**
- Triangles, shaders, textures
- 3D graphics, lighting
- Buffers, models, rendering
- Any OpenGL programming topic

*Note: This is a demo response. The full AI system would analyze your specific question and provide detailed answers based on the course content.*"""

    return {"response": response}

@app.route("/", methods = ["GET","POST"])
def result():
    if request.method == "POST":
        incoming_query = request.form["queryInput"]
        
        # Simple text search to find relevant content
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
        
        if top_videos:
            # Create context from relevant videos
            video_context = []
            for video in top_videos:
                video_context.append(f"**Video {video['number']}: {video['title']}** (Time: {video['start']}s-{video['end']}s)\n{video['text']}")
            
            context = "\n\n".join(video_context)
            prompt = f"""Student asked: "{incoming_query}"

Relevant course content:
{context}

Please provide a helpful response based on this content."""
        else:
            prompt = f"""Student asked: "{incoming_query}"

Please provide a helpful OpenGL response."""

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
