#!/bin/bash

# Start Ollama in background
ollama serve &

# Wait for Ollama to start
sleep 10

# Pull required models
echo "Pulling llama3.2 model..."
ollama pull llama3.2

echo "Pulling bge-m3 model..."
ollama pull bge-m3

# Start the Flask application
echo "Starting Flask application..."
python3 main.py
