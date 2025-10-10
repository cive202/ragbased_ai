#!/bin/bash

# Start Ollama in background
echo "Starting Ollama server..."
ollama serve &

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 15

# Check if Ollama is running
echo "Checking Ollama status..."
ollama list

# Try smaller models first
echo "Pulling smaller models for Render compatibility..."

# Try tinyllama (much smaller)
echo "Pulling tinyllama model..."
ollama pull tinyllama || echo "Failed to pull tinyllama"

# Try smaller embedding model
echo "Pulling nomic-embed-text model (smaller alternative)..."
ollama pull nomic-embed-text || echo "Failed to pull nomic-embed-text"

# Check what models are available
echo "Available models:"
ollama list

# Start the Flask application
echo "Starting Flask application..."
python3 main.py
