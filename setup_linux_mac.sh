#!/bin/bash

echo "🎯 RAG Ollama + ngrok Setup Script for Linux/Mac"
echo "================================================"

# Install Ollama
echo "🔧 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
echo "🚀 Starting Ollama server..."
ollama serve &
sleep 10

# Download models
echo "📥 Downloading AI models..."
ollama pull llama3.2
ollama pull bge-m3

# Install ngrok
echo "🔧 Installing ngrok..."
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Start ngrok
echo "🌐 Starting ngrok tunnel..."
ngrok http 5000 &

# Set environment variables
export OLLAMA_URL=http://localhost:11434
export FLASK_ENV=production

# Start Flask app
echo "🚀 Starting Flask app..."
python3 main.py
