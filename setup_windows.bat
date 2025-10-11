@echo off
echo 🎯 RAG Ollama + ngrok Setup Script for Windows
echo ================================================

echo 🔧 Installing Ollama...
curl -L https://ollama.com/download/windows -o ollama-windows-amd64.zip
if exist ollama-windows-amd64.zip (
    echo ✅ Ollama downloaded. Please install it manually.
    echo 📥 Run the installer and then restart this script.
    pause
    exit /b
)

echo 🚀 Starting Ollama server...
start /B ollama serve
timeout /t 10 /nobreak

echo 📥 Downloading AI models...
ollama pull llama3.2
ollama pull bge-m3

echo 🔧 Installing ngrok...
curl -L https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip -o ngrok.zip
powershell Expand-Archive ngrok.zip -DestinationPath .

echo 🌐 Starting ngrok tunnel...
start /B ngrok.exe http 5000

echo 🚀 Starting Flask app...
set OLLAMA_URL=http://localhost:11434
set FLASK_ENV=production
python main.py

pause
