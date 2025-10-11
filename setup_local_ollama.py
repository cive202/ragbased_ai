#!/usr/bin/env python3
"""
Ready-to-run Ollama + ngrok setup script
This will set up your RAG app with real Ollama locally and make it public
"""

import subprocess
import time
import os
import sys
import requests
import threading
from pathlib import Path

def run_command(command, shell=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_ollama_installed():
    """Check if Ollama is already installed"""
    success, _, _ = run_command("ollama --version")
    return success

def install_ollama():
    """Install Ollama"""
    print("🔧 Installing Ollama...")
    
    if os.name == 'nt':  # Windows
        print("📥 Downloading Ollama for Windows...")
        success, _, error = run_command("curl -L https://ollama.com/download/windows -o ollama-windows-amd64.zip")
        if success:
            print("✅ Ollama downloaded. Please install it manually from the downloaded file.")
            return False
    else:  # Linux/Mac
        success, _, error = run_command("curl -fsSL https://ollama.com/install.sh | sh")
        if success:
            print("✅ Ollama installed successfully!")
            return True
        else:
            print(f"❌ Failed to install Ollama: {error}")
            return False

def start_ollama():
    """Start Ollama server"""
    print("🚀 Starting Ollama server...")
    
    # Start Ollama in background
    if os.name == 'nt':  # Windows
        subprocess.Popen(["ollama", "serve"], shell=True)
    else:  # Linux/Mac
        subprocess.Popen(["ollama", "serve"], shell=True)
    
    # Wait for Ollama to start
    print("⏳ Waiting for Ollama to start...")
    time.sleep(10)
    
    # Check if Ollama is running
    success, _, _ = run_command("ollama list")
    if success:
        print("✅ Ollama server is running!")
        return True
    else:
        print("❌ Failed to start Ollama server")
        return False

def download_models():
    """Download required models"""
    print("📥 Downloading AI models...")
    
    models = ["llama3.2", "bge-m3"]
    
    for model in models:
        print(f"📦 Downloading {model}...")
        success, _, error = run_command(f"ollama pull {model}")
        if success:
            print(f"✅ {model} downloaded successfully!")
        else:
            print(f"❌ Failed to download {model}: {error}")
            return False
    
    return True

def install_ngrok():
    """Install ngrok"""
    print("🔧 Installing ngrok...")
    
    if os.name == 'nt':  # Windows
        # Download ngrok for Windows
        success, _, error = run_command("curl -L https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip -o ngrok.zip")
        if success:
            success, _, error = run_command("powershell Expand-Archive ngrok.zip")
            if success:
                print("✅ ngrok installed!")
                return True
    else:  # Linux/Mac
        success, _, error = run_command("curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok")
        if success:
            print("✅ ngrok installed!")
            return True
    
    print(f"❌ Failed to install ngrok: {error}")
    return False

def start_ngrok(port=5000):
    """Start ngrok tunnel"""
    print("🌐 Starting ngrok tunnel...")
    
    # Start ngrok in background
    if os.name == 'nt':  # Windows
        ngrok_process = subprocess.Popen(["./ngrok.exe", "http", str(port)], shell=True)
    else:  # Linux/Mac
        ngrok_process = subprocess.Popen(["ngrok", "http", str(port)], shell=True)
    
    # Wait for ngrok to start
    time.sleep(5)
    
    # Get ngrok URL
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        if response.status_code == 200:
            data = response.json()
            if data['tunnels']:
                public_url = data['tunnels'][0]['public_url']
                print(f"🌍 Your app is now public at: {public_url}")
                return public_url, ngrok_process
    except:
        pass
    
    print("⚠️ Could not get ngrok URL automatically. Check ngrok dashboard at http://localhost:4040")
    return None, ngrok_process

def start_flask_app():
    """Start Flask app"""
    print("🚀 Starting Flask app...")
    
    # Set environment variables
    os.environ['OLLAMA_URL'] = 'http://localhost:11434'
    os.environ['FLASK_ENV'] = 'production'
    
    # Start Flask app
    subprocess.run([sys.executable, "main.py"])

def main():
    """Main setup function"""
    print("🎯 RAG Ollama + ngrok Setup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script in your RAG project directory.")
        return
    
    # Step 1: Install Ollama
    if not check_ollama_installed():
        if not install_ollama():
            print("❌ Setup failed at Ollama installation")
            return
    else:
        print("✅ Ollama is already installed")
    
    # Step 2: Start Ollama
    if not start_ollama():
        print("❌ Setup failed at starting Ollama")
        return
    
    # Step 3: Download models
    if not download_models():
        print("❌ Setup failed at downloading models")
        return
    
    # Step 4: Install ngrok
    if not install_ngrok():
        print("❌ Setup failed at installing ngrok")
        return
    
    # Step 5: Start ngrok
    public_url, ngrok_process = start_ngrok()
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print(f"🌍 Your RAG app is now public at: {public_url}")
    print("🔗 Ollama is running locally")
    print("🤖 AI models are ready")
    print("\n📝 To stop the services:")
    print("   - Press Ctrl+C to stop Flask")
    print("   - Close this terminal to stop ngrok")
    
    # Step 6: Start Flask app
    try:
        start_flask_app()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        if ngrok_process:
            ngrok_process.terminate()

if __name__ == "__main__":
    main()
