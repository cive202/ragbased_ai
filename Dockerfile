# Use Ubuntu as base image
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-simple.txt .
RUN pip3 install -r requirements-simple.txt

# Copy application files
COPY . .

# Expose ports
EXPOSE 8000 11434

# Make start script executable and run it
RUN chmod +x start.sh

# Set environment variables
ENV FLASK_ENV=production
ENV OLLAMA_URL=http://localhost:11434

# Start the application
CMD ./start.sh
