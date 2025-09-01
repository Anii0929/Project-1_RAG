#!/bin/bash

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your RDSEC_API_KEY and RDSEC_API_ENDPOINT in .env"
echo "(Anthropic API key can be used as fallback)"

# Change to backend directory and start the server
cd backend && uv run uvicorn app:app --reload --port 8000