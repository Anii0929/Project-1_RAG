#!/bin/bash

# Run linting checks
echo "Running flake8..."
uv run flake8 . --max-line-length=88 --extend-ignore=E203,W503 --exclude=.venv,chroma_db

echo "✅ Linting complete!"
