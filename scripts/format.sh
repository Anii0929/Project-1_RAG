#!/bin/bash

# Code Formatting Script
# Auto-format the codebase with black and isort

set -e

echo "🎨 Formatting codebase..."

echo "📏 Running Black (code formatting)..."
uv run black .

echo "📦 Running isort (import sorting)..."
uv run isort .

echo "✅ Code formatting complete!"