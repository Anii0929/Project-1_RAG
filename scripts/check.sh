#!/bin/bash
# Quick format and quality check script for development

echo "🚀 Running quick development checks..."

echo "Step 1: Formatting code..."
uv run black .
uv run isort .

echo "Step 2: Running linting..."
uv run flake8 backend/ --show-source --max-line-length=88 --extend-ignore=E501,E203,W503

echo "Step 3: Type checking..."
uv run mypy backend/ --ignore-missing-imports

echo "✅ Development checks completed!"