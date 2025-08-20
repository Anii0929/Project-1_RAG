#!/bin/bash
# Run all code quality checks

echo "Running code quality checks..."

echo "🔍 Running flake8 linting..."
uv run flake8 backend/ --show-source --statistics

echo "🔧 Checking black formatting..."
uv run black --check --diff backend/

echo "📦 Checking import sorting..."
uv run isort --check-only --diff backend/

echo "🧪 Running type checking with mypy..."
uv run mypy backend/ --ignore-missing-imports

echo "✅ All quality checks completed!"