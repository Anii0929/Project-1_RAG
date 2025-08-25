#!/bin/bash

# Code Quality Script
# Run all quality checks for the project

set -e

echo "🔍 Running code quality checks..."

echo "📏 Running Black (code formatting)..."
uv run black --check --diff .

echo "📦 Running isort (import sorting)..."
uv run isort --check-only --diff .

echo "🔎 Running flake8 (linting)..."
uv run flake8 .

echo "🏷️  Running mypy (type checking)..."
uv run mypy backend/ main.py

echo "✅ All quality checks passed!"