#!/bin/bash

# Comprehensive code quality checks
set -e

echo "🔍 Running comprehensive code quality checks..."
echo ""

echo "1️⃣ Running black (code formatting)..."
uv run black --check --diff .
echo "✅ Black check passed!"
echo ""

echo "2️⃣ Running isort (import sorting)..."
uv run isort --check-only --diff .
echo "✅ Import sorting check passed!"
echo ""

echo "3️⃣ Running flake8 (linting)..."
uv run flake8 . --max-line-length=88 --extend-ignore=E203,W503 --exclude=.venv,chroma_db
echo "✅ Flake8 check passed!"
echo ""

echo "4️⃣ Running pytest (tests)..."
if [ -d "backend/tests" ]; then
    cd backend && uv run pytest tests/ -v
    cd ..
    echo "✅ Tests passed!"
else
    echo "⚠️ No tests directory found, skipping tests"
fi

echo ""
echo "🎉 All quality checks passed!"
