@echo off
REM Format all Python code using black and isort

echo Running code formatting...

echo 🔧 Formatting code with black...
uv run black .

echo 📦 Sorting imports with isort...
uv run isort .

echo ✅ Code formatting completed!