# Code Quality Check Script
# Runs formatting and linting tools for the RAG chatbot project

Write-Host "Running code quality checks..." -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "Error: pyproject.toml not found. Please run from project root." -ForegroundColor Red
    exit 1
}

# Run isort to sort imports
Write-Host "1. Sorting imports with isort..." -ForegroundColor Yellow
uv run isort backend/ main.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: isort failed" -ForegroundColor Red
    exit 1
}

# Run black to format code
Write-Host "2. Formatting code with black..." -ForegroundColor Yellow
uv run black backend/ main.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: black formatting failed" -ForegroundColor Red
    exit 1
}

# Run flake8 to check for style issues
Write-Host "3. Checking code style with flake8..." -ForegroundColor Yellow
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: flake8 found style issues" -ForegroundColor Yellow
    # Don't exit on flake8 warnings, just warn
}

Write-Host "Code quality checks completed!" -ForegroundColor Green