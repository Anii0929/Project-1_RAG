# Quality Check Script
param()

Write-Host "Running quality checks..." -ForegroundColor Green

if (-not (Test-Path "pyproject.toml")) {
    Write-Host "Error: pyproject.toml not found. Please run from project root." -ForegroundColor Red
    exit 1
}

Write-Host "1. Checking import order..." -ForegroundColor Yellow
uv run isort backend/ main.py --check-only --diff
$isort_result = $LASTEXITCODE

Write-Host "2. Checking code formatting..." -ForegroundColor Yellow  
uv run black backend/ main.py --check --diff
$black_result = $LASTEXITCODE

Write-Host "3. Checking code style..." -ForegroundColor Yellow
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503
$flake8_result = $LASTEXITCODE

Write-Host "--- Quality Check Summary ---" -ForegroundColor Blue

if ($isort_result -eq 0) {
    Write-Host "Import order: PASSED" -ForegroundColor Green
} else {
    Write-Host "Import order: NEEDS ATTENTION" -ForegroundColor Red
}

if ($black_result -eq 0) {
    Write-Host "Code formatting: PASSED" -ForegroundColor Green  
} else {
    Write-Host "Code formatting: NEEDS ATTENTION" -ForegroundColor Red
}

if ($flake8_result -eq 0) {
    Write-Host "Code style: PASSED" -ForegroundColor Green
} else {
    Write-Host "Code style: HAS WARNINGS" -ForegroundColor Yellow
}

if ($isort_result -eq 0 -and $black_result -eq 0) {
    Write-Host "All quality checks passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Run format.ps1 to fix formatting issues." -ForegroundColor Yellow
    exit 1
}