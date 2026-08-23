$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "InsightFlow AI Release Checks" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

Write-Host ""
Write-Host "1. Checking Python version..." -ForegroundColor Yellow

python --version

if ($LASTEXITCODE -ne 0) {
    throw "Python is unavailable."
}

Write-Host ""
Write-Host "2. Compiling application source..." -ForegroundColor Yellow

python -m compileall `
    app `
    core `
    tests `
    scripts

if ($LASTEXITCODE -ne 0) {
    throw "Python compilation failed."
}

Write-Host ""
Write-Host "3. Checking critical imports..." -ForegroundColor Yellow

$importCheck = @"
from app.agents.ai_planner import AIAnalysisPlan
from app.agents.ai_planner import AIPlannerResult
from app.agents.ai_planner import create_ai_plan
from app.agents.ai_planner import plan_with_qwen
from app.services.filtered_executor import execute_filtered_analysis
from core.analytics_quality_gate import validate_profit_readiness
from core.analytics_quality_gate import validate_revenue_readiness
from core.unified_file_loader import process_uploaded_file
print('Critical imports passed.')
"@

python -c $importCheck

if ($LASTEXITCODE -ne 0) {
    throw "Critical import checks failed."
}

Write-Host ""
Write-Host "4. Running deterministic tests..." -ForegroundColor Yellow

python -m pytest `
    -q `
    -ra `
    -m "not ollama"

if ($LASTEXITCODE -ne 0) {
    throw "Deterministic tests failed."
}

Write-Host ""
Write-Host "5. Checking optional Ollama environment..." -ForegroundColor Yellow

$ollamaCommand = Get-Command `
    ollama `
    -ErrorAction SilentlyContinue

if ($ollamaCommand) {
    ollama --version

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Ollama was found but did not return its version."
    }

    ollama list

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Ollama was found but the model list was unavailable."
    }
}
else {
    Write-Host (
        "Ollama is unavailable. " +
        "Optional local-LLM tests were not executed."
    ) -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "InsightFlow AI release checks passed." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
