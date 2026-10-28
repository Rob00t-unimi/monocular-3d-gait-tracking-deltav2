# PowerShell Setup Script for DELTAv2 & Analysis Conda Environments
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

Write-Host "=== Setting up Gait360 DELTAv2 & Analysis Conda Environments ===" -ForegroundColor Cyan
conda env update -f envs/analysis_env.yml --prune
conda env update -f envs/delta_env.yml --prune

Write-Host "=== Installing editable gait360 package ===" -ForegroundColor Cyan
conda run -n analysis_env pip install -e .
Write-Host "=== Setup completed successfully ===" -ForegroundColor Green
