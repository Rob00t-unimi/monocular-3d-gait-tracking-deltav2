# PowerShell Setup Script for Exploratory SpaTrackerV2 & VGGT Conda Environments
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

Write-Host "=== Setting up Exploratory SpaTrackerV2 & VGGT Environments ===" -ForegroundColor Cyan
conda env update -f envs/spatrack_env.yml --prune
conda env update -f envs/vggt_env.yml --prune
Write-Host "=== Exploratory environments configured ===" -ForegroundColor Green
