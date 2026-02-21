# Deployment script for CMS Anomaly Detection System (Windows PowerShell)
# Usage: .\deploy.ps1 [-Environment prod]

param(
    [Parameter(Mandatory = $false)]
    [string]$Environment = "prod"
)

$ErrorActionPreference = "Stop"

Write-Host "Deploying CMS Anomaly Detection System" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Load environment variables from .env
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Item -Path "env:$name" -Value $value
        }
    }
}

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

if (-not (Get-Command terraform -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Terraform not found" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: AWS CLI not found" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Node.js not found" -ForegroundColor Red
    exit 1
}

# Verify AWS credentials
Write-Host "Verifying AWS credentials..." -ForegroundColor Yellow
try {
    aws sts get-caller-identity | Out-Null
}
catch {
    Write-Host "ERROR: AWS credentials not configured" -ForegroundColor Red
    exit 1
}

# Navigate to Terraform directory
Set-Location (Join-Path $ProjectRoot "terraform")

# Initialize Terraform
Write-Host "Initializing Terraform..." -ForegroundColor Yellow
terraform init

# Validate configuration
Write-Host "Validating Terraform configuration..." -ForegroundColor Yellow
terraform validate

# Plan deployment
Write-Host "Planning deployment..." -ForegroundColor Yellow
terraform plan `
    -var="environment=$Environment" `
    -var="sagemaker_endpoint_name=$env:SAGEMAKER_ENDPOINT" `
    -var="github_repository_url=$env:GITHUB_REPOSITORY_URL" `
    -var="github_access_token=$env:GITHUB_ACCESS_TOKEN" `
    -out=tfplan

# Apply deployment
Write-Host "Applying deployment..." -ForegroundColor Yellow
terraform apply -auto-approve tfplan

# Get outputs
Write-Host "Deployment outputs:" -ForegroundColor Yellow
$ApiUrl = terraform output -raw api_gateway_endpoint
$AmplifyUrl = terraform output -raw amplify_app_url

Write-Host "API Gateway URL: $ApiUrl" -ForegroundColor Green
Write-Host "Amplify App URL: $AmplifyUrl" -ForegroundColor Green

# Update frontend environment
Write-Host "Updating frontend environment..." -ForegroundColor Yellow
$EnvFile = Join-Path $ProjectRoot "frontend\src\environments\environment.prod.ts"
(Get-Content $EnvFile) -replace 'YOUR_API_GATEWAY_URL', $ApiUrl | Set-Content $EnvFile

# Run smoke test
Write-Host "Running smoke test..." -ForegroundColor Yellow
$Body = @{ record_count = 10 } | ConvertTo-Json
try {
    $Response = Invoke-RestMethod -Uri $ApiUrl -Method Post -Body $Body -ContentType "application/json"
    if ($Response.success) {
        Write-Host "Deployment successful!" -ForegroundColor Green
        Write-Host "Frontend URL: $AmplifyUrl" -ForegroundColor Cyan
    }
    else {
        Write-Host "WARNING: API smoke test failed" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "WARNING: Could not reach API endpoint" -ForegroundColor Yellow
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Deployment complete!" -ForegroundColor Green
