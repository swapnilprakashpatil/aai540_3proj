#!/bin/bash

# Deployment script for CMS Anomaly Detection System
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-prod}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Deploying CMS Anomaly Detection System"
echo "Environment: $ENVIRONMENT"
echo "========================================"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
fi

# Check prerequisites
echo "Checking prerequisites..."
command -v terraform >/dev/null 2>&1 || { echo "ERROR: Terraform not found"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "ERROR: AWS CLI not found"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "ERROR: Node.js not found"; exit 1; }

# Verify AWS credentials
echo "Verifying AWS credentials..."
aws sts get-caller-identity > /dev/null || { echo "ERROR: AWS credentials not configured"; exit 1; }

# Navigate to Terraform directory
cd "$PROJECT_ROOT/terraform"

# Initialize Terraform
echo "Initializing Terraform..."
terraform init

# Validate configuration
echo "Validating Terraform configuration..."
terraform validate

# Plan deployment
echo "Planning deployment..."
terraform plan \
    -var="environment=$ENVIRONMENT" \
    -var="sagemaker_endpoint_name=$SAGEMAKER_ENDPOINT" \
    -var="github_repository_url=$GITHUB_REPOSITORY_URL" \
    -var="github_access_token=$GITHUB_ACCESS_TOKEN" \
    -out=tfplan

# Apply deployment
echo "Applying deployment..."
terraform apply -auto-approve tfplan

# Get outputs
echo "Deployment outputs:"
API_URL=$(terraform output -raw api_gateway_endpoint)
AMPLIFY_URL=$(terraform output -raw amplify_app_url)

echo "API Gateway URL: $API_URL"
echo "Amplify App URL: $AMPLIFY_URL"

# Update frontend environment
echo "Updating frontend environment..."
sed -i.bak "s|YOUR_API_GATEWAY_URL|$API_URL|g" \
    "$PROJECT_ROOT/frontend/src/environments/environment.prod.ts"

# Run smoke test
echo "Running smoke test..."
RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"record_count": 10}' \
    "$API_URL")

if echo "$RESPONSE" | grep -q "success"; then
    echo "Deployment successful!"
    echo "Frontend URL: $AMPLIFY_URL"
else
    echo "WARNING: API smoke test failed"
    echo "Response: $RESPONSE"
fi

echo "========================================"
echo "Deployment complete!"
