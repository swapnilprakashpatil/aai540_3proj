# Lambda Function - Anomaly Detection

This Lambda function serves as the API backend for the CMS Anomaly Detection application.

## Functionality

1. **Athena Query**: Fetches random records from the CMS Open Payments dataset stored in Athena
2. **Feature Preparation**: Prepares features for the ML model
3. **SageMaker Inference**: Calls the deployed SageMaker endpoint for anomaly detection
4. **Response Processing**: Combines results and returns structured JSON response

## Environment Variables

- `ATHENA_DATABASE`: Athena database name (default: `cms_open_payments`)
- `ATHENA_TABLE`: Athena table name (default: `general_payments`)
- `ATHENA_OUTPUT_BUCKET`: S3 bucket for Athena query results
- `SAGEMAKER_ENDPOINT`: SageMaker endpoint name

## IAM Permissions Required

The Lambda execution role needs the following permissions:

- `athena:StartQueryExecution`
- `athena:GetQueryExecution`
- `athena:GetQueryResults`
- `s3:GetObject`
- `s3:PutObject`
- `glue:GetTable`
- `glue:GetDatabase`
- `sagemaker:InvokeEndpoint`

## Request Format

```json
{
  "record_count": 10000
}
```

## Response Format

```json
{
  "success": true,
  "total_records": 10000,
  "anomaly_count": 150,
  "anomaly_percentage": 1.5,
  "results": [
    {
      "record": {
        "Record_ID": "123456",
        "Recipient_State": "CA",
        "Total_Amount_of_Payment_USDollars": 5000.00,
        ...
      },
      "anomaly_score": -0.85,
      "is_anomaly": true,
      "features": {}
    }
  ],
  "execution_time": 12.5,
  "timestamp": "2026-02-16T10:30:00Z"
}
```

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ATHENA_DATABASE=cms_open_payments
export ATHENA_TABLE=general_payments
export ATHENA_OUTPUT_BUCKET=your-bucket-name
export SAGEMAKER_ENDPOINT=your-endpoint-name

# Test locally (requires valid AWS credentials)
python -c "from handler import lambda_handler; import json; print(lambda_handler({'body': json.dumps({'record_count': 100})}, {}))"
```

## Deployment

This function is deployed using Terraform. See the `terraform/` directory for infrastructure configuration.

## Monitoring

CloudWatch logs are automatically created for this function with the log group:
`/aws/lambda/cms-anomaly-detection`
