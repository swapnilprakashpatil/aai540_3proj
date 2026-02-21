output "api_gateway_url" {
  description = "API Gateway invoke URL"
  value       = aws_api_gateway_stage.api.invoke_url
}

output "api_gateway_endpoint" {
  description = "Full API endpoint for anomaly detection"
  value       = "${aws_api_gateway_stage.api.invoke_url}/anomaly-detection"
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.anomaly_detection.function_name
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.anomaly_detection.arn
}

output "athena_results_bucket" {
  description = "S3 bucket for Athena query results"
  value       = aws_s3_bucket.athena_results.bucket
}

output "amplify_app_id" {
  description = "Amplify app ID"
  value       = aws_amplify_app.frontend.id
}

output "amplify_default_domain" {
  description = "Amplify default domain"
  value       = aws_amplify_app.frontend.default_domain
}

output "amplify_app_url" {
  description = "Amplify application URL"
  value       = "https://${aws_amplify_branch.main.branch_name}.${aws_amplify_app.frontend.default_domain}"
}

output "cloudwatch_log_group_lambda" {
  description = "CloudWatch log group for Lambda function"
  value       = aws_cloudwatch_log_group.lambda_logs.name
}

output "cloudwatch_log_group_api_gateway" {
  description = "CloudWatch log group for API Gateway"
  value       = aws_cloudwatch_log_group.api_gateway.name
}
