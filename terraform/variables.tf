variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "cms-anomaly-detection"
}

variable "athena_database" {
  description = "Athena database name"
  type        = string
  default     = "cms_open_payments"
}

variable "athena_table" {
  description = "Athena table name"
  type        = string
  default     = "general_payments"
}

variable "sagemaker_endpoint_name" {
  description = "SageMaker endpoint name for anomaly detection"
  type        = string
}

variable "github_repository_url" {
  description = "GitHub repository URL for the frontend"
  type        = string
}

variable "github_access_token" {
  description = "GitHub personal access token for Amplify"
  type        = string
  sensitive   = true
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 300
}

variable "lambda_memory_size" {
  description = "Lambda function memory size in MB"
  type        = number
  default     = 1024
}
