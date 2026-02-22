# IAM role for Amplify
resource "aws_iam_role" "amplify" {
  name = "${var.project_name}-amplify-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "amplify.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "amplify_backend_deployment" {
  role       = aws_iam_role.amplify.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess-Amplify"
}

# Amplify App
resource "aws_amplify_app" "frontend" {
  name       = "${var.project_name}-${var.environment}"
  repository = var.github_repository_url

  access_token = var.github_access_token
  
  iam_service_role_arn = aws_iam_role.amplify.arn

  build_spec = yamlencode({
    version = "1.0"
    frontend = {
      phases = {
        preBuild = {
          commands = [
            "cd frontend",
            "npm ci"
          ]
        }
        build = {
          commands = [
            "npm run build:prod"
          ]
        }
      }
      artifacts = {
        baseDirectory = "frontend/dist/cms-anomaly-detection-app"
        files = [
          "**/*"
        ]
      }
      cache = {
        paths = [
          "frontend/node_modules/**/*"
        ]
      }
    }
  })

  environment_variables = {
    AMPLIFY_MONOREPO_APP_ROOT = "frontend"
    API_URL                   = aws_api_gateway_stage.api.invoke_url
  }

  custom_rule {
    source = "/<*>"
    status = "404"
    target = "/index.html"
  }

  custom_rule {
    source = "</^[^.]+$|\\.(?!(css|gif|ico|jpg|js|png|txt|svg|woff|woff2|ttf|map|json)$)([^.]+$)/>"
    status = "200"
    target = "/index.html"
  }
}

# Amplify Branch
resource "aws_amplify_branch" "main" {
  app_id      = aws_amplify_app.frontend.id
  branch_name = "main"

  enable_auto_build = true

  framework = "Angular"
  stage     = var.environment == "prod" ? "PRODUCTION" : "DEVELOPMENT"
}

# Amplify Domain Association (optional - configure if you have a custom domain)
# resource "aws_amplify_domain_association" "main" {
#   app_id      = aws_amplify_app.frontend.id
#   domain_name = "yourdomain.com"
#
#   sub_domain {
#     branch_name = aws_amplify_branch.main.branch_name
#     prefix      = var.environment == "prod" ? "" : var.environment
#   }
# }
