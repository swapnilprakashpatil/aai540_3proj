# Quick Start Scripts

This directory contains helper scripts for common deployment tasks.

## Available Scripts

### setup.sh (Linux/Mac) or setup.ps1 (Windows)

Initial setup script that:

- Checks prerequisites
- Configures AWS CLI
- Creates Terraform backend
- Sets up environment variables

### deploy.sh (Linux/Mac) or deploy.ps1 (Windows)

Deployment script that:

- Validates Terraform configuration
- Deploys infrastructure
- Updates frontend with API URL
- Runs smoke tests

### cleanup.sh (Linux/Mac) or cleanup.ps1 (Windows)

Cleanup script that:

- Destroys all Terraform resources
- Removes temporary files
- Optionally deletes Terraform backend

## Usage

### Linux/Mac

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/deploy.sh
```

### Windows PowerShell

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\scripts\setup.ps1
.\scripts\deploy.ps1
```
