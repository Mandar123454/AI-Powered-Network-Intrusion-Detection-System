# Azure Deployment Script for AI-NIDS (PowerShell)
# =================================================
# This script deploys AI-NIDS to Azure App Service

param(
    [string]$ResourceGroup = "ai-nids-rg",
    [string]$Location = "eastus",
    [string]$AppName = "ai-nids-app",
    [string]$AppPlan = "ai-nids-plan",
    [string]$Sku = "B2",
    [string]$Registry = "",
    [string]$ImageName = "ai-nids",
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check Azure CLI
    if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
        Write-Error "Azure CLI not found. Please install it from https://docs.microsoft.com/cli/azure/install-azure-cli"
        exit 1
    }
    
    # Check Docker
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker not found. Please install Docker Desktop."
        exit 1
    }
    
    # Check Azure login
    $account = az account show 2>$null | ConvertFrom-Json
    if (-not $account) {
        Write-Warn "Not logged into Azure. Initiating login..."
        az login
    }
    
    Write-Info "Prerequisites check passed!"
}

# Create resource group
function New-ResourceGroup {
    Write-Info "Creating resource group: $ResourceGroup"
    
    $exists = az group exists --name $ResourceGroup
    if ($exists -eq "true") {
        Write-Info "Resource group already exists"
    } else {
        az group create --name $ResourceGroup --location $Location
        Write-Info "Resource group created"
    }
}

# Create App Service Plan
function New-AppServicePlan {
    Write-Info "Creating App Service Plan: $AppPlan"
    
    $plan = az appservice plan show --name $AppPlan --resource-group $ResourceGroup 2>$null
    if ($plan) {
        Write-Info "App Service Plan already exists"
    } else {
        az appservice plan create `
            --name $AppPlan `
            --resource-group $ResourceGroup `
            --sku $Sku `
            --is-linux
        Write-Info "App Service Plan created"
    }
}

# Build and push Docker image
function Build-AndPushImage {
    Write-Info "Building Docker image..."
    
    # Build image
    docker build -t "${ImageName}:${ImageTag}" .
    
    if ($Registry) {
        Write-Info "Pushing to registry: $Registry"
        
        # Tag for registry
        docker tag "${ImageName}:${ImageTag}" "${Registry}/${ImageName}:${ImageTag}"
        
        # Push to registry
        docker push "${Registry}/${ImageName}:${ImageTag}"
        
        Write-Info "Image pushed to registry"
    } else {
        Write-Warn "No registry specified. Image built locally only."
        Write-Warn "Set -Registry parameter to push to a registry."
    }
}

# Create Web App
function New-WebApp {
    Write-Info "Creating Web App: $AppName"
    
    if (-not $Registry) {
        Write-Error "Registry not specified. Cannot create Web App with container."
        Write-Info "Please set -Registry parameter and push the image first."
        return
    }
    
    $ContainerImage = "${Registry}/${ImageName}:${ImageTag}"
    
    $app = az webapp show --name $AppName --resource-group $ResourceGroup 2>$null
    if ($app) {
        Write-Info "Web App already exists. Updating container..."
        az webapp config container set `
            --name $AppName `
            --resource-group $ResourceGroup `
            --docker-custom-image-name $ContainerImage
    } else {
        az webapp create `
            --name $AppName `
            --resource-group $ResourceGroup `
            --plan $AppPlan `
            --deployment-container-image-name $ContainerImage
        Write-Info "Web App created"
    }
}

# Configure environment variables
function Set-EnvVars {
    Write-Info "Configuring environment variables..."
    
    # Generate secret key
    $SecretKey = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | ForEach-Object {[char]$_})
    
    az webapp config appsettings set `
        --name $AppName `
        --resource-group $ResourceGroup `
        --settings `
            FLASK_ENV=production `
            SECRET_KEY=$SecretKey `
            LOG_LEVEL=INFO `
            WEBSITES_PORT=8000
    
    Write-Info "Environment variables configured"
    Write-Warn "Note: Remember to configure DATABASE_URL for production database"
}

# Enable logging
function Enable-Logging {
    Write-Info "Enabling logging..."
    
    az webapp log config `
        --name $AppName `
        --resource-group $ResourceGroup `
        --docker-container-logging filesystem `
        --level information
    
    Write-Info "Logging enabled"
}

# Enable HTTPS
function Enable-Https {
    Write-Info "Enabling HTTPS only..."
    
    az webapp update `
        --name $AppName `
        --resource-group $ResourceGroup `
        --https-only true
    
    Write-Info "HTTPS enabled"
}

# Create Azure SQL Database
function New-AzureSql {
    param(
        [string]$SqlServer = "ai-nids-sql",
        [string]$SqlDatabase = "ai_nids",
        [string]$SqlAdmin = "nidsadmin",
        [string]$SqlPassword
    )
    
    if (-not $SqlPassword) {
        Write-Warn "SQL Password not provided. Skipping database creation."
        return
    }
    
    Write-Info "Creating Azure SQL Server: $SqlServer"
    
    az sql server create `
        --name $SqlServer `
        --resource-group $ResourceGroup `
        --location $Location `
        --admin-user $SqlAdmin `
        --admin-password $SqlPassword
    
    Write-Info "Creating database: $SqlDatabase"
    
    az sql db create `
        --name $SqlDatabase `
        --server $SqlServer `
        --resource-group $ResourceGroup `
        --service-objective S0
    
    # Configure firewall
    az sql server firewall-rule create `
        --server $SqlServer `
        --resource-group $ResourceGroup `
        --name "AllowAzureServices" `
        --start-ip-address 0.0.0.0 `
        --end-ip-address 0.0.0.0
    
    # Set connection string
    $ConnectionString = "mssql+pyodbc://${SqlAdmin}:${SqlPassword}@${SqlServer}.database.windows.net/${SqlDatabase}?driver=ODBC+Driver+17+for+SQL+Server"
    
    az webapp config appsettings set `
        --name $AppName `
        --resource-group $ResourceGroup `
        --settings DATABASE_URL=$ConnectionString
    
    Write-Info "Azure SQL configured"
}

# Print deployment info
function Show-Info {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "           AI-NIDS Deployment Complete            " -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Resource Group: $ResourceGroup"
    Write-Host "App Service:    $AppName"
    Write-Host "URL:            https://${AppName}.azurewebsites.net"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Configure DATABASE_URL for production database"
    Write-Host "  2. Train and upload ML models"
    Write-Host "  3. Configure notification channels"
    Write-Host "  4. Change default admin password"
    Write-Host ""
    Write-Host "Useful commands:" -ForegroundColor Yellow
    Write-Host "  View logs:    az webapp log tail --name $AppName --resource-group $ResourceGroup"
    Write-Host "  Restart:      az webapp restart --name $AppName --resource-group $ResourceGroup"
    Write-Host "  SSH:          az webapp ssh --name $AppName --resource-group $ResourceGroup"
    Write-Host ""
}

# Main deployment
function Main {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "       AI-NIDS Azure Deployment Script            " -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ""
    
    Test-Prerequisites
    New-ResourceGroup
    New-AppServicePlan
    Build-AndPushImage
    New-WebApp
    Set-EnvVars
    Enable-Logging
    Enable-Https
    
    # Optional: Setup database
    # New-AzureSql -SqlPassword "YourSecurePassword123!"
    
    Show-Info
}

# Run main
Main
