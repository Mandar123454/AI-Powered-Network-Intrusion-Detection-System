#!/bin/bash
# Azure Deployment Script for AI-NIDS
# ====================================
# This script deploys AI-NIDS to Azure App Service

set -e

# Configuration
RESOURCE_GROUP="${RESOURCE_GROUP:-ai-nids-rg}"
LOCATION="${LOCATION:-eastus}"
APP_NAME="${APP_NAME:-ai-nids-app}"
APP_PLAN="${APP_PLAN:-ai-nids-plan}"
SKU="${SKU:-B2}"
REGISTRY="${REGISTRY:-}"
IMAGE_NAME="${IMAGE_NAME:-ai-nids}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."
    
    if ! command -v az &> /dev/null; then
        echo_error "Azure CLI not found. Please install it from https://docs.microsoft.com/cli/azure/install-azure-cli"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check Azure login
    if ! az account show &> /dev/null; then
        echo_warn "Not logged into Azure. Initiating login..."
        az login
    fi
    
    echo_info "Prerequisites check passed!"
}

# Create resource group
create_resource_group() {
    echo_info "Creating resource group: $RESOURCE_GROUP"
    
    if az group exists --name "$RESOURCE_GROUP" | grep -q "true"; then
        echo_info "Resource group already exists"
    else
        az group create \
            --name "$RESOURCE_GROUP" \
            --location "$LOCATION"
        echo_info "Resource group created"
    fi
}

# Create App Service Plan
create_app_service_plan() {
    echo_info "Creating App Service Plan: $APP_PLAN"
    
    if az appservice plan show --name "$APP_PLAN" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        echo_info "App Service Plan already exists"
    else
        az appservice plan create \
            --name "$APP_PLAN" \
            --resource-group "$RESOURCE_GROUP" \
            --sku "$SKU" \
            --is-linux
        echo_info "App Service Plan created"
    fi
}

# Build and push Docker image
build_and_push_image() {
    echo_info "Building Docker image..."
    
    # Build image
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
    
    if [ -n "$REGISTRY" ]; then
        echo_info "Pushing to registry: $REGISTRY"
        
        # Tag for registry
        docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        
        # Push to registry
        docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        
        echo_info "Image pushed to registry"
    else
        echo_warn "No registry specified. Image built locally only."
        echo_warn "Set REGISTRY environment variable to push to a registry."
    fi
}

# Create Web App
create_web_app() {
    echo_info "Creating Web App: $APP_NAME"
    
    local CONTAINER_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    if [ -z "$REGISTRY" ]; then
        echo_error "Registry not specified. Cannot create Web App with container."
        echo_info "Please set REGISTRY environment variable and push the image first."
        return 1
    fi
    
    if az webapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        echo_info "Web App already exists. Updating container..."
        az webapp config container set \
            --name "$APP_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --docker-custom-image-name "$CONTAINER_IMAGE"
    else
        az webapp create \
            --name "$APP_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --plan "$APP_PLAN" \
            --deployment-container-image-name "$CONTAINER_IMAGE"
        echo_info "Web App created"
    fi
}

# Configure environment variables
configure_env_vars() {
    echo_info "Configuring environment variables..."
    
    # Generate secret key if not provided
    SECRET_KEY="${SECRET_KEY:-$(openssl rand -hex 32)}"
    
    az webapp config appsettings set \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --settings \
            FLASK_ENV=production \
            SECRET_KEY="$SECRET_KEY" \
            LOG_LEVEL=INFO \
            WEBSITES_PORT=8000
    
    echo_info "Environment variables configured"
    echo_warn "Note: Remember to configure DATABASE_URL for production database"
}

# Enable logging
enable_logging() {
    echo_info "Enabling logging..."
    
    az webapp log config \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --docker-container-logging filesystem \
        --level information
    
    echo_info "Logging enabled"
}

# Setup custom domain (optional)
setup_custom_domain() {
    local DOMAIN="$1"
    
    if [ -z "$DOMAIN" ]; then
        echo_info "No custom domain specified. Skipping..."
        return
    fi
    
    echo_info "Setting up custom domain: $DOMAIN"
    
    az webapp config hostname add \
        --webapp-name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --hostname "$DOMAIN"
    
    echo_info "Custom domain configured"
    echo_info "Please update your DNS records to point to: ${APP_NAME}.azurewebsites.net"
}

# Setup SSL (optional)
setup_ssl() {
    echo_info "Enabling HTTPS only..."
    
    az webapp update \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --https-only true
    
    echo_info "HTTPS enabled"
}

# Create Azure SQL Database (optional)
create_azure_sql() {
    local SQL_SERVER="${SQL_SERVER:-ai-nids-sql}"
    local SQL_DATABASE="${SQL_DATABASE:-ai_nids}"
    local SQL_ADMIN="${SQL_ADMIN:-nidsadmin}"
    local SQL_PASSWORD="${SQL_PASSWORD:-}"
    
    if [ -z "$SQL_PASSWORD" ]; then
        echo_warn "SQL_PASSWORD not set. Skipping database creation."
        return
    fi
    
    echo_info "Creating Azure SQL Server: $SQL_SERVER"
    
    az sql server create \
        --name "$SQL_SERVER" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --admin-user "$SQL_ADMIN" \
        --admin-password "$SQL_PASSWORD"
    
    echo_info "Creating database: $SQL_DATABASE"
    
    az sql db create \
        --name "$SQL_DATABASE" \
        --server "$SQL_SERVER" \
        --resource-group "$RESOURCE_GROUP" \
        --service-objective S0
    
    # Configure firewall to allow Azure services
    az sql server firewall-rule create \
        --server "$SQL_SERVER" \
        --resource-group "$RESOURCE_GROUP" \
        --name "AllowAzureServices" \
        --start-ip-address 0.0.0.0 \
        --end-ip-address 0.0.0.0
    
    # Get connection string
    local CONNECTION_STRING="mssql+pyodbc://${SQL_ADMIN}:${SQL_PASSWORD}@${SQL_SERVER}.database.windows.net/${SQL_DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
    
    echo_info "Setting DATABASE_URL..."
    az webapp config appsettings set \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --settings DATABASE_URL="$CONNECTION_STRING"
    
    echo_info "Azure SQL configured"
}

# Print deployment info
print_info() {
    echo ""
    echo "=================================================="
    echo "           AI-NIDS Deployment Complete            "
    echo "=================================================="
    echo ""
    echo "Resource Group: $RESOURCE_GROUP"
    echo "App Service:    $APP_NAME"
    echo "URL:            https://${APP_NAME}.azurewebsites.net"
    echo ""
    echo "Next steps:"
    echo "  1. Configure DATABASE_URL for production database"
    echo "  2. Train and upload ML models"
    echo "  3. Configure notification channels"
    echo "  4. Change default admin password"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
    echo "  Restart:      az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP"
    echo "  SSH:          az webapp ssh --name $APP_NAME --resource-group $RESOURCE_GROUP"
    echo ""
}

# Main deployment flow
main() {
    echo ""
    echo "=================================================="
    echo "       AI-NIDS Azure Deployment Script            "
    echo "=================================================="
    echo ""
    
    check_prerequisites
    create_resource_group
    create_app_service_plan
    build_and_push_image
    create_web_app
    configure_env_vars
    enable_logging
    setup_ssl
    
    # Optional: Setup database
    # create_azure_sql
    
    print_info
}

# Run main function
main "$@"
