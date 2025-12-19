# Azure Deployment Guide for AI-NIDS

This guide provides comprehensive instructions for deploying AI-NIDS to Microsoft Azure.

## Prerequisites

1. **Azure Account** - Azure subscription (Student Premium works!)
2. **Azure CLI** - [Install Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
3. **Docker** - [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
4. **Git** - For cloning the repository

## Quick Deploy (Automated)

### Using PowerShell (Windows)
```powershell
cd deployment
.\azure-deploy.ps1 -ResourceGroup "ai-nids-rg" -AppName "ai-nids-app" -Registry "yourdockerhub"
```

### Using Bash (Linux/Mac)
```bash
cd deployment
chmod +x azure-deploy.sh
./azure-deploy.sh
```

## Manual Deployment Steps

### Step 1: Login to Azure
```bash
az login
```

### Step 2: Create Resource Group
```bash
az group create --name ai-nids-rg --location eastus
```

### Step 3: Create Container Registry (Optional)
Using Azure Container Registry (ACR):
```bash
az acr create --resource-group ai-nids-rg --name ainidsregistry --sku Basic
az acr login --name ainidsregistry
```

Or use Docker Hub.

### Step 4: Build and Push Docker Image

**Using Docker Hub:**
```bash
docker login
docker build -t yourusername/ai-nids:latest .
docker push yourusername/ai-nids:latest
```

**Using Azure Container Registry:**
```bash
docker build -t ainidsregistry.azurecr.io/ai-nids:latest .
docker push ainidsregistry.azurecr.io/ai-nids:latest
```

### Step 5: Create App Service Plan
```bash
az appservice plan create \
    --name ai-nids-plan \
    --resource-group ai-nids-rg \
    --sku B2 \
    --is-linux
```

**SKU Options:**
- `B1`, `B2`, `B3` - Basic (development)
- `S1`, `S2`, `S3` - Standard (production)
- `P1V2`, `P2V2`, `P3V2` - Premium (high performance)

### Step 6: Create Web App
```bash
az webapp create \
    --name ai-nids-app \
    --resource-group ai-nids-rg \
    --plan ai-nids-plan \
    --deployment-container-image-name yourusername/ai-nids:latest
```

### Step 7: Configure Environment Variables
```bash
az webapp config appsettings set \
    --resource-group ai-nids-rg \
    --name ai-nids-app \
    --settings \
        FLASK_ENV=production \
        SECRET_KEY="$(openssl rand -hex 32)" \
        LOG_LEVEL=INFO \
        WEBSITES_PORT=8000
```

### Step 8: Enable HTTPS
```bash
az webapp update \
    --name ai-nids-app \
    --resource-group ai-nids-rg \
    --https-only true
```

### Step 9: Enable Logging
```bash
az webapp log config \
    --name ai-nids-app \
    --resource-group ai-nids-rg \
    --docker-container-logging filesystem
```

## Database Setup

### Option 1: Azure SQL Database (Recommended)
```bash
# Create SQL Server
az sql server create \
    --name ai-nids-sql \
    --resource-group ai-nids-rg \
    --location eastus \
    --admin-user nidsadmin \
    --admin-password "YourSecurePassword123!"

# Create Database
az sql db create \
    --name ai_nids \
    --server ai-nids-sql \
    --resource-group ai-nids-rg \
    --service-objective S0

# Allow Azure services
az sql server firewall-rule create \
    --server ai-nids-sql \
    --resource-group ai-nids-rg \
    --name AllowAzureServices \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0

# Set connection string
az webapp config appsettings set \
    --name ai-nids-app \
    --resource-group ai-nids-rg \
    --settings DATABASE_URL="mssql+pyodbc://nidsadmin:YourSecurePassword123!@ai-nids-sql.database.windows.net/ai_nids?driver=ODBC+Driver+17+for+SQL+Server"
```

### Option 2: Azure Database for PostgreSQL
```bash
az postgres server create \
    --name ai-nids-postgres \
    --resource-group ai-nids-rg \
    --location eastus \
    --admin-user nidsadmin \
    --admin-password "YourSecurePassword123!" \
    --sku-name B_Gen5_1

az postgres db create \
    --name ai_nids \
    --server-name ai-nids-postgres \
    --resource-group ai-nids-rg

# Set connection string
az webapp config appsettings set \
    --name ai-nids-app \
    --resource-group ai-nids-rg \
    --settings DATABASE_URL="postgresql://nidsadmin:YourSecurePassword123!@ai-nids-postgres.postgres.database.azure.com/ai_nids?sslmode=require"
```

## Custom Domain Setup

### Add Custom Domain
```bash
az webapp config hostname add \
    --webapp-name ai-nids-app \
    --resource-group ai-nids-rg \
    --hostname yourdomain.com
```

### Configure DNS
Add a CNAME record pointing to `ai-nids-app.azurewebsites.net`

### Add SSL Certificate
```bash
# Using managed certificate (free)
az webapp config ssl bind \
    --certificate-type ManagedCertificate \
    --name ai-nids-app \
    --resource-group ai-nids-rg \
    --hostname yourdomain.com
```

## Monitoring & Logging

### View Logs
```bash
az webapp log tail --name ai-nids-app --resource-group ai-nids-rg
```

### Enable Application Insights
```bash
az monitor app-insights component create \
    --app ai-nids-insights \
    --location eastus \
    --resource-group ai-nids-rg

# Get instrumentation key and add to app settings
```

## Scaling

### Scale Up (More Power)
```bash
az appservice plan update \
    --name ai-nids-plan \
    --resource-group ai-nids-rg \
    --sku S2
```

### Scale Out (More Instances)
```bash
az webapp scale \
    --instance-count 3 \
    --name ai-nids-app \
    --resource-group ai-nids-rg
```

### Auto-Scale
```bash
az monitor autoscale create \
    --resource-group ai-nids-rg \
    --resource ai-nids-plan \
    --resource-type Microsoft.Web/serverfarms \
    --name ai-nids-autoscale \
    --min-count 1 \
    --max-count 5 \
    --count 1
```

## CI/CD with GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and Push
      run: |
        docker build -t ${{ secrets.DOCKER_USERNAME }}/ai-nids:${{ github.sha }} .
        docker push ${{ secrets.DOCKER_USERNAME }}/ai-nids:${{ github.sha }}
    
    - name: Login to Azure
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: ai-nids-app
        images: ${{ secrets.DOCKER_USERNAME }}/ai-nids:${{ github.sha }}
```

## Cost Optimization

### Student Plan Recommendations
- Use **B1** or **B2** App Service Plan
- Use **Basic** tier for SQL Database
- Enable auto-shutdown for dev/test resources
- Use **Azure Free Services** where available

### Estimated Monthly Costs (Student)
| Resource | SKU | Est. Cost |
|----------|-----|-----------|
| App Service | B2 | ~$55/month |
| SQL Database | Basic | ~$5/month |
| Storage | Standard | ~$1/month |
| **Total** | | **~$61/month** |

## Troubleshooting

### Container Not Starting
```bash
# Check logs
az webapp log tail --name ai-nids-app --resource-group ai-nids-rg

# Check container settings
az webapp config container show --name ai-nids-app --resource-group ai-nids-rg
```

### Database Connection Issues
1. Verify firewall rules allow Azure services
2. Check connection string format
3. Ensure SSL is configured properly

### Slow Performance
1. Scale up App Service Plan
2. Enable Application Insights
3. Check database query performance

## Security Checklist

- [ ] Change default admin password
- [ ] Use managed identity for Azure resources
- [ ] Enable HTTPS only
- [ ] Configure IP restrictions if needed
- [ ] Enable Azure Defender
- [ ] Set up alerts for suspicious activity
- [ ] Regularly rotate secrets

## Useful Commands

```bash
# Restart app
az webapp restart --name ai-nids-app --resource-group ai-nids-rg

# SSH into container
az webapp ssh --name ai-nids-app --resource-group ai-nids-rg

# View app settings
az webapp config appsettings list --name ai-nids-app --resource-group ai-nids-rg

# Delete all resources
az group delete --name ai-nids-rg --yes --no-wait
```

## Support

- Azure Documentation: https://docs.microsoft.com/azure
- Azure Student: https://azure.microsoft.com/free/students
- AI-NIDS Issues: https://github.com/yourusername/ai-nids/issues
