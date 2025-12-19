#!/bin/bash

# Azure Startup Script
echo "Starting AI-NIDS Application..."

# Navigate to the app directory
cd /home/site/wwwroot

# Install dependencies if not already installed
if [ ! -d "antenv" ]; then
    echo "Creating virtual environment..."
    python -m venv antenv
fi

# Activate virtual environment
source antenv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-azure.txt

# Start the application
echo "Starting Gunicorn..."
gunicorn --bind=0.0.0.0:8000 --timeout 600 application:app
