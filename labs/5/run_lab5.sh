#!/bin/bash
# Script to run Lab 5 experiments with MLflow on GCP

echo "Lab 5: MLflow on GCP with Cloud Run"
echo "-----------------------------------"

# Check if MLFLOW_URL is provided
if [ -z "$1" ]; then
    echo "Error: Please provide your MLflow server URL"
    echo "Usage: ./run_lab5.sh <MLFLOW_SERVER_URL>"
    echo "Example: ./run_lab5.sh https://mlflow-server-abcdefghij-uw.a.run.app"
    exit 1
fi

MLFLOW_URL=$1

# Install required dependencies
echo "Installing required dependencies..."
pip install -r requirements.txt

# Update the tracking URI in the script
echo "Updating MLflow tracking URI in script..."

# Check if running on macOS or Linux and use appropriate sed syntax
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' -e "s|MLFLOW_TRACKING_URI = \"https://mlflow-server-YOUR-CLOUD-RUN-URL.a.run.app\"|MLFLOW_TRACKING_URI = \"$MLFLOW_URL\"|g" run_experiments.py
else
    # Linux and others
    sed -i "s|MLFLOW_TRACKING_URI = \"https://mlflow-server-YOUR-CLOUD-RUN-URL.a.run.app\"|MLFLOW_TRACKING_URI = \"$MLFLOW_URL\"|g" run_experiments.py
fi

# Run the experiments
echo "Running experiments..."
python run_experiments.py

echo "-----------------------------------"
echo "Lab 5 completed!"
echo "Visit your MLflow server at $MLFLOW_URL to view the results" 