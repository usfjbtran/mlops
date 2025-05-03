# Lab 7 Submission: MLflow on GCP

## GitHub Repository Link
The code for Lab 7 is available in the `src` folder of the repository:
[https://github.com/usfjbtran/mlops/tree/main/src](https://github.com/usfjbtran/mlops/tree/main/src)

## Required Screenshots
Please include the following screenshots in your PDF submission to Canvas:

1. **Screenshot 1**: Terminal showing Kubernetes job creation
   - This should show the message indicating the creation of Kubernetes resources
   - Look for messages like: "Creating Kubernetes resources..." and "Launching Kubernetes Job: metaflow-..."

2. **Screenshot 2**: MLflow UI showing the registered model
   - Access the MLflow UI at: http://mlflow-labseven.35.236.43.131.nip.io
   - Navigate to the Models section and show the registered "lab7yipee" model
   - Make sure the screenshot clearly shows the model name and version

## Implementation Details

The following files were implemented for this lab:

1. `lab7_mlflow_register.py`: 
   - Simple script that trains a Linear Regression model on the wine dataset
   - Registers the model with MLflow with name "lab7yipee"
   - Connects to the MLflow server running on GCP

2. `gcp_mlflow_simple.py`:
   - Metaflow pipeline that runs on GCP using Kubernetes
   - Trains multiple Lasso regression models with different alpha values
   - Selects the best model and registers it with MLflow

3. `trainingflow_final.py`:
   - More complex Metaflow pipeline that runs training in parallel
   - Uses Kubernetes, timeout, retry, and resource decorators
   - Trains, evaluates, and registers the model with MLflow

## Summary of Accomplishments
- Successfully deployed MLflow on Google Kubernetes Engine (GKE)
- Implemented machine learning training flows on GCP using Metaflow and Kubernetes
- Registered trained models with MLflow for model versioning and tracking
- Connected a GCP-hosted MLflow instance with training jobs running on GKE 