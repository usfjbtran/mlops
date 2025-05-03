# MLOps Lab 7 - Training and Registering Models on GCP with MLflow

This directory contains the code for training machine learning models on GCP and registering them with MLflow.

## Files Description

- `lab7_mlflow_register.py`: Trains a Linear Regression model on the wine dataset and registers it with MLflow.
- `gcp_mlflow_simple.py`: Metaflow pipeline to train Lasso models on GCP using Kubernetes and register with MLflow.
- `trainingflow_final.py`: Parallel training pipeline for Lasso models on GCP with MLflow registration.

## Screenshots

Screenshots showing the successful execution of the lab are available in the [screenshots](./screenshots) directory:

1. [Kubernetes Job Creation](./screenshots/Kubernetes%20Job%20Creation.png) - Shows the MLFlowGCP job running on GCP with Kubernetes.
2. [MLflow Registered Model](./screenshots/MLflow%20Registered%20Model.png) - Shows the successfully registered "lab7yipee" model in MLflow.

## How to Run

### Running on GCP with Kubernetes

```bash
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
kubectl apply -f mlflow-simple.yaml
kubectl port-forward service/mlflow-tracking 5000:5000
python gcp_mlflow_simple.py run --with kubernetes
```

### Running the Simple Registration Script

```bash
python lab7_mlflow_register.py
```

## MLflow Access

- http://mlflow-labseven.35.236.43.131.nip.io
- http://localhost:5000 (with port forwarding)

## Lab Requirements

1. Training an ML model on GCP using Kubernetes
2. Registering the trained model with MLflow 