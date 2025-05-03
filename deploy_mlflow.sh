#!/bin/bash

# Set project variables
PROJECT_ID="still-mesh-458705-t0"
REGION="us-west2"

# Create Kubernetes secrets
kubectl create secret generic database-url \
  --from-literal=POSTGRESQL_URL="postgresql://postgres:mlflow-password@/mlflow?host=/cloudsql/still-mesh-458705-t0:us-west2:mlflow-db"

kubectl create secret generic bucket-url \
  --from-literal=STORAGE_URL="gs://still-mesh-458705-t0-mlflow-artifacts"

kubectl create secret generic access-keys \
  --from-file=access-keys.json=mlflow-key.json

# Deploy MLflow
kubectl apply -f mlflow-deployment.yaml
kubectl apply -f mlflow-service.yaml

# Wait for the service to be ready
echo "Waiting for MLflow service to be ready..."
kubectl wait --for=condition=available deployment/mlflow-deployment --timeout=300s

# Get the external IP
echo "MLflow service is ready. Getting external IP..."
kubectl get service mlflow-service 