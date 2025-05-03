#!/bin/bash

# Set project variables
PROJECT_ID="still-mesh-458705-t0"
REGION="us-west2"
REPOSITORY="mlflow-repo"
IMAGE="mlflow:v1"

# Build and push the Docker image
gcloud builds submit --tag us-west2-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE

# Deploy to Cloud Run
gcloud run deploy mlflow-server \
  --image=us-west2-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE \
  --platform=managed \
  --region=$REGION \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300s \
  --allow-unauthenticated \
  --set-secrets POSTGRESQL_URL=database_url:latest,STORAGE_URL=bucket_url:latest \
  --set-secrets GOOGLE_APPLICATION_CREDENTIALS=access_keys:latest \
  --service-account=mlflow-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --vpc-connector=projects/$PROJECT_ID/locations/$REGION/connectors/mlflow-connector 