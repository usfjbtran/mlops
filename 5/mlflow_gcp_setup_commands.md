# MLflow on GCP Setup Commands

This document contains all commands used to set up MLflow on Google Cloud Platform using Cloud Run, as required for Lab 5.

## 1. Setting up GCP Environment

```bash
# Install gcloud CLI if not already installed
# Instructions: https://cloud.google.com/sdk/docs/install-sdk

# Initialize gcloud configuration
gcloud init

# Create a new project in GCP Console
# Go to https://console.cloud.google.com/
# Create a new project and connect it to a billing account
# Note down your PROJECT_ID

# Set your project ID in gcloud
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudsql.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable compute.googleapis.com
```

## 2. Setting up PostgreSQL Instance

```bash
# Create PostgreSQL instance (adjust parameters as needed)
gcloud sql instances create mlflow-postgres \
--database-version=POSTGRES_15 \
--region=us-west2 \
--tier=db-f1-micro \
--storage-type=HDD \
--storage-size=10GB \
--authorized-networks=0.0.0.0/0

# Create user for PostgreSQL
gcloud sql users create mlflow-user \
--instance=mlflow-postgres \
--password=YOUR_PASSWORD

# Create database for MLflow
gcloud sql databases create mlflow-db --instance=mlflow-postgres

# Get the IP address of your PostgreSQL instance
gcloud sql instances describe mlflow-postgres
# Note down the IP address
```

## 3. Creating a Storage Bucket

```bash
# Create a bucket for MLflow artifacts
gcloud storage buckets create gs://YOUR_PROJECT_ID-mlflow-artifacts

# Create mlruns folder in the bucket (using the console)
# Go to Cloud Storage > Buckets > YOUR_PROJECT_ID-mlflow-artifacts
# Click "Create Folder" and name it "mlruns"
```

## 4. Creating Artifact Registry Repository

```bash
# Create a Docker repository
gcloud artifacts repositories create mlflow-repo \
--location=us-west2 \
--repository-format=docker
```

## 5. Creating Service Account and Assigning Roles

```bash
# Create a service account for MLflow
gcloud iam service-accounts create mlflow-sa

# Get project ID
PROJECT_ID=$(gcloud config get-value project)

# Assign necessary roles to the service account
gcloud projects add-iam-policy-binding $PROJECT_ID --member='serviceAccount:mlflow-sa@'$PROJECT_ID'.iam.gserviceaccount.com' --role='roles/cloudsql.editor'
gcloud projects add-iam-policy-binding $PROJECT_ID --member='serviceAccount:mlflow-sa@'$PROJECT_ID'.iam.gserviceaccount.com' --role='roles/storage.objectAdmin'
gcloud projects add-iam-policy-binding $PROJECT_ID --member='serviceAccount:mlflow-sa@'$PROJECT_ID'.iam.gserviceaccount.com' --role='roles/secretmanager.secretAccessor'
gcloud projects add-iam-policy-binding $PROJECT_ID --member='serviceAccount:mlflow-sa@'$PROJECT_ID'.iam.gserviceaccount.com' --role='roles/artifactregistry.admin'
gcloud projects add-iam-policy-binding $PROJECT_ID --member='serviceAccount:mlflow-sa@'$PROJECT_ID'.iam.gserviceaccount.com' --role='roles/clouddeploy.serviceAgent'
gcloud projects add-iam-policy-binding $PROJECT_ID --member='serviceAccount:mlflow-sa@'$PROJECT_ID'.iam.gserviceaccount.com' --role='roles/cloudfunctions.admin'
```

## 6. Setting up Secrets

```bash
# Create a key file for service account
gcloud iam service-accounts keys create sa-private-key.json --iam-account=mlflow-sa@$PROJECT_ID.iam.gserviceaccount.com

# Create secrets for service account key
gcloud secrets create access_keys --data-file=sa-private-key.json

# Get PostgreSQL instance IP address
SQL_IP=$(gcloud sql instances describe mlflow-postgres --format='value(ipAddresses[0].ipAddress)')

# Create secret for database URL
gcloud secrets create database_url
echo -n "postgresql://mlflow-user:YOUR_PASSWORD@$SQL_IP/mlflow-db" | \
    gcloud secrets versions add database_url --data-file=-

# Create secret for storage bucket URL
gcloud secrets create bucket_url
echo -n "gs://$PROJECT_ID-mlflow-artifacts/mlruns" | \
    gcloud secrets versions add bucket_url --data-file=-
```

## 7. Creating Docker Image for MLflow Server

First, create the following files in a new directory:

**requirements.txt**:
```
setuptools
mlflow==2.15.1
psycopg2-binary==2.9.9
google-cloud-storage==2.18.2
```

**Dockerfile**:
```
FROM python:3.12-slim

WORKDIR /

COPY requirements.txt requirements.txt   
COPY server.sh server.sh

ENV GOOGLE_APPLICATION_CREDENTIALS='./secrets/credentials'

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

RUN chmod +x server.sh

ENTRYPOINT ["./server.sh"]
```

**server.sh**:
```bash
#!/bin/bash  

mlflow db upgrade $POSTGRESQL_URL
mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri $POSTGRESQL_URL \
  --artifacts-destination $STORAGE_URL
```

Then build and push the Docker image:

```bash
# Configure Docker to use Google Cloud's artifact registry
gcloud auth configure-docker us-west2-docker.pkg.dev

# Build and push the Docker image
docker build --platform linux/amd64 -t "us-west2-docker.pkg.dev/$PROJECT_ID/mlflow-repo/mlflow:v1" .
docker push "us-west2-docker.pkg.dev/$PROJECT_ID/mlflow-repo/mlflow:v1"

# Alternative method if Docker build has issues
gcloud builds submit --tag us-west2-docker.pkg.dev/$PROJECT_ID/mlflow-repo/mlflow:v1
```

## 8. Deploying to Cloud Run

```bash
# Deploy the MLflow server to Cloud Run
gcloud run deploy mlflow-server \
  --image=us-west2-docker.pkg.dev/$PROJECT_ID/mlflow-repo/mlflow:v1 \
  --platform=managed \
  --region=us-west2 \
  --allow-unauthenticated \
  --set-secrets=POSTGRESQL_URL=database_url:latest,STORAGE_URL=bucket_url:latest,GOOGLE_APPLICATION_CREDENTIALS=/secrets/credentials=access_keys:latest \
  --service-account=mlflow-sa@$PROJECT_ID.iam.gserviceaccount.com
```

After deployment, you'll get a URL for your MLflow server. Note it down for configuring your experiments.

## 9. Setting Up the Client to Use Remote MLflow Server

In your Jupyter notebook, modify the MLflow tracking URI to point to your Cloud Run instance:

```python
import mlflow
mlflow.set_tracking_uri('https://mlflow-server-YOUR-CLOUD-RUN-URL.a.run.app')
```

## 10. Running Lab 2 Experiments with Remote MLflow Server

Modify your existing Lab 2 Jupyter notebook to use the remote MLflow server:

1. Change `mlflow.set_tracking_uri('sqlite:///mlflow.db')` to `mlflow.set_tracking_uri('https://mlflow-server-YOUR-CLOUD-RUN-URL.a.run.app')`
2. Run the experiments as before - they will now be tracked in the GCP MLflow instance
3. Verify that experiments are appearing in the MLflow UI by visiting your Cloud Run URL in a browser
4. Register your final model in the MLflow UI or via the API as required

## GCP Console Steps
- Create a new project and link a billing account
- Navigate to Storage > Browser and create a folder "mlruns" in your artifact bucket after creating it
- You may need to enable APIs through the console if the CLI commands fail
- You can monitor your Cloud Run deployment and logs in the console
- You can check your MLflow UI by visiting the Cloud Run service URL 