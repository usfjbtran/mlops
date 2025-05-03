# MLflow Server Docker Files

This directory contains the necessary files to build a Docker image for the MLflow server to run on Google Cloud Run.

## Files

- `Dockerfile`: The Docker configuration file
- `requirements.txt`: Python dependencies for the MLflow server
- `server.sh`: Shell script that starts the MLflow server

## Building the Docker Image

1. Make sure you have Docker installed locally
2. Configure Docker to use Google Cloud's artifact registry:
   ```bash
   gcloud auth configure-docker us-west2-docker.pkg.dev
   ```
3. Build the Docker image:
   ```bash
   docker build --platform linux/amd64 -t "us-west2-docker.pkg.dev/YOUR_PROJECT_ID/mlflow-repo/mlflow:v1" .
   ```
4. Push the image to Google Container Registry:
   ```bash
   docker push "us-west2-docker.pkg.dev/YOUR_PROJECT_ID/mlflow-repo/mlflow:v1"
   ```

If you're having issues with Docker or want to use Google's build service:

```bash
gcloud builds submit --tag us-west2-docker.pkg.dev/YOUR_PROJECT_ID/mlflow-repo/mlflow:v1
```

## Deployment

The image is deployed to Cloud Run with the following environment variables:

- `POSTGRESQL_URL`: Secret reference to the PostgreSQL connection string
- `STORAGE_URL`: Secret reference to the GCS bucket URL for artifact storage
- `GOOGLE_APPLICATION_CREDENTIALS`: Secret reference to the service account key

Refer to the main instructions in `mlflow_gcp_setup_commands.md` for the complete deployment process. 