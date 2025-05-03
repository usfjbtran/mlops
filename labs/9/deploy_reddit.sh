#!/bin/bash

# Set project variables
PROJECT_ID="still-mesh-458705-t0"
REGION="us-west2"
IMAGE_NAME="reddit-app"
TAG="v1"

# Build and push the Docker image
docker build -t $IMAGE_NAME:$TAG -f reddit.Dockerfile .
docker tag $IMAGE_NAME:$TAG us-west2-docker.pkg.dev/$PROJECT_ID/reddit-repo/$IMAGE_NAME:$TAG
docker push us-west2-docker.pkg.dev/$PROJECT_ID/reddit-repo/$IMAGE_NAME:$TAG

# Update the deployment file with the correct image
sed -i '' "s|image: reddit-app:latest|image: us-west2-docker.pkg.dev/$PROJECT_ID/reddit-repo/$IMAGE_NAME:$TAG|" reddit-deployment.yaml

# Deploy to Kubernetes
kubectl apply -f reddit-deployment.yaml
kubectl apply -f reddit-service.yaml

# Wait for the deployment to be ready
echo "Waiting for Reddit app deployment to be ready..."
kubectl wait --for=condition=available deployment/reddit-app --timeout=300s

# Get the external IP
echo "Reddit app is ready. Getting external IP..."
kubectl get service reddit-service 