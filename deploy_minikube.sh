#!/bin/bash

# Build the Docker image
docker build -t reddit-app:local -f reddit.Dockerfile .

# Load the image into Minikube
minikube image load reddit-app:local

# Create local deployment file
cat > reddit-minikube-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reddit-app
  labels:
    app: reddit
spec:
  replicas: 1
  selector:
    matchLabels:
      app: reddit
  template:
    metadata:
      labels:
        app: reddit
    spec:
      containers:
      - name: reddit-app
        image: reddit-app:local
        imagePullPolicy: Never
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: reddit-service
spec:
  selector:
    app: reddit
  ports:
  - port: 8000
    targetPort: 8000
  type: NodePort
EOF

# Deploy to Minikube
kubectl apply -f reddit-minikube-deployment.yaml

# Wait for deployment to be ready
echo "Waiting for Reddit app deployment to be ready..."
kubectl wait --for=condition=available deployment/reddit-app --timeout=300s

# Get the service URL
echo "Reddit app is ready. Getting service URL..."
minikube service reddit-service --url 