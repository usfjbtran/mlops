FROM python:3.9-slim

# Install mlflow
RUN pip install mlflow

# Expose the port MLflow will run on
EXPOSE 5000

# Command to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"] 