import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri(f"http://mlflow-labseven.35.236.43.131.nip.io")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
RUN_NAME = "MLflow-Lab7-Final"

with mlflow.start_run(run_name=RUN_NAME) as run:
    print(f"Started run: {run.info.run_id}")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"RMSE: {rmse}")
    
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("dataset", "Wine")
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
    
    model_uri = f"runs:/{run.info.run_id}/model"
    registered_model_name = "lab7yipee"
    registered_model = mlflow.register_model(model_uri, registered_model_name)
    
    print(f"\nSuccess! Run ID: {run.info.run_id}")
    print(f"Model registered as: {registered_model_name}")
    print(f"View the run at: http://mlflow-labseven.35.236.43.131.nip.io/#/experiments/0/runs/{run.info.run_id}")
    print(f"View the model at: http://mlflow-labseven.35.236.43.131.nip.io/#/models/{registered_model_name}")
    print("\n*** LAB 7 - MLflow on GCP completed successfully! ***") 