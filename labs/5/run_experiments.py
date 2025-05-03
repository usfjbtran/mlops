#!/usr/bin/env python3
"""
This script runs the experiments from Lab 2 using the remote MLflow tracking server on GCP.
It loads the wine dataset from sklearn and trains multiple decision tree classifiers
with different hyperparameters, logging the results to MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import sys

# Change this to your MLflow server URL from Cloud Run
MLFLOW_TRACKING_URI = "https://mlflow-server-193277309796.us-west2.run.app"

def load_data():
    """Load and prepare the wine dataset from sklearn"""
    wine = load_wine()
    df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    X = df_wine
    y = wine.target
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, wine.feature_names

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def run_experiment(X_train, X_test, y_train, y_test, feature_names, max_depth, min_samples_split, 
                   criterion, experiment_name="wine-classification"):
    """Run an experiment with specified parameters and log to MLflow"""
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Start a new run
    with mlflow.start_run():
        # Train the model
        dt_clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=42
        )
        dt_clf.fit(X_train, y_train)
        
        # Evaluate the model
        metrics = evaluate_model(dt_clf, X_test, y_test)
        
        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("criterion", criterion)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Calculate feature importances
        feature_importances = dt_clf.feature_importances_
        
        # Create feature importance dictionary
        importance_dict = {feature: importance for feature, importance 
                           in zip(feature_names, feature_importances)}
        
        # Log feature importances as parameters
        for feature, importance in importance_dict.items():
            mlflow.log_param(f"importance_{feature.replace(' ', '_')}", importance)
        
        # Log the model
        mlflow.sklearn.log_model(dt_clf, "model")
        
        # Return metrics for comparison
        return metrics, dt_clf

def main():
    """Main function to run experiments"""
    # Set the MLflow tracking URI
    print(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load the data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Define parameter combinations for experiments
    experiment_params = [
        # max_depth, min_samples_split, criterion
        (None, 2, "gini"),
        (3, 2, "gini"),
        (5, 2, "gini"),
        (10, 2, "gini"),
        (None, 5, "gini"),
        (None, 10, "gini"),
        (None, 2, "entropy"),
        (5, 5, "entropy"),
        (10, 10, "entropy")
    ]
    
    # Run all experiments
    print("Running experiments...")
    best_accuracy = 0
    best_model = None
    best_params = None
    
    for max_depth, min_samples_split, criterion in experiment_params:
        print(f"Running experiment with max_depth={max_depth}, min_samples_split={min_samples_split}, criterion={criterion}")
        metrics, model = run_experiment(
            X_train, X_test, y_train, y_test, feature_names,
            max_depth, min_samples_split, criterion
        )
        
        # Track the best model
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_model = model
            best_params = {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "criterion": criterion
            }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("------------------------------")
    
    # Register the best model
    print(f"Best model had accuracy: {best_accuracy:.4f} with parameters: {best_params}")
    print("Registering the best model...")
    
    mlflow.set_experiment("wine-classification-production")
    with mlflow.start_run():
        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics for the best model
        metrics = evaluate_model(best_model, X_test, y_test)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log the model
        mlflow.sklearn.log_model(best_model, "model",
                                registered_model_name="WineClassifierProduction")
    
    print("Best model registered as 'WineClassifierProduction'")
    print("Experiments completed successfully!")

if __name__ == "__main__":
    main() 