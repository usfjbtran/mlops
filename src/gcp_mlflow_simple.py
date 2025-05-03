from metaflow import FlowSpec, step, current, kubernetes

class GCPMLFlowSimple(FlowSpec):
    @step
    def start(self):
        import sys
        import subprocess
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "scikit-learn", "pandas", "mlflow"])
        
        import numpy as np
        import pandas as pd
        import mlflow
        import mlflow.sklearn
        from sklearn.datasets import load_wine
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        
        tracking_uri = "http://mlflow-tracking.default.svc.cluster.local:5000"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        
        wine = load_wine()
        X = wine.data
        y = wine.target
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        alphas = [0.1, 0.01, 0.001, 0.0001]
        best_score = -float('inf')
        best_model = None
        best_alpha = None
        
        for alpha in alphas:
            model = Lasso(alpha=alpha)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            print(f"Alpha: {alpha}, Score: {score}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_alpha = alpha
        
        print(f"Best alpha: {best_alpha}, Best score: {best_score}")
        
        try:
            with mlflow.start_run(run_name="GCP-MLFlow-Simple") as run:
                mlflow.log_param("model", "Lasso")
                mlflow.log_param("alpha", best_alpha)
                mlflow.log_param("dataset", "Wine")
                
                mlflow.log_metric("score", best_score)
                
                mlflow.sklearn.log_model(best_model, "model")
                
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model_name = "wine-lasso-gcp"
                registered_model = mlflow.register_model(model_uri, registered_model_name)
                
                print(f"\nSuccess! Run ID: {run.info.run_id}")
                print(f"Model registered as: {registered_model_name}")
                print(f"View the run at: {tracking_uri}/#/experiments/0/runs/{run.info.run_id}")
                print(f"View the model at: {tracking_uri}/#/models/{registered_model_name}")
                print("\nNote: You need to set up port forwarding to access the MLflow UI:")
                print("kubectl port-forward service/mlflow-tracking 5000:5000")
                
                self.best_model = best_model
                self.best_score = best_score
                self.run_id = run.info.run_id
                
        except Exception as e:
            print(f"Error registering model: {e}")
            self.best_model = best_model
            self.best_score = best_score
            self.run_id = None
            
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow completed!")
        print(f"Best model: {self.best_model}")
        print(f"Score: {self.best_score}")
        
        if hasattr(self, 'run_id') and self.run_id:
            print(f"View registered model at: http://mlflow-tracking.default.svc.cluster.local:5000/#/experiments/0/runs/{self.run_id}")
        
if __name__ == "__main__":
    GCPMLFlowSimple() 