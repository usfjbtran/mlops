from metaflow import FlowSpec, step, Parameter
import os
import mlflow
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_adult_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class AdultIncomeTrainingFlow(FlowSpec):
    """
    A Metaflow workflow for training income prediction models on the Adult Census dataset.
    This flow implements a complete ML training pipeline including:
    1. Data preprocessing
    2. Parallel model training (Random Forest and Logistic Regression)
    3. Model selection based on F1 score
    4. Model registration with MLflow
    """
    
    data_path = Parameter('data_path', 
                          default=os.path.join(PROJECT_ROOT, 'data/adult.data'),
                          help='Path to the input data file')
    
    test_size = Parameter('test_size',
                          default=0.2,
                          help='Proportion of data to use for testing')
    
    random_state = Parameter('random_state',
                             default=42,
                             help='Random seed for reproducibility')
    
    rf_n_estimators = Parameter('rf_n_estimators',
                               default=100,
                               help='Number of trees in the random forest')
    
    rf_max_depth = Parameter('rf_max_depth',
                            default=None,
                            help='Max depth of trees in the random forest')
    
    @step
    def start(self):
        """
        The starting step of the workflow.
        This step loads and preprocesses the data, making it ready for model training.
        """
        print(f"Loading data from {self.data_path}")
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names, self.encoders, self.scaler = \
            preprocess_adult_data(self.data_path, test_size=self.test_size, random_state=self.random_state)
            
        print(f"Data preprocessed: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
        self.next(self.train_random_forest, self.train_logistic_regression)
    
    @step
    def train_random_forest(self):
        """
        Train a Random Forest classifier model.
        This step trains and evaluates a Random Forest model on the prepared data.
        """
        print("Training Random Forest classifier...")
        
        self.model = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=self.random_state
        )
        
        self.model.fit(self.X_train, self.y_train)        
        y_pred = self.model.predict(self.X_test)
        
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.f1 = f1_score(self.y_test, y_pred)        
        self.model_type = 'random_forest'
        print(f"Random Forest - Accuracy: {self.accuracy:.4f}, F1: {self.f1:.4f}")
        
        self.next(self.choose_model)
    
    @step
    def train_logistic_regression(self):
        """
        Train a Logistic Regression model
        """
        print("Training Logistic Regression classifier...")
        
        self.model = LogisticRegression(
            max_iter=1000,  # Increased max_iter to ensure convergence
            random_state=self.random_state
        )
        
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)        
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.f1 = f1_score(self.y_test, y_pred)
        
        self.model_type = 'logistic_regression'
        print(f"Logistic Regression - Accuracy: {self.accuracy:.4f}, F1: {self.f1:.4f}")
        
        self.next(self.choose_model)
    
    @step
    def choose_model(self, inputs):
        """
        Compare trained models and select the best performing one.
        This is a join step that merges the parallel training branches and selects
        the best model based on F1 score.
        """
        print("Choosing the best model based on F1 score...")
        self.recall = None
        self.precision = None 
        self.model_type = None
        self.model = None
        self.f1 = None
        self.accuracy = None
        
        self.merge_artifacts(inputs)
        
        models = [
            {
                'type': inp.model_type,
                'model': inp.model,
                'accuracy': inp.accuracy,
                'precision': inp.precision,
                'recall': inp.recall,
                'f1': inp.f1
            }
            for inp in inputs
        ]
        
        self.models = sorted(models, key=lambda x: x['f1'], reverse=True)        
        self.best_model = self.models[0]
        
        print(f"Selected model: {self.best_model['type']} with F1: {self.best_model['f1']:.4f}")
        
        self.next(self.register_model)
    
    @step
    def register_model(self):
        """
        Register the best performing model with MLflow.
        This step saves the model and its metadata to the MLflow model registry
        and stores preprocessing artifacts for later use in inference.
        """
        mlflow_dir = os.path.join(PROJECT_ROOT, 'mlflow_test')
        mlflow.set_tracking_uri(f'file://{mlflow_dir}')
        mlflow.set_experiment('adult-income-prediction')
        
        with mlflow.start_run(run_name=f"metaflow_{self.best_model['type']}") as run:
            if self.best_model['type'] == 'random_forest':
                mlflow.log_param('model_type', 'RandomForestClassifier')
                mlflow.log_param('n_estimators', self.rf_n_estimators)
                mlflow.log_param('max_depth', self.rf_max_depth)
            else:
                mlflow.log_param('model_type', 'LogisticRegression')
                mlflow.log_param('max_iter', 1000)
                
            mlflow.log_param('random_state', self.random_state)
            
            # Log performance metrics
            mlflow.log_metric('accuracy', self.best_model['accuracy'])
            mlflow.log_metric('precision', self.best_model['precision'])
            mlflow.log_metric('recall', self.best_model['recall'])
            mlflow.log_metric('f1', self.best_model['f1'])
            
            # Log the model to MLflow model registry
            model_info = mlflow.sklearn.log_model(
                self.best_model['model'],
                artifact_path="model",
                registered_model_name="adult-income-classifier"
            )
            
            self.model_uri = model_info.model_uri
            self.run_id = run.info.run_id

            artifacts_path = os.path.join(PROJECT_ROOT, f"artifacts/{self.run_id}")
            os.makedirs(artifacts_path, exist_ok=True)
            
            joblib.dump(self.encoders, f"{artifacts_path}/encoders.joblib")
            joblib.dump(self.scaler, f"{artifacts_path}/scaler.joblib")
            joblib.dump(self.feature_names, f"{artifacts_path}/feature_names.joblib")
            
            print(f"Model registered with MLflow - Run ID: {self.run_id}")
            print(f"Model URI: {self.model_uri}")
            print(f"Preprocessing artifacts saved to: {artifacts_path}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        The final step of the workflow.
        This step summarizes the training results and completes the flow.
        """
        print("Training flow completed!")
        print(f"Best model: {self.best_model['type']} with F1 score: {self.best_model['f1']:.4f}")
        print(f"Model registered with MLflow - Run ID: {self.run_id}")
        print(f"Model URI: {self.model_uri}")

if __name__ == '__main__':
    AdultIncomeTrainingFlow() 