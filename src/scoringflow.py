from metaflow import FlowSpec, step, Parameter, Flow
import os
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import transform_new_data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class AdultIncomeScoringFlow(FlowSpec):
    """
    A Metaflow workflow for scoring new data using a trained model from AdultIncomeTrainingFlow.
    This flow loads a trained model and applies it to a test dataset.
    """
    
    data_path = Parameter('data_path', 
                          default=os.path.join(PROJECT_ROOT, 'data/adult.test'),
                          help='path to the test data file')
    
    run_id = Parameter('run_id',
                      default=None,
                      help='run id of the training flow to use(default: latest)')
    
    @step
    def start(self):
        """
        The starting step of the workflow.
        This step loads the trained model and preprocessing artifacts from MLflow.
        """
        print("starting scoring flow...")
        if self.run_id is None:
            print("no run id specified")
            train_run = Flow('AdultIncomeTrainingFlow').latest_run
            self.train_run_id = train_run.id
        else:
            print(f"Using specified run_id: {self.run_id}")
            self.train_run_id = self.run_id
        
        print(f"loading model from training flow run: {self.train_run_id}")
        train_flow_end_step = Flow('AdultIncomeTrainingFlow')[self.train_run_id]['end']
        mlflow_run_id = train_flow_end_step.task.data.run_id
        mlflow_dir = os.path.join(PROJECT_ROOT, 'mlflow_test')
        mlflow.set_tracking_uri(f'file://{mlflow_dir}')
        print(f"loading model from MLflow run: {mlflow_run_id}")
        self.model = mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/model")        
        artifacts_path = os.path.join(PROJECT_ROOT, f"artifacts/{mlflow_run_id}")
        self.encoders = joblib.load(f"{artifacts_path}/encoders.joblib")
        self.scaler = joblib.load(f"{artifacts_path}/scaler.joblib")
        self.feature_names = joblib.load(f"{artifacts_path}/feature_names.joblib")
        
        print("model and preprocessing artifacts loaded successfully!!!")
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """
        Load and preprocess the test data.
        This step loads the test data and performs initial cleaning.
        """
        print(f"Loading test data from {self.data_path}")        
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race',
            'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
            'native_country', 'income'
        ]
        df = pd.read_csv(self.data_path, names=column_names, skipinitialspace=True, skiprows=1, header=None)
        df = df.replace(' ?', np.nan) 
        df = df.dropna()          
        df['income'] = df['income'].str.rstrip('.')
        df['income'] = df['income'].str.strip()
        
        self.X_test = df.drop('income', axis=1)
        self.y_test = df['income'].apply(lambda x: 1 if x.endswith('>50K') else 0)
        
        print(f"test set class distribution: {np.bincount(self.y_test)}")
        print(f"Test data loaded: {self.X_test.shape[0]} samples")
        self.next(self.score_data)
    
    @step
    def score_data(self):
        """
        Apply the model to the test data and calculate metrics.
        This step transforms the test data, makes predictions, and evaluates model performance.
        """
        print("Transforming test data...")
        
        self.X_transformed = transform_new_data(self.X_test, self.encoders, self.scaler, self.feature_names)
        
        print("making predictions...")
        
        self.y_pred = self.model.predict(self.X_transformed)
        self.y_prob = self.model.predict_proba(self.X_transformed)[:,1] if hasattr(self.model, "predict_proba") else None                
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred)        
        self.report = classification_report(self.y_test, self.y_pred, output_dict=True)        
        print(f"Metrics on test data:")
        print(f"  Accuracy:  {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1 Score:  {self.f1:.4f}")        
        results = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.y_pred
        })        
        if self.y_prob is not None:
            results['probability'] = self.y_prob
        
        results_path = os.path.join(PROJECT_ROOT, f"results/model_predictions_{self.train_run_id}.csv")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results.to_csv(results_path, index=False)
        
        self.results_path = results_path
        print(f"Results saved to: {self.results_path}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        The final step of the workflow.
        This step prints a summary of the scoring results.
        """
        print("scoring flow completed!")
        print(f"model from training run {self.train_run_id} applied to {self.X_test.shape[0]} test samples")
        print(f"results saved to: {self.results_path}")
        print(f"performance metrics:")
        print(f"  Accuracy:  {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1 Score:  {self.f1:.4f}")

if __name__ == '__main__':
    AdultIncomeScoringFlow() 