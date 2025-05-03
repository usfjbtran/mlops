from metaflow import FlowSpec, step, kubernetes, timeout, retry, resources

class ClassifierTrainFlowGCP(FlowSpec):
    
    @step
    def start(self):
        from sklearn.datasets import load_wine
        import numpy as np

        dataset = load_wine()
        self.X = dataset.data
        self.y = dataset.target
        
        self.alphas = np.linspace(0.001, 1.0, 5)
        
        self.next(self.train_lasso, foreach='alphas')

    @kubernetes
    @timeout(minutes=10)
    @retry(times=2)
    @resources(cpu=2, memory=1000)
    @step
    def train_lasso(self):
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import Lasso
        
        self.alpha = self.input
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        model = Lasso(alpha=self.alpha)
        model.fit(X_train, y_train)
        
        self.score = model.score(X_test, y_test)
        self.model = model
        
        print(f"Trained Lasso model with alpha={self.alpha}, Score: {self.score}")
        
        self.next(self.choose_model)
    
    @step
    def choose_model(self, inputs):
        import numpy as np
        
        self.scores = {inp.alpha: inp.score for inp in inputs}
        
        self.best_alpha = max(self.scores.items(), key=lambda x: x[1])[0]
        self.best_score = self.scores[self.best_alpha]
        self.best_model = [inp.model for inp in inputs if inp.alpha == self.best_alpha][0]
        
        print(f"Best alpha: {self.best_alpha}, Best score: {self.best_score}")
        
        self.next(self.register_model)
    
    @step
    def register_model(self):
        import mlflow
        import mlflow.sklearn
        
        mlflow.set_tracking_uri("http://localhost:5000")
        
        with mlflow.start_run(run_name="GCP-Training-Flow") as run:
            mlflow.log_param("model", "Lasso")
            mlflow.log_param("alpha", self.best_alpha)
            mlflow.log_metric("score", self.best_score)
            
            mlflow.sklearn.log_model(self.best_model, "model")
            
            self.run_id = run.info.run_id
            self.experiment_id = run.info.experiment_id
            
            print(f"Model registered with MLflow, run_id: {self.run_id}")
            print(f"View at: http://localhost:5000/#/experiments/{self.experiment_id}/runs/{self.run_id}")
        
        self.next(self.end)
    
    @step
    def end(self):
        print("Training flow completed!")
        print(f"Best model alpha: {self.best_alpha}")
        print(f"Best model score: {self.best_score}")
        
if __name__ == "__main__":
    ClassifierTrainFlowGCP() 