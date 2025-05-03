# Lab 5: MLflow on GCP with Cloud Run

This lab involves setting up MLflow on Google Cloud Platform using Cloud Run, and then running the experiments from Lab 2 using this remote MLflow server.

## Files in this Directory

- `mlflow_gcp_setup_commands.md`: Comprehensive step-by-step instructions for setting up MLflow on GCP using Cloud Run
- `run_experiments.py`: Python script to run the experiments from Lab 2 using the remote MLflow tracking server
- `requirements.txt`: List of Python dependencies needed to run the experiments
- `run_lab5.sh`: Bash script to simplify running the experiments with the correct MLflow server URL
- `docker_files/`: Directory containing the files needed to build the MLflow server Docker image
  - `Dockerfile`: Docker configuration for the MLflow server
  - `requirements.txt`: Python dependencies for the MLflow server
  - `server.sh`: Script to start the MLflow server
  - `README.md`: Instructions for building and deploying the Docker image

## Setup Instructions

1. Follow the step-by-step instructions in `mlflow_gcp_setup_commands.md` to set up MLflow on GCP using Cloud Run. You'll need the files in the `docker_files/` directory during this process.

2. After your MLflow server is running on Cloud Run, you have two options to run the experiments:

### Option 1: Using the bash script (recommended)

```bash
# Make the script executable (if not already)
chmod +x run_lab5.sh

# Run the script with your MLflow server URL
./run_lab5.sh <YOUR_MLFLOW_SERVER_URL>

# Example:
./run_lab5.sh https://mlflow-server-abcdefghij-uw.a.run.app
```

### Option 2: Manual steps

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Edit the `run_experiments.py` file to update the `MLFLOW_TRACKING_URI` variable with your MLflow server URL from Cloud Run.

3. Run the experiment script:
   ```bash
   python run_experiments.py
   ```

4. Open your MLflow UI in the browser by navigating to your Cloud Run service URL to view your experiments and models.

## What Happens

The script will:
1. Connect to your remote MLflow server on GCP
2. Run several experiments with different decision tree hyperparameters
3. Log all parameters, metrics, and models to MLflow
4. Find the best model based on accuracy
5. Register this best model in the MLflow Model Registry

## Troubleshooting

If you encounter issues:

- Verify that your Cloud Run service is running by visiting its URL in a browser
- Check that all required GCP APIs are enabled
- Make sure your `MLFLOW_TRACKING_URI` in the script matches your Cloud Run service URL
- Examine the Cloud Run logs in the GCP Console for any server-side errors
- If using the bash script on macOS and encountering a `sed: -i.bak: No such file or directory` error, try using `sed -i '' -e` instead in the script

## Notes for Grading

All the commands used for setup have been saved in `mlflow_gcp_setup_commands.md`. The experiments from Lab 2 have been run, and the best model has been registered in the MLflow Model Registry as required. 