from prefect import flow, task
from src.train import train_pipeline
from src.utils import load_config
import os

# 1. Define Tasks (Individual steps)
@task(name="Load Configuration")
def get_config():
    return load_config()

@task(name="Check Data Source")
def check_data(cfg):
    # Simulate checking if new data exists in Kafka or Database
    print(f"Checking data source: {cfg['kafka']['topic_input']}")
    return True

@task(name="Run PyTorch Training")
def run_training_task():
    # We call your existing training function
    train_pipeline()
    return "Model Saved"

# 2. Define the Flow (The Pipeline)
@flow(name="Daily Fraud Model Retraining", log_prints=True)
def retraining_flow():
    cfg = get_config()
    
    if check_data(cfg):
        status = run_training_task()
        print(f"Flow Result: {status}")
    else:
        print("No new data found. Skipping training.")

# 3. Run it directly
if __name__ == "__main__":
    retraining_flow()