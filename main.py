import argparse
import os  # <--- This was missing!
from src.train import train_pipeline
from src.inference import run_inference
from src.producer import start_stream

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline Orchestrator")
    parser.add_argument('mode', choices=['train', 'stream', 'predict', 'web'], help="Choose action: train model, stream data, predict, or launch web UI")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_pipeline()
    elif args.mode == 'stream':
        start_stream()
    elif args.mode == 'predict':
        run_inference()
    elif args.mode == 'web':
        print("Launching Web Dashboard...")
        # Now this will work because 'os' is imported
        os.system("streamlit run web/app.py")

if __name__ == "__main__":
    main()