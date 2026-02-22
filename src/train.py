import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
import os
from .model import FraudNet
from .dataset import FraudDataset
from .utils import load_config

def train_pipeline():
    # 1. Load Config
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Training on {device}...")

    # 2. Generate & Prep Data
    X, y = make_classification(n_samples=cfg['data']['num_samples'], 
                               n_features=cfg['data']['num_features'], 
                               n_classes=2, weights=[0.95, 0.05])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg['data']['test_size'])
    
    train_loader = DataLoader(FraudDataset(X_train, y_train), batch_size=cfg['model']['batch_size'], shuffle=True)

    # 3. Setup Model & MLflow
    mlflow.set_experiment(cfg['project']['experiment_name'])
    
    with mlflow.start_run():
        model = FraudNet(cfg['model']['input_dim'], cfg['model']['hidden_dim'], cfg['model']['output_dim']).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg['model']['learning_rate'])

        # Log Params
        mlflow.log_params(cfg['model'])

        # 4. Training Loop
        for epoch in range(cfg['model']['epochs']):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
                mlflow.log_metric("loss", loss.item(), step=epoch)

        # 5. Save Artifacts
        if not os.path.exists(cfg['data']['save_path']):
            os.makedirs(cfg['data']['save_path'])
            
        model_path = f"{cfg['data']['save_path']}fraud_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")
        print("✅ Training Complete. Model Saved.")