import torch
import json
from kafka import KafkaConsumer
from .model import FraudNet
from .utils import load_config

def run_inference():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model Structure & Weights
    model = FraudNet(cfg['model']['input_dim'], cfg['model']['hidden_dim'], cfg['model']['output_dim']).to(device)
    try:
        model.load_state_dict(torch.load(f"{cfg['data']['save_path']}fraud_model.pth"))
        model.eval()
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("❌ Model not found! Run training first.")
        return

    # Setup Consumer
    consumer = KafkaConsumer(
        cfg['kafka']['topic_input'],
        bootstrap_servers=cfg['kafka']['bootstrap_servers'],
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print(f"📡 Listening on topic: {cfg['kafka']['topic_input']}...")
    
    with torch.no_grad():
        for message in consumer:
            data = message.value
            features = torch.tensor(data['features'], dtype=torch.float32).to(device)
            
            # Predict
            prob = model(features.unsqueeze(0)).item()
            
            status = "🚨 FRAUD" if prob > 0.85 else "✅ LEGIT"
            print(f"ID: {data['id']} | Risk: {prob:.4f} | {status}")