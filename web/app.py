import streamlit as st
import pandas as pd
import json
import time
import torch
import sys
import os

# Allow importing from src/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import FraudNet
from src.utils import load_config
from kafka import KafkaConsumer

# 1. Setup & Config
st.set_page_config(page_title="Fraud Guard AI", layout="wide")
st.title("🛡️ Real-Time Banking Fraud Detection")

cfg = load_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load Model (Cached to prevent reloading every second)
@st.cache_resource
def load_model():
    model = FraudNet(cfg['model']['input_dim'], cfg['model']['hidden_dim'], cfg['model']['output_dim'])
    try:
        model.load_state_dict(torch.load(f"{cfg['data']['save_path']}fraud_model.pth"))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model not found! Please run 'python main.py train' first.")
        return None

model = load_model()

# 3. Sidebar Stats
st.sidebar.header("System Status")
status_indicator = st.sidebar.empty()
counter_placeholder = st.sidebar.empty()
fraud_counter_placeholder = st.sidebar.empty()

# 4. Main Dashboard Layout
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Live Transaction Stream")
    chart_holder = st.empty()
with col2:
    st.subheader("Recent Alerts")
    alert_holder = st.empty()

# 5. Kafka Consumer Setup
consumer = KafkaConsumer(
    cfg['kafka']['topic_input'],
    bootstrap_servers=cfg['kafka']['bootstrap_servers'],
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# 6. Real-Time Loop
data_buffer = []
fraud_count = 0
total_count = 0

if st.button("Start Live Feed"):
    status_indicator.success("🟢 System Online")
    
    with torch.no_grad():
        for message in consumer:
            data = message.value
            features = torch.tensor(data['features'], dtype=torch.float32).to(device)
            
            # Inference
            prob = model(features.unsqueeze(0)).item()
            is_fraud = prob > 0.85
            
            # Update Stats
            total_count += 1
            if is_fraud:
                fraud_count += 1
                alert_holder.error(f"🚨 FRAUD DETECTED! ID: {data['id']} (Risk: {prob:.2f})")
            
            # Update Buffer for Chart
            data_buffer.append({"ID": data['id'], "Risk Score": prob, "Time": data['timestamp']})
            if len(data_buffer) > 50: data_buffer.pop(0) # Keep last 50 points
            
            # Refresh UI
            df = pd.DataFrame(data_buffer)
            chart_holder.line_chart(df.set_index("ID")["Risk Score"])
            
            counter_placeholder.metric("Total Transactions", total_count)
            fraud_counter_placeholder.metric("Fraud Detected", fraud_count, delta_color="inverse")
            
            # Little sleep to not freeze the browser
            time.sleep(0.1)