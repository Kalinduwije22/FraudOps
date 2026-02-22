import time
import json
import random
import numpy as np
from kafka import KafkaProducer
from .utils import load_config

def start_stream():
    cfg = load_config()
    producer = KafkaProducer(
        bootstrap_servers=cfg['kafka']['bootstrap_servers'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    
    print("Generating transactions...")
    trans_id = 1
    
    try:
        while True:
            # Generate fake features
            feats = np.random.randn(cfg['data']['num_features']).tolist()
            
            # Simulate fraud sometimes
            if random.random() < 0.1: 
                feats = [x * 4 for x in feats]

            payload = {'id': trans_id, 'features': feats, 'timestamp': time.time()}
            producer.send(cfg['kafka']['topic_input'], value=payload)
            print(f"Sent ID: {trans_id}")
            trans_id += 1
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        producer.close()