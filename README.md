# AI-Powered-Anomaly-Detection-for-IoT-Networks
This project leverages machine learning (ML) to detect unusual patterns in IoT device behavior, signaling potential cybersecurity threats like DDoS attacks, malware infections, or unauthorized access.
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import socket
import threading
import json
import time

# Simulated IoT Network Traffic
def generate_fake_iot_data():
    """
    Generates simulated IoT network traffic data for testing.
    """
    np.random.seed(42)
    data = {
        "device_id": np.random.choice(["Device_A", "Device_B", "Device_C"], 1000),
        "packet_size": np.random.normal(500, 100, 1000),
        "connection_duration": np.random.normal(30, 10, 1000),
        "request_rate": np.random.normal(50, 10, 1000),
    }
    df = pd.DataFrame(data)
    return df

# Train Isolation Forest for anomaly detection
def train_anomaly_detector(data):
    """
    Trains an Isolation Forest model to detect anomalies in IoT traffic.
    """
    model = IsolationForest(contamination=0.05, random_state=42)
    features = data[["packet_size", "connection_duration", "request_rate"]]
    model.fit(features)
    return model

# Monitor IoT traffic and detect anomalies
def monitor_traffic(model):
    """
    Simulates real-time IoT traffic monitoring and detects anomalies.
    """
    while True:
        # Simulated incoming data
        new_data = {
            "device_id": np.random.choice(["Device_A", "Device_B", "Device_C"]),
            "packet_size": np.random.normal(500, 100),
            "connection_duration": np.random.normal(30, 10),
            "request_rate": np.random.normal(50, 10),
        }

        data_df = pd.DataFrame([new_data])
        prediction = model.predict(data_df[["packet_size", "connection_duration", "request_rate"]])
        if prediction[0] == -1:
            print(f"[ALERT] Anomaly Detected! Data: {new_data}")
        else:
            print(f"[INFO] Normal Activity: {new_data}")

        time.sleep(1)  # Simulate real-time delay

if __name__ == "__main__":
    # Step 1: Generate and train
    print("Generating IoT traffic data...")
    traffic_data = generate_fake_iot_data()
    print("Training anomaly detection model...")
    detector_model = train_anomaly_detector(traffic_data)

    # Step 2: Monitor IoT traffic
    print("Monitoring IoT traffic...")
    monitor_thread = threading.Thread(target=monitor_traffic, args=(detector_model,))
    monitor_thread.start()
