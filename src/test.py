import pickle
import logging
import yaml
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_data
from utils.encode_categories import categories_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_FILE = config["model_file"]
DATA_FILE = config["test_data"]
PREDICTIONS_FILE = config["predictions_file"]
FEATURES = config["features"]


def main():
    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
    data = load_data(DATA_FILE)
    test = data[FEATURES]
    test_data = categories_encoder(test)    
    
    predictions = model.predict(test_data)
    
    data['predicted_delivery_time'] = predictions


    data.to_csv(PREDICTIONS_FILE, index=False)
    logging.info(f"Predictions saved to {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
