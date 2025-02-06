import logging
import pandas as pd
import yaml
import json
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import numpy as np
# Add the parent directory of utils to the Python path
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath("/Users/au713932/Documents/mlportfolio/food-delivery-ml"))
from utils.data_loader import load_data
from utils.encode_categories import categories_encoder
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
DATA_FILE = config["train_data"]
MODEL_FILE = config["model_file"]
SCORES_FILE = config["scores_file"]
TEST_SIZE = config["test_size"]
RANDOM_STATE = config["random_state"]
FEATURES = config["features"]
TARGET = config["target"]
MODEL_NAME = config["model"]["name"]
MODEL_PARAMS = config["model"]["params"]


def main():
    logging.info("Loading data...")
    data = load_data(DATA_FILE)

    X = data[FEATURES]
    X = categories_encoder(X)
    y = data[TARGET]
    

    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logging.info("Training model...")
    if MODEL_NAME == "ridge":
        MODEL = Ridge(**MODEL_PARAMS)
    else:
        raise ValueError(f"{MODEL_NAME} is not supported.")
    
    pipeline = Pipeline([
        ("regression", MODEL)
    ])
    pipeline.fit(X_train, y_train)
    
    logging.info("Testing model...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    mae = mean_absolute_error(y_test, y_pred)  
    r2 = r2_score(y_test, y_pred) 


    logging.info("Calculating scores...")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"R-squared: {r2:.4f}")

    # Save scores to a JSON file
    scores = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=4)

    logging.info("Saving model...")
    

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(pipeline, file)
    logging.info("Model saved successfully.")
    logging.info(f"Model results saved to {SCORES_FILE}")

if __name__ == "__main__":
    main()