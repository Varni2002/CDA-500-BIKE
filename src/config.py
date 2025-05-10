import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define project directory structure
PARENT_DIR = Path(_file_).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Hopsworks configuration
# -----------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "CDA500FINAL")

# Feature group for lag features
FEATURE_GROUP_NAME = "citibike_daily_lagged"
FEATURE_GROUP_VERSION = 1

# Feature group to store predictions
FEATURE_GROUP_PREDICTION_NAME = "citibike_predictions"
FEATURE_GROUP_PREDICTION_VERSION = 1

# Model registry info
MODEL_NAME = "citibike_predictor"
MODEL_VERSION = 1

# MLflow experiment name
MLFLOW_EXPERIMENT_NAME = "citi-bike-trip-prediction"