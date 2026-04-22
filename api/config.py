"""
Configuration module for Solar Energy Prediction API
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Feature configuration
REQUIRED_FEATURES = [
    "month_cos",
    "lag1",
    "lag2",
    "roll_mean_3",
    "roll_max_7",
    "roll_min_7",
    "diff1",
    "diff7",
    "lag1_roll7"
]

# MLflow configuration
# MLFLOW_TRACKING_URI = os.getenv("set_tracking_uri", "")
# MLFLOW_REPO_OWNER = os.getenv("repo_owner", "")
# MLFLOW_REPO_NAME = os.getenv("repo_name", "")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
# Model configuration
KWH_MODEL_NAME = "final_model_kwh"
ERROR_MODEL_NAME = "final_model_error"
MODEL_ALIAS = "production"

# API configuration
API_TITLE = "Solar Energy Prediction API"
API_DESCRIPTION = "API for predicting solar energy (KWH and Error) with forecasting capability"
API_VERSION = "2.2.0"
API_HOST = "0.0.0.0"
API_PORT = 8000

# Data column names
DATE_COLUMN = "date"
KWH_COLUMN = "E-Today(KWH)"
