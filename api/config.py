"""
Configuration module for Solar Energy Prediction API
"""
import os
from dotenv import load_dotenv

load_dotenv()


# =========================
# FEATURE CONFIGURATION
# =========================
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


# =========================
# ENV VALIDATION (IMPORTANT FIX)
# =========================
def get_env(name: str, required: bool = True, default=None):
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        raise ValueError(f"❌ Missing required environment variable: {name}")
    return value.strip() if isinstance(value, str) else value


# =========================
# DATABASE CONFIG (ADD THIS FIX)
# =========================
AWS_HOST = get_env("AWS_HOST")
AWS_PORT = get_env("AWS_PORT", required=False, default="5432")
AWS_DB = get_env("AWS_DB")
AWS_USER = get_env("AWS_USER")
AWS_PASS = get_env("AWS_PASS")


# =========================
# MLFLOW / DAGSHUB
# =========================
DAGSHUB_TOKEN = get_env("DAGSHUB_TOKEN", required=False)


# =========================
# MODEL CONFIG
# =========================
KWH_MODEL_NAME = "final_model_kwh"
ERROR_MODEL_NAME = "final_model_error"
MODEL_ALIAS = "production"


# =========================
# API CONFIG
# =========================
API_TITLE = "Solar Energy Prediction API"
API_DESCRIPTION = "API for predicting solar energy (KWH and Error)"
API_VERSION = "2.2.0"
API_HOST = "0.0.0.0"
API_PORT = 8000


# =========================
# DATA COLUMNS
# =========================
DATE_COLUMN = "date"
KWH_COLUMN = "E-Today(KWH)"