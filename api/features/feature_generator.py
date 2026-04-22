"""
Feature engineering module - handles feature calculation from raw data
"""
import pandas as pd
import numpy as np
from config import KWH_COLUMN


def calculate_features_for_prediction(
    df: pd.DataFrame,
    current_date: pd.Timestamp
) -> dict:
    """
    Calculate all required features for a single prediction
    
    Args:
        df: DataFrame containing historical KWH data with date column
        current_date: Current date for which to calculate features
    
    Returns:
        dict: Dictionary with all required features
    """
    try:
        # Time-based features
        month = current_date.month
        month_cos = np.cos(2 * np.pi * month / 12)

        # Lag features
        lag1 = df[KWH_COLUMN].iloc[-1]
        lag2 = df[KWH_COLUMN].iloc[-2]

        # Rolling features
        roll_mean_3 = df[KWH_COLUMN].iloc[-3:].mean()
        roll_mean_7 = df[KWH_COLUMN].iloc[-7:].mean()
        roll_max_7 = df[KWH_COLUMN].iloc[-7:].max()
        roll_min_7 = df[KWH_COLUMN].iloc[-7:].min()

        # Trend features
        diff1 = lag1 - df[KWH_COLUMN].iloc[-2]
        diff7 = lag1 - df[KWH_COLUMN].iloc[-7]

        # Interaction features
        lag1_roll7 = lag1 * roll_mean_7

        return {
            "month_cos": month_cos,
            "lag1": lag1,
            "lag2": lag2,
            "roll_mean_3": roll_mean_3,
            "roll_max_7": roll_max_7,
            "roll_min_7": roll_min_7,
            "diff1": diff1,
            "diff7": diff7,
            "lag1_roll7": lag1_roll7
        }

    except Exception as e:
        print(f"❌ Feature calculation error: {e}")
        raise


def prepare_features_for_model(features: dict) -> np.ndarray:
    """
    Convert feature dictionary to numpy array in correct order for model
    
    Args:
        features: Dictionary of features
    
    Returns:
        np.ndarray: Properly ordered feature array
    """
    feature_order = [
        "month_cos", "lag1", "lag2",
        "roll_mean_3", "roll_max_7", "roll_min_7",
        "diff1", "diff7", "lag1_roll7"
    ]
    
    return np.array([[features[key] for key in feature_order]])


def validate_data(df: pd.DataFrame, date_column: str = 'date', kwh_column: str = KWH_COLUMN) -> bool:
    """
    Validate that DataFrame has required columns and sufficient data
    
    Args:
        df: DataFrame to validate
        date_column: Name of date column
        kwh_column: Name of KWH column
    
    Returns:
        bool: True if valid, raises exception otherwise
    """
    if date_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{date_column}' column")
    
    if kwh_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{kwh_column}' column")
    
    if len(df) < 7:
        raise ValueError(f"DataFrame must have at least 7 rows, found {len(df)}")
    
    return True
