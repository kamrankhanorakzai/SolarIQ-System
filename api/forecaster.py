"""
Forecasting module - handles multi-step ahead forecasting
Production-safe version (no None crashes, no side effects)
"""

import pandas as pd
from typing import Tuple, List

from api.feature_generator import (
    calculate_features_for_prediction,
    prepare_features_for_model,
    validate_data
)

from api.config import DATE_COLUMN, KWH_COLUMN


# ==============================
# MAIN FORECAST FUNCTION
# ==============================
def forecast(
    model,
    df: pd.DataFrame,
    days: int = 7,
    data_column: str = KWH_COLUMN,
    date_column: str = DATE_COLUMN
) -> Tuple[pd.DataFrame, bool]:

    try:
        # ------------------------------
        # Safety checks
        # ------------------------------
        if df is None or df.empty:
            return pd.DataFrame(), False

        df = df.copy()  # prevent mutation bugs

        # ------------------------------
        # Validate data
        # ------------------------------
        validate_data(df, date_column=date_column, kwh_column=data_column)

        # ------------------------------
        # Prepare dates
        # ------------------------------
        df[date_column] = pd.to_datetime(df[date_column])
        last_date = df[date_column].max()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )

        # ------------------------------
        # Recursive forecasting setup
        # ------------------------------
        df_full = df.copy()
        predictions = []

        # ------------------------------
        # Forecast loop
        # ------------------------------
        for current_date in future_dates:

            # Feature engineering for current step
            features = calculate_features_for_prediction(df_full, current_date)
            X_future = prepare_features_for_model(features)

            # Prediction
            pred = model.predict(X_future)[0]
            pred = float(pred)

            predictions.append(pred)

            # Append prediction for next step
            df_full = pd.concat([
                df_full,
                pd.DataFrame({
                    date_column: [current_date],
                    data_column: [pred]
                })
            ], ignore_index=True)

        # ------------------------------
        # Final output
        # ------------------------------
        result_df = pd.DataFrame({
            date_column: future_dates,
            "prediction": predictions
        })

        print(f"✅ Forecast completed for {days} days")

        return result_df, True

    except Exception as e:
        print(f"❌ Forecast error: {e}")
        return pd.DataFrame(), False


# ==============================
# CSV FORECAST WRAPPER
# ==============================
def forecast_from_csv(
    model,
    csv_path: str,
    days: int = 7,
    data_column: str = KWH_COLUMN,
    date_column: str = DATE_COLUMN
) -> Tuple[pd.DataFrame, bool]:

    try:
        df = pd.read_csv(csv_path)

        if df is None or df.empty:
            return pd.DataFrame(), False

        return forecast(model, df, days, data_column, date_column)

    except Exception as e:
        print(f"❌ CSV forecast error: {e}")
        return pd.DataFrame(), False


# ==============================
# RESPONSE FORMATTER
# ==============================
def format_forecast_response(forecast_df: pd.DataFrame) -> List[dict]:

    if forecast_df is None or forecast_df.empty:
        return []

    return [
        {
            "date": pd.to_datetime(row[DATE_COLUMN]).isoformat(),
            "prediction": round(float(row["prediction"]), 2)
        }
        for _, row in forecast_df.iterrows()
    ]