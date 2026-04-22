"""
Forecasting module - handles multi-step ahead forecasting
"""
import pandas as pd
from typing import Tuple, List
from features.feature_generator import calculate_features_for_prediction, prepare_features_for_model, validate_data
from config import DATE_COLUMN, KWH_COLUMN


def forecast(
    model,
    df: pd.DataFrame,
    days: int = 7,
    data_column: str = KWH_COLUMN,
    date_column: str = DATE_COLUMN
) -> Tuple[pd.DataFrame, bool]:
    """
    Generate multi-step ahead forecast using recursive feature engineering
    
    Args:
        model: Trained ML model with predict method
        df: Historical data with date and KWH columns
        days: Number of days to forecast (default: 7)
        data_column: Name of the column containing KWH values
        date_column: Name of the date column
    
    Returns:
        Tuple[pd.DataFrame, bool]: (Forecast DataFrame, Success flag)
                                  DataFrame contains 'date' and 'prediction' columns
    """
    try:
        # Validate input data
        validate_data(df, date_column=date_column, kwh_column=data_column)
        
        # Prepare data
        future = pd.DataFrame()
        df[date_column] = pd.to_datetime(df[date_column])
        last_date = df[date_column].max()
        
        # Generate future dates
        future[date_column] = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )

        # Create working copy of dataframe for recursive forecasting
        df_full = df.copy()
        predictions = []

        # Forecast loop
        for i in range(days):
            try:
                current_date = future.loc[i, date_column]

                # Calculate features for current prediction
                features = calculate_features_for_prediction(df_full, current_date)
                
                # Prepare features array for model
                X_future = prepare_features_for_model(features)

                # Make prediction
                pred = model.predict(X_future)[0]
                predictions.append(float(pred))

                # Append prediction to dataframe for next iteration
                new_row = pd.DataFrame({
                    date_column: [current_date],
                    data_column: [pred]
                })
                df_full = pd.concat([df_full, new_row], ignore_index=True)

            except Exception as e:
                print(f"❌ Error forecasting day {i+1}: {e}")
                return None, False

        # Prepare result
        future['prediction'] = predictions
        print(f"✅ Forecast completed for {days} days")
        
        return future, True

    except Exception as e:
        print(f"❌ Forecast error: {e}")
        return None, False


def forecast_from_csv(
    model,
    csv_path: str,
    days: int = 7,
    data_column: str = KWH_COLUMN,
    date_column: str = DATE_COLUMN
) -> Tuple[pd.DataFrame, bool]:
    """
    Generate forecast from CSV file
    
    Args:
        model: Trained ML model with predict method
        csv_path: Path to CSV file with historical data
        days: Number of days to forecast
        data_column: Name of the column containing KWH values
        date_column: Name of the date column
    
    Returns:
        Tuple[pd.DataFrame, bool]: (Forecast DataFrame, Success flag)
    """
    try:
        df = pd.read_csv(csv_path)
        return forecast(model, df, days, data_column, date_column)
    except Exception as e:
        print(f"❌ Error loading CSV for forecast: {e}")
        return None, False


def format_forecast_response(forecast_df: pd.DataFrame) -> List[dict]:
    """
    Format forecast DataFrame into response list
    
    Args:
        forecast_df: DataFrame with 'date' and 'prediction' columns
    
    Returns:
        List of dictionaries with date and prediction
    """
    if forecast_df is None:
        return []
    
    result = []
    for idx, row in forecast_df.iterrows():
        result.append({
            "date": row[DATE_COLUMN].isoformat(),
            "prediction": float(row['prediction'])
        })
    
    return result
