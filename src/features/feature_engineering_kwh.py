
import pandas as pd
import numpy as np
import os 



# 1. LOAD DATA
# -------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)

        if 'date' not in df.columns or 'E-Today(KWH)' not in df.columns:
            raise ValueError("Missing required columns")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        print("✅ Data loaded successfully")
        return df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


# -------------------------
# 2. FEATURE ENGINEERING
# -------------------------
def create_features(df):
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        error_cols = [
            "A0-Grid over voltage",
            "A1-Grid under voltage",
            "A2-Grid absent",
            "A3-Grid over frequency",
            "A4-Grid under frequency",
            "A6-Grid abnormal"
        ]
        df.drop(columns=error_cols, inplace=True, errors='ignore')
        # Time feature
        df['month'] = df['date'].dt.month
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Lag features
        df['lag1'] = df['E-Today(KWH)'].shift(1)
        df['lag2'] = df['E-Today(KWH)'].shift(2)

        # Rolling features
        df['roll_mean_3'] = df['E-Today(KWH)'].rolling(3).mean()
        df['roll_mean_7'] = df['E-Today(KWH)'].rolling(7).mean()
        df['roll_max_7'] = df['E-Today(KWH)'].rolling(7).max()
        df['roll_min_7'] = df['E-Today(KWH)'].rolling(7).min()

        # Trend
        df['diff1'] = df['E-Today(KWH)'].diff(1)
        df['diff7'] = df['E-Today(KWH)'].diff(7)

        # Interaction
        df['lag1_roll7'] = df['lag1'] * df['roll_mean_7']
        df.drop(columns=["date","month","roll_mean_7"], inplace=True)
        df = df.dropna()

        print("✅ Features created")
        # print(f"📊 Final feature set: {df.columns.tolist()}")
        return df

    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return None

def data_save(df, path):
    try:
        df.to_csv(os.path.join(path, "feature_engineered_data_kwh.csv"), index=False)
        print(f"✅ Data saved to {os.path.join(path, 'featured_data_kwh.csv')}")
    except Exception as e:
        print(f"❌ Error saving data: {e}")



BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(BASE_DIR, "data", "processed","cleaned_data")
path=os.path.join(DIR, "final_data_forecasting.csv")
output_dir = os.path.join(BASE_DIR, "data", "processed","feature_data","kwh_data")
os.makedirs(output_dir, exist_ok=True)

df=load_data(path)
df=create_features(df)
data_save(df, output_dir)