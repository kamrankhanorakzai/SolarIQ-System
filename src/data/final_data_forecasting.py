import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(BASE_DIR, "data", "interim", "Preprocessing_data")

df_unit=pd.read_csv(os.path.join(DIR, "df_unit.csv"))
df_error=pd.read_csv(os.path.join(DIR, "df_error.csv"))

def preprocess_error_data(df_error):
    df_error.drop(columns=["time/date"], inplace=True)
    error_counts = df_error.groupby(['date', 'event']).size().unstack(fill_value=0)
    return error_counts


def merge_unit_error_data(df_unit, error_counts):
    df = df_unit.merge(error_counts, on='date', how='left').fillna(0)
    return df

error_counts = preprocess_error_data(df_error)
df = merge_unit_error_data(df_unit, error_counts)

output_dir = os.path.join(BASE_DIR, "data", "processed","cleaned_data")
os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, "final_data_forecasting.csv"), index=False)