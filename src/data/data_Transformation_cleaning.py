import pandas as pd
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
combine_data_dir = os.path.join(BASE_DIR, "data", "interim","Fetch_join_data")

df_unit=pd.read_csv(os.path.join(combine_data_dir, "unit_data.csv"))
df_error=pd.read_csv(os.path.join(combine_data_dir, "error_data.csv"))


def clean_unit_data(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def clean_error_data(df):
    df['time/date'] = pd.to_datetime(df['time/date'], errors='coerce')
    df.dropna(subset=['time/date'], inplace=True)
    df['date'] = df['time/date'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    df.drop_duplicates(inplace=True)
    return df


df_unit = clean_unit_data(df_unit)
df_error = clean_error_data(df_error)

output_dir = os.path.join(BASE_DIR, "data", "interim","Preprocessing_data")
os.makedirs(output_dir, exist_ok=True)

df_unit.to_csv(os.path.join(output_dir, "df_unit.csv"), index=False)
df_error.to_csv(os.path.join(output_dir, "df_error.csv"), index=False)