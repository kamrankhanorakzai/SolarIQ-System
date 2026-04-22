import pandas as pd
import glob
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR_UNIT = os.path.join(BASE_DIR ,"data", "raw", "unit_files")
RAW_DIR_ERROR = os.path.join(BASE_DIR ,"data", "raw", "error_files")


# electricity unit generation data

# all files starting with a number (0–9)
def fetch_and_combine_unit_data(RAW_DIR_UNIT):
        files = glob.glob(os.path.join(RAW_DIR_UNIT, "[0-9]*.xlsx"))

        df_unit = pd.DataFrame()

        for file in files:
            try:
                temp_df = pd.read_excel(
                    file,
                    sheet_name=0,
                    names=["date", "E-Today(KWH)"],
                    usecols=[0, 1],
                    skiprows=3
                )
                df_unit = pd.concat([df_unit, temp_df], ignore_index=True)

            except Exception as e:
                print(f"⚠️ Skipped {os.path.basename(file)} → {e}")
        df_unit["date"] = pd.to_datetime(df_unit["date"], errors="coerce")
        return df_unit



# error per day occurence data


# find all error files starting with "e" or "error"
def fetch_and_combine_error_data(RAW_DIR_ERROR):
            files = glob.glob(os.path.join(RAW_DIR_ERROR, "e*.xlsx"))
            # OR more strict:
            # files = glob.glob(os.path.join(RAW_DIR, "error_*.xlsx"))

            df_error = pd.DataFrame()

            for file in files:
                temp_df = pd.read_excel(
                    file,
                    usecols=["time", "event"]
                )
                df_error = pd.concat([df_error, temp_df], ignore_index=True)

            # -------------------------------
            # Cleaning & transformations
            # -------------------------------
            df_error.rename(columns={"time": "time/date"}, inplace=True)
            df_error["time/date"] = pd.to_datetime(df_error["time/date"])
            df_error["date"] = df_error["time/date"].dt.date
            df_error["date"] = pd.to_datetime(df_error["date"])
            return df_error
            

df_unit = fetch_and_combine_unit_data(RAW_DIR_UNIT).sort_values("date")
df_error = fetch_and_combine_error_data(RAW_DIR_ERROR).sort_values("date")

output_dir = os.path.join(BASE_DIR, "data", "interim","Fetch_join_data")
os.makedirs(output_dir, exist_ok=True)

df_unit.to_csv(os.path.join(output_dir, "unit_data.csv"), index=False)
df_error.to_csv(os.path.join(output_dir, "error_data.csv"), index=False)
