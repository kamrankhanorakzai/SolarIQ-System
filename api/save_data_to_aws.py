
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
from load_data_aws import  load_from_postgres
load_dotenv()

# ==============================
# OUTPUT DIRECTORY (LOCAL SAVE)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

file_path = os.path.join(BASE_DIR, "scrape_data", "final_data_forecasting.csv")
df = pd.read_csv(file_path)


# ==============================
# DATABASE CONNECTION
# ==============================
def get_db_engine():
    try:
        db_user = os.getenv("AWS_USER")
        db_password = os.getenv("AWS_PASS")
        db_host = os.getenv("AWS_HOST")
        db_port = os.getenv("AWS_PORT", 5432)
        db_name = os.getenv("AWS_DB")

        if not all([db_user, db_password, db_host, db_name]):
            raise ValueError("❌ Missing AWS DB environment variables")

        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        print("✅ Connected to AWS RDS PostgreSQL")
        return engine

    except Exception as e:
        print(f"❌ DB connection error: {e}")
        return None





# ==============================
# SAVE TO RDS (IMPORTANT PART)
# ==============================
def save_to_postgres(df, engine, table_name="final_data_forecasting"):
    try:
        # ======================
        # Load existing data
        # ======================
        rds_df = load_from_postgres(engine=engine)

        # ======================
        # Convert to datetime
        # ======================
        df["date"] = pd.to_datetime(df["date"])
        rds_df["date"] = pd.to_datetime(rds_df["date"])

        # ======================
        # Handle empty DB
        # ======================
        if rds_df.empty:
            print("🆕 RDS is empty → inserting full data")
            df.to_sql(table_name, engine, if_exists="append", index=False)
            return

        # ======================
        # Get max dates
        # ======================
        local_max = df["date"].max()
        rds_max = rds_df["date"].max()

        print(f"📅 Local max: {local_max}")
        print(f"📅 RDS max: {rds_max}")

        # ======================
        # Filter only new data
        # ======================
        new_data = df[df["date"] > rds_max]

        if new_data.empty:
            print("⚠️ No new data to insert")
            return

        # ======================
        # Insert ONLY new rows
        # ======================
        new_data.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",
            index=False
        )

        print(f"✅ Inserted {len(new_data)} new rows")

    except Exception as e:
        print(f"❌ Error saving to DB: {e}")

save_to_postgres(df, get_db_engine())