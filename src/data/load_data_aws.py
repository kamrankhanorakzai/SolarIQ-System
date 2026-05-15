import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()


# -------------------------------
# DATABASE CONNECTION (AWS RDS)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = os.path.join(BASE_DIR, "data", "processed","cleaned_data")
os.makedirs(output_dir, exist_ok=True)



def get_db_engine():
    """Create AWS RDS PostgreSQL connection"""
    try:
        db_user = os.getenv("AWS_USER")
        db_password = os.getenv("AWS_PASS")
        db_host = os.getenv("AWS_HOST")
        db_port = os.getenv("AWS_PORT")
        db_name = os.getenv("AWS_DB")

        # safety check (IMPORTANT)
        if not all([db_user, db_password, db_host, db_name]):
            raise ValueError("Missing AWS DB environment variables")

        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        print("✅ Connected to AWS RDS PostgreSQL")
        return engine

    except Exception as e:
        print(f"❌ DB connection error: {e}")
        return None


# -------------------------------
# LOAD DATA FROM AWS RDS
# -------------------------------
def load_from_postgres(engine=get_db_engine(),table_name="final_data_forecasting"):
    """Fetch data from PostgreSQL table"""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        # remove duplicatede
        df.drop_duplicates(inplace=True)
        print(f"📥 Data loaded from DB: {df.shape}")
        return df

    except Exception as e:
        print(f"❌ Error loading data from DB: {e}")
        return None


if __name__ == "__main__":
    df=load_from_postgres()
    df.to_csv(os.path.join(output_dir, "final_data_forecasting.csv"), index=False)

# -------------------------------



