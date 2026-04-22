import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()


# -------------------------------
# DATABASE CONNECTION (AWS RDS)
# -------------------------------
def get_db_engine():
    """Create AWS RDS PostgreSQL connection"""
    try:
        db_user = os.getenv("AWS_USER")
        db_password = os.getenv("AWS_PASS")
        db_host = os.getenv("AWS_HOST")
        db_port = os.getenv("AWS_PORT", 5432)
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
def load_from_postgres(table_name, engine):
    """Fetch data from PostgreSQL table"""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)

        print(f"📥 Data loaded from DB: {df.shape}")
        return df

    except Exception as e:
        print(f"❌ Error loading data from DB: {e}")
        return None






# -------------------------------



