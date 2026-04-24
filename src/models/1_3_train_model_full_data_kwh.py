import pandas as pd
import os
from xgboost import XGBRegressor
import yaml
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.utils.mlflow_setup import setup_mlflow

# -------------------------------
# MLFLOW SETUP
# -------------------------------
setup_mlflow("solar-performance-monitoring_kwh")

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
df_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_data",
    "kwh_data",
    "feature_engineered_data_kwh.csv"
)

output_model = os.path.join(
    BASE_DIR,
    "models",
    "final_model",
    "kwh_model",
    "model_kwh.pkl"
)

os.makedirs(os.path.dirname(output_model), exist_ok=True)

# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_full_model(df_path):
    try:
        df = pd.read_csv(df_path)

        # 🔥 IMPORTANT: Drop non-numeric columns
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        X = df.drop(columns=["E-Today(KWH)"])
        y = df["E-Today(KWH)"]

        model = XGBRegressor(
            n_estimators=params["model_train_kwh"]["n_estimators"],
            learning_rate=params["model_train_kwh"]["learning_rate"],
            max_depth=params["model_train_kwh"]["max_depth"],
            random_state=42
        )

        model.fit(X, y)

        print("✅ Model trained successfully")
        return model, X.shape[1]

    except Exception as e:
        print(f"❌ Training error: {e}")
        return None, None


# -------------------------------
# SAVE MODEL LOCALLY (DVC)
# -------------------------------
def save_model(model, path):
    try:
        joblib.dump(model, path)
        print(f"✅ Model saved locally at {path}")
    except Exception as e:
        print(f"❌ Save model error: {e}")


# -------------------------------
# MAIN TRAINING + REGISTRY LOGIC
# -------------------------------
with mlflow.start_run(run_name="deployment_model_kwh") as run:

    final_model, feature_length = train_full_model(df_path)

    if final_model is None:
        raise Exception("❌ Model training failed")

    # -------------------------------
    # LOG PARAMETERS
    # -------------------------------
    mlflow.log_params({
        "n_estimators": params["model_train_kwh"]["n_estimators"],
        "learning_rate": params["model_train_kwh"]["learning_rate"],
        "max_depth": params["model_train_kwh"]["max_depth"],
        "num_features": feature_length
    })

    model_name = "final_model_kwh"

    # -------------------------------
    # LOG MODEL TO MLFLOW
    # -------------------------------
    mlflow.sklearn.log_model(
        sk_model=final_model,
        name=model_name,
        registered_model_name=model_name
    )

    mlflow.set_tag("stage", "training")
    mlflow.set_tag("model_type", "xgboost")

    print("✔ Model logged and registered")

    client = MlflowClient()

    # -------------------------------
    # GET ALL MODEL VERSIONS
    # -------------------------------
    versions = client.search_model_versions(f"name='{model_name}'")

    # Get latest version
    latest_version = max(versions, key=lambda x: int(x.version))

    print(f"📦 Latest model version: {latest_version.version}")

    # -------------------------------
    # HANDLE OLD PRODUCTION MODEL
    # -------------------------------
    try:
        old_prod = client.get_model_version_by_alias(model_name, "production")

        # Remove alias
        client.delete_registered_model_alias(model_name, "production")
        print(f"🔁 Removed production alias from version {old_prod.version}")

        # Archive old version
        client.transition_model_version_stage(
            name=model_name,
            version=old_prod.version,
            stage="Archived"
        )
        print(f"📦 Archived old version {old_prod.version}")

    except Exception:
        print("ℹ️ No previous production model found")

    # -------------------------------
    # PROMOTE NEW MODEL
    # -------------------------------
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production"
    )

    # Set alias
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version.version
    )

    print(f"🚀 Model version {latest_version.version} is now PRODUCTION")

    # -------------------------------
    # SAVE LOCAL MODEL
    # -------------------------------
    save_model(final_model, output_model)

    print(f"🆔 Run ID: {run.info.run_id}")