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
setup_mlflow("solar-performance-monitoring_error")

# Load params
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
    "error_data",
    "feature_engineered_data_error.csv"
)

output_model = os.path.join(
    BASE_DIR,
    "models",
    "final_model",
    "error_model",
    "final_model_error.pkl"
)

os.makedirs(os.path.dirname(output_model), exist_ok=True)

# -------------------------------
# FEATURES
# -------------------------------
def get_features():
    return [
        "month_cos",
        "lag1",
        "lag2",
        "roll_mean_3",
        "roll_max_7",
        "roll_min_7",
        "diff1",
        "diff7",
        "lag1_roll7"
    ]

# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_full_model(df_path, features):
    try:
        df = pd.read_csv(df_path)

        # Safety: remove non-numeric columns if exist
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        X = df[features]
        y = df["total_error"]

        model = XGBRegressor(
            n_estimators=params["model_train_error"]["n_estimators"],
            learning_rate=params["model_train_error"]["learning_rate"],
            max_depth=params["model_train_error"]["max_depth"],
            random_state=42
        )

        model.fit(X, y)

        print("✅ Error model trained successfully")
        return model, len(features)

    except Exception as e:
        print(f"❌ Training error: {e}")
        return None, None


# -------------------------------
# SAVE LOCAL MODEL (DVC backup)
# -------------------------------
def save_model(model, path):
    try:
        joblib.dump(model, path)
        print(f"✅ Model saved locally at {path}")
    except Exception as e:
        print(f"❌ Save model error: {e}")


# -------------------------------
# MAIN TRAINING + MLFLOW
# -------------------------------
with mlflow.start_run(run_name="deployment_model_error") as run:

    features_list = get_features()
    final_model, feature_length = train_full_model(df_path, features_list)

    if final_model is None:
        raise Exception("❌ Model training failed")

    # -------------------------------
    # LOG PARAMETERS
    # -------------------------------
    mlflow.log_params({
        "n_estimators": params["model_train_error"]["n_estimators"],
        "learning_rate": params["model_train_error"]["learning_rate"],
        "max_depth": params["model_train_error"]["max_depth"],
        "num_features": feature_length
    })

    model_name = "final_model_error"

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

    print("✔ Error model logged and registered")

    client = MlflowClient()

    # -------------------------------
    # GET ALL VERSIONS
    # -------------------------------
    versions = client.search_model_versions(f"name='{model_name}'")

    latest_version = max(versions, key=lambda x: int(x.version))

    print(f"📦 Latest version: {latest_version.version}")

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

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version.version
    )

    print(f"🚀 Error model version {latest_version.version} is now PRODUCTION")

    # -------------------------------
    # SAVE LOCAL MODEL
    # -------------------------------
    save_model(final_model, output_model)

    print(f"🆔 Run ID: {run.info.run_id}")