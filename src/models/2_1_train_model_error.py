from xgboost import XGBRegressor
import os
import yaml
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from src.utils.mlflow_setup import setup_mlflow


# -------------------------------
# MLFLOW SETUP
# -------------------------------
setup_mlflow("solar-performance-monitoring_error")

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path = os.path.join(BASE_DIR, "data", "train_test_data", "train_test_error")

train_model_path = os.path.join(BASE_DIR, "models","train_model","error_model","train_model_error.pkl")
os.makedirs(os.path.dirname(train_model_path), exist_ok=True)

# -------------------------------
# LOAD PARAMS
# -------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


# -------------------------------
# FUNCTIONS
# -------------------------------
def train_on_train_data(X_train, y_train):
    try:
        model = XGBRegressor(
            n_estimators=params["model_train_error"]["n_estimators"],
            learning_rate=params["model_train_error"]["learning_rate"],
            max_depth=params["model_train_error"]["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train.values.ravel())  # ✅ fix shape issue

        return model

    except Exception as e:
        print(f"❌ Train error: {e}")
        return None
    

def load_train_test(path):
    try:
        X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(path, "y_train.csv"))

        print("📊 X_train shape:", X_train.shape)
        print("📊 y_train shape:", y_train.shape)

        return X_train, y_train

    except Exception as e:
        print(f"❌ Load train/test error: {e}")
        return None, None    


def save_train_test_model(model, path):
    try:
        joblib.dump(model, path)
        print(f"✅ Model saved at {path}")
    except Exception as e:
        print(f"❌ Save model error: {e}")


# -------------------------------
# MAIN WITH MLFLOW
# -------------------------------
with mlflow.start_run(run_name="train_test_error_model"):

    X_train, y_train = load_train_test(path)

    model = train_on_train_data(X_train, y_train)

    if model is not None:

        # ✅ LOG PARAMETERS
        mlflow.log_params({
            "n_estimators": params['model_train_error']['n_estimators'],
            "learning_rate": params['model_train_error']['learning_rate'],
            "max_depth": params['model_train_error']['max_depth']
        })


        mlflow.log_param("num_features", X_train.shape[1])

        mlflow.sklearn.log_model(model, "train_test_error_model")

        # ✅ TAGS
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_type", "xgboost")

        # SAVE LOCALLY (DVC)
        save_train_test_model(model, train_model_path)

    else:
        print("❌ Model training failed")