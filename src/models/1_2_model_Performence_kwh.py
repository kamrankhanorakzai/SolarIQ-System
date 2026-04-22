# from turtle import pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import json
import os
import yaml
import joblib
import pandas as pd
import mlflow
from src.utils.mlflow_setup import setup_mlflow_kwh   # ✅ added

# -------------------------------
# MLFLOW SETUP
# -------------------------------
setup_mlflow_kwh()   # ✅ added

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

train_test_model_path = os.path.join(BASE_DIR, "models","train_model","kwh_model","train_model_kwh.pkl")
test_data_path = os.path.join(BASE_DIR, "data", "train_test_data" ,"train_test_kwh")

output_matrics = os.path.join(BASE_DIR, "accuracy_report","kwh_matrics","metrics_kwh.json")
os.makedirs(os.path.dirname(output_matrics), exist_ok=True)


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_test = y_test.values.flatten()

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100

        metrics = {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape)
        }

        print("📊 Evaluation Done:", metrics)
        return metrics

    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return None
    

def load_model(path):
    try:
        model = joblib.load(path)
        print(f"✅ Model loaded at {path}")
        return model
    except Exception as e:
        print(f"❌ Load model error: {e}")
        
        return None

def load_train_test(path):
    try:
        X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(path, "y_test.csv"))
        print(f"✅ Train/test data loaded at {path}")
        return X_test, y_test
    except Exception as e:
        print(f"❌ Load train/test error: {e}")
        return None, None
    
def save_metrics(metrics, path):
    try:
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"✅ Metrics saved at {path}")

    except Exception as e:
        print(f"❌ Save metrics error: {e}")



with mlflow.start_run(run_name="model_evaluation_kwh"):

    load_model_path = train_test_model_path
    model = load_model(load_model_path) 
    
    X_test, y_test = load_train_test(test_data_path)

    metrics = evaluate_model(model, X_test, y_test)    

    if metrics is not None:

        # ✅ LOG METRICS (MAIN PART)
        mlflow.log_metrics({
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"]
        })



        # ✅ LOG METRICS FILE AS ARTIFACT
        save_metrics(metrics, output_matrics)
        mlflow.log_artifact(output_matrics)

        # ✅ TAG
        mlflow.set_tag("stage", "evaluation")

    else:
        print("❌ Evaluation failed")