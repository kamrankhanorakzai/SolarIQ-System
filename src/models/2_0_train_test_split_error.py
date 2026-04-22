import pandas as pd
import os
import yaml
import mlflow
from src.utils.mlflow_setup import setup_mlflow_error

# -------------------------------
# MLFLOW SETUP
# -------------------------------
setup_mlflow_error()

# -------------------------------
# LOAD PARAMS
# -------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_data",
    "error_data",
    "feature_engineered_data_error.csv"
)

output_train_test = os.path.join(BASE_DIR, "data", "train_test_data", "train_test_error")
os.makedirs(output_train_test, exist_ok=True)

# -------------------------------
# FUNCTIONS
# -------------------------------
def read_data(path):
    try:
        df = pd.read_csv(path)
        print("📊 Data shape:", df.shape)

        return df
    except Exception as e:
        print(f"❌ Read data error: {e}")
        return None


def split_data(df):
    try:
        train_size_ratio = params['split_data_error']['train_size']
        train_size = int(len(df) * train_size_ratio)

        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

        X_train = train.drop(columns=['total_error'])
        y_train = train['total_error']

        X_test = test.drop(columns=['total_error'])
        y_test = test['total_error']

        return X_train, X_test, y_train, y_test, train_size_ratio

    except Exception as e:
        print(f"❌ Split error: {e}")
        return None, None, None, None, None


def save_train_test(X_train, X_test, y_train, y_test, path):
    try:
        X_train.to_csv(os.path.join(path, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(path, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(path, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(path, "y_test.csv"), index=False)

        print(f"✅ Train/test data saved at {path}")

    except Exception as e:
        print(f"❌ Save train/test error: {e}")



# -------------------------------
# MAIN WITH MLFLOW
# -------------------------------
with mlflow.start_run(run_name="train_test_split_error"):

    df = read_data(path)


    X_train, X_test, y_train, y_test, split_ratio = split_data(df)

    if X_train is not None:

        # ✅ LOG PARAMETERS
        mlflow.log_param("train_size_ratio", split_ratio)
        mlflow.log_param("num_features", len(X_train.columns))

        # ✅ LOG DATA INFO
        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))

        # ✅ TAG
        mlflow.set_tag("stage", "data_split")

        save_train_test(X_train, X_test, y_train, y_test, output_train_test)

    else:
        print("❌ Split failed")