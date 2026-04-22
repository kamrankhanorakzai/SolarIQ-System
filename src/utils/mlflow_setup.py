import dagshub
import mlflow
import dotenv
import os
dotenv.load_dotenv()

def setup_mlflow_kwh():
    dagshub.init(
        repo_owner=os.getenv("repo_owner"),
        repo_name=os.getenv("repo_name"),
        mlflow=True
    )

    mlflow.set_tracking_uri(
        os.getenv("set_tracking_uri")
    )

    mlflow.set_experiment(os.getenv("experiment_kwh"))

def setup_mlflow_error():
    dagshub.init(
        repo_owner=os.getenv("repo_owner"),
        repo_name=os.getenv("repo_name"),
        mlflow=True
    )

    mlflow.set_tracking_uri(
        os.getenv("set_tracking_uri")
    )

    mlflow.set_experiment(os.getenv("experiment_error"))
