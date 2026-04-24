
import mlflow
import os
import dotenv



dotenv.load_dotenv()

def setup_mlflow(experiment_name: str):
    try:
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "kamrankhanorakzai"
        repo_name = "SolarIQ-System"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
        mlflow.set_experiment(experiment_name)

        print(f"✅ MLflow connected: {experiment_name}")
        return True

    except Exception as e:
        print(f"❌ MLflow setup error: {e}")
        return False





    
