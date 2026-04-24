"""
Model loading module - handles MLflow model loading and caching
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
from api.config import (
    KWH_MODEL_NAME,
    ERROR_MODEL_NAME,
    MODEL_ALIAS
)


def setup_mlflow() -> bool:
    """
    Setup MLflow connection to DagShub
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        dagshub_url = "https://dagshub.com"
        repo_owner = "kamrankhanorakzai"
        repo_name = "solar-performance-monitoring"
        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


        print("✅ MLflow tracking URI configured")
        return True
    except Exception as e:
        print(f"❌ MLflow setup error: {e}")
        return False


def load_model_from_registry(model_name: str, alias: str = "production"):
    """
    Load model from MLflow registry
    
    Args:
        model_name: Name of the registered model (e.g., 'final_model_kwh')
        alias: Model alias to load (default: 'production')
    
    Returns:
        Loaded model or None if failed
    """
    try:
        client = MlflowClient()
        
        # Try to load model with the specified alias
        try:
            model_uri = f"models:/{model_name}@{alias}"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✅ Model '{model_name}' (alias: {alias}) loaded from MLflow registry")
            return model
        except Exception as e:
            print(f"⚠️ Alias '{alias}' not found, trying latest version: {e}")
            # If alias doesn't exist, get the latest version
            latest_versions = client.get_latest_versions(model_name)
            if latest_versions:
                latest_version = latest_versions[0]
                model_uri = f"models:/{model_name}/{latest_version.version}"
                model = mlflow.pyfunc.load_model(model_uri)
                print(f"✅ Model '{model_name}' (version: {latest_version.version}) loaded from MLflow registry")
                return model
            else:
                print(f"❌ No versions found for model '{model_name}'")
                return None
                
    except Exception as e:
        print(f"❌ Error loading model '{model_name}' from registry: {e}")
        return None


def load_both_models():
    """
    Load both KWH and Error models from MLflow registry
    
    Returns:
        tuple: (kwh_model, error_model)
    """
    kwh_model = None
    error_model = None
    
    if setup_mlflow():
        kwh_model = load_model_from_registry(KWH_MODEL_NAME, alias=MODEL_ALIAS)
        error_model = load_model_from_registry(ERROR_MODEL_NAME, alias=MODEL_ALIAS)
    
    return kwh_model, error_model
