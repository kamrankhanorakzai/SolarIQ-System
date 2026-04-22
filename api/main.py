"""
Solar Energy Prediction API - Main Application
Endpoints for predicting solar energy (KWH and Error) with forecasting capability
"""
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from load_data_aws import get_db_engine,load_from_postgres
from config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_HOST,
    API_PORT,
    REQUIRED_FEATURES,
    KWH_MODEL_NAME,
    ERROR_MODEL_NAME,
    KWH_COLUMN,
    DATE_COLUMN
)

from schemas import (
    ForecastRequest,
    KWHForecastResponse,
    ForecastedValue,
    ErrorForecastResponse,
    HealthResponse,
    ModelInfoResponse
)

from models.model_loader import load_both_models
from forecasting.forecaster import forecast


# ==================== Initialize FastAPI App ====================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Load Models ====================

print("🔄 Loading models from MLflow registry...")
kwh_model, error_model = load_both_models()


# ==================== Health & Info Endpoints ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API and models are running correctly"""
    if kwh_model is None and error_model is None:
        raise HTTPException(status_code=503, detail="No models loaded from MLflow registry")
    
    return HealthResponse(
        status="healthy",
        kwh_model_loaded=kwh_model is not None,
        error_model_loaded=error_model is not None,
        mlflow_connected=True,
        message="API is running with models from MLflow registry"
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "mlflow_integration": {
            "enabled": True,
            "repository": "solar-performance-monitoring",
            "tracking_uri": "https://dagshub.com/kamrankhanorakzai/solar-performance-monitoring.mlflow"
        },
        "models": {
            "kwh": {
                "status": "loaded" if kwh_model is not None else "not loaded",
                "registered_name": KWH_MODEL_NAME,
                "endpoints": [
                    "/kwh/forecast",
                    "/kwh/model-info"
                ]
            },
            "error": {
                "status": "loaded" if error_model is not None else "not loaded",
                "registered_name": ERROR_MODEL_NAME,
                "endpoints": [
                    "/error/forecast",
                    "/error/model-info"
                ]
            }
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# ==================== KWH FORECASTING ENDPOINTS ====================

@app.post("/kwh/forecast", response_model=KWHForecastResponse, tags=["KWH Forecasting"])
async def forecast_kwh(request: ForecastRequest):
    """
    Forecast KWH energy output for specified number of days
    Requires CSV file path in environment variable or request
    
    Args:
        request: ForecastRequest with days parameter (default: 7)
    
    Returns:
        KWHForecastResponse: List of forecasted values with dates
    """
    if kwh_model is None:
        raise HTTPException(status_code=503, detail="KWH model not loaded from MLflow registry")
    
    try:
        # Try to load data from environment or default path
        engine = get_db_engine()
        df_db = load_from_postgres("final_data_forecasting", engine)
        data_path = df_db
   
        df = data_path
        
        # Generate forecast
        forecast_df, success = forecast(
            kwh_model,
            df,
            days=request.days,
            data_column=KWH_COLUMN,
            date_column=DATE_COLUMN
        )
        
        if not success or forecast_df is None:
            raise Exception("Forecasting failed")
        
        # Format response
        forecast_list = [
            ForecastedValue(
                date=pd.to_datetime(row[DATE_COLUMN]),
                prediction=round(float(row['prediction']), 2)
            )
            for idx, row in forecast_df.iterrows()
        ]
        
        return KWHForecastResponse(
            forecast=forecast_list,
            days=request.days,
            message=f"KWH forecast generated successfully for {request.days} days"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"KWH forecast error: {str(e)}")


@app.get("/kwh/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
async def kwh_model_info():
    """Get information about the KWH model"""
    if kwh_model is None:
        raise HTTPException(status_code=503, detail="KWH model not loaded")
    
    return ModelInfoResponse(
        model_type="MLflow PyfuncModel",
        model_name=KWH_MODEL_NAME,
        source="MLflow Registry",
        alias="production (or latest version)",
        required_features=REQUIRED_FEATURES,
        feature_count=len(REQUIRED_FEATURES),
        tracking_uri="https://dagshub.com/kamrankhanorakzai/solar-performance-monitoring.mlflow",
        repo_owner="kamrankhanorakzai",
        repo_name="solar-performance-monitoring"
    )


# ==================== ERROR FORECASTING ENDPOINTS ====================

@app.post("/error/forecast", response_model=ErrorForecastResponse, tags=["Error Forecasting"])
async def forecast_error(request: ForecastRequest):
    """
    Forecast error values for specified number of days
    Uses KWH forecast as basis for feature engineering
    
    Args:
        request: ForecastRequest with days parameter (default: 7)
    
    Returns:
        ErrorForecastResponse: List of forecasted error values with dates
    """
    if error_model is None:
        raise HTTPException(status_code=503, detail="Error model not loaded from MLflow registry")
    
    try:
        engine = get_db_engine()
        df_db = load_from_postgres("final_data_forecasting", engine)
        data_path = df_db

        # Load data
        df = data_path
        
        # Generate forecast using error model with KWH column
        forecast_df, success = forecast(
            error_model,
            df,
            days=request.days,
            data_column=KWH_COLUMN,  # Use KWH column for feature engineering
            date_column=DATE_COLUMN
        )
        
        if not success or forecast_df is None:
            raise Exception("Forecasting failed")
        
        # Format response
        forecast_list = [
            ForecastedValue(
                date=pd.to_datetime(row[DATE_COLUMN]),
                prediction=int(round(float(row['prediction'])))
            )
            for idx, row in forecast_df.iterrows()
        ]
        
        return ErrorForecastResponse(
            forecast=forecast_list,
            days=request.days,
            message=f"Error forecast generated successfully for {request.days} days"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error forecast error: {str(e)}")


@app.get("/error/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
async def error_model_info():
    """Get information about the Error model"""
    if error_model is None:
        raise HTTPException(status_code=503, detail="Error model not loaded")
    
    return ModelInfoResponse(
        model_type="MLflow PyfuncModel",
        model_name=ERROR_MODEL_NAME,
        source="MLflow Registry",
        alias="production (or latest version)",
        required_features=REQUIRED_FEATURES,
        feature_count=len(REQUIRED_FEATURES),
        tracking_uri="https://dagshub.com/kamrankhanorakzai/solar-performance-monitoring.mlflow",
        repo_owner="kamrankhanorakzai",
        repo_name="solar-performance-monitoring"
    )


# ==================== Run Application ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
