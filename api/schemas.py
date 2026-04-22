"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel
from typing import List
from datetime import datetime


# ==================== Feature Input Models ====================

class FeaturesInput(BaseModel):
    """Features input model for single predictions"""
    month_cos: float
    lag1: float
    lag2: float
    roll_mean_3: float
    roll_max_7: float
    roll_min_7: float
    diff1: float
    diff7: float
    lag1_roll7: float

    class Config:
        json_schema_extra = {
            "example": {
                "month_cos": 0.5,
                "lag1": 100.5,
                "lag2": 98.3,
                "roll_mean_3": 99.5,
                "roll_max_7": 105.0,
                "roll_min_7": 95.0,
                "diff1": 2.2,
                "diff7": 5.5,
                "lag1_roll7": 99.0
            }
        }


class BatchFeaturesInput(BaseModel):
    """Batch input for multiple predictions"""
    predictions: List[FeaturesInput]


# ==================== Forecast Input Models ====================

class ForecastRequest(BaseModel):
    """Forecast request model"""
    days: int = 7

    class Config:
        json_schema_extra = {
            "example": {
                "days": 7
            }
        }


# ==================== KWH Prediction Response Models ====================

class KWHPredictionResponse(BaseModel):
    """Response model for single KWH prediction"""
    predicted_kwh: float
    message: str


class KWHBatchPredictionResponse(BaseModel):
    """Response model for batch KWH predictions"""
    predictions: List[float]
    count: int


class ForecastedValue(BaseModel):
    """Single forecasted value"""
    date: datetime
    prediction: float


class KWHForecastResponse(BaseModel):
    """Response model for KWH forecast"""
    forecast: List[ForecastedValue]
    days: int
    message: str


# ==================== Error Prediction Response Models ====================

class ErrorPredictionResponse(BaseModel):
    """Response model for single Error prediction"""
    predicted_error: float
    message: str


class ErrorBatchPredictionResponse(BaseModel):
    """Response model for batch Error predictions"""
    predictions: List[float]
    count: int


class ErrorForecastResponse(BaseModel):
    """Response model for Error forecast"""
    forecast: List[ForecastedValue]
    days: int
    message: str


# ==================== Health and Info Response Models ====================

class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    kwh_model_loaded: bool
    error_model_loaded: bool
    mlflow_connected: bool
    message: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    model_name: str
    source: str
    alias: str
    required_features: List[str]
    feature_count: int
    tracking_uri: str
    repo_owner: str
    repo_name: str
