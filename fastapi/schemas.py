"""
Pydantic schemas for CarPriceML API
Defines request/response models with validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import datetime


class CarPricePredictionRequest(BaseModel):
    """
    Request schema for car price prediction
    
    Features MUST be in this EXACT order to match the trained model:
    1. Numerical features (6): max_power_bhp, year, engine_cc, power_per_cc, seats, km_driven
    2. Categorical features (5): brand, owner, fuel, seller_type, transmission
    
    Note: mileage_mpg was EXCLUDED (correlation < 0.1)
    """
    
    # ========== NUMERICAL FEATURES (order critical - matches training) ==========
    max_power_bhp: float = Field(
        ..., 
        gt=0, 
        le=200,
        description="Maximum power in brake horsepower (BHP), max 200 BHP",
        example=74.0
    )
    
    year: int = Field(
        ..., 
        ge=1983, 
        le=2025,
        description="Manufacturing year of the vehicle",
        example=2014
    )
    
    engine_cc: int = Field(
        ..., 
        gt=0, 
        le=2500,
        description="Engine capacity in cubic centimeters (cc), max 2500cc",
        example=1248
    )
    
    power_per_cc: float = Field(
        ..., 
        gt=0,
        le=0.2,
        description="Power per cc ratio (calculated: max_power_bhp / engine_cc)",
        example=0.059247
    )
    
    seats: int = Field(
        ..., 
        ge=2, 
        le=7,
        description="Number of seats (2-7, excludes minibuses)",
        example=5
    )
    
    km_driven: int = Field(
        ..., 
        ge=0, 
        le=200000,
        description="Total kilometers driven (max 200,000 km based on training data)",
        example=145500
    )
    
    # ========== CATEGORICAL FEATURES (order critical) ==========
    brand: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Car brand/manufacturer (e.g., Maruti, Hyundai, BMW)",
        example="Maruti"
    )
    
    owner: Literal[
        "First", 
        "Second", 
        "Third", 
        "Fourth & Above", 
        "Test Drive Car"
    ] = Field(
        ...,
        description="Ownership history of the vehicle",
        example="First"
    )
    
    fuel: Literal["Diesel", "Petrol", "CNG", "LPG"] = Field(
        ...,
        description="Type of fuel used by the vehicle",
        example="Diesel"
    )
    
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"] = Field(
        ...,
        description="Type of seller",
        example="Individual"
    )
    
    transmission: Literal["Manual", "Automatic"] = Field(
        ...,
        description="Transmission type",
        example="Manual"
    )
    
    @field_validator('power_per_cc')
    @classmethod
    def validate_power_per_cc(cls, v, info):
        """Validate that power_per_cc is reasonable"""
        if v < 0.01 or v > 0.2:
            raise ValueError('power_per_cc must be between 0.01 and 0.2 (typical range for cars)')
        return v
    
    @field_validator('brand')
    @classmethod
    def clean_brand(cls, v):
        """Clean brand name (strip whitespace, capitalize)"""
        return v.strip().title()
    
    class Config:
        json_schema_extra = {
            "example": {
                "max_power_bhp": 74.0,
                "year": 2014,
                "engine_cc": 1248,
                "power_per_cc": 0.059247,
                "seats": 5,
                "km_driven": 145500,
                "brand": "Maruti",
                "owner": "First",
                "fuel": "Diesel",
                "seller_type": "Individual",
                "transmission": "Manual"
            }
        }


class CarPricePredictionResponse(BaseModel):
    """Response schema for car price prediction"""
    
    predicted_price_mad: float = Field(
        ...,
        description="Predicted price in Moroccan Dirhams (MAD)",
        example=75000.50
    )
    
    confidence_interval: dict = Field(
        ...,
        description="95% confidence interval for the prediction (lower and upper bounds)",
        example={"lower": 68000.0, "upper": 82000.0}
    )
    
    prediction_std: float = Field(
        ...,
        description="Standard deviation of predictions from individual trees (uncertainty measure)",
        example=3500.75
    )
    
    confidence_score: float = Field(
        ...,
        description="Confidence score (0-100): higher means more confident. Based on prediction variance.",
        example=85.5
    )
    
    model_version: str = Field(
        ...,
        description="Version of the ML model used for prediction",
        example="3.0_optimized_outliers"
    )
    
    prediction_timestamp: datetime = Field(
        ...,
        description="Timestamp when the prediction was made",
        example="2025-11-07T14:30:00"
    )
    
    input_features: dict = Field(
        ...,
        description="Echo of input features used for prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price_mad": 75000.50,
                "confidence_interval": {
                    "lower": 68000.0,
                    "upper": 82000.0
                },
                "prediction_std": 3500.75,
                "confidence_score": 85.5,
                "model_version": "3.0_optimized_outliers",
                "prediction_timestamp": "2025-11-07T14:30:00",
                "input_features": {
                    "year": 2018,
                    "brand": "Maruti",
                    "km_driven": 35000
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint"""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Health status of the API",
        example="healthy"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready",
        example=True
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Version of the loaded model",
        example="3.0_optimized_outliers"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp of the health check",
        example="2025-11-07T14:30:00"
    )
    
    message: Optional[str] = Field(
        None,
        description="Additional status message",
        example="API is operational"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "3.0_optimized_outliers",
                "timestamp": "2025-11-07T14:30:00",
                "message": "API is operational"
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for error cases"""
    
    error: str = Field(
        ...,
        description="Error type/category",
        example="ValidationError"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Invalid input: year must be between 2000 and 2025"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp when the error occurred",
        example="2025-11-07T14:30:00"
    )
    
    details: Optional[dict] = Field(
        None,
        description="Additional error details (if available)"
    )
