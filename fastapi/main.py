"""
CarPriceML FastAPI Backend
API pour la prédiction du prix des voitures d'occasion

Endpoints:
- POST /predict: Prédiction du prix d'une voiture
- GET /health: Vérification du statut de l'API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from loguru import logger
import sys

from schemas import (
    CarPricePredictionRequest,
    CarPricePredictionResponse,
    HealthCheckResponse,
    ErrorResponse
)

# ========== CONFIGURATION ==========
MODEL_PATH = Path("../models/rfmodel.joblib")
METADATA_PATH = Path("../models/model_metadata.json")

# Global variable to store loaded model
ml_model = None
model_metadata = None

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


# ========== LIFESPAN MANAGEMENT ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    Load model on startup, cleanup on shutdown
    """
    global ml_model, model_metadata
    
    # Startup: Load model
    logger.info("🚀 Starting CarPriceML API...")
    try:
        # Load the trained model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        ml_model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        
        # Load metadata
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)
            logger.info(f"✅ Metadata loaded: version {model_metadata.get('version', 'unknown')}")
        else:
            logger.warning(f"⚠️ Metadata file not found at {METADATA_PATH}")
            model_metadata = {"version": "unknown"}
        
        logger.info("🎉 API startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        ml_model = None
        model_metadata = {"version": "unknown", "error": str(e)}
    
    yield
    
    # Shutdown: Cleanup
    logger.info("🛑 Shutting down CarPriceML API...")
    ml_model = None
    model_metadata = None


# ========== FASTAPI APP ==========
app = FastAPI(
    title="CarPriceML API",
    description="API de prédiction du prix des voitures d'occasion avec ML",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration (allow Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== HELPER FUNCTIONS ==========
def prepare_input_dataframe(request: CarPricePredictionRequest) -> pd.DataFrame:
    """
    Convert Pydantic request to pandas DataFrame with correct feature order
    
    Order MUST match training: numerical_features_final + categorical_features_engineered
    Feature order from training (after correlation selection):
    - Numerical (6): max_power_bhp, year, engine_cc, power_per_cc, seats, km_driven
    - Categorical (5): brand, owner, fuel, seller_type, transmission
    Note: mileage_mpg was EXCLUDED (correlation < 0.1)
    """
    # Feature order (CRITICAL - must match training exactly!)
    feature_order = [
        'max_power_bhp', 'year', 'engine_cc', 'power_per_cc', 'seats', 'km_driven',
        'brand', 'owner', 'fuel', 'seller_type', 'transmission'
    ]
    
    # Convert to dict
    data = request.model_dump()
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([data])[feature_order]
    
    logger.info(f"📊 Input DataFrame prepared with shape {df.shape}")
    logger.debug(f"Features: {df.columns.tolist()}")
    
    return df


# ========== API ENDPOINTS ==========

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API info"""
    return {
        "message": "CarPriceML API - Prédiction du prix des voitures d'occasion",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Vérifier le statut de l'API",
    description="Endpoint pour vérifier si l'API et le modèle ML sont opérationnels"
)
async def health_check():
    """
    Health check endpoint
    Returns API status and model availability
    """
    global ml_model, model_metadata
    
    # Determine status
    if ml_model is not None:
        status_value = "healthy"
        message = "API is operational and model is loaded"
        model_loaded = True
    else:
        status_value = "unhealthy"
        message = "API is running but model is not loaded"
        model_loaded = False
    
    return HealthCheckResponse(
        status=status_value,
        model_loaded=model_loaded,
        model_version=model_metadata.get('version', 'unknown') if model_metadata else 'unknown',
        timestamp=datetime.now(),
        message=message
    )


@app.post(
    "/predict",
    response_model=CarPricePredictionResponse,
    tags=["Prediction"],
    summary="Prédire le prix d'une voiture",
    description="Prédit le prix d'une voiture d'occasion en MAD à partir de ses caractéristiques",
    responses={
        200: {"description": "Prédiction réussie"},
        400: {"model": ErrorResponse, "description": "Erreur de validation des données"},
        500: {"model": ErrorResponse, "description": "Erreur interne du serveur"},
        503: {"model": ErrorResponse, "description": "Modèle non disponible"}
    }
)
async def predict_price(request: CarPricePredictionRequest):
    """
    Predict car price endpoint
    
    Args:
        request: CarPricePredictionRequest with all required features
        
    Returns:
        CarPricePredictionResponse with predicted price in MAD
    """
    global ml_model, model_metadata
    
    # Check if model is loaded
    if ml_model is None:
        logger.error("❌ Model not loaded, cannot make prediction")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please check server logs."
        )
    
    try:
        # Log request
        logger.info(f"📥 Received prediction request for {request.brand} {request.year}")
        
        # Prepare input DataFrame
        input_df = prepare_input_dataframe(request)
        
        # Make prediction with the full pipeline
        prediction = ml_model.predict(input_df)[0]
        
        # Calculate prediction confidence using individual tree predictions
        # Extract the RandomForest model from the pipeline
        rf_model = ml_model.named_steps['model']
        
        # Get predictions from all individual trees
        tree_predictions = np.array([tree.predict(ml_model.named_steps['preprocessor'].transform(input_df))[0] 
                                     for tree in rf_model.estimators_])
        
        # Calculate standard deviation (uncertainty measure)
        prediction_std = float(np.std(tree_predictions))
        
        # Calculate 95% confidence interval (mean ± 1.96 * std)
        confidence_interval = {
            "lower": round(float(prediction - 1.96 * prediction_std), 2),
            "upper": round(float(prediction + 1.96 * prediction_std), 2)
        }
        
        # Calculate confidence score (0-100)
        # Lower std = higher confidence. Normalize based on price magnitude.
        relative_std = (prediction_std / prediction) * 100  # Coefficient of variation %
        confidence_score = round(max(0, min(100, 100 - relative_std)), 2)
        
        # Round to 2 decimal places
        predicted_price = round(float(prediction), 2)
        prediction_std = round(prediction_std, 2)
        
        logger.info(f"✅ Prediction successful: {predicted_price:,.2f} MAD (±{prediction_std:,.2f}, confidence: {confidence_score}%)")
        
        # Prepare response with ALL input features
        response = CarPricePredictionResponse(
            predicted_price_mad=predicted_price,
            confidence_interval=confidence_interval,
            prediction_std=prediction_std,
            confidence_score=confidence_score,
            model_version=model_metadata.get('version', 'unknown'),
            prediction_timestamp=datetime.now(),
            input_features={
                "max_power_bhp": request.max_power_bhp,
                "year": request.year,
                "engine_cc": request.engine_cc,
                "power_per_cc": request.power_per_cc,
                "seats": request.seats,
                "km_driven": request.km_driven,
                "brand": request.brand,
                "owner": request.owner,
                "fuel": request.fuel,
                "seller_type": request.seller_type,
                "transmission": request.transmission
            }
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during prediction: {str(e)}"
        )


# ========== EXCEPTION HANDLERS ==========
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Custom handler for uncaught exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ========== RUN SERVER ==========
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting FastAPI server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
