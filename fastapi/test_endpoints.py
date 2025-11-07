"""
Unit Tests for CarPriceML FastAPI Endpoints
Tests /health and /predict endpoints
"""

import pytest
from fastapi.testclient import TestClient
import json
import os
from pathlib import Path

# Set up model paths before importing app
os.environ["MODEL_PATH"] = str(Path("../models/rfmodel.joblib").resolve())
os.environ["PREPROCESSOR_PATH"] = str(Path("../models/preprocessor.joblib").resolve())

from main import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test suite for /health endpoint"""
    
    def test_health_endpoint_exists(self):
        """Test that /health endpoint is accessible"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_json(self):
        """Test that /health returns JSON"""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"
    
    def test_health_structure(self):
        """Test that /health returns correct structure"""
        response = client.get("/health")
        data = response.json()
        
        # Check required fields
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
    
    def test_health_status_healthy(self):
        """Test that status is 'healthy'"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_model_loaded(self):
        """Test that model is loaded"""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
    
    def test_health_model_version(self):
        """Test that model version is present"""
        response = client.get("/health")
        data = response.json()
        assert data["model_version"] == "3.0_optimized_outliers"


class TestPredictEndpoint:
    """Test suite for /predict endpoint"""
    
    @pytest.fixture
    def valid_payload(self):
        """Fixture providing valid prediction payload"""
        return {
            "max_power_bhp": 80.0,
            "year": 2015,
            "engine_cc": 1200,
            "power_per_cc": 0.0667,
            "seats": 5,
            "km_driven": 50000,
            "brand": "Maruti",
            "owner": "First",
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual"
        }
    
    def test_predict_endpoint_exists(self, valid_payload):
        """Test that /predict endpoint is accessible"""
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
    
    def test_predict_returns_json(self, valid_payload):
        """Test that /predict returns JSON"""
        response = client.post("/predict", json=valid_payload)
        assert response.headers["content-type"] == "application/json"
    
    def test_predict_response_structure(self, valid_payload):
        """Test that /predict returns correct structure"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        
        # Check required fields
        assert "predicted_price_mad" in data
        assert "confidence_interval" in data
        assert "prediction_std" in data
        assert "confidence_score" in data
        assert "model_version" in data
        assert "prediction_timestamp" in data
        assert "input_features" in data
    
    def test_predict_price_is_positive(self, valid_payload):
        """Test that predicted price is positive"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert data["predicted_price_mad"] > 0
    
    def test_predict_confidence_interval_structure(self, valid_payload):
        """Test confidence interval has lower and upper bounds"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        
        ci = data["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] < ci["upper"]
    
    def test_predict_confidence_score_range(self, valid_payload):
        """Test confidence score is between 0 and 100"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        
        score = data["confidence_score"]
        assert 0 <= score <= 100
    
    def test_predict_input_features_echoed(self, valid_payload):
        """Test that input features are echoed back"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        
        input_features = data["input_features"]
        assert input_features["brand"] == "Maruti"
        assert input_features["year"] == 2015
        assert input_features["fuel"] == "Petrol"
    
    def test_predict_missing_field(self):
        """Test that missing required field returns validation error"""
        invalid_payload = {
            "max_power_bhp": 80.0,
            "year": 2015,
            # Missing engine_cc and other fields
        }
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_year(self, valid_payload):
        """Test that invalid year returns validation error"""
        invalid_payload = valid_payload.copy()
        invalid_payload["year"] = 1900  # Too old
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422
    
    def test_predict_invalid_fuel_type(self, valid_payload):
        """Test that invalid fuel type returns validation error"""
        invalid_payload = valid_payload.copy()
        invalid_payload["fuel"] = "Nuclear"  # Invalid
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422
    
    def test_predict_invalid_brand(self, valid_payload):
        """Test that invalid brand returns validation error"""
        invalid_payload = valid_payload.copy()
        invalid_payload["brand"] = "InvalidBrand123"
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422
    
    def test_predict_negative_power(self, valid_payload):
        """Test that negative power returns validation error"""
        invalid_payload = valid_payload.copy()
        invalid_payload["max_power_bhp"] = -50.0
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422


class TestPredictVariousScenarios:
    """Test different car scenarios"""
    
    def test_predict_luxury_car(self):
        """Test prediction for luxury car"""
        payload = {
            "max_power_bhp": 150.0,
            "year": 2020,
            "engine_cc": 2000,
            "power_per_cc": 0.075,
            "seats": 5,
            "km_driven": 20000,
            "brand": "BMW",
            "owner": "First",
            "fuel": "Diesel",
            "seller_type": "Dealer",
            "transmission": "Automatic"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Luxury car should have higher price
        assert data["predicted_price_mad"] > 50000
    
    def test_predict_old_car(self):
        """Test prediction for old car"""
        payload = {
            "max_power_bhp": 60.0,
            "year": 2005,
            "engine_cc": 1000,
            "power_per_cc": 0.06,
            "seats": 5,
            "km_driven": 150000,
            "brand": "Maruti",
            "owner": "Third",
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Old car should have lower price
        assert data["predicted_price_mad"] < 50000
    
    def test_predict_diesel_car(self):
        """Test prediction for diesel car"""
        payload = {
            "max_power_bhp": 90.0,
            "year": 2018,
            "engine_cc": 1500,
            "power_per_cc": 0.06,
            "seats": 5,
            "km_driven": 40000,
            "brand": "Toyota",
            "owner": "First",
            "fuel": "Diesel",
            "seller_type": "Dealer",
            "transmission": "Manual"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "predicted_price_mad" in response.json()
    
    def test_predict_automatic_transmission(self):
        """Test prediction for automatic transmission"""
        payload = {
            "max_power_bhp": 100.0,
            "year": 2019,
            "engine_cc": 1500,
            "power_per_cc": 0.0667,
            "seats": 5,
            "km_driven": 30000,
            "brand": "Hyundai",
            "owner": "First",
            "fuel": "Petrol",
            "seller_type": "Dealer",
            "transmission": "Automatic"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "predicted_price_mad" in response.json()


class TestResponseConsistency:
    """Test response consistency and data types"""
    
    def test_predict_response_data_types(self):
        """Test that response contains correct data types"""
        payload = {
            "max_power_bhp": 80.0,
            "year": 2015,
            "engine_cc": 1200,
            "power_per_cc": 0.0667,
            "seats": 5,
            "km_driven": 50000,
            "brand": "Maruti",
            "owner": "First",
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual"
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        
        # Check data types
        assert isinstance(data["predicted_price_mad"], (int, float))
        assert isinstance(data["confidence_score"], (int, float))
        assert isinstance(data["prediction_std"], (int, float))
        assert isinstance(data["model_version"], str)
        assert isinstance(data["prediction_timestamp"], str)
        assert isinstance(data["confidence_interval"], dict)
        assert isinstance(data["input_features"], dict)
    
    def test_multiple_predictions_consistency(self):
        """Test that same input gives same prediction"""
        payload = {
            "max_power_bhp": 80.0,
            "year": 2015,
            "engine_cc": 1200,
            "power_per_cc": 0.0667,
            "seats": 5,
            "km_driven": 50000,
            "brand": "Maruti",
            "owner": "First",
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual"
        }
        
        # Make two predictions
        response1 = client.post("/predict", json=payload)
        response2 = client.post("/predict", json=payload)
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Same input should give same prediction
        assert data1["predicted_price_mad"] == data2["predicted_price_mad"]


class TestEdgeCases:
    """Test edge cases and boundary values"""
    
    def test_predict_max_values(self):
        """Test with maximum allowed values"""
        payload = {
            "max_power_bhp": 200.0,
            "year": 2025,
            "engine_cc": 2500,
            "power_per_cc": 0.08,
            "seats": 7,
            "km_driven": 200000,
            "brand": "Mercedes-Benz",
            "owner": "First",
            "fuel": "Diesel",
            "seller_type": "Trustmark Dealer",
            "transmission": "Automatic"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    
    def test_predict_min_values(self):
        """Test with minimum allowed values"""
        payload = {
            "max_power_bhp": 30.0,
            "year": 1983,
            "engine_cc": 500,
            "power_per_cc": 0.06,
            "seats": 2,
            "km_driven": 0,
            "brand": "Maruti",
            "owner": "Fourth & Above",
            "fuel": "CNG",
            "seller_type": "Individual",
            "transmission": "Manual"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
