from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.pipeline.prediction_pipeline import PredictionPipeline
from typing import List, Dict, Optional
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata for Swagger
app = FastAPI(
    title="Emotion Classification API",
    description="""
    This API provides emotion classification for text using a transformer-based model.
    
    Features:
    * Single text prediction
    * Batch prediction
    * Model information
    * Health check
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path
MODEL_PATH = "artifacts/model_trainer/best_model"

# Initialize predictor globally
try:
    predictor = PredictionPipeline(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., 
        example="I am feeling really happy today!",
        description="The text to analyze for emotion"
    )

class BatchTextInput(BaseModel):
    texts: List[str] = Field(...,
        example=["I am happy!", "I am sad."],
        description="List of texts to analyze"
    )

class PredictionResponse(BaseModel):
    text: str = Field(..., description="Input text")
    emotion: str = Field(..., description="Predicted emotion")
    confidence: float = Field(..., description="Prediction confidence score")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int = Field(..., description="Total number of texts processed")
    average_confidence: float = Field(..., description="Average confidence score")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    supported_emotions: List[str] = Field(..., description="List of supported emotions")
    max_text_length: int = Field(..., description="Maximum text length supported")
    version: str = Field(..., description="Model version")

class HealthCheck(BaseModel):
    status: str = Field(..., description="API status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Model loading status")

# API endpoints
@app.get("/", 
    response_model=HealthCheck,
    tags=["Health"],
    summary="Health Check",
    description="Check if the API is running and model is loaded"
)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None
    }

@app.get("/model-info",
    response_model=ModelInfo,
    tags=["Model"],
    summary="Get Model Information",
    description="Get information about the emotion classification model"
)
async def get_model_info():
    return {
        "model_name": "Emotion Classifier",
        "supported_emotions": list(predictor.emotion_mapping.values()),
        "max_text_length": 512,
        "version": "1.0.0"
    }

@app.post("/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Single Text Prediction",
    description="Predict emotion for a single text input"
)
async def predict_emotion(
    input_data: TextInput
):
    try:
        logger.info(f"Received prediction request for text: {input_data.text}")
        result = predictor.predict(input_data.text)
        result["timestamp"] = datetime.now().isoformat()
        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch Text Prediction",
    description="Predict emotions for multiple texts"
)
async def predict_batch(
    input_data: BatchTextInput
):
    try:
        predictions = []
        total_confidence = 0

        for text in input_data.texts:
            result = predictor.predict(text)
            result["timestamp"] = datetime.now().isoformat()
            predictions.append(result)
            total_confidence += result["confidence"]

        return {
            "predictions": predictions,
            "total_processed": len(predictions),
            "average_confidence": total_confidence / len(predictions) if predictions else 0
        }
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    """Function to start the server"""
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise e

if __name__ == "__main__":
    start_server() 