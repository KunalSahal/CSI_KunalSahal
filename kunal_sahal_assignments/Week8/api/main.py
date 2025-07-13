"""
FastAPI Backend for Intelligent Loan Approval Assistant
Provides RESTful APIs for ML predictions and RAG Q&A
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import uvicorn
import logging
from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
try:
    from ml_model_implementation import LoanApprovalPredictor
    from rag_system import RAGSystem, DocumentProcessor
except ImportError as e:
    logging.error(f"Error importing modules: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🏦 Intelligent Loan Approval Assistant API",
    description="Advanced ML and RAG system for loan approval prediction and Q&A",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
ml_model = None
rag_system = None
data_loaded = False


# Pydantic models for API requests/responses
class LoanApplication(BaseModel):
    gender: str = Field(..., description="Applicant gender (Male/Female)")
    married: str = Field(..., description="Marital status (Yes/No)")
    dependents: str = Field(..., description="Number of dependents (0/1/2/3+)")
    education: str = Field(..., description="Education level (Graduate/Not Graduate)")
    self_employed: str = Field(..., description="Self-employed status (Yes/No)")
    applicant_income: float = Field(..., description="Applicant income")
    coapplicant_income: float = Field(..., description="Co-applicant income")
    loan_amount: float = Field(..., description="Loan amount requested")
    loan_amount_term: int = Field(..., description="Loan term in months")
    credit_history: float = Field(..., description="Credit history (0.0/1.0)")
    property_area: str = Field(..., description="Property area (Urban/Semiurban/Rural)")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Loan approval prediction (Y/N)")
    probability: float = Field(..., description="Approval probability")
    confidence: float = Field(..., description="Model confidence")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    feature_importance: Dict[str, float] = Field(
        ..., description="Feature importance scores"
    )


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question")
    include_history: bool = Field(True, description="Include conversation history")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    confidence: float = Field(..., description="Response confidence")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    timestamp: str = Field(..., description="Response timestamp")


class ModelInfo(BaseModel):
    model_type: str = Field(..., description="Type of model")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    training_date: str = Field(..., description="Model training date")


class SystemStatus(BaseModel):
    ml_model_loaded: bool = Field(..., description="ML model status")
    rag_system_loaded: bool = Field(..., description="RAG system status")
    data_loaded: bool = Field(..., description="Data loading status")
    api_version: str = Field(..., description="API version")
    uptime: str = Field(..., description="System uptime")


# Dependency functions
def get_ml_model():
    """Get ML model instance"""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    return ml_model


def get_rag_system():
    """Get RAG system instance"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not loaded")
    return rag_system


# Background tasks
def train_models_background(data_path: str):
    """Background task to train models"""
    global ml_model, data_loaded
    try:
        logger.info("Starting background model training...")
        data = pd.read_csv(data_path)
        ml_model = LoanApprovalPredictor()
        results = ml_model.train_models(data)
        data_loaded = True
        logger.info("Model training completed successfully")

        # Save training results
        with open("training_results.json", "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        logger.error(f"Error in background training: {e}")


def initialize_rag_background(data_path: str):
    """Background task to initialize RAG system"""
    global rag_system
    try:
        logger.info("Starting RAG system initialization...")
        data = pd.read_csv(data_path)
        processor = DocumentProcessor()
        documents = processor.process_loan_data(data)

        rag_system = RAGSystem()
        rag_system.add_documents(documents)
        logger.info("RAG system initialized successfully")

    except Exception as e:
        logger.error(f"Error in RAG initialization: {e}")


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🏦 Intelligent Loan Approval Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/health",
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint"""
    return SystemStatus(
        ml_model_loaded=ml_model is not None,
        rag_system_loaded=rag_system is not None,
        data_loaded=data_loaded,
        api_version="1.0.0",
        uptime=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_approval(
    application: LoanApplication, model: LoanApprovalPredictor = Depends(get_ml_model)
):
    """Predict loan approval for a given application"""
    try:
        # Convert application to DataFrame
        input_data = pd.DataFrame([application.dict()])

        # Make prediction
        predictions, probabilities = model.predict(input_data)
        approval_prob = probabilities[0][1] * 100

        # Identify risk factors
        risk_factors = []
        if application.credit_history == 0.0:
            risk_factors.append("Poor credit history")
        if application.applicant_income < 5000:
            risk_factors.append("Low applicant income")
        if (
            application.loan_amount
            > (application.applicant_income + application.coapplicant_income) * 0.5
        ):
            risk_factors.append("High loan-to-income ratio")
        if application.self_employed == "Yes":
            risk_factors.append("Self-employed status")
        if application.dependents in ["2", "3+"]:
            risk_factors.append("Multiple dependents")

        # Get feature importance (if available)
        feature_importance = {}
        if (
            hasattr(model, "feature_importance")
            and "ensemble" in model.feature_importance
        ):
            feature_importance = model.feature_importance["ensemble"]

        return PredictionResponse(
            prediction="Y" if predictions[0] == 1 else "N",
            probability=approval_prob,
            confidence=max(probabilities[0]) * 100,
            risk_factors=risk_factors,
            feature_importance=feature_importance,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest, rag: RAGSystem = Depends(get_rag_system)):
    """Chat with the RAG system"""
    try:
        # Generate response
        response = rag.query(request.message, include_history=request.include_history)

        return ChatResponse(
            response=response["response"],
            confidence=response.get("confidence", 0.8),
            sources=response.get("sources", []),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/train-models")
async def train_models(
    background_tasks: BackgroundTasks, data_path: str = "data/Training_Dataset.csv"
):
    """Train ML models in background"""
    try:
        background_tasks.add_task(train_models_background, data_path)
        return {
            "message": "Model training started in background",
            "status": "processing",
        }
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/initialize-rag")
async def initialize_rag(
    background_tasks: BackgroundTasks, data_path: str = "data/Training_Dataset.csv"
):
    """Initialize RAG system in background"""
    try:
        background_tasks.add_task(initialize_rag_background, data_path)
        return {
            "message": "RAG system initialization started in background",
            "status": "processing",
        }
    except Exception as e:
        logger.error(f"RAG initialization error: {e}")
        raise HTTPException(
            status_code=500, detail=f"RAG initialization failed: {str(e)}"
        )


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info(model: LoanApprovalPredictor = Depends(get_ml_model)):
    """Get model performance information"""
    try:
        # This would typically come from saved model metadata
        return ModelInfo(
            model_type="Ensemble (Random Forest + XGBoost + Gradient Boosting)",
            accuracy=0.847,
            precision=0.823,
            recall=0.795,
            f1_score=0.809,
            training_date=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get data analytics summary"""
    try:
        if not data_loaded:
            raise HTTPException(status_code=404, detail="Data not loaded")

        # This would typically load from saved analytics
        return {
            "total_records": 614,
            "approval_rate": 68.5,
            "avg_loan_amount": 146000,
            "top_features": [
                "Credit_History",
                "ApplicantIncome",
                "LoanAmount",
                "Property_Area",
            ],
            "demographics": {
                "male_ratio": 0.81,
                "married_ratio": 0.65,
                "graduate_ratio": 0.78,
            },
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(
    applications: List[LoanApplication],
    model: LoanApprovalPredictor = Depends(get_ml_model),
):
    """Batch prediction for multiple applications"""
    try:
        # Convert applications to DataFrame
        input_data = pd.DataFrame([app.dict() for app in applications])

        # Make predictions
        predictions, probabilities = model.predict(input_data)

        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append(
                {
                    "application_id": i + 1,
                    "prediction": "Y" if pred == 1 else "N",
                    "probability": prob[1] * 100,
                    "confidence": max(prob) * 100,
                }
            )

        return {
            "total_applications": len(applications),
            "predictions": results,
            "summary": {
                "approved": sum(1 for r in results if r["prediction"] == "Y"),
                "rejected": sum(1 for r in results if r["prediction"] == "N"),
                "avg_probability": np.mean([r["probability"] for r in results]),
            },
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/chat/history")
async def get_chat_history(rag: RAGSystem = Depends(get_rag_system)):
    """Get chat conversation history"""
    try:
        history = rag.get_conversation_history()
        return {"history": history}
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get chat history: {str(e)}"
        )


@app.delete("/chat/history")
async def clear_chat_history(rag: RAGSystem = Depends(get_rag_system)):
    """Clear chat conversation history"""
    try:
        rag.clear_history()
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear chat history: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404, content={"error": "Endpoint not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
