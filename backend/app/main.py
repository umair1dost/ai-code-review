"""
AI-Powered Code Review and Bug Prediction System
FastAPI Backend — Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
from app.api.health import router as health_router

app = FastAPI(
    title="AI Code Review API",
    description=(
        "REST API for the AI-Powered Code Review and Bug Prediction System. "
        "Exposes endpoints for metric-based Random Forest prediction and "
        "source-code-based LSTM prediction, combined via a soft-voting ensemble."
    ),
    version="1.0.0",
    contact={
        "name": "IMSciences FYP Team — Muhammad Nasir Khan, Ahmed Khan Khisro, Umair Dost",
        "email": "fyp@imsciences.edu.pk",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, prefix="/predict", tags=["Prediction"])
