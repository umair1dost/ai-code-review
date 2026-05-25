"""
Pydantic request and response schemas for the prediction API.
"""

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /predict/metrics  — structured metric payload
# ---------------------------------------------------------------------------

class MetricsPayload(BaseModel):
    """Software metrics extracted from a single code module."""

    # Size metrics
    loc: int = Field(..., ge=0, description="Lines of code")
    lloc: int = Field(..., ge=0, description="Logical lines of code")
    num_methods: int = Field(..., ge=0, description="Number of methods/functions")
    num_attributes: int = Field(..., ge=0, description="Number of class attributes")

    # Complexity metrics
    cyclomatic_complexity: float = Field(..., ge=0, description="McCabe cyclomatic complexity")
    halstead_volume: float = Field(..., ge=0, description="Halstead volume")
    halstead_effort: float = Field(..., ge=0, description="Halstead effort")
    halstead_difficulty: float = Field(..., ge=0, description="Halstead difficulty")

    # Object-oriented metrics (CK suite)
    cbo: float = Field(..., ge=0, description="Coupling Between Objects")
    wmc: float = Field(..., ge=0, description="Weighted Methods per Class")
    rfc: float = Field(..., ge=0, description="Response For a Class")
    lcom: float = Field(..., ge=0, le=1, description="Lack of Cohesion of Methods (0–1)")
    dit: int = Field(..., ge=0, description="Depth of Inheritance Tree")
    noc: int = Field(..., ge=0, description="Number of Children")

    # Process metrics
    num_authors: int = Field(..., ge=0, description="Number of distinct authors")
    code_churn: int = Field(..., ge=0, description="Total lines added + deleted")
    prior_defects: int = Field(..., ge=0, description="Number of prior bug-fix commits")
    module_age_days: int = Field(..., ge=0, description="Age of module in days")

    # Dependency metrics
    fan_in: int = Field(..., ge=0, description="Number of modules that depend on this one")
    fan_out: int = Field(..., ge=0, description="Number of modules this one depends on")
    instability: float = Field(..., ge=0, le=1, description="Instability index (fan_out / (fan_in + fan_out))")

    class Config:
        json_schema_extra = {
            "example": {
                "loc": 245, "lloc": 198, "num_methods": 12, "num_attributes": 8,
                "cyclomatic_complexity": 18, "halstead_volume": 4200,
                "halstead_effort": 95000, "halstead_difficulty": 22.6,
                "cbo": 12, "wmc": 24, "rfc": 45, "lcom": 0.73, "dit": 3, "noc": 1,
                "num_authors": 4, "code_churn": 312, "prior_defects": 2, "module_age_days": 540,
                "fan_in": 6, "fan_out": 9, "instability": 0.6,
            }
        }


class FeatureImportance(BaseModel):
    feature: str
    shap_value: float


class MetricsPredictionResponse(BaseModel):
    bug_probability: float = Field(..., ge=0, le=1)
    prediction: Literal["BUGGY", "CLEAN"]
    severity: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    top_features: List[FeatureImportance]
    review_suggestion: str


# ---------------------------------------------------------------------------
# /predict/code  — raw source code payload
# ---------------------------------------------------------------------------

class CodePayload(BaseModel):
    source_code: str = Field(..., min_length=10, description="Raw source code string")
    language: Literal["python", "java"] = Field("python", description="Programming language")

    class Config:
        json_schema_extra = {
            "example": {
                "source_code": "def process_data(df):\n    # TODO: add validation\n    result = []\n    for i in range(len(df)):\n        for j in range(len(df)):\n            result.append(df[i] * df[j])\n    return result",
                "language": "python",
            }
        }


class CodePredictionResponse(BaseModel):
    bug_probability: float = Field(..., ge=0, le=1)
    prediction: Literal["BUGGY", "CLEAN"]
    severity: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    code_smells: List[str]
    review_suggestion: str
    ensemble_weights: dict = Field(
        default={"random_forest": 0.55, "lstm": 0.45},
        description="Weights used in soft-voting ensemble",
    )
