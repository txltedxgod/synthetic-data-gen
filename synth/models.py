"""Data models and API schemas for differential privacy synthesis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ColumnType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL_FLOAT = "numerical_float"
    NUMERICAL_INT = "numerical_int"


class ColumnSpec(BaseModel):
    name: str
    col_type: ColumnType
    bounds: Optional[Tuple[float, float]] = None
    categories: Optional[List[str]] = None


class SynthesizeRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1, description="Source tabular records")
    columns: Optional[List[ColumnSpec]] = Field(
        default=None, description="Optional explicit column schemas"
    )
    sample_size: int = Field(default=50, ge=1, le=100000, description="Number of synthetic rows to generate")
    epsilon: float = Field(default=1.0, gt=0, le=50.0, description="Privacy budget epsilon")
    delta: float = Field(default=1e-5, ge=0.0, lt=1.0, description="Privacy budget delta (0 for pure DP)")
    random_seed: Optional[int] = Field(default=None, description="Deterministic PRNG seed")


class PrivacyReport(BaseModel):
    spent_epsilon: float
    spent_delta: float
    remaining_epsilon: float
    total_epsilon: float
    is_exhausted: bool


class SynthesizeResponse(BaseModel):
    records: List[Dict[str, Any]]
    sample_size: int
    privacy_report: PrivacyReport
    generation_time_ms: float


class EvaluateRequest(BaseModel):
    real_records: List[Dict[str, Any]] = Field(..., min_length=2)
    synthetic_records: List[Dict[str, Any]] = Field(..., min_length=2)


class EvaluateResponse(BaseModel):
    marginal_divergences: Dict[str, float] = Field(
        description="Per-column divergence (Wasserstein or Total Variation Distance)"
    )
    correlation_error: float = Field(
        description="Frobenius norm distance between Pearson correlation matrices"
    )
    fidelity_score: float = Field(
        ge=0.0, le=100.0, description="Overall fidelity quality score (0-100%)"
    )
    memorization_risk_score: float = Field(
        ge=0.0, le=1.0, description="Risk of identical or near-identical record replication"
    )
    passed_quality_gate: bool = Field(
        description="True if fidelity >= 60% and memorization risk <= 0.05"
    )
