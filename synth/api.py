"""REST API service for differentially private synthetic data generation and evaluation."""

from __future__ import annotations

import time
from fastapi import FastAPI, HTTPException
from synth.evaluator import FidelityEvaluator
from synth.generator import TabularSynthesizer
from synth.models import (
    EvaluateRequest,
    EvaluateResponse,
    PrivacyReport,
    SynthesizeRequest,
    SynthesizeResponse,
)
from synth.privacy import PrivacyBudget

app = FastAPI(
    title="Privacy-Preserving Tabular Synthetic Data Generator",
    description="Mathematical Differential Privacy engine with Gaussian/Laplace mechanisms and fidelity evaluation.",
    version="1.0.0",
)

global_budget = PrivacyBudget(total_epsilon=100.0, total_delta=1e-2)
evaluator = FidelityEvaluator()


@app.get("/healthz")
def health_check():
    """Service health and liveness probe."""
    return {
        "status": "healthy",
        "service": "synthetic-data-gen",
        "version": "1.0.0",
        "budget_remaining_eps": global_budget.remaining_epsilon,
    }


@app.post("/api/v1/synthesize", response_model=SynthesizeResponse)
def synthesize_data(request: SynthesizeRequest):
    """Generate differentially private synthetic tabular records from source data."""
    start_time = time.perf_counter()
    try:
        synth = TabularSynthesizer(budget=global_budget)
        synth.fit(
            records=request.records,
            columns=request.columns,
            epsilon=request.epsilon,
            delta=request.delta,
            random_seed=request.random_seed,
        )
        samples = synth.generate(
            sample_size=request.sample_size,
            random_seed=request.random_seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    report = PrivacyReport(
        spent_epsilon=round(global_budget.spent_epsilon, 4),
        spent_delta=round(global_budget.spent_delta, 6),
        remaining_epsilon=round(global_budget.remaining_epsilon, 4),
        total_epsilon=round(global_budget.total_epsilon, 4),
        is_exhausted=global_budget.is_exhausted,
    )

    return SynthesizeResponse(
        records=samples,
        sample_size=len(samples),
        privacy_report=report,
        generation_time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/v1/evaluate", response_model=EvaluateResponse)
def evaluate_fidelity(request: EvaluateRequest):
    """Evaluate statistical fidelity and privacy leakage risk between real and synthetic data."""
    try:
        return evaluator.evaluate(
            real_records=request.real_records,
            synthetic_records=request.synthetic_records,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/privacy/budget", response_model=PrivacyReport)
def get_privacy_budget():
    """Retrieve cumulative privacy budget consumption."""
    return PrivacyReport(
        spent_epsilon=round(global_budget.spent_epsilon, 4),
        spent_delta=round(global_budget.spent_delta, 6),
        remaining_epsilon=round(global_budget.remaining_epsilon, 4),
        total_epsilon=round(global_budget.total_epsilon, 4),
        is_exhausted=global_budget.is_exhausted,
    )


@app.post("/api/v1/privacy/reset")
def reset_privacy_budget():
    """Reset spent privacy budget counter."""
    global_budget.reset()
    return {"status": "budget_reset", "remaining_epsilon": global_budget.remaining_epsilon}


# Backward compatibility with initial mock
@app.get("/api/v1/generate")
def legacy_generate(count: int = 10, epsilon: float = 0.5):
    """Legacy endpoint for generating basic gaussian samples."""
    synth = TabularSynthesizer(budget=global_budget)
    return {
        "samples": synth.generate_gaussian_samples(
            sample_size=count, epsilon=epsilon
        )
    }
