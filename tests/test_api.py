"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from synth.api import app

client = TestClient(app)


def test_healthz_endpoint():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "synthetic-data-gen"


def test_synthesize_endpoint():
    payload = {
        "records": [
            {"age": 25, "city": "Berlin", "income": 45000.0},
            {"age": 35, "city": "Munich", "income": 65000.0},
            {"age": 45, "city": "Hamburg", "income": 75000.0},
            {"age": 55, "city": "Berlin", "income": 85000.0},
        ],
        "sample_size": 10,
        "epsilon": 2.0,
        "delta": 1e-5,
        "random_seed": 42,
    }

    resp = client.post("/api/v1/synthesize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["records"]) == 10
    assert data["sample_size"] == 10
    assert "privacy_report" in data
    assert data["privacy_report"]["spent_epsilon"] > 0
    assert data["generation_time_ms"] >= 0


def test_evaluate_endpoint():
    payload = {
        "real_records": [
            {"val": 10.0, "type": "A"},
            {"val": 20.0, "type": "B"},
            {"val": 30.0, "type": "A"},
        ],
        "synthetic_records": [
            {"val": 12.0, "type": "A"},
            {"val": 21.0, "type": "B"},
            {"val": 29.0, "type": "A"},
        ],
    }

    resp = client.post("/api/v1/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "marginal_divergences" in data
    assert "fidelity_score" in data
    assert "passed_quality_gate" in data


def test_privacy_budget_endpoints():
    resp = client.get("/api/v1/privacy/budget")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_epsilon" in data

    reset_resp = client.post("/api/v1/privacy/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "budget_reset"


def test_legacy_generate_endpoint():
    resp = client.get("/api/v1/generate?count=8&epsilon=1.0")
    assert resp.status_code == 200
    data = resp.json()
    assert "samples" in data
    assert len(data["samples"]) == 8
