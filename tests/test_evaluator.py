"""Unit tests for FidelityEvaluator and quality checks."""

import pytest
from synth.evaluator import FidelityEvaluator


def test_evaluator_identical_datasets():
    evaluator = FidelityEvaluator()
    data = [
        {"role": "dev", "xp": 5.0, "rating": 90.0},
        {"role": "qa", "xp": 3.0, "rating": 85.0},
        {"role": "dev", "xp": 8.0, "rating": 95.0},
        {"role": "manager", "xp": 10.0, "rating": 88.0},
    ]

    res = evaluator.evaluate(real_records=data, synthetic_records=data)
    # Correlation error between identical sets is 0
    assert res.correlation_error == 0.0
    # Fidelity should be close to 100%
    assert res.fidelity_score >= 95.0
    # Memorization risk: identical records mean high memorization risk (1.0)
    assert res.memorization_risk_score == 1.0
    # Thus quality gate will fail because memorization risk > 0.05
    assert not res.passed_quality_gate


def test_evaluator_quality_gate_pass():
    evaluator = FidelityEvaluator()
    real = [
        {"score": 50.0, "cat": "A"},
        {"score": 60.0, "cat": "B"},
        {"score": 70.0, "cat": "A"},
        {"score": 80.0, "cat": "B"},
    ]
    # Synthetic has similar distribution but different values (no exact replica)
    synth = [
        {"score": 52.0, "cat": "A"},
        {"score": 58.0, "cat": "B"},
        {"score": 68.0, "cat": "A"},
        {"score": 82.0, "cat": "B"},
    ]

    res = evaluator.evaluate(real_records=real, synthetic_records=synth)
    assert res.fidelity_score >= 80.0
    assert res.memorization_risk_score == 0.0
    assert res.passed_quality_gate


def test_evaluator_invalid_input():
    evaluator = FidelityEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate([], [])

    with pytest.raises(ValueError):
        evaluator.evaluate([{"a": 1}], [{"b": 2}])
