"""Unit tests for TabularSynthesizer with Differential Privacy."""

import numpy as np
import pytest

from synth.generator import (
    SyntheticDataGenerator,
    TabularSynthesizer,
    _ensure_positive_semidefinite,
)
from synth.models import ColumnSpec, ColumnType
from synth.privacy import PrivacyBudget


def test_positive_semidefinite_projection():
    # Matrix with negative eigenvalue
    non_psd = np.array([[1.0, 3.0], [3.0, 1.0]])
    eigvals_orig = np.linalg.eigvalsh(non_psd)
    assert np.min(eigvals_orig) < 0  # 1-3 = -2

    psd = _ensure_positive_semidefinite(non_psd, min_eigenvalue=1e-3)
    eigvals_fixed = np.linalg.eigvalsh(psd)
    assert np.all(eigvals_fixed >= 1e-3)
    assert np.allclose(psd, psd.T)


def test_tabular_synthesizer_categorical():
    records = [
        {"department": "Engineering"},
        {"department": "Engineering"},
        {"department": "Engineering"},
        {"department": "Marketing"},
        {"department": "Sales"},
    ]

    synth = TabularSynthesizer()
    synth.fit(records=records, epsilon=2.0, random_seed=42)
    samples = synth.generate(sample_size=30, random_seed=42)

    assert len(samples) == 30
    valid_departments = {"Engineering", "Marketing", "Sales"}
    for row in samples:
        assert row["department"] in valid_departments


def test_tabular_synthesizer_numerical_bounds():
    records = [
        {"age": 20, "salary": 30000.0},
        {"age": 30, "salary": 50000.0},
        {"age": 40, "salary": 70000.0},
        {"age": 50, "salary": 90000.0},
    ]

    columns = [
        ColumnSpec(name="age", col_type=ColumnType.NUMERICAL_INT, bounds=(18.0, 65.0)),
        ColumnSpec(name="salary", col_type=ColumnType.NUMERICAL_FLOAT, bounds=(25000.0, 100000.0)),
    ]

    synth = TabularSynthesizer()
    synth.fit(records=records, columns=columns, epsilon=2.0, random_seed=123)
    samples = synth.generate(sample_size=40, random_seed=123)

    assert len(samples) == 40
    for row in samples:
        assert isinstance(row["age"], int)
        assert 18 <= row["age"] <= 65
        assert isinstance(row["salary"], float)
        assert 25000.0 <= row["salary"] <= 100000.0


def test_tabular_synthesizer_mixed_schema_auto_infer():
    records = [
        {"tier": "silver", "score": 75, "active": "yes"},
        {"tier": "gold", "score": 90, "active": "yes"},
        {"tier": "bronze", "score": 45, "active": "no"},
        {"tier": "gold", "score": 95, "active": "yes"},
        {"tier": "silver", "score": 70, "active": "no"},
    ]

    synth = TabularSynthesizer()
    synth.fit(records=records, epsilon=1.5, random_seed=99)
    samples = synth.generate(sample_size=20, random_seed=99)

    assert len(samples) == 20
    for row in samples:
        assert row["tier"] in {"bronze", "silver", "gold"}
        assert row["active"] in {"yes", "no"}
        assert isinstance(row["score"], int)


def test_reproducibility_with_seed():
    records = [{"val": 10.0}, {"val": 20.0}, {"val": 30.0}]

    synth1 = TabularSynthesizer()
    synth1.fit(records, epsilon=1.0, random_seed=42)
    s1 = synth1.generate(10, random_seed=100)

    synth2 = TabularSynthesizer()
    synth2.fit(records, epsilon=1.0, random_seed=42)
    s2 = synth2.generate(10, random_seed=100)

    assert s1 == s2


def test_legacy_generate_gaussian_samples():
    gen = SyntheticDataGenerator()
    samples = gen.generate_gaussian_samples(sample_size=15, mean=100.0, std=5.0, epsilon=1.0)
    assert len(samples) == 15
    assert "synthetic_value" in samples[0]
    assert samples[0]["privacy_budget_eps"] == 1.0

    # Also test count keyword argument
    samples_count = gen.generate_gaussian_samples(count=5)
    assert len(samples_count) == 5
