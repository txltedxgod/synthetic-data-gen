"""Unit tests for Differential Privacy noise mechanisms and budget accounting."""

import numpy as np
import pytest

from synth.privacy import (
    BudgetExceededError,
    GaussianMechanism,
    LaplaceMechanism,
    PrivacyBudget,
    clip_bounds,
    exponential_mechanism,
)


def test_laplace_mechanism_noise_properties():
    rng = np.random.default_rng(42)
    laplace = LaplaceMechanism(epsilon=1.0, sensitivity=2.0, rng=rng)
    assert laplace.scale == 2.0

    samples = np.array([laplace.add_noise(10.0) for _ in range(2000)])
    # Mean of Laplace(0, b) should be centered near 0, so samples mean ~ 10.0
    assert abs(np.mean(samples) - 10.0) < 0.25
    # Variance of Laplace(0, b) is 2 * b^2 = 2 * 4 = 8.0
    assert abs(np.var(samples) - 8.0) < 1.5


def test_laplace_array_input():
    rng = np.random.default_rng(42)
    laplace = LaplaceMechanism(epsilon=0.5, sensitivity=1.0, rng=rng)
    data = np.array([1.0, 2.0, 3.0])
    noisy = laplace.add_noise(data)
    assert noisy.shape == data.shape
    assert not np.allclose(noisy, data)


def test_gaussian_mechanism():
    rng = np.random.default_rng(42)
    gauss = GaussianMechanism(epsilon=1.0, delta=1e-4, sensitivity=1.0, rng=rng)
    assert gauss.sigma > 0.0

    samples = np.array([gauss.add_noise(0.0) for _ in range(2000)])
    assert abs(np.mean(samples)) < 0.5


def test_clip_bounds():
    assert clip_bounds(5.0, 0.0, 10.0) == 5.0
    assert clip_bounds(-2.0, 0.0, 10.0) == 0.0
    assert clip_bounds(15.0, 0.0, 10.0) == 10.0

    arr = np.array([-5.0, 5.0, 20.0])
    clipped = clip_bounds(arr, 0.0, 10.0)
    np.testing.assert_array_equal(clipped, np.array([0.0, 5.0, 10.0]))

    with pytest.raises(ValueError):
        clip_bounds(5.0, 10.0, 0.0)


def test_privacy_budget_spend_and_exhaustion():
    budget = PrivacyBudget(total_epsilon=2.0, total_delta=1e-3)
    assert budget.remaining_epsilon == 2.0
    assert not budget.is_exhausted

    budget.spend(epsilon=1.5, delta=5e-4)
    assert round(budget.spent_epsilon, 2) == 1.5
    assert round(budget.remaining_epsilon, 2) == 0.5
    assert not budget.is_exhausted

    with pytest.raises(BudgetExceededError):
        budget.spend(epsilon=1.0)  # Exceeds remaining 0.5

    budget.spend(epsilon=0.5)
    assert budget.is_exhausted

    budget.reset()
    assert budget.remaining_epsilon == 2.0
    assert not budget.is_exhausted


def test_exponential_mechanism():
    rng = np.random.default_rng(42)
    candidates = ["low", "medium", "high"]
    # Give 'high' a much higher utility
    utility = {"low": 0.0, "medium": 1.0, "high": 10.0}

    selected = [
        exponential_mechanism(
            candidates=candidates,
            utility_fn=lambda c: utility[c],
            sensitivity=1.0,
            epsilon=2.0,
            rng=rng,
        )
        for _ in range(100)
    ]
    # 'high' should be selected the vast majority of times
    assert selected.count("high") > 70
