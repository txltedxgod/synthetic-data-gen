"""Differential Privacy primitives, noise mechanisms, and budget accounting.

Implements pure epsilon-Differential Privacy (Laplace Mechanism) and
approximate (epsilon, delta)-Differential Privacy (Gaussian Mechanism)
with calibrated sensitivity clipping and compositional privacy budgeting.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, TypeVar, Union
import numpy as np

T = TypeVar("T")


class BudgetExceededError(ValueError):
    """Raised when an operation requests more privacy budget than available."""
    pass


class PrivacyBudget:
    """Tracks cumulative privacy expenditure and enforces budget guarantees."""

    def __init__(self, total_epsilon: float = 10.0, total_delta: float = 1e-4) -> None:
        if total_epsilon <= 0:
            raise ValueError(f"total_epsilon must be positive, got {total_epsilon}")
        if total_delta < 0:
            raise ValueError(f"total_delta must be non-negative, got {total_delta}")

        self.total_epsilon = float(total_epsilon)
        self.total_delta = float(total_delta)
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self.spent_delta)

    @property
    def is_exhausted(self) -> bool:
        return self.spent_epsilon >= self.total_epsilon or (
            self.total_delta > 0 and self.spent_delta >= self.total_delta
        )

    def spend(self, epsilon: float, delta: float = 0.0) -> None:
        """Deduct privacy budget using basic composition theorem.

        Args:
            epsilon: Epsilon parameter to spend.
            delta: Delta parameter to spend.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon to spend must be positive, got {epsilon}")
        if delta < 0:
            raise ValueError(f"delta to spend must be non-negative, got {delta}")

        if self.spent_epsilon + epsilon > self.total_epsilon + 1e-9:
            raise BudgetExceededError(
                f"Requested epsilon {epsilon:.4f} exceeds remaining budget {self.remaining_epsilon:.4f}"
            )
        if self.total_delta > 0 and self.spent_delta + delta > self.total_delta + 1e-9:
            raise BudgetExceededError(
                f"Requested delta {delta:.2e} exceeds remaining budget {self.remaining_delta:.2e}"
            )

        self.spent_epsilon += epsilon
        self.spent_delta += delta

    def reset(self) -> None:
        """Reset spent budget back to zero."""
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0

    def summary(self) -> dict[str, Any]:
        """Return a structured summary of budget consumption."""
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "spent_epsilon": round(self.spent_epsilon, 6),
            "spent_delta": round(self.spent_delta, 8),
            "remaining_epsilon": round(self.remaining_epsilon, 6),
            "remaining_delta": round(self.remaining_delta, 8),
            "is_exhausted": self.is_exhausted,
        }


def clip_bounds(
    data: Union[float, np.ndarray],
    lower: float,
    upper: float,
) -> Union[float, np.ndarray]:
    """Clips data to [lower, upper] to enforce bounded global sensitivity.

    Args:
        data: Scalar or numpy array to clip.
        lower: Minimum acceptable bound.
        upper: Maximum acceptable bound.
    """
    if lower > upper:
        raise ValueError(f"lower ({lower}) cannot exceed upper ({upper})")
    if isinstance(data, (int, float)):
        return max(lower, min(upper, float(data)))
    return np.clip(data, lower, upper)


class LaplaceMechanism:
    """Laplace mechanism for pure epsilon-Differential Privacy.

    Adds noise from Laplace(0, b) where scale b = sensitivity / epsilon.
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")

        self.epsilon = float(epsilon)
        self.sensitivity = float(sensitivity)
        self.scale = self.sensitivity / self.epsilon
        self.rng = rng or np.random.default_rng()

    def add_noise(
        self,
        value: Union[float, int, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Add Laplace noise to input value or array."""
        arr = np.asarray(value, dtype=float)
        noise = self.rng.laplace(loc=0.0, scale=self.scale, size=arr.shape)
        result = arr + noise
        if result.ndim == 0:
            return float(result)
        return result


class GaussianMechanism:
    """Gaussian mechanism for (epsilon, delta)-Differential Privacy.

    Adds noise from Normal(0, sigma^2) where
    sigma = sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be strictly in (0, 1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self.sigma = (
            math.sqrt(2.0 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        )
        self.rng = rng or np.random.default_rng()

    def add_noise(
        self,
        value: Union[float, int, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Add Gaussian noise calibrated to (epsilon, delta)."""
        arr = np.asarray(value, dtype=float)
        noise = self.rng.normal(loc=0.0, scale=self.sigma, size=arr.shape)
        result = arr + noise
        if result.ndim == 0:
            return float(result)
        return result


def exponential_mechanism(
    candidates: List[T],
    utility_fn: Callable[[T], float],
    sensitivity: float,
    epsilon: float,
    rng: Optional[np.random.Generator] = None,
) -> T:
    """Exponential mechanism for differentially private selection over discrete candidates.

    Selects candidate c with probability proportional to exp(epsilon * u(c) / (2 * Delta u)).
    """
    if not candidates:
        raise ValueError("Candidates list cannot be empty")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")

    rng = rng or np.random.default_rng()
    utilities = np.array([utility_fn(c) for c in candidates], dtype=float)
    scaled_utilities = (epsilon * utilities) / (2.0 * sensitivity)

    # Numerical stability shift
    max_scaled = np.max(scaled_utilities)
    exp_utilities = np.exp(scaled_utilities - max_scaled)
    probs = exp_utilities / np.sum(exp_utilities)

    selected_idx = rng.choice(len(candidates), p=probs)
    return candidates[selected_idx]
