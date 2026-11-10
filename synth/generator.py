"""Privacy-preserving tabular synthetic data generator."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from synth.models import ColumnSpec, ColumnType
from synth.privacy import (
    LaplaceMechanism,
    GaussianMechanism,
    PrivacyBudget,
    clip_bounds,
)


def _ensure_positive_semidefinite(
    matrix: np.ndarray,
    min_eigenvalue: float = 1e-4,
) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semi-definite cone."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped_eigvals = np.maximum(eigvals, min_eigenvalue)
    psd = eigvecs @ np.diag(clipped_eigvals) @ eigvecs.T
    # Re-symmetrize and ensure non-negative diagonal
    psd = (psd + psd.T) / 2.0
    for i in range(psd.shape[0]):
        if psd[i, i] < min_eigenvalue:
            psd[i, i] = min_eigenvalue
    return psd


class TabularSynthesizer:
    """Generates synthetic tabular datasets with mathematically rigorous Differential Privacy.

    Models categorical attributes via DP-perturbed marginal distributions and
    continuous numerical attributes via DP-perturbed multivariate Gaussian covariance.
    """

    def __init__(self, budget: Optional[PrivacyBudget] = None) -> None:
        self.budget = budget or PrivacyBudget(total_epsilon=50.0, total_delta=1e-3)
        self.columns: List[ColumnSpec] = []
        self._categorical_dist: Dict[str, Tuple[List[str], np.ndarray]] = {}
        self._numeric_cols: List[str] = []
        self._numeric_means: np.ndarray = np.array([])
        self._numeric_cov: np.ndarray = np.array([[]])
        self._numeric_bounds: Dict[str, Tuple[float, float]] = {}
        self._is_fitted: bool = False

    def _infer_column_spec(
        self, records: List[Dict[str, Any]]
    ) -> List[ColumnSpec]:
        """Infer column schema types, categories, and bounds from input data."""
        if not records:
            return []

        col_names = list(records[0].keys())
        specs: List[ColumnSpec] = []

        for name in col_names:
            values = [r[name] for r in records if r.get(name) is not None]
            if not values:
                specs.append(ColumnSpec(name=name, col_type=ColumnType.CATEGORICAL))
                continue

            first_val = values[0]
            is_int = all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values)
            is_float = all(
                isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
                for v in values
            )

            if is_int:
                numeric_vals = [float(v) for v in values]
                low, high = min(numeric_vals), max(numeric_vals)
                if low == high:
                    low -= 1.0
                    high += 1.0
                specs.append(
                    ColumnSpec(
                        name=name,
                        col_type=ColumnType.NUMERICAL_INT,
                        bounds=(low, high),
                    )
                )
            elif is_float:
                numeric_vals = [float(v) for v in values]
                low, high = min(numeric_vals), max(numeric_vals)
                if low == high:
                    low -= 1.0
                    high += 1.0
                specs.append(
                    ColumnSpec(
                        name=name,
                        col_type=ColumnType.NUMERICAL_FLOAT,
                        bounds=(low, high),
                    )
                )
            else:
                str_cats = sorted(list({str(v) for v in values}))
                specs.append(
                    ColumnSpec(
                        name=name,
                        col_type=ColumnType.CATEGORICAL,
                        categories=str_cats,
                    )
                )

        return specs

    def fit(
        self,
        records: List[Dict[str, Any]],
        columns: Optional[List[ColumnSpec]] = None,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        random_seed: Optional[int] = None,
    ) -> "TabularSynthesizer":
        """Fit synthesizer to input records under differential privacy guarantees.

        Args:
            records: List of tabular row dictionaries.
            columns: Optional explicit column schemas.
            epsilon: Privacy parameter epsilon.
            delta: Privacy parameter delta.
            random_seed: Optional seed for reproducible DP noise.
        """
        if not records:
            raise ValueError("Input records cannot be empty")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.budget.spend(epsilon=epsilon, delta=delta)
        rng = np.random.default_rng(random_seed)

        self.columns = columns if columns is not None else self._infer_column_spec(records)
        num_records = len(records)

        # Categorize columns
        cat_specs = [c for c in self.columns if c.col_type == ColumnType.CATEGORICAL]
        num_specs = [
            c
            for c in self.columns
            if c.col_type in (ColumnType.NUMERICAL_FLOAT, ColumnType.NUMERICAL_INT)
        ]

        # Allocate budget between categorical and numerical features
        if cat_specs and num_specs:
            eps_cat = epsilon * 0.4
            eps_num = epsilon * 0.6
        elif cat_specs:
            eps_cat = epsilon
            eps_num = 0.0
        else:
            eps_cat = 0.0
            eps_num = epsilon

        # 1. Fit Categorical Marginals with DP Laplace Mechanism
        self._categorical_dist.clear()
        if cat_specs:
            eps_per_cat = eps_cat / len(cat_specs)
            for spec in cat_specs:
                vals = [str(r.get(spec.name, "")) for r in records]
                categories = spec.categories or sorted(list(set(vals)))
                if not categories:
                    categories = ["default"]

                # Count frequencies: sensitivity of count histogram is 1
                counts = np.array([vals.count(c) for c in categories], dtype=float)
                laplace = LaplaceMechanism(epsilon=eps_per_cat, sensitivity=1.0, rng=rng)
                noisy_counts = laplace.add_noise(counts)
                # Clip negative counts to zero
                clipped_counts = np.maximum(0.0, noisy_counts)

                total_count = np.sum(clipped_counts)
                if total_count <= 0:
                    probs = np.full(len(categories), 1.0 / len(categories))
                else:
                    probs = clipped_counts / total_count

                self._categorical_dist[spec.name] = (categories, probs)

        # 2. Fit Numerical Multivariate Distributions with DP Gaussian / Laplace
        self._numeric_cols = [c.name for c in num_specs]
        self._numeric_bounds.clear()

        if num_specs:
            # Build matrix of clipped numerical values
            matrix_data = []
            for spec in num_specs:
                raw = [float(r[spec.name]) for r in records if r.get(spec.name) is not None]
                low, high = spec.bounds or (min(raw), max(raw))
                if low == high:
                    low -= 1.0
                    high += 1.0
                self._numeric_bounds[spec.name] = (low, high)
                clipped = np.clip([float(r.get(spec.name, (low + high) / 2)) for r in records], low, high)
                matrix_data.append(clipped)

            X = np.column_stack(matrix_data)  # shape (N, D)
            d = X.shape[1]

            # Split numeric budget: 35% for mean, 65% for covariance
            eps_mean = eps_num * 0.35
            eps_cov = eps_num * 0.65
            delta_cov = delta if delta > 0 else 1e-6

            # Compute DP Mean: sensitivity of sum is (high - low), mean sensitivity is (high - low)/N
            emp_mean = np.mean(X, axis=0)
            ranges = np.array(
                [self._numeric_bounds[name][1] - self._numeric_bounds[name][0] for name in self._numeric_cols]
            )
            mean_sens = ranges / float(num_records)
            lap_mean = LaplaceMechanism(epsilon=eps_mean, sensitivity=1.0, rng=rng)
            mean_noise = np.array([lap_mean.add_noise(0.0) * sens for sens in mean_sens])
            dp_mean = emp_mean + mean_noise

            # Clip DP mean to valid bounds
            for i, name in enumerate(self._numeric_cols):
                dp_mean[i] = clip_bounds(
                    dp_mean[i],
                    self._numeric_bounds[name][0],
                    self._numeric_bounds[name][1],
                )
            self._numeric_means = dp_mean

            # Compute DP Covariance
            if num_records > 1:
                centered = X - emp_mean
                emp_cov = (centered.T @ centered) / float(num_records - 1)
            else:
                emp_cov = np.eye(d) * 1.0

            # Sensitivity for covariance entries bounded by (range_i * range_j) / N
            cov_sens = float(np.max(ranges) ** 2) / float(num_records)
            cov_noise_mech = GaussianMechanism(
                epsilon=eps_cov, delta=delta_cov, sensitivity=cov_sens, rng=rng
            )

            raw_noise = rng.normal(0.0, cov_noise_mech.sigma, size=(d, d))
            sym_noise = (raw_noise + raw_noise.T) / 2.0
            dp_cov = emp_cov + sym_noise

            # Project to PSD cone
            self._numeric_cov = _ensure_positive_semidefinite(dp_cov, min_eigenvalue=1e-4)

        self._is_fitted = True
        return self

    def generate(
        self,
        sample_size: int = 50,
        random_seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Synthesize tabular records from fitted differentially private models.

        Args:
            sample_size: Number of synthetic rows to generate.
            random_seed: Optional seed for PRNG sampling.
        """
        if not self._is_fitted:
            raise RuntimeError("Synthesizer must be fitted before generating samples")
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")

        rng = np.random.default_rng(random_seed)
        rows: List[Dict[str, Any]] = [{} for _ in range(sample_size)]

        # Sample Categorical Features
        for col_name, (cats, probs) in self._categorical_dist.items():
            sampled_cats = rng.choice(cats, size=sample_size, p=probs)
            for idx, val in enumerate(sampled_cats):
                rows[idx][col_name] = str(val)

        # Sample Numerical Features
        if self._numeric_cols:
            d = len(self._numeric_cols)
            if d == 1:
                variance = float(self._numeric_cov[0, 0])
                std = np.sqrt(max(1e-4, variance))
                sampled_vals = rng.normal(
                    loc=self._numeric_means[0], scale=std, size=(sample_size, 1)
                )
            else:
                sampled_vals = rng.multivariate_normal(
                    mean=self._numeric_means, cov=self._numeric_cov, size=sample_size
                )

            # Assign and post-process numerical columns
            spec_map = {c.name: c for c in self.columns}
            for col_idx, col_name in enumerate(self._numeric_cols):
                spec = spec_map[col_name]
                low, high = self._numeric_bounds[col_name]
                col_data = np.clip(sampled_vals[:, col_idx], low, high)

                for row_idx in range(sample_size):
                    val = float(col_data[row_idx])
                    if spec.col_type == ColumnType.NUMERICAL_INT:
                        rows[row_idx][col_name] = int(round(val))
                    else:
                        rows[row_idx][col_name] = round(val, 4)

        return rows

    # Backward compatibility with initial mock
    def generate_gaussian_samples(
        self,
        count: Optional[int] = None,
        sample_size: Optional[int] = None,
        mean: float = 50.0,
        std: float = 10.0,
        epsilon: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Legacy helper for single-feature gaussian sampling."""
        n = count if count is not None else (sample_size if sample_size is not None else 10)
        rng = np.random.default_rng()
        laplace = LaplaceMechanism(epsilon=epsilon, sensitivity=1.0, rng=rng)

        records = []
        base = rng.normal(mean, std, n)
        noisy = laplace.add_noise(base)
        for idx, val in enumerate(noisy):
            records.append({
                "id": idx + 1,
                "synthetic_value": round(float(val), 2),
                "privacy_budget_eps": epsilon,
            })
        return records


# Convenience alias for backwards compatibility
SyntheticDataGenerator = TabularSynthesizer
