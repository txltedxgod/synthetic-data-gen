"""Statistical fidelity and privacy leakage evaluation for synthetic datasets."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from synth.models import EvaluateResponse


class FidelityEvaluator:
    """Evaluates mathematical distribution fidelity and privacy memorization risk."""

    def evaluate(
        self,
        real_records: List[Dict[str, Any]],
        synthetic_records: List[Dict[str, Any]],
    ) -> EvaluateResponse:
        """Run fidelity metrics and privacy risk audit comparing real and synthetic data."""
        if not real_records or not synthetic_records:
            raise ValueError("Datasets cannot be empty")

        common_keys = sorted(
            list(set(real_records[0].keys()).intersection(synthetic_records[0].keys()))
        )
        if not common_keys:
            raise ValueError("No common columns found between real and synthetic records")

        marginal_divergences: Dict[str, float] = {}
        marginal_fidelities: List[float] = []

        numeric_cols: List[str] = []

        for key in common_keys:
            real_vals = [r[key] for r in real_records if r.get(key) is not None]
            synth_vals = [r[key] for r in synthetic_records if r.get(key) is not None]

            if not real_vals or not synth_vals:
                continue

            is_numeric = all(
                isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
                for v in real_vals + synth_vals
            )

            if is_numeric:
                numeric_cols.append(key)
                r_arr = np.array(real_vals, dtype=float)
                s_arr = np.array(synth_vals, dtype=float)

                r_min, r_max = np.min(r_arr), np.max(r_arr)
                val_range = max(1e-6, r_max - r_min)

                # Quantile Wasserstein approximation
                quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
                r_q = np.quantile(r_arr, quantiles)
                s_q = np.quantile(s_arr, quantiles)
                norm_diff = float(np.mean(np.abs(r_q - s_q)) / val_range)
                norm_diff = min(1.0, norm_diff)

                marginal_divergences[key] = round(norm_diff, 4)
                marginal_fidelities.append(1.0 - norm_diff)
            else:
                # Categorical Total Variation Distance
                all_cats = sorted(list(set([str(v) for v in real_vals + synth_vals])))
                r_counts = np.array([real_vals.count(c) for c in all_cats], dtype=float)
                s_counts = np.array([synth_vals.count(c) for c in all_cats], dtype=float)

                r_prob = r_counts / max(1.0, np.sum(r_counts))
                s_prob = s_counts / max(1.0, np.sum(s_counts))

                tvd = float(0.5 * np.sum(np.abs(r_prob - s_prob)))
                marginal_divergences[key] = round(tvd, 4)
                marginal_fidelities.append(1.0 - tvd)

        # 2D Correlation Matrix Error
        correlation_error = 0.0
        if len(numeric_cols) > 1:
            r_mat = np.column_stack(
                [[float(r.get(c, 0.0)) for r in real_records] for c in numeric_cols]
            )
            s_mat = np.column_stack(
                [[float(s.get(c, 0.0)) for s in synthetic_records] for c in numeric_cols]
            )

            # Safeguard standard deviation
            r_std = np.std(r_mat, axis=0)
            s_std = np.std(s_mat, axis=0)

            if np.all(r_std > 1e-6) and np.all(s_std > 1e-6):
                r_corr = np.corrcoef(r_mat, rowvar=False)
                s_corr = np.corrcoef(s_mat, rowvar=False)
                d = len(numeric_cols)
                frobenius_diff = float(np.linalg.norm(r_corr - s_corr, ord="fro"))
                norm_factor = np.sqrt(d * (d - 1)) if d > 1 else 1.0
                correlation_error = min(1.0, frobenius_diff / norm_factor)

        # Memorization / Privacy Leakage Risk (Nearest Neighbor Check)
        memorization_risk = 0.0
        if numeric_cols:
            r_mat = np.column_stack(
                [[float(r.get(c, 0.0)) for r in real_records] for c in numeric_cols]
            )
            s_mat = np.column_stack(
                [[float(s.get(c, 0.0)) for s in synthetic_records] for c in numeric_cols]
            )

            # Normalize columns by range for isotropic distance
            ptp = np.ptp(r_mat, axis=0)
            ptp = np.where(ptp < 1e-6, 1.0, ptp)
            r_norm = r_mat / ptp
            s_norm = s_mat / ptp

            exact_matches = 0
            for s_vec in s_norm:
                dists = np.linalg.norm(r_norm - s_vec, axis=1)
                if np.min(dists) < 1e-4:  # Exact or near-identical duplicate
                    exact_matches += 1

            memorization_risk = float(exact_matches / len(synthetic_records))

        # Overall Fidelity Score calculation
        mean_marginal = np.mean(marginal_fidelities) if marginal_fidelities else 0.5
        overall_score = 100.0 * (0.65 * mean_marginal + 0.35 * (1.0 - correlation_error))
        overall_score = max(0.0, min(100.0, overall_score))

        passed = overall_score >= 60.0 and memorization_risk <= 0.05

        return EvaluateResponse(
            marginal_divergences=marginal_divergences,
            correlation_error=round(correlation_error, 4),
            fidelity_score=round(overall_score, 2),
            memorization_risk_score=round(memorization_risk, 4),
            passed_quality_gate=passed,
        )
