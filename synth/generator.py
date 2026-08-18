import numpy as np
from typing import List, Dict, Any

class SyntheticDataGenerator:
    def generate_gaussian_samples(self, sample_size: int = 10, mean: float = 50.0, std: float = 10.0, epsilon: float = 0.5) -> List[Dict[str, Any]]:
        # Add Laplace differential privacy noise
        dp_noise = np.random.laplace(0, 1.0 / epsilon, sample_size)
        base_samples = np.random.normal(mean, std, sample_size) + dp_noise

        records = []
        for idx, val in enumerate(base_samples):
            records.append({
                "id": idx + 1,
                "synthetic_value": round(float(val), 2),
                "privacy_budget_eps": epsilon
            })
        return records
