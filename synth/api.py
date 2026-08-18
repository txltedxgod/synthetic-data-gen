from fastapi import FastAPI
from synth.generator import SyntheticDataGenerator

app = FastAPI(title="Synthetic Data Generator", version="0.1.0")
gen = SyntheticDataGenerator()

@app.get("/api/v1/generate")
def generate(count: int = 10, epsilon: float = 0.5):
    return {"samples": gen.generate_gaussian_samples(sample_size=count, epsilon=epsilon)}
