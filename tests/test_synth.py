from synth.generator import SyntheticDataGenerator

def test_synthetic_gen():
    g = SyntheticDataGenerator()
    samples = g.generate_gaussian_samples(count=5)
    assert len(samples) == 5
    assert "synthetic_value" in samples[0]
