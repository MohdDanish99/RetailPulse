import numpy as np

from src.monitoring.drift import psi


def test_psi_low_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.1


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(2, 1, size=5000)) > 0.2
