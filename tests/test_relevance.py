import numpy as np

from axobench.relevance import (
    adjusted_box_fences,
    fivenum,
    iron_relevance,
    iron_sera,
    iron_threshold_weights,
    medcouple,
)


def _medcouple_reference(values):
    """O(n^2) medcouple with robustbase's tie kernel and (nr/2 + 1)-th largest rule."""
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    n = x.size
    median = 0.5 * (x[(n - 1) // 2] + x[n // 2])
    z = x - median
    zplus, zminus = z[z >= 0], z[z <= 0]
    p = zplus.size
    kernel = []
    for i, a in enumerate(zplus):
        for j, b in enumerate(zminus):
            kernel.append(np.sign(p - 1 - i - j) if a == b else (a + b) / (a - b))
    kernel = np.sort(kernel)[::-1]
    return float(kernel[len(kernel) // 2])


def test_fivenum_matches_r():
    np.testing.assert_allclose(fivenum(np.arange(1, 11)), [1, 3, 5.5, 8, 10])
    np.testing.assert_allclose(fivenum(np.arange(1, 10)), [1, 3, 5, 7, 9])


def test_medcouple_matches_quadratic_reference_including_ties():
    rng = np.random.default_rng(3)
    samples = [
        rng.normal(size=57),
        rng.exponential(size=120),
        rng.integers(0, 5, size=90).astype(float),
        np.r_[rng.normal(size=60), np.full(7, 4.0)],
    ]
    for sample in samples:
        assert abs(medcouple(sample) - _medcouple_reference(sample)) < 1e-12


def test_medcouple_sign_follows_skew():
    rng = np.random.default_rng(4)
    assert medcouple(rng.exponential(size=2000)) > 0.2
    assert medcouple(-rng.exponential(size=2000)) < -0.2
    assert abs(medcouple(rng.normal(size=2000))) < 0.05


def test_iron_relevance_is_zero_at_median_and_one_beyond_fences():
    rng = np.random.default_rng(5)
    y = np.r_[rng.normal(-65, 2, 5000), rng.uniform(-40, 20, 50)]
    low, median, high = adjusted_box_fences(y)
    phi = iron_relevance(np.array([median, low - 1, high + 1, 0.5 * (median + high)]), reference=y)
    np.testing.assert_allclose(phi[:3], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(phi[3], 0.5)  # 3t^2 - 2t^3 at t = 1/2


def test_threshold_weights_reproduce_iron_sera():
    rng = np.random.default_rng(6)
    error = rng.normal(size=4000)
    phi = rng.uniform(size=4000)
    phi[:50] = 1.0
    phi[50:100] = 0.0
    weighted = float(np.sum(error**2 * iron_threshold_weights(phi)))
    np.testing.assert_allclose(weighted, iron_sera(error, phi), rtol=1e-12)
