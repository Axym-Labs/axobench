import numpy as np
import pytest

from axobench.benchmark.defect_injection import (
    DEFAULT_DOSES,
    DEFECTS,
    axobench_scores,
    defect_dose_rows,
    inject_defect,
)
from axobench.metrics import coincidence_factor


def synthetic_teacher(traces: int = 4, steps: int = 3000, seed: int = 0):
    """Leaky subthreshold voltage with stereotyped 1 ms action potentials."""
    rng = np.random.default_rng(seed)
    v = np.full((traces, steps), -70.0)
    noise = rng.normal(0.0, 1.0, size=(traces, steps))
    for t in range(1, steps):
        v[:, t] = v[:, t - 1] + 0.1 * (-70.0 - v[:, t - 1]) + noise[:, t]
    spikes = np.zeros((traces, steps), dtype=np.float32)
    shape = np.array([10.0, 40.0, 100.0, 60.0, 10.0, -5.0, -8.0])
    for i in range(traces):
        for t in np.sort(rng.choice(np.arange(20, steps - 60, 37), size=12, replace=False)):
            spikes[i, t] = 1.0
            v[i, t - 2 : t + 5] += shape
            v[i, t + 5 : t + 40] -= 5.0 * np.exp(-np.arange(35) / 10.0)
    return spikes, v


def test_coincidence_factor_is_one_for_identical_and_near_zero_for_chance():
    spikes, _ = synthetic_teacher()
    assert coincidence_factor(spikes, spikes)["gamma"] == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    random = np.zeros_like(spikes)
    for i, n in enumerate(spikes.sum(axis=1).astype(int)):
        random[i, rng.choice(spikes.shape[1], size=n, replace=False)] = 1.0
    assert abs(coincidence_factor(random, spikes)["gamma"]) < 0.15


def test_coincidence_factor_counts_one_to_one_within_delta():
    target = np.zeros((1, 100))
    target[0, [10, 50]] = 1
    pred = np.zeros((1, 100))
    pred[0, [12, 13, 80]] = 1
    result = coincidence_factor(pred, target, delta_steps=2)
    assert result["coincidences"] == 1
    assert result["expected_coincidences"] == pytest.approx(2 * 0.03 * 2 * 2)


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_zero_dose_is_identity(defect):
    spikes, v = synthetic_teacher()
    out_s, out_v = inject_defect(spikes, v, defect, 0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out_s, spikes)
    np.testing.assert_allclose(out_v, v)


def test_perfect_prediction_scores_one():
    spikes, v = synthetic_teacher()
    scores = axobench_scores(spikes, v, spikes, v)
    assert scores["mean_f1"] == pytest.approx(1.0)
    assert scores["gamma_2"] == pytest.approx(1.0)
    assert scores["voltage_skill"] == pytest.approx(1.0)
    assert scores["dynamics_skill"] == pytest.approx(1.0)


def test_spike_defects_change_counts_and_move_action_potentials():
    spikes, v = synthetic_teacher()
    n = int(spikes.sum())
    deleted, v_del = inject_defect(spikes, v, "spike_deletion", 1.0, rng=np.random.default_rng(0))
    assert deleted.sum() == 0
    assert v_del.max() < v.max() - 50
    inserted, _ = inject_defect(spikes, v, "spike_insertion", 1.0, rng=np.random.default_rng(0))
    assert n < inserted.sum() <= 2 * n
    lagged, v_lag = inject_defect(spikes, v, "global_lag", 3)
    np.testing.assert_array_equal(np.flatnonzero(lagged[0]), np.flatnonzero(spikes[0]) + 3)
    np.testing.assert_allclose(v_lag[0, 3:], v[0, :-3])


@pytest.mark.parametrize(
    "defect,score",
    [
        ("spike_jitter", "mean_f1"),
        ("spike_deletion", "gamma_2"),
        ("spike_insertion", "mean_f1"),
        ("global_lag", "mean_f1"),
        ("ap_clip", "voltage_skill"),
        ("voltage_lowpass", "dynamics_skill"),
        ("voltage_offset", "voltage_skill"),
        ("voltage_gain", "voltage_skill"),
    ],
)
def test_axobench_scores_fall_with_dose(defect, score):
    spikes, v = synthetic_teacher()
    rows = defect_dose_rows(spikes, v, {defect: DEFAULT_DOSES[defect]}, seed=0)
    values = [row[score] for row in rows]
    assert values[0] == pytest.approx(1.0)
    assert values[-1] < values[0] - 0.05
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
