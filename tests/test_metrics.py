import numpy as np
import pytest

from axobench.relevance import iron_relevance, iron_threshold_weights
from axobench.metrics import (
    binary_auc,
    compute_core_metric_set,
    iron_dynamics_sera,
    iron_voltage_sera,
    response_sera,
    response_sera_trace_terms,
    soma_rmse,
    spike_auc,
    spike_event_confusion,
    spike_f1_curve,
    event_count_matched_threshold,
    spike_event_peaks,
    spike_events,
)


def test_soma_rmse_uses_second_output_channel():
    prediction = np.array([[[0.0, 1.0], [0.0, 3.0]]], dtype=np.float32)
    target = np.array([[[0.0, 2.0], [0.0, 5.0]]], dtype=np.float32)

    assert np.isclose(soma_rmse(prediction, target), np.sqrt(2.5))


def test_binary_auc_handles_perfect_ranking():
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    scores = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)

    assert binary_auc(scores, labels) == 1.0


def test_binary_auc_averages_tied_ranks():
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    scores = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    assert binary_auc(scores, labels) == 0.5


def test_spike_auc_uses_first_output_channel():
    prediction = np.array([[[0.1, 100.0], [0.4, -100.0], [0.8, 0.0]]], dtype=np.float32)
    target = np.array([[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]], dtype=np.float32)

    assert spike_auc(prediction, target) == 1.0


def test_iron_voltage_sera_is_relevance_area_per_teacher_spike():
    rng = np.random.default_rng(11)
    raw_soma_mv = -67.0 + rng.normal(size=(3, 40))
    raw_soma_mv[:, 20] = -20.0
    target = np.zeros((3, 40, 2), dtype=np.float64)
    target[..., 1] = (raw_soma_mv + 67.7) * 0.1
    target[:, 20, 0] = 1.0
    prediction = target.copy()

    exact = iron_voltage_sera(prediction, target, raw_soma_mv=raw_soma_mv)
    prediction[..., 1] += 0.2  # +2 mV everywhere
    offset = iron_voltage_sera(prediction, target, raw_soma_mv=raw_soma_mv)

    weights = iron_threshold_weights(iron_relevance(raw_soma_mv))
    assert exact["voltage_sera_iron_per_spike_mv2"] == 0.0
    assert offset["teacher_spikes"] == 3
    assert np.isclose(offset["voltage_sera_iron_mv2"], np.sum(4.0 * weights))
    assert np.isclose(
        offset["voltage_sera_iron_per_spike_mv2"], np.sum(4.0 * weights) / 3.0
    )


def test_iron_sera_per_spike_is_undefined_without_teacher_spikes():
    target = np.zeros((1, 12, 2), dtype=np.float32)
    target[0, :, 1] = np.linspace(-0.4, 0.4, 12)

    result = iron_voltage_sera(target, target)

    assert result["teacher_spikes"] == 0
    assert result["voltage_sera_iron_per_spike_mv2"] is None


def test_iron_dynamics_sera_complements_voltage_level_error():
    target = np.zeros((1, 12, 2), dtype=np.float32)
    target[0, :, 1] = np.linspace(-0.4, 0.4, 12)
    target[0, 6, 0] = 1.0
    shifted = target.copy()
    shifted[..., 1] += 0.5
    distorted = target.copy()
    distorted[0, 5:7, 1] += np.array([0.5, -0.5])

    key = "dynamics_sera_iron_per_spike_mv2_per_ms2"
    shifted_result = iron_dynamics_sera(shifted, target)
    distorted_result = iron_dynamics_sera(distorted, target)
    faster_sampling_result = iron_dynamics_sera(distorted, target, timestep_ms=0.5)

    assert np.isclose(shifted_result[key], 0.0, atol=1e-9)
    assert distorted_result[key] > 0.0
    # Relevance is scale invariant, so halving the timestep quadruples SERA.
    assert np.isclose(faster_sampling_result[key], 4.0 * distorted_result[key])


def test_spike_f1_curve_uses_one_to_one_event_matching():
    predicted = np.array([[0.0, 2.0, 2.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0, 1.0]])

    exact = spike_f1_curve(predicted, target, threshold=1.0, tolerances_steps=(0,), refractory_steps=None)
    loose = spike_f1_curve(predicted, target, threshold=1.0, tolerances_steps=(0, 1), refractory_steps=None)
    confusion = spike_event_confusion(predicted >= 1.0, target, tolerance_steps=1)

    assert exact["rows"][0]["f1"] == 0.5
    assert loose["rows"][1]["f1"] == 1.0
    assert loose["mean_f1"] == 0.75
    assert loose["f1_area"] == 0.75
    assert confusion == {"tp": 2, "fp": 0, "fn": 0}


def test_spike_mean_f1_is_the_arithmetic_mean_over_declared_tolerances():
    predicted = np.array([[0.0, 0.0, 0.0, 2.0]])
    target = np.array([[0.0, 1.0, 0.0, 0.0]])

    result = spike_f1_curve(
        predicted,
        target,
        threshold=1.0,
        tolerances_steps=(0, 1, 2),
    )

    assert result["mean_f1"] == pytest.approx(1.0 / 3.0)
    assert result["f1_area"] == 0.25


def test_spike_f1_curve_requires_strictly_increasing_tolerances():
    scores = np.zeros((1, 4))
    target = np.zeros((1, 4))

    with pytest.raises(ValueError, match="strictly increasing"):
        spike_f1_curve(
            scores,
            target,
            tolerances_steps=(0, 2, 1),
        )


def test_compute_core_metric_set_returns_promoted_metrics():
    target = np.zeros((1, 8, 2), dtype=np.float32)
    prediction = target.copy()
    target[0, 4, 0] = 1.0
    prediction[0, 4, 0] = 3.0

    result = compute_core_metric_set(
        prediction, target, ignore_start=0, spike_tolerances_steps=(0, 1, 2)
    )

    assert result["voltage_sera_iron_per_spike_mv2"] == 0.0
    assert result["dynamics_sera_iron_per_spike_mv2_per_ms2"] == 0.0
    assert result["spike_mean_f1"] == 1.0
    assert result["spike_f1_area"] == 1.0
    assert "spike_squared_f1_deficit_area" not in result
    assert "spike_squared_deficit_f1_score" not in result
    assert result["spike_threshold_kind"] == "prevalence_matched"
    assert "voltage_rmse_mv" not in result
    assert "voltage_sera_mv2" not in result


def test_compute_core_metric_set_includes_response_sera_when_pair_is_given():
    baseline = np.zeros((1, 8, 2), dtype=np.float32)
    intervention = baseline.copy()
    intervention[..., 1] = 0.5
    intervention[0, 5, 0] = 1.0

    result = compute_core_metric_set(
        baseline,
        baseline,
        intervention_prediction=intervention,
        intervention_target=intervention,
        ignore_start=2,
    )

    assert result["response_sera_iron_per_spike_mv2"] == 0.0
    assert result["response"]["response_sera_iron_per_spike_mv2"] == 0.0
    assert "response_sera_mv2" not in result


def _response_fixture():
    rng = np.random.default_rng(13)
    baseline_target = np.zeros((4, 30, 2), dtype=np.float64)
    baseline_target[..., 1] = rng.normal(0.0, 0.2, size=(4, 30))
    baseline_target[:, 10, 0] = 1.0
    intervention_target = baseline_target.copy()
    intervention_target[..., 1] += 0.5 + rng.normal(0.0, 0.05, size=(4, 30))
    intervention_target[:, 20, 0] = 1.0
    return baseline_target, intervention_target


def test_response_sera_is_zero_for_exact_teacher_response():
    baseline_target, intervention_target = _response_fixture()

    correct = response_sera(
        baseline_target, intervention_target, baseline_target, intervention_target
    )

    assert correct["response_sera_iron_per_spike_mv2"] == 0.0
    assert correct["response_sera_iron_mv2"] == 0.0
    assert correct["teacher_spikes"] == 12  # 1 baseline + 2 intervention spikes per pair
    assert "response_root_sera_mv" not in correct


def test_response_sera_zero_response_control_is_positive_and_matches_ignored_change():
    baseline_target, intervention_target = _response_fixture()

    ignored = response_sera(
        baseline_target, baseline_target, baseline_target, intervention_target
    )

    teacher_delta = (intervention_target[..., 1] - baseline_target[..., 1]) / 0.1
    weights = iron_threshold_weights(iron_relevance(teacher_delta))
    expected_area = float(np.sum(teacher_delta**2 * weights))
    assert ignored["zero_response_sera_iron_per_spike_mv2"] > 0.0
    assert np.isclose(ignored["response_sera_iron_mv2"], expected_area)
    assert np.isclose(
        ignored["response_sera_iron_per_spike_mv2"],
        ignored["zero_response_sera_iron_per_spike_mv2"],
    )
    assert np.isclose(ignored["response_sera_iron_per_spike_mv2"], expected_area / 12)


def test_response_sera_trace_terms_sum_to_response_sera():
    baseline_target, intervention_target = _response_fixture()
    intervention_prediction = intervention_target.copy()
    intervention_prediction[..., 1] *= 0.7

    terms = response_sera_trace_terms(
        baseline_target, intervention_prediction, baseline_target, intervention_target
    )
    result = response_sera(
        baseline_target, intervention_prediction, baseline_target, intervention_target
    )

    assert terms["area"].shape == (4,)
    np.testing.assert_array_equal(terms["teacher_spikes"], [3, 3, 3, 3])
    assert np.isclose(
        terms["area"].sum() / terms["teacher_spikes"].sum(),
        result["response_sera_iron_per_spike_mv2"],
    )
    assert 0.0 < result["response_sera_iron_mv2"] < result["zero_response_sera_iron_mv2"]


def test_spike_events_count_a_supra_threshold_run_once():
    scores = np.array([[0.1, 0.6, 0.9, 0.7, 0.1, 0.2, 0.8, 0.1]])
    events = spike_events(scores, 0.5)
    np.testing.assert_array_equal(np.flatnonzero(events[0]), [2, 6])
    np.testing.assert_array_equal(np.flatnonzero(spike_events(scores, 0.5, refractory_steps=None)[0]), [1, 2, 3, 6])


def test_spike_event_peaks_suppress_within_refractory_window():
    scores = np.array([[0.0, 0.9, 0.2, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0]])
    np.testing.assert_array_equal(np.flatnonzero(spike_event_peaks(scores, refractory_steps=2)[0]), [1, 7])
    np.testing.assert_array_equal(np.flatnonzero(spike_event_peaks(scores, refractory_steps=0)[0]), [1, 3, 7])


def test_event_count_matched_threshold_yields_requested_event_count():
    rng = np.random.default_rng(0)
    scores = rng.random((3, 400))
    for count in (1, 10, 40):
        thr = event_count_matched_threshold(scores, count)
        assert spike_events(scores, thr).sum() == count


def test_spike_f1_curve_scores_a_plateau_prediction_as_one_event():
    predicted = np.array([[0.0, 2.0, 2.0, 2.0, 0.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    curve = spike_f1_curve(predicted, target, threshold=1.0, tolerances_steps=(0,))
    assert curve["rows"][0]["f1"] == 1.0
