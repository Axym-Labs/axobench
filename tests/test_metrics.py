import numpy as np
import pytest

from axobench.metrics import (
    binary_auc,
    compute_core_metric_set,
    dynamics_sera,
    response_sera,
    root_sera,
    sera,
    soma_rmse,
    spike_auc,
    spike_event_confusion,
    spike_f1_curve,
    voltage_sera,
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


def test_root_sera_matches_rmse_when_relevance_is_constant():
    error = np.array([1.0, -2.0, 2.0])
    relevance = np.ones_like(error)

    assert np.isclose(root_sera(error, relevance), np.sqrt(np.mean(error ** 2)))


def test_sera_matches_mse_and_is_root_sera_squared():
    error = np.array([1.0, -2.0, 2.0])
    relevance = np.ones_like(error)

    squared = sera(error, relevance)

    assert np.isclose(squared, np.mean(error**2))
    assert np.isclose(squared, root_sera(error, relevance) ** 2)


def test_voltage_sera_emphasizes_relevant_teacher_regions():
    target = np.zeros((1, 12, 2), dtype=np.float32)
    prediction = target.copy()
    raw_soma_mv = np.full((1, 12), -67.0, dtype=np.float32)
    raw_soma_mv[0, 6] = -20.0
    target[0, 6, 0] = 1.0
    prediction[0, 6, 1] = (-67.0 + 67.7) * 0.1
    target[..., 1] = (-67.0 + 67.7) * 0.1

    result = voltage_sera(prediction, target, raw_soma_mv=raw_soma_mv, ignore_start=0)

    assert result["voltage_sera_mv"] > result["voltage_rmse_mv"]
    assert result["relevance_support_ge_0_9"] > 0


def test_dynamics_sera_complements_voltage_level_error():
    target = np.zeros((1, 12, 2), dtype=np.float32)
    target[0, :, 1] = np.linspace(-0.4, 0.4, 12)
    shifted = target.copy()
    shifted[..., 1] += 0.5
    distorted = target.copy()
    distorted[0, 5:7, 1] += np.array([0.5, -0.5])

    shifted_result = dynamics_sera(shifted, target, ignore_start=0)
    distorted_result = dynamics_sera(distorted, target, ignore_start=0)
    faster_sampling_result = dynamics_sera(distorted, target, timestep_ms=0.5, ignore_start=0)

    assert np.isclose(shifted_result["dynamics_sera_mv_per_ms"], 0.0, atol=1e-6)
    assert distorted_result["dynamics_sera_mv_per_ms"] > 0.0
    assert np.isclose(
        faster_sampling_result["dynamics_sera_mv_per_ms"],
        2.0 * distorted_result["dynamics_sera_mv_per_ms"],
    )


def test_spike_f1_curve_uses_one_to_one_event_matching():
    predicted = np.array([[0.0, 2.0, 2.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0, 1.0]])

    exact = spike_f1_curve(predicted, target, threshold=1.0, tolerances_steps=(0,))
    loose = spike_f1_curve(predicted, target, threshold=1.0, tolerances_steps=(0, 1))
    confusion = spike_event_confusion(predicted >= 1.0, target, tolerance_steps=1)

    assert exact["rows"][0]["f1"] == 0.5
    assert loose["rows"][1]["f1"] == 1.0
    assert loose["f1_area"] == 0.75
    assert loose["squared_f1_deficit_area"] == 0.125
    assert loose["squared_deficit_f1_score"] == 0.875
    assert confusion == {"tp": 2, "fp": 0, "fn": 0}


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

    result = compute_core_metric_set(prediction, target, ignore_start=0, spike_tolerances_steps=(0, 1, 2))

    assert result["voltage_sera_mv"] == 0.0
    assert result["voltage_sera_mv2"] == 0.0
    assert result["dynamics_sera_mv_per_ms"] == 0.0
    assert result["dynamics_sera_mv2_per_ms2"] == 0.0
    assert result["spike_f1_area"] == 1.0
    assert result["spike_squared_f1_deficit_area"] == 0.0
    assert result["spike_squared_deficit_f1_score"] == 1.0
    assert result["spike_threshold_kind"] == "prevalence_matched"
    assert "voltage_rmse_mv" not in result


def test_compute_core_metric_set_includes_response_sera_when_pair_is_given():
    baseline = np.zeros((1, 8, 2), dtype=np.float32)
    intervention = baseline.copy()
    intervention[..., 1] = 0.5

    result = compute_core_metric_set(
        baseline,
        baseline,
        intervention_prediction=intervention,
        intervention_target=intervention,
        ignore_start=2,
    )

    assert result["response_sera_mv"] == 0.0
    assert result["response_sera_mv2"] == 0.0
    assert result["response"]["response_sera_mv"] == 0.0


def test_response_sera_is_zero_for_correct_intervention_delta():
    baseline_target = np.zeros((1, 8, 2), dtype=np.float32)
    intervention_target = baseline_target.copy()
    intervention_target[..., 1] = 0.5
    baseline_prediction = baseline_target.copy()
    intervention_prediction = intervention_target.copy()

    correct = response_sera(baseline_prediction, intervention_prediction, baseline_target, intervention_target)
    wrong = response_sera(baseline_prediction, baseline_prediction, baseline_target, intervention_target)

    assert correct["response_sera_mv"] == 0.0
    assert wrong["response_sera_mv"] > 0.0
