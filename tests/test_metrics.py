import numpy as np
import pytest

from axobench.relevance import iron_relevance, iron_threshold_weights
from axobench.metrics import (
    AP_POST_STEPS,
    ap_window_ends,
    ap_window_mask,
    binary_auc,
    compute_core_metric_set,
    iron_dynamics_sera,
    iron_voltage_sera,
    response_sera,
    response_sera_trace_terms,
    sera_skill_trace_terms,
    sera_skills,
    soma_rmse,
    spike_auc,
    spike_event_confusion,
    spike_event_matches,
    spike_f1_curve,
    event_count_matched_threshold,
    spike_event_peaks,
    spike_events,
    metric_set,
    subthreshold_voltage_error,
    voltage_event_steps,
    voltage_event_thresholds,
    subthreshold_voltage_point_terms,
    subthreshold_voltage_trace_terms,
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
    assert result["subthreshold_rmse_mv"] == 0.0
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


def _subthreshold_case(steps: int = 60):
    rng = np.random.default_rng(3)
    raw = -70.0 + rng.normal(size=(2, steps))
    raw[0, 20:22] = (-20.0, -40.0)  # action potential
    raw[1, 35] = -25.0
    target = np.zeros((2, steps, 2))
    target[0, 20, 0] = 1.0
    target[1, 35, 0] = 1.0
    target[..., 1] = (raw + 67.7) * 0.1
    return target, raw


def _with_voltage(target: np.ndarray, voltage_mv: np.ndarray) -> np.ndarray:
    prediction = target.copy()
    prediction[..., 1] = (voltage_mv + 67.7) * 0.1
    return prediction


def test_subthreshold_error_is_zero_for_perfect_prediction():
    target, raw = _subthreshold_case()
    result = subthreshold_voltage_error(
        target, target, predicted_spikes=target[..., 0], raw_soma_mv=raw
    )
    assert result["subthreshold_rmse_mv"] == pytest.approx(0.0, abs=1e-12)
    assert result["subthreshold_skill"] == pytest.approx(1.0)
    assert result["retained_points"] == 2 * (60 - 7)


def test_subthreshold_skill_is_zero_for_per_trace_mean_prediction():
    target, raw = _subthreshold_case()
    keep = np.ones(raw.shape, dtype=bool)
    keep[0, 18:25] = keep[1, 33:40] = False
    mean = np.array([raw[i, keep[i]].mean() for i in range(2)])
    prediction = _with_voltage(target, np.repeat(mean[:, None], raw.shape[1], axis=1))
    result = subthreshold_voltage_error(
        prediction, target, predicted_spikes=target[..., 0], raw_soma_mv=raw
    )
    assert result["subthreshold_skill"] == pytest.approx(0.0, abs=1e-9)
    assert result["subthreshold_rmse_mv"] == pytest.approx(
        np.sqrt(np.mean((raw - mean[:, None])[keep] ** 2))
    )


def test_subthreshold_rmse_equals_constant_offset():
    target, raw = _subthreshold_case()
    prediction = _with_voltage(target, raw + 2.5)
    result = subthreshold_voltage_error(
        prediction, target, predicted_spikes=target[..., 0], raw_soma_mv=raw
    )
    assert result["subthreshold_rmse_mv"] == pytest.approx(2.5)


def test_subthreshold_error_ignores_teacher_and_predicted_ap_windows():
    target, raw = _subthreshold_case()
    predicted_spikes = target[..., 0].copy()
    predicted_spikes[0, 45] = 1.0  # spurious predicted spike
    voltage = raw.copy()
    voltage[0, 18:25] += 30.0  # teacher AP window [t-2, t+4]
    voltage[0, 43:50] -= 30.0  # predicted AP window
    voltage[1, 33:40] -= 30.0
    prediction = _with_voltage(target, voltage)
    result = subthreshold_voltage_error(
        prediction, target, predicted_spikes=predicted_spikes, raw_soma_mv=raw
    )
    assert result["subthreshold_rmse_mv"] == pytest.approx(0.0, abs=1e-9)
    assert result["retained_points"] == 2 * (60 - 7) - 7

    voltage[0, 50] -= 3.0  # first step after the predicted window counts
    prediction = _with_voltage(target, voltage)
    terms = subthreshold_voltage_trace_terms(
        prediction, target, predicted_spikes=predicted_spikes, raw_soma_mv=raw
    )
    assert terms["squared_error"][0] == pytest.approx(9.0)
    assert terms["squared_error"][1] == pytest.approx(0.0, abs=1e-9)


def test_subthreshold_error_masks_teacher_spikes_before_ignore_start():
    target, raw = _subthreshold_case()
    voltage = raw.copy()
    voltage[0, 22:25] += 5.0  # after ignore_start but inside the teacher AP window
    prediction = _with_voltage(target, voltage)
    result = subthreshold_voltage_error(
        prediction,
        target,
        predicted_spikes=target[:, 22:, 0],
        raw_soma_mv=raw,
        ignore_start=22,
    )
    assert result["subthreshold_rmse_mv"] == pytest.approx(0.0, abs=1e-9)


def test_subthreshold_error_clips_both_voltages_at_minus_55_mv():
    target, raw = _subthreshold_case()
    raw[0, 10] = -50.0
    target[0, 10, 1] = (-50.0 + 67.7) * 0.1
    voltage = raw.copy()
    voltage[0, 10] = -30.0
    voltage[1, 10] = -55.0 + (-55.0 - raw[1, 10])  # above the clip
    voltage[1, 12] = raw[1, 12] - 4.0  # below the clip, error kept
    prediction = _with_voltage(target, voltage)
    terms = subthreshold_voltage_trace_terms(
        prediction, target, predicted_spikes=target[..., 0], raw_soma_mv=raw
    )
    assert terms["squared_error"][0] == pytest.approx(0.0, abs=1e-9)
    assert terms["squared_error"][1] == pytest.approx(
        (-55.0 - raw[1, 10]) ** 2 + 16.0
    )


SLOW_RECOVERY_MV = [-80.0, -77.0, -75.0, -73.4, -72.2, -71.1, -70.5, -70.2]  # flat from +7
FAST_RECOVERY_MV = [-75.0, -71.5, -71.0]  # flat from +3


def _isolated_spike_teacher(recovery_mv, *, traces=2, spikes_per_trace=12, steps=600):
    """Flat -70 mV teacher; each spike (+20 mV) is followed by `recovery_mv` at lags 1, 2, ..."""
    raw = np.full((traces, steps), -70.0)
    spikes = np.zeros((traces, steps))
    for t in 10 + 40 * np.arange(spikes_per_trace):
        spikes[:, t] = 1.0
        raw[:, t] = 20.0
        raw[:, t + 1 : t + 1 + len(recovery_mv)] = recovery_mv
    return raw, spikes


def test_ap_window_mask_accepts_per_trace_window_ends():
    spikes = np.zeros((2, 12))
    spikes[:, 5] = 1.0
    mask = ap_window_mask(spikes, np.array([1, 3]))
    assert np.flatnonzero(mask[0]).tolist() == [3, 4, 5, 6]
    assert np.flatnonzero(mask[1]).tolist() == [3, 4, 5, 6, 7, 8]
    np.testing.assert_array_equal(
        ap_window_mask(spikes), ap_window_mask(spikes, np.full(2, AP_POST_STEPS))
    )


def test_ap_window_end_is_first_flat_lag_of_mean_isolated_waveform():
    raw, spikes = _isolated_spike_teacher(SLOW_RECOVERY_MV)
    windows = ap_window_ends(raw, spikes)
    assert windows["end_steps_by_cell_type"] == {"all": 7}
    assert windows["fallback_cell_types"] == []
    assert windows["post_steps"].tolist() == [7, 7]
    # The flatness threshold is 1 mV/ms times the time step.
    assert ap_window_ends(raw, spikes, timestep_ms=2.0)["end_steps_by_cell_type"] == {"all": 4}


def test_ap_window_ends_group_traces_by_trace_id_prefix():
    slow_raw, slow_spikes = _isolated_spike_teacher(SLOW_RECOVERY_MV)
    fast_raw, fast_spikes = _isolated_spike_teacher(FAST_RECOVERY_MV)
    raw = np.concatenate([slow_raw, fast_raw])[[0, 2, 1, 3]]
    spikes = np.concatenate([slow_spikes, fast_spikes])[[0, 2, 1, 3]]
    ids = ["cellA:val:0", "cellB:val:1", "cellA:val:2", "cellB:val:3"]
    windows = ap_window_ends(raw, spikes, trace_ids=ids)
    assert windows["end_steps_by_cell_type"] == {"cellA": 7, "cellB": 3}
    assert windows["post_steps"].tolist() == [7, 3, 7, 3]
    pooled = ap_window_ends(raw, spikes, trace_ids=["0", "1", "2", "3"])
    assert list(pooled["end_steps_by_cell_type"]) == ["all"]


def test_ap_window_ends_fall_back_without_enough_isolated_spikes_or_voltage():
    slow_raw, slow_spikes = _isolated_spike_teacher(SLOW_RECOVERY_MV)
    few_raw, few_spikes = _isolated_spike_teacher(SLOW_RECOVERY_MV, spikes_per_trace=9)
    windows = ap_window_ends(
        np.concatenate([slow_raw, few_raw]),
        np.concatenate([slow_spikes, few_spikes]),
        trace_ids=["a:0", "a:1", "b:0", "b:1"],  # b has 18 isolated spikes
    )
    assert windows["end_steps_by_cell_type"] == {"a": 7, "b": AP_POST_STEPS}
    assert windows["fallback_cell_types"] == ["b"]
    no_voltage = ap_window_ends(None, slow_spikes)
    assert no_voltage["end_steps_by_cell_type"] == {"all": AP_POST_STEPS}
    assert no_voltage["fallback_cell_types"] == ["all"]


def test_subthreshold_error_uses_the_teacher_window_end():
    raw, spikes = _isolated_spike_teacher(SLOW_RECOVERY_MV)
    target = np.zeros(raw.shape + (2,))
    target[..., 0] = spikes
    target[..., 1] = (raw + 67.7) * 0.1
    voltage = raw.copy()
    voltage[0, 10 + 6] += 3.0  # inside the +7 window, outside the fixed +4 window
    prediction = _with_voltage(target, voltage)
    result = subthreshold_voltage_error(
        prediction, target, predicted_spikes=spikes, raw_soma_mv=raw
    )
    assert result["subthreshold_rmse_mv"] == pytest.approx(0.0, abs=1e-12)
    assert result["retained_points"] == 2 * (600 - 12 * 10)
    fixed = subthreshold_voltage_trace_terms(
        prediction,
        target,
        predicted_spikes=spikes,
        raw_soma_mv=raw,
        ap_post_steps=np.full(2, AP_POST_STEPS),
    )
    assert fixed["squared_error"][0] == pytest.approx(9.0)


def test_sera_skill_is_one_for_perfect_and_zero_for_reference_prediction():
    target, raw = _subthreshold_case()
    reference = float(np.minimum(raw, -55.0).mean())
    perfect = sera_skills(sera_skill_trace_terms(_with_voltage(target, raw), target, reference_mv=reference, raw_soma_mv=raw))
    assert perfect["voltage_sera_skill"] == pytest.approx(1.0)
    assert perfect["dynamics_sera_skill"] == pytest.approx(1.0)
    ceiling = perfect["voltage_sera_ceiling_skill"]
    assert 0.0 < ceiling < 1.0
    assert perfect["voltage_sera_skill_of_ceiling"] == pytest.approx(1.0 / ceiling)

    constant = sera_skills(
        sera_skill_trace_terms(_with_voltage(target, np.full_like(raw, reference)), target, reference_mv=reference, raw_soma_mv=raw)
    )
    assert constant["voltage_sera_skill"] == pytest.approx(0.0)
    assert constant["dynamics_sera_skill"] == pytest.approx(0.0)
    assert constant["voltage_sera_skill_of_ceiling"] == pytest.approx(0.0)


def test_sera_skill_of_clipped_teacher_is_the_ceiling():
    target, raw = _subthreshold_case()
    terms = sera_skill_trace_terms(
        _with_voltage(target, np.minimum(raw, -55.0)), target, reference_mv=-70.0, raw_soma_mv=raw
    )
    skills = sera_skills(terms)
    assert skills["voltage_sera_skill"] == pytest.approx(skills["voltage_sera_ceiling_skill"])
    assert skills["voltage_sera_skill_of_ceiling"] == pytest.approx(1.0)
    # The voltage area per spike is the reported Voltage SERA.
    sera = iron_voltage_sera(_with_voltage(target, np.minimum(raw, -55.0)), target, raw_soma_mv=raw)
    assert terms["voltage"].sum() / terms["teacher_spikes"].sum() == pytest.approx(sera["voltage_sera_iron_per_spike_mv2"])
    # Subsets are ratios of summed trace areas.
    first = sera_skills(terms, np.array([0]))
    assert first["voltage_sera_skill"] == pytest.approx(1.0 - terms["voltage"][0] / terms["voltage_reference"][0])


def test_spike_event_matches_pair_events_one_to_one():
    predicted = np.zeros((2, 20))
    target = np.zeros((2, 20))
    predicted[0, [3, 4, 15]] = 1.0
    target[0, [4, 10]] = 1.0
    predicted[1, 7] = 1.0
    target[1, 9] = 1.0
    matches = spike_event_matches(predicted, target, tolerance_steps=2)
    assert matches.tolist() == [[0, 3, 4], [1, 7, 9]]
    assert spike_event_confusion(predicted, target, tolerance_steps=2) == {"tp": 2, "fp": 2, "fn": 1}


def test_subthreshold_point_terms_sum_to_trace_terms():
    target, raw = _subthreshold_case()
    prediction = _with_voltage(target, raw + 1.0)
    kwargs = dict(predicted_spikes=target[:, 5:, 0], raw_soma_mv=raw, ignore_start=5)
    points = subthreshold_voltage_point_terms(prediction, target, **kwargs)
    terms = subthreshold_voltage_trace_terms(prediction, target, **kwargs)
    assert points["keep"].shape == (2, 55)
    assert np.allclose((points["error"] ** 2).sum(axis=1), terms["squared_error"])
    assert np.allclose(points["error"][points["keep"]], 1.0)
    assert np.all(points["error"][~points["keep"]] == 0.0)


def _metric_set_case(traces: int = 4, steps: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    voltage = -70.0 + np.cumsum(rng.normal(0.0, 0.3, (traces, steps)), axis=1) * 0.1
    spikes = np.zeros((traces, steps), dtype=bool)
    spikes[:, [800, 1500, 2400]] = True
    voltage[spikes] += 80.0
    scores = np.where(spikes, 5.0, -5.0)
    return scores, voltage, spikes


def test_metric_set_is_perfect_for_the_true_output():
    scores, voltage, spikes = _metric_set_case()
    result = metric_set(scores, voltage, spikes, voltage, spike_threshold=0.0)
    assert result["mean_f1"] == pytest.approx(1.0)
    assert result["event_error_mv"] == pytest.approx(0.0)
    assert result["subthreshold_error_mv"] == pytest.approx(0.0)


def test_missing_action_potential_raises_only_the_event_error():
    scores, voltage, spikes = _metric_set_case()
    predicted = voltage.copy()
    predicted[spikes] -= 80.0
    result = metric_set(scores, predicted, spikes, voltage, spike_threshold=0.0)
    assert result["event_error_mv"] > 1.0
    assert result["subthreshold_error_mv"] == pytest.approx(0.0)


def test_event_and_subthreshold_error_recombine_to_the_rmse():
    scores, voltage, spikes = _metric_set_case()
    predicted = voltage * 0.9 + 1.0
    result = metric_set(scores, predicted, spikes, voltage, spike_threshold=0.0)
    share = result["event_share"]
    rmse2 = np.mean((predicted[:, 500:] - voltage[:, 500:]) ** 2)
    assert share * result["event_error_mv"] ** 2 + (1 - share) * result[
        "subthreshold_error_mv"
    ] ** 2 == pytest.approx(rmse2)


def test_event_steps_cover_spike_windows_and_rare_voltages_with_fixed_thresholds():
    _, voltage, spikes = _metric_set_case()
    thresholds = voltage_event_thresholds(voltage[:, 500:], spikes[:, 500:])
    events = voltage_event_steps(voltage[:, 500:], spikes[:, 500:], thresholds)
    assert events[:, 300 - 5 : 300 + 6].all()
    assert 0.01 < events.mean() < 0.05
    shifted = voltage[:, 500:] + 5.0
    assert voltage_event_steps(shifted, spikes[:, 500:], thresholds).mean() > 0.5
