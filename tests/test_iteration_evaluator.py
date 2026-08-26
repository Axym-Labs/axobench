from __future__ import annotations

import csv
import json
from threading import Lock
import time

import numpy as np

import axobench.benchmark as benchmark
from axobench.benchmark import iteration_evaluator
from axobench.metrics import root_sera, sera

from axobench.benchmark.iteration_evaluator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INTERVENTION_SAMPLES,
    DEFAULT_MULTIPLE_MORPHOLOGY_INTERVENTION_SAMPLES,
    DEFAULT_MULTIPLE_MORPHOLOGY_STANDARD_SAMPLES,
    DEFAULT_INTERVENTION_WORKERS,
    DEFAULT_STANDARD_SAMPLES,
    EvaluationConfig,
    PairedInterventionPredictions,
    create_model_evaluation_cache,
    evaluate_model,
    evaluate_model_against_cache,
    evaluate_predictions,
    default_iteration_sample_limits,
    compare_predictions,
    format_model_comparison,
    format_evaluation_report,
    load_model_evaluation_cache,
    write_evaluation_report,
    write_model_evaluation_cache,
    write_model_comparison,
)


def _ordinary_fixture(*, delayed_spikes: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    traces, steps = 8, 80
    time = np.arange(steps, dtype=np.float64)
    soma_mv = np.stack(
        [-67.7 + 3.0 * np.sin((time + phase) / 9.0) for phase in range(traces)]
    )
    target = np.zeros((traces, steps, 2), dtype=np.float64)
    target[..., 1] = (soma_mv + 67.7) * 0.1
    for trace in range(traces):
        target[trace, (20 + trace) % steps, 0] = 1.0
        target[trace, (55 + trace) % steps, 0] = 1.0

    prediction = target.copy()
    if delayed_spikes:
        prediction[..., 0] = 0.0
        for trace in range(traces):
            for spike in np.flatnonzero(target[trace, :, 0]):
                prediction[trace, min(steps - 1, spike + delayed_spikes), 0] = 1.0

    inputs = np.zeros((traces, steps, 4), dtype=np.int8)
    inputs[:, 10::10, 0] = 1
    inputs[:, 15::20, 2] = -1
    trace_ids = [f"specimen_{trace % 2}:trace_{trace}" for trace in range(traces)]
    return prediction, target, inputs, trace_ids


def _intervention_fixture(*, ignored: bool = False) -> PairedInterventionPredictions:
    prediction, target, inputs, _ = _ordinary_fixture()
    baseline_target = target.copy()
    intervention_target = target.copy()
    intervention_target[..., 1] += 0.5  # +5 mV response
    intervention_target[:, 40, 0] = 1.0
    baseline_prediction = prediction.copy()
    intervention_prediction = baseline_prediction.copy() if ignored else intervention_target.copy()
    return PairedInterventionPredictions(
        name="excitatory_drive",
        baseline_prediction=baseline_prediction,
        intervention_prediction=intervention_prediction,
        baseline_target=baseline_target,
        intervention_target=intervention_target,
    )


def _by_id(report: dict, section: str) -> dict[str, dict]:
    return {row["id"]: row for row in report[section]}


def test_iteration_defaults_match_uncertainty_and_throughput_study() -> None:
    assert DEFAULT_STANDARD_SAMPLES == 120
    assert DEFAULT_INTERVENTION_SAMPLES == 40
    assert DEFAULT_MULTIPLE_MORPHOLOGY_STANDARD_SAMPLES == 120
    assert DEFAULT_MULTIPLE_MORPHOLOGY_INTERVENTION_SAMPLES == 40
    assert default_iteration_sample_limits() == (120, 40)
    assert default_iteration_sample_limits(multiple_morphologies=True) == (120, 40)
    assert DEFAULT_INTERVENTION_WORKERS == 4
    assert DEFAULT_BATCH_SIZE == 128


def test_iteration_evaluator_is_available_from_public_benchmark_api() -> None:
    assert benchmark.EvaluationConfig is EvaluationConfig
    assert benchmark.evaluate_model is evaluate_model
    assert benchmark.compare_predictions is compare_predictions
    assert benchmark.format_model_comparison is format_model_comparison
    assert benchmark.write_model_comparison is write_model_comparison
    assert benchmark.evaluate_predictions is evaluate_predictions
    assert benchmark.default_iteration_sample_limits is default_iteration_sample_limits
    assert benchmark.format_evaluation_report is format_evaluation_report
    assert benchmark.write_evaluation_report is write_evaluation_report


def test_perfect_predictions_clear_available_attention_rows() -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    report = evaluate_predictions(
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        interventions=[_intervention_fixture()],
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )

    metrics = _by_id(report, "metrics")
    assert metrics["spike_f1_exact"]["value"] == 1.0
    assert metrics["spike_f1_5ms"]["value"] == 1.0
    assert metrics["spike_mean_f1_0_5ms"]["value"] == 1.0
    assert metrics["spike_f1_area_0_5ms"]["value"] == 1.0
    assert metrics["voltage_sera_mv2"]["value"] == 0.0
    assert metrics["voltage_sera_mv"]["value"] == 0.0
    assert metrics["dynamics_sera_mv2_per_ms2"]["value"] == 0.0
    assert metrics["dynamics_sera_mv_per_ms"]["value"] == 0.0
    assert metrics["response_sera_mv2"]["value"] == 0.0
    assert metrics["response_sera_mv"]["value"] == 0.0
    assert [row["tolerance_steps"] for row in report["spike_f1_curve"]] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert "raw_voltage_rmse_mv" not in metrics
    assert "clipped_voltage_rmse_mv" not in metrics
    assert "spike_auc" not in metrics

    diagnostics = report["diagnostics"]
    assert len(diagnostics) == 24
    assert all(row["attention"] in {"NO", "NA"} for row in diagnostics)
    assert all(row["basis"] for row in diagnostics)


def test_delayed_spikes_separate_exact_and_tolerant_event_fidelity() -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture(delayed_spikes=3)
    report = evaluate_predictions(
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )
    metrics = _by_id(report, "metrics")
    diagnostics = _by_id(report, "diagnostics")

    assert metrics["spike_f1_exact"]["value"] == 0.0
    assert metrics["spike_f1_5ms"]["value"] == 1.0
    assert diagnostics["spike_exact_fidelity"]["attention"] == "YES"
    assert diagnostics["spike_timing_tolerance_gap"]["attention"] == "YES"
    assert diagnostics["spike_timing_bias"]["attention"] == "YES"


def test_paired_comparison_identifies_consistent_candidate_improvement() -> None:
    candidate, target, inputs, trace_ids = _ordinary_fixture()
    baseline, _, _, _ = _ordinary_fixture(delayed_spikes=3)
    baseline[..., 1] = 0.0
    candidate_intervention = _intervention_fixture()
    baseline_intervention = _intervention_fixture(ignored=True)

    comparison = compare_predictions(
        candidate,
        baseline,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        candidate_interventions=[candidate_intervention],
        baseline_interventions=[baseline_intervention],
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
        bootstrap_replicates=50,
        seed=17,
    )

    rows = {row["id"]: row for row in comparison["comparison"]}
    assert set(rows) == {
        "voltage_sera_mv2",
        "voltage_sera_mv",
        "dynamics_sera_mv2_per_ms2",
        "dynamics_sera_mv_per_ms",
        "spike_mean_f1_0_5ms",
        "spike_f1_area_0_5ms",
        "spike_f1_exact",
        "spike_f1_5ms",
        "response_sera_mv2",
        "response_sera_mv",
    }
    assert all(
        row["assessment"] == "improved"
        for identifier, row in rows.items()
        if identifier != "spike_f1_5ms"
    )
    assert rows["spike_f1_5ms"]["assessment"] == "uncertain"
    assert rows["voltage_sera_mv"]["delta"] < 0.0
    assert rows["spike_f1_exact"]["delta"] > 0.0
    rendered = format_model_comparison(comparison)
    assert "PAIRED COMPARISON" in rendered
    assert "95% CI" in rendered


def test_paired_comparison_marks_identical_models_uncertain() -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()

    comparison = compare_predictions(
        prediction,
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
        bootstrap_replicates=20,
        seed=19,
    )

    assert all(
        row["delta"] == 0.0
        for row in comparison["comparison"]
        if row["delta"] is not None
    )
    assert all(row["assessment"] == "uncertain" for row in comparison["comparison"])


def test_paired_comparison_writes_json_and_tidy_csv(tmp_path) -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    comparison = compare_predictions(
        prediction,
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
        bootstrap_replicates=10,
    )
    json_path = write_model_comparison(comparison, tmp_path / "comparison.json")
    csv_path = write_model_comparison(comparison, tmp_path / "comparison.csv")

    assert json.loads(json_path.read_text(encoding="utf-8"))["bootstrap_replicates"] == 10
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {row["assessment"] for row in rows} == {"uncertain"}


def test_bootstrap_sera_statistics_match_metric_definition() -> None:
    rng = np.random.default_rng(29)
    error = rng.normal(size=(4, 30))
    relevance = rng.uniform(size=(4, 30))
    sums, counts = iteration_evaluator._sera_trace_statistics(error, relevance)
    indices = np.arange(error.shape[0])

    from_statistics = iteration_evaluator._root_sera_from_trace_statistics(sums, counts, indices)
    direct = root_sera(
        error,
        relevance,
        thresholds=iteration_evaluator._BOOTSTRAP_SERA_THRESHOLDS,
    )

    assert np.isclose(from_statistics, direct)
    assert np.isclose(
        iteration_evaluator._sera_from_trace_statistics(
            sums,
            counts,
            indices,
        ),
        sera(
            error,
            relevance,
            thresholds=iteration_evaluator._BOOTSTRAP_SERA_THRESHOLDS,
        ),
    )


def test_invalid_outputs_raise_numerical_and_physiological_attention_without_crashing() -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    prediction[4, 30, 1] = np.nan
    prediction[5, 31, 1] = (80.0 + 67.7) * 0.1
    prediction[6, 32, 1] = (60.0 + 67.7) * 0.1
    prediction[6, 33, 1] = (-100.0 + 67.7) * 0.1

    report = evaluate_predictions(
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )
    diagnostics = _by_id(report, "diagnostics")

    assert diagnostics["nonfinite_outputs"]["attention"] == "YES"
    assert diagnostics["voltage_bounds"]["attention"] == "YES"
    assert diagnostics["voltage_slew"]["attention"] == "YES"


def test_ignored_teacher_intervention_raises_response_attention() -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    report = evaluate_predictions(
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        interventions=[_intervention_fixture(ignored=True)],
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )
    diagnostics = _by_id(report, "diagnostics")

    assert diagnostics["intervention_response_skill"]["attention"] == "YES"
    assert diagnostics["intervention_direction"]["attention"] == "YES"
    assert diagnostics["intervention_gain"]["attention"] == "YES"


def test_report_format_is_deterministic_and_bounded() -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    report = evaluate_predictions(
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )

    rendered = format_evaluation_report(report)

    assert "CORE METRICS" in rendered
    assert "ATTENTION PANEL" in rendered
    assert "spike_f1_exact" in rendered
    assert "mV/ms" in rendered
    assert "YES = early watch" in rendered
    assert len(rendered.splitlines()) <= 45
    assert rendered == format_evaluation_report(report)


def test_model_runner_batches_inference_and_discovers_interventions(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    val_root = dataset_root / "val"
    intervention_root = dataset_root / "interventions" / "drive"
    val_root.mkdir(parents=True)
    intervention_root.mkdir(parents=True)

    prediction, target, inputs, trace_ids = _ordinary_fixture()
    inputs = np.zeros((8, 80, 4), dtype=np.float32)
    inputs[..., :2] = target
    np.savez_compressed(
        val_root / "val.npz",
        inputs=inputs,
        targets=target,
        soma_mv=target[..., 1] / 0.1 - 67.7,
        sample_ids=np.asarray(trace_ids),
    )

    intervention = _intervention_fixture()
    baseline_inputs = np.zeros((8, 80, 4), dtype=np.float32)
    baseline_inputs[..., :2] = intervention.baseline_target
    intervention_inputs = np.zeros((8, 80, 4), dtype=np.float32)
    intervention_inputs[..., :2] = intervention.intervention_target
    np.savez_compressed(
        intervention_root / "pair.npz",
        baseline_inputs=baseline_inputs,
        baseline_targets=intervention.baseline_target,
        intervention_inputs=intervention_inputs,
        intervention_targets=intervention.intervention_target,
        sample_ids=np.asarray(trace_ids),
    )

    batch_sizes: list[int] = []

    def model_fn(batch: np.ndarray) -> np.ndarray:
        batch_sizes.append(batch.shape[0])
        return batch[..., :2]

    report = evaluate_model(
        model_fn,
        dataset_root=dataset_root,
        model_id="identity",
        max_samples=6,
        batch_size=2,
        max_intervention_samples=4,
        multiple_morphologies=True,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )

    assert report["samples"] == 6
    assert report["evaluation_seconds"] > 0.0
    assert [row["name"] for row in report["interventions"]] == ["drive"]
    assert max(batch_sizes) <= 2
    assert len(batch_sizes) == 7  # 3 val + 2 baseline + 2 intervention batches


def test_cached_baseline_comparison_reuses_predictions_and_report(tmp_path) -> None:
    val_root = tmp_path / "dataset" / "val"
    val_root.mkdir(parents=True)
    _, target, _, trace_ids = _ordinary_fixture()
    inputs = np.zeros((8, 80, 4), dtype=np.float32)
    inputs[..., :2] = target
    np.savez_compressed(
        val_root / "val.npz",
        inputs=inputs,
        targets=target,
        soma_mv=target[..., 1] / 0.1 - 67.7,
        sample_ids=np.asarray(trace_ids),
    )
    baseline_calls = 0

    def baseline_fn(batch: np.ndarray) -> np.ndarray:
        nonlocal baseline_calls
        baseline_calls += 1
        return np.zeros((*batch.shape[:2], 2), dtype=np.float32)

    cache = create_model_evaluation_cache(
        baseline_fn,
        dataset_root=tmp_path / "dataset",
        model_id="frozen-baseline",
        max_samples=6,
        max_interventions=0,
        multiple_morphologies=True,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )
    cache_path = write_model_evaluation_cache(cache, tmp_path / "baseline-cache.npz")
    calls_after_cache = baseline_calls

    loaded = load_model_evaluation_cache(cache_path)
    comparison = evaluate_model_against_cache(
        lambda batch: batch[..., :2],
        baseline_cache=loaded,
        dataset_root=tmp_path / "dataset",
        candidate_id="candidate",
        max_samples=6,
        max_interventions=0,
        multiple_morphologies=True,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
        bootstrap_replicates=20,
        bootstrap_seed=23,
    )

    assert baseline_calls == calls_after_cache
    assert comparison["baseline"] == loaded.report
    assert comparison["baseline_cache"] == str(cache_path.resolve())
    voltage = next(row for row in comparison["comparison"] if row["id"] == "voltage_sera_mv")
    assert voltage["assessment"] == "improved"


def test_cached_baseline_rejects_changed_trace_selection(tmp_path) -> None:
    val_root = tmp_path / "dataset" / "val"
    val_root.mkdir(parents=True)
    _, target, _, trace_ids = _ordinary_fixture()
    inputs = np.zeros((8, 80, 4), dtype=np.float32)
    np.savez_compressed(
        val_root / "val.npz",
        inputs=inputs,
        targets=target,
        sample_ids=np.asarray(trace_ids),
    )
    cache = create_model_evaluation_cache(
        lambda batch: np.zeros((*batch.shape[:2], 2), dtype=np.float32),
        dataset_root=tmp_path / "dataset",
        max_samples=6,
        max_interventions=0,
        multiple_morphologies=True,
        config=EvaluationConfig(ignore_start=0),
    )

    with np.testing.assert_raises_regex(ValueError, "baseline cache protocol"):
        evaluate_model_against_cache(
            lambda batch: np.zeros((*batch.shape[:2], 2), dtype=np.float32),
            baseline_cache=cache,
            dataset_root=tmp_path / "dataset",
            max_samples=4,
            max_interventions=0,
            multiple_morphologies=True,
            config=EvaluationConfig(ignore_start=0),
            bootstrap_replicates=10,
        )


def test_spike_free_evidence_is_marked_unavailable_instead_of_failed() -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    prediction[..., 0] = 0.0
    target[..., 0] = 0.0

    report = evaluate_predictions(
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )
    diagnostics = _by_id(report, "diagnostics")

    for identifier in (
        "spike_exact_fidelity",
        "spike_5ms_fidelity",
        "spike_timing_tolerance_gap",
        "spike_count_error",
        "spike_timing_bias",
        "default_threshold_count",
    ):
        assert diagnostics[identifier]["attention"] == "NA"


def test_morphology_disparity_compares_error_across_trace_groups() -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    prediction[1::2, :, 1] += 1.0

    report = evaluate_predictions(
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )
    morphology = _by_id(report, "diagnostics")["morphology_error_disparity"]

    assert morphology["attention"] == "YES"
    assert morphology["value"] > 1.25


def test_model_runner_balances_grouped_morphology_shards(tmp_path) -> None:
    val_root = tmp_path / "dataset" / "val"
    val_root.mkdir(parents=True)
    traces, steps = 12, 20
    targets = np.zeros((traces, steps, 2), dtype=np.float32)
    targets[..., 1] = np.linspace(-0.2, 0.2, steps)
    inputs = np.zeros((traces, steps, 4), dtype=np.float32)
    inputs[..., :2] = targets
    sample_ids = np.asarray(
        [f"specimen_a:trace_{index}" for index in range(6)]
        + [f"specimen_b:trace_{index}" for index in range(6)]
    )
    np.savez_compressed(val_root / "val.npz", inputs=inputs, targets=targets, sample_ids=sample_ids)

    report = evaluate_model(
        lambda batch: batch[..., :2],
        dataset_root=tmp_path / "dataset",
        max_samples=6,
        batch_size=2,
        max_interventions=0,
        multiple_morphologies=True,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )

    morphology = _by_id(report, "diagnostics")["morphology_error_disparity"]
    assert morphology["attention"] == "NO"
    assert morphology["value"] == 1.0


def test_machine_readable_report_writes_json_or_tidy_csv_only_when_called(tmp_path) -> None:
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    report = evaluate_predictions(
        prediction,
        target,
        inputs=inputs,
        trace_ids=trace_ids,
        interventions=[_intervention_fixture()],
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
        model_id="candidate",
    )
    assert list(tmp_path.iterdir()) == []

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    write_evaluation_report(report, json_path)
    write_evaluation_report(report, csv_path)

    assert json.loads(json_path.read_text())["model_id"] == "candidate"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["row_type"] for row in rows} == {"metric", "diagnostic", "intervention"}
    assert len([row for row in rows if row["row_type"] == "metric"]) == 10
    assert len([row for row in rows if row["row_type"] == "diagnostic"]) == 24
    exact = next(row for row in rows if row["id"] == "spike_f1_exact")
    assert exact["model_id"] == "candidate"
    assert float(exact["value"]) == 1.0
    assert "morphology_mode" in exact


def test_intervention_loading_can_overlap_without_concurrent_model_calls(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    val_root = dataset_root / "val"
    val_root.mkdir(parents=True)
    prediction, target, inputs, trace_ids = _ordinary_fixture()
    inputs = np.zeros((8, 80, 4), dtype=np.float32)
    inputs[..., :2] = target
    np.savez_compressed(val_root / "val.npz", inputs=inputs, targets=target, sample_ids=np.asarray(trace_ids))
    for name in ("a", "b"):
        folder = dataset_root / "interventions" / name
        folder.mkdir(parents=True)
        (folder / "placeholder.npz").touch()

    active_loaders = 0
    max_active_loaders = 0
    model_active = 0
    lock = Lock()

    def fake_loader(path, *, max_samples, multiple_morphologies, morphology_id):
        nonlocal active_loaders, max_active_loaders
        with lock:
            active_loaders += 1
            max_active_loaders = max(max_active_loaders, active_loaders)
        time.sleep(0.03)
        with lock:
            active_loaders -= 1
        pair_inputs = np.zeros((1, 80, 4), dtype=np.float32)
        pair_inputs[..., :2] = target[:1]
        return {
            "baseline_inputs": pair_inputs,
            "baseline_targets": target[:1],
            "intervention_inputs": pair_inputs,
            "intervention_targets": target[:1],
        }

    def model_fn(batch):
        nonlocal model_active
        with lock:
            assert model_active == 0
            model_active += 1
        output = batch[..., :2]
        with lock:
            model_active -= 1
        return output

    monkeypatch.setattr(
        "axobench.benchmark.iteration_evaluator._load_intervention_directory",
        fake_loader,
    )
    evaluate_model(
        model_fn,
        dataset_root=dataset_root,
        max_samples=6,
        batch_size=2,
        max_intervention_samples=1,
        intervention_workers=2,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )

    assert max_active_loaders == 2


def test_intervention_sample_is_balanced_across_morphology_files(tmp_path) -> None:
    folder = tmp_path / "intervention"
    folder.mkdir()
    for morphology_index, morphology in enumerate(("specimen_100", "specimen_200"), start=1):
        inputs = np.full((4, 6, 3), morphology_index, dtype=np.int8)
        targets = np.zeros((4, 6, 2), dtype=np.float32)
        targets[..., 1] = morphology_index
        np.savez_compressed(
            folder / f"{morphology}_intervention_pair_00001.npz",
            baseline_inputs=inputs,
            baseline_targets=targets,
            intervention_inputs=inputs,
            intervention_targets=targets,
        )

    pair = iteration_evaluator._load_intervention_directory(
        folder,
        max_samples=4,
        multiple_morphologies=True,
    )

    assert pair["baseline_inputs"].shape[0] == 4
    assert np.count_nonzero(pair["baseline_targets"][:, 0, 1] == 1) == 2
    assert np.count_nonzero(pair["baseline_targets"][:, 0, 1] == 2) == 2
    assert pair["morphology_ids"].tolist() == ["specimen_100"] * 2 + ["specimen_200"] * 2


def test_single_morphology_is_default_for_validation_and_interventions(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    val_root = dataset_root / "val"
    intervention_root = dataset_root / "interventions" / "drive"
    val_root.mkdir(parents=True)
    intervention_root.mkdir(parents=True)
    traces, steps = 8, 12
    inputs = np.zeros((traces, steps, 4), dtype=np.float32)
    targets = np.zeros((traces, steps, 2), dtype=np.float32)
    sample_ids = np.asarray(
        [f"specimen_100:trace_{index}" for index in range(4)]
        + [f"specimen_200:trace_{index}" for index in range(4)]
    )
    np.savez_compressed(val_root / "val.npz", inputs=inputs, targets=targets, sample_ids=sample_ids)
    for morphology, value in (("specimen_100", 1.0), ("specimen_200", 2.0)):
        pair_inputs = np.full((4, steps, 4), value, dtype=np.float32)
        np.savez_compressed(
            intervention_root / f"{morphology}_pair.npz",
            baseline_inputs=pair_inputs,
            baseline_targets=targets[:4],
            intervention_inputs=pair_inputs,
            intervention_targets=targets[:4],
        )

    seen_values: list[float] = []

    def model_fn(batch: np.ndarray) -> np.ndarray:
        seen_values.extend(np.unique(batch).tolist())
        return np.zeros((*batch.shape[:2], 2), dtype=np.float32)

    report = evaluate_model(
        model_fn,
        dataset_root=dataset_root,
        max_samples=4,
        max_intervention_samples=4,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )

    assert report["morphology_mode"] == "single"
    assert report["morphology_ids"] == ["specimen_100"]
    assert report["interventions"][0]["samples"] == 4
    assert 2.0 not in seen_values


def test_requested_single_morphology_is_used_consistently(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    val_root = dataset_root / "val"
    val_root.mkdir(parents=True)
    traces, steps = 8, 12
    targets = np.zeros((traces, steps, 2), dtype=np.float32)
    inputs = np.zeros((traces, steps, 4), dtype=np.float32)
    inputs[4:] = 2.0
    sample_ids = np.asarray(
        [f"specimen_100:trace_{index}" for index in range(4)]
        + [f"specimen_200:trace_{index}" for index in range(4)]
    )
    np.savez_compressed(val_root / "val.npz", inputs=inputs, targets=targets, sample_ids=sample_ids)

    seen: list[float] = []

    def model_fn(batch: np.ndarray) -> np.ndarray:
        seen.extend(np.unique(batch).tolist())
        return np.zeros((*batch.shape[:2], 2), dtype=np.float32)

    report = evaluate_model(
        model_fn,
        dataset_root=dataset_root,
        morphology_id="specimen_200",
        max_samples=4,
        max_interventions=0,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )

    assert report["morphology_ids"] == ["specimen_200"]
    assert report["sample_profile"] == "iteration"
    assert report["intervention_sample_limit_per_condition"] == 40
    assert seen == [2.0]


def test_context_aware_model_receives_per_sample_morphology_ids(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    val_root = dataset_root / "val"
    val_root.mkdir(parents=True)
    inputs = np.zeros((4, 8, 3), dtype=np.float32)
    targets = np.zeros((4, 8, 2), dtype=np.float32)
    sample_ids = np.asarray(
        ["specimen_100:trace_0", "specimen_100:trace_1", "specimen_200:trace_0", "specimen_200:trace_1"]
    )
    np.savez_compressed(val_root / "val.npz", inputs=inputs, targets=targets, sample_ids=sample_ids)
    seen: list[str] = []

    def model_fn(batch, *, morphology_ids=None):
        assert morphology_ids is not None
        seen.extend(str(value) for value in morphology_ids)
        return np.zeros((*batch.shape[:2], 2), dtype=np.float32)

    model_fn.supports_morphology_ids = True
    report = evaluate_model(
        model_fn,
        dataset_root=dataset_root,
        max_samples=4,
        max_interventions=0,
        multiple_morphologies=True,
        config=EvaluationConfig(ignore_start=0, calibration_fraction=0.25),
    )

    assert report["morphology_ids"] == ["specimen_100", "specimen_200"]
    assert seen == ["specimen_100", "specimen_100", "specimen_200", "specimen_200"]
