"""Bounded model-iteration metrics and early-attention diagnostics.

The array-level API is intentionally independent of model frameworks and
checkpoint formats. Model repositories supply predictions; AxoBench owns the
measurement protocol and terminal report.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np

from axobench.metrics import (
    binary_auc,
    prevalence_matched_threshold,
    response_sera,
    spike_f1_curve,
    voltage_sera,
)
from axobench.data import ShardedNeuronIODataset


DEFAULT_STANDARD_SAMPLES = 240
DEFAULT_INTERVENTION_SAMPLES = 60
DEFAULT_INTERVENTION_WORKERS = 4
DEFAULT_BATCH_SIZE = 16


ATTENTION_BASES = {
    "nonfinite_outputs": "Numerical validity requires every emitted score and voltage to be finite.",
    "voltage_bounds": "The broad soma-voltage envelope is -120 to +60 mV.",
    "voltage_slew": "A one-step soma change above 120 mV/ms exceeds a broad action-potential slew envelope.",
    "voltage_null_skill": (
        "Skill is 1-MSE(model)/MSE(per-trace teacher mean); non-positive skill does not beat the null."
    ),
    "voltage_bias": "A mean offset above 0.10 teacher standard deviations is an early systematic-error watch.",
    "voltage_gain": "A least-squares voltage gain outside 0.90-1.10 indicates more than 10% amplitude distortion.",
    "residual_lag1": (
        "Absolute lag-1 correlation above 0.10 makes more than 1% of residual variance linearly predictable."
    ),
    "low_input_state_error": "A 1.10 RMSE ratio corresponds to at least 21% MSE inflation in the low-input regime.",
    "hyperpolarized_state_error": (
        "A 1.10 RMSE ratio corresponds to at least 21% MSE inflation in the lower voltage tail."
    ),
    "depolarized_state_error": (
        "A 1.10 RMSE ratio corresponds to at least 21% MSE inflation in the upper voltage tail."
    ),
    "high_dvdt_error": "A 1.10 RMSE ratio corresponds to at least 21% MSE inflation in the high-slope tail.",
    "near_spike_error": "A 1.10 RMSE ratio corresponds to at least 21% MSE inflation within 5 ms of teacher spikes.",
    "post_input_burst_error": (
        "A 1.10 RMSE ratio corresponds to at least 21% MSE inflation after upper-tail input load."
    ),
    "spike_exact_fidelity": "Exact F1 below 0.90 leaves more than 10% harmonic precision-recall deficit.",
    "spike_5ms_fidelity": "F1 below 0.95 at 5 ms leaves more than 5% tolerant event-fidelity deficit.",
    "spike_timing_tolerance_gap": (
        "A gain above 0.05 from timing tolerance indicates materially timing-dependent fidelity."
    ),
    "spike_count_error": "Per-trace count MAE above 10% of the mean teacher count indicates rate inconsistency.",
    "spike_timing_bias": (
        "Median matched-event displacement above half a 1 ms sample indicates systematic timing offset."
    ),
    "default_threshold_count": "The interval 0.80-1.25 is a symmetric multiplicative 25% count-calibration envelope.",
    "refractory_violations": "Predicted inter-spike intervals below 2 ms violate a broad absolute refractory bound.",
    "intervention_response_skill": "Skill below 0.80 leaves more than 20% of zero-response root-SERA error.",
    "intervention_direction": "Agreement below 0.95 reverses at least 5% of material teacher response samples.",
    "intervention_gain": "Response gain outside 0.80-1.20 distorts intervention magnitude by more than 20%.",
    "morphology_error_disparity": (
        "Worst-morphology RMSE above 1.25 times the median indicates a 25% transfer disparity."
    ),
}


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for a compact model-iteration evaluation."""

    soma_scale: float = 0.1
    soma_bias_mv: float = -67.7
    ignore_start: int = 500
    timestep_ms: float = 1.0
    calibration_fraction: float = 0.25
    default_spike_threshold: float = 0.5
    f1_tolerances_steps: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    selected_f1_steps: tuple[int, int] = (0, 5)
    physiological_voltage_bounds_mv: tuple[float, float] = (-120.0, 60.0)
    physiological_slew_bound_mv_per_ms: float = 120.0
    refractory_period_ms: float = 2.0
    state_error_ratio_watch: float = 1.10
    state_error_floor_mv: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")
        if self.soma_scale <= 0.0:
            raise ValueError("soma_scale must be positive")
        if self.ignore_start < 0:
            raise ValueError("ignore_start must be non-negative")
        if self.timestep_ms <= 0.0:
            raise ValueError("timestep_ms must be positive")
        if not self.f1_tolerances_steps:
            raise ValueError("f1_tolerances_steps must not be empty")
        missing = set(self.selected_f1_steps) - set(self.f1_tolerances_steps)
        if missing:
            raise ValueError(f"selected_f1_steps are absent from f1_tolerances_steps: {sorted(missing)}")


@dataclass(frozen=True)
class PairedInterventionPredictions:
    """Predictions and teacher traces for one paired intervention."""

    name: str
    baseline_prediction: np.ndarray
    intervention_prediction: np.ndarray
    baseline_target: np.ndarray
    intervention_target: np.ndarray
    baseline_raw_soma_mv: np.ndarray | None = None
    intervention_raw_soma_mv: np.ndarray | None = None


def evaluate_model(
    model_fn: Callable[[np.ndarray], np.ndarray],
    *,
    dataset_root: str | Path,
    model_id: str = "model",
    max_samples: int = DEFAULT_STANDARD_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_intervention_samples: int = DEFAULT_INTERVENTION_SAMPLES,
    max_interventions: int = 8,
    intervention_workers: int = DEFAULT_INTERVENTION_WORKERS,
    intervention_names: Iterable[str] | None = None,
    cache_shards: int = 1,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Run batched model inference and evaluate a v1-style dataset.

    Full synaptic input tensors are released after each batch. Only model
    outputs, targets, raw soma voltage, and two compact E/I event-count channels
    are retained for the diagnostic pass.
    """
    if max_samples < 2:
        raise ValueError("max_samples must be at least two")
    if batch_size <= 0 or max_intervention_samples <= 0 or max_interventions < 0:
        raise ValueError("batch and intervention limits must be positive")
    if intervention_workers <= 0:
        raise ValueError("intervention_workers must be positive")
    started_at = time.perf_counter()
    root = Path(dataset_root)
    validation = ShardedNeuronIODataset(root / "val", cache_shards=cache_shards)
    selected_indices = _balanced_validation_indices(validation, min(int(max_samples), len(validation)))
    sample_count = len(selected_indices)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    event_indicators: list[np.ndarray] = []
    raw_soma_chunks: list[np.ndarray] = []
    raw_soma_available = True
    trace_ids: list[str] = []
    for start in range(0, sample_count, batch_size):
        indices = selected_indices[start : start + batch_size]
        inputs, batch_targets = validation.get_batch(indices)
        batch_prediction = np.asarray(model_fn(inputs))
        if batch_prediction.shape != batch_targets.shape:
            raise ValueError(
                f"model prediction must match target shape, got {batch_prediction.shape} and {batch_targets.shape}"
            )
        predictions.append(batch_prediction.astype(np.float32, copy=False))
        targets.append(np.asarray(batch_targets, dtype=np.float32))
        event_indicators.append(
            np.stack(
                [np.count_nonzero(inputs > 0, axis=-1), np.count_nonzero(inputs < 0, axis=-1)],
                axis=-1,
            ).astype(np.int16, copy=False)
        )
        raw = validation.get_optional_array_batch("soma_mv", indices)
        if raw is None:
            raw_soma_available = False
        elif raw_soma_available:
            raw_soma_chunks.append(np.asarray(raw, dtype=np.float32))
        trace_ids.extend(validation.get_sample_ids(indices))

    interventions = _run_dataset_interventions(
        model_fn,
        root=root,
        batch_size=batch_size,
        max_samples=max_intervention_samples,
        max_interventions=max_interventions,
        workers=intervention_workers,
        intervention_names=intervention_names,
    )
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    event_array = np.concatenate(event_indicators, axis=0)
    raw_array = np.concatenate(raw_soma_chunks, axis=0) if raw_soma_available and raw_soma_chunks else None
    report_order = _round_robin_group_order(trace_ids)
    report = evaluate_predictions(
        prediction_array[report_order],
        target_array[report_order],
        inputs=event_array[report_order],
        raw_soma_mv=raw_array[report_order] if raw_array is not None else None,
        trace_ids=[trace_ids[index] for index in report_order],
        interventions=interventions,
        config=config,
        model_id=model_id,
    )
    report["evaluation_seconds"] = float(time.perf_counter() - started_at)
    return report


def evaluate_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    inputs: np.ndarray | None = None,
    raw_soma_mv: np.ndarray | None = None,
    trace_ids: list[str] | None = None,
    interventions: Iterable[PairedInterventionPredictions] = (),
    config: EvaluationConfig | None = None,
    model_id: str = "model",
) -> dict[str, Any]:
    """Evaluate stored predictions and return a fixed-size terminal report."""
    cfg = config or EvaluationConfig()
    pred = _as_traces(prediction, "prediction")
    tgt = _as_traces(target, "target")
    if pred.shape != tgt.shape or pred.shape[-1] != 2:
        raise ValueError(
            "prediction and target must match with shape (trace, time, 2), "
            f"got {pred.shape} and {tgt.shape}"
        )
    if pred.shape[0] < 2:
        raise ValueError("at least two traces are required to separate threshold calibration and scoring")
    if cfg.ignore_start >= pred.shape[1]:
        raise ValueError("ignore_start leaves no timesteps to evaluate")

    n_calibration = max(1, min(pred.shape[0] - 1, int(round(pred.shape[0] * cfg.calibration_fraction))))
    calibration_scores = pred[:n_calibration, cfg.ignore_start :, 0]
    calibration_labels = tgt[:n_calibration, cfg.ignore_start :, 0] > 0.5
    finite_calibration = np.isfinite(calibration_scores)
    threshold = prevalence_matched_threshold(
        calibration_scores[finite_calibration],
        int(calibration_labels[finite_calibration].sum()),
    )

    score_slice = slice(n_calibration, None)
    pred_score = pred[score_slice, cfg.ignore_start :, :]
    tgt_score = tgt[score_slice, cfg.ignore_start :, :]
    raw_score = _slice_optional_2d(raw_soma_mv, score_slice, cfg.ignore_start, pred.shape[:2], "raw_soma_mv")
    inputs_score = _slice_optional_inputs(inputs, score_slice, cfg.ignore_start, pred.shape[:2])
    ids = trace_ids or [str(index) for index in range(pred.shape[0])]
    if len(ids) != pred.shape[0]:
        raise ValueError("trace_ids length must match trace count")
    score_ids = ids[n_calibration:]

    spike_scores = pred_score[..., 0]
    spike_labels = tgt_score[..., 0] > 0.5
    safe_spike_scores = np.where(np.isfinite(spike_scores), spike_scores, -np.inf)
    f1 = spike_f1_curve(
        safe_spike_scores,
        spike_labels,
        threshold=threshold,
        tolerances_steps=cfg.f1_tolerances_steps,
    )
    f1_by_step = {int(row["tolerance_steps"]): row for row in f1["rows"]}
    exact_step, tolerant_step = cfg.selected_f1_steps

    voltage = voltage_sera(
        pred_score,
        tgt_score,
        raw_soma_mv=raw_score,
        soma_scale=cfg.soma_scale,
        soma_bias_mv=cfg.soma_bias_mv,
    )
    clipped_rmse = _finite_rmse((pred_score[..., 1] - tgt_score[..., 1]) / cfg.soma_scale)
    finite_spikes = np.isfinite(spike_scores)
    auc = binary_auc(spike_scores[finite_spikes], spike_labels[finite_spikes]) if finite_spikes.any() else float("nan")

    intervention_reports = [
        _evaluate_intervention(item, cfg)
        for item in interventions
    ]
    response_sera_value = _root_mean_square(
        [item["response_sera_mv"] for item in intervention_reports]
    )

    metrics = [
        _metric("voltage_sera_mv", voltage["voltage_sera_mv"], "mV", "teacher-relevance voltage error"),
        _metric("spike_f1_area_0_5ms", f1["f1_area"], "fraction", "mean F1 over timing tolerance"),
        _metric(
            "spike_f1_exact",
            f1_by_step[exact_step]["f1"],
            "fraction",
            f"F1 at {exact_step * cfg.timestep_ms:g} ms",
        ),
        _metric(
            "spike_f1_5ms",
            f1_by_step[tolerant_step]["f1"],
            "fraction",
            f"F1 at {tolerant_step * cfg.timestep_ms:g} ms",
        ),
        _metric("response_sera_mv", response_sera_value, "mV", "paired intervention response error"),
        _metric("raw_voltage_rmse_mv", voltage["voltage_rmse_mv"], "mV", "raw-voltage compatibility"),
        _metric("clipped_voltage_rmse_mv", clipped_rmse, "mV", "legacy target-view compatibility"),
        _metric("spike_auc", auc, "fraction", "legacy ranking compatibility"),
    ]

    diagnostics = _ordinary_attention_rows(
        pred_score,
        tgt_score,
        inputs_score=inputs_score,
        raw_soma_mv=raw_score,
        threshold=threshold,
        f1_by_step=f1_by_step,
        config=cfg,
    )
    diagnostics.extend(_intervention_attention_rows(intervention_reports))
    diagnostics.append(_morphology_attention_row(pred_score, tgt_score, raw_score, score_ids, cfg))
    if len(diagnostics) != 24:
        raise RuntimeError(f"internal diagnostic panel must contain 24 rows, got {len(diagnostics)}")

    return {
        "model_id": model_id,
        "samples": int(pred.shape[0]),
        "calibration_samples": int(n_calibration),
        "scored_samples": int(pred_score.shape[0]),
        "spike_threshold": float(threshold),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "interventions": intervention_reports,
    }


def format_evaluation_report(report: dict[str, Any]) -> str:
    """Render a deterministic, bounded plain-text report."""
    runtime = "" if "evaluation_seconds" not in report else f"  runtime={report['evaluation_seconds']:.2f}s"
    lines = [
        f"MODEL: {report['model_id']}  samples={report['samples']} "
        f"(calibration={report['calibration_samples']}, scored={report['scored_samples']}) "
        f"spike_threshold={_format_number(report['spike_threshold'])}{runtime}",
        "",
        "CORE METRICS",
    ]
    metric_rows = [
        [row["id"], _format_value(row["value"], row["unit"]), row["meaning"]]
        for row in report["metrics"]
    ]
    lines.extend(_ascii_table(["metric", "value", "role"], metric_rows, widths=(27, 14, 35)))
    lines.extend(["", "ATTENTION PANEL", "YES = early watch; NO = clear at this trigger; NA = evidence unavailable."])
    diagnostic_rows = [
        [
            row["area"],
            row["id"],
            row["attention"],
            row["primary"],
            row["trigger"],
        ]
        for row in report["diagnostics"]
    ]
    lines.extend(
        _ascii_table(
            ["area", "diagnostic", "attention", "primary", "trigger"],
            diagnostic_rows,
            widths=(12, 29, 9, 28, 25),
        )
    )
    yes = sum(row["attention"] == "YES" for row in report["diagnostics"])
    na = sum(row["attention"] == "NA" for row in report["diagnostics"])
    lines.append(f"Attention summary: {yes} YES, {24 - yes - na} NO, {na} NA (24 fixed rows).")
    return "\n".join(lines)


def write_evaluation_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the report as strict JSON or tidy CSV, selected by suffix."""
    output = Path(path)
    suffix = output.suffix.lower()
    if suffix not in {".json", ".csv"}:
        raise ValueError("evaluation report path must end in .json or .csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        output.write_text(
            json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return output

    fieldnames = (
        "model_id",
        "samples",
        "calibration_samples",
        "scored_samples",
        "spike_threshold",
        "evaluation_seconds",
        "row_type",
        "id",
        "value",
        "unit",
        "attention",
        "condition",
        "trigger",
        "basis",
        "meaning",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_report_csv_rows(report))
    return output


def _report_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "model_id": report.get("model_id"),
        "samples": report.get("samples"),
        "calibration_samples": report.get("calibration_samples"),
        "scored_samples": report.get("scored_samples"),
        "spike_threshold": _csv_value(report.get("spike_threshold")),
        "evaluation_seconds": _csv_value(report.get("evaluation_seconds")),
    }
    rows: list[dict[str, Any]] = []
    for metric in report.get("metrics", []):
        rows.append(
            {
                **common,
                "row_type": "metric",
                "id": metric.get("id"),
                "value": _csv_value(metric.get("value")),
                "unit": metric.get("unit"),
                "meaning": metric.get("meaning"),
            }
        )
    for diagnostic in report.get("diagnostics", []):
        rows.append(
            {
                **common,
                "row_type": "diagnostic",
                "id": diagnostic.get("id"),
                "value": _csv_value(diagnostic.get("value")),
                "unit": diagnostic.get("unit"),
                "attention": diagnostic.get("attention"),
                "condition": _diagnostic_condition(diagnostic.get("primary")),
                "trigger": diagnostic.get("trigger"),
                "basis": diagnostic.get("basis"),
            }
        )
    intervention_units = {
        "response_sera_mv": "mV",
        "zero_response_sera_mv": "mV",
        "response_skill": "skill",
        "direction_agreement": "fraction",
        "response_gain": "ratio",
    }
    for intervention in report.get("interventions", []):
        for identifier, unit in intervention_units.items():
            rows.append(
                {
                    **common,
                    "row_type": "intervention",
                    "id": identifier,
                    "value": _csv_value(intervention.get(identifier)),
                    "unit": unit,
                    "condition": intervention.get("name"),
                }
            )
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def _diagnostic_condition(primary: Any) -> str:
    text = str(primary or "")
    if "[" not in text or not text.endswith("]"):
        return ""
    return text.rsplit("[", 1)[1][:-1]


def _run_dataset_interventions(
    model_fn: Callable[[np.ndarray], np.ndarray],
    *,
    root: Path,
    batch_size: int,
    max_samples: int,
    max_interventions: int,
    workers: int,
    intervention_names: Iterable[str] | None,
) -> list[PairedInterventionPredictions]:
    interventions_root = root / "interventions"
    if not interventions_root.exists() or max_interventions == 0:
        return []
    requested = set(intervention_names) if intervention_names is not None else None
    directories = [
        path for path in sorted(interventions_root.iterdir())
        if path.is_dir() and (requested is None or path.name in requested)
    ][:max_interventions]
    missing = requested - {path.name for path in directories} if requested is not None else set()
    if missing:
        raise FileNotFoundError(f"intervention folders not found: {sorted(missing)}")

    results = []
    worker_count = min(workers, len(directories))
    if worker_count <= 1:
        loaded_pairs = (_load_intervention_directory(path, max_samples=max_samples) for path in directories)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="axobench-intervention")
        loaded_pairs = executor.map(
            lambda path: _load_intervention_directory(path, max_samples=max_samples),
            directories,
        )
    try:
        for directory, pair in zip(directories, loaded_pairs, strict=True):
            base_prediction = _predict_in_batches(model_fn, pair["baseline_inputs"], batch_size=batch_size)
            intervention_prediction = _predict_in_batches(
                model_fn,
                pair["intervention_inputs"],
                batch_size=batch_size,
            )
            results.append(
                PairedInterventionPredictions(
                    name=directory.name,
                    baseline_prediction=base_prediction,
                    intervention_prediction=intervention_prediction,
                    baseline_target=pair["baseline_targets"],
                    intervention_target=pair["intervention_targets"],
                    baseline_raw_soma_mv=pair.get("baseline_soma_mv"),
                    intervention_raw_soma_mv=pair.get("intervention_soma_mv"),
                )
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return results


def _balanced_validation_indices(dataset: ShardedNeuronIODataset, sample_count: int) -> list[int]:
    all_indices = list(range(len(dataset)))
    all_ids = dataset.get_sample_ids(all_indices)
    groups: dict[str, list[int]] = {}
    for index, trace_id in zip(all_indices, all_ids, strict=True):
        groups.setdefault(_morphology_id(trace_id), []).append(index)
    if len(groups) <= 1:
        return all_indices[:sample_count]

    names = sorted(groups)
    base, extra = divmod(sample_count, len(names))
    selected: list[int] = []
    unused: list[int] = []
    for position, name in enumerate(names):
        quota = base + int(position < extra)
        selected.extend(groups[name][:quota])
        unused.extend(groups[name][quota:])
    if len(selected) < sample_count:
        selected.extend(sorted(unused)[: sample_count - len(selected)])
    return sorted(selected)


def _round_robin_group_order(trace_ids: list[str]) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for index, trace_id in enumerate(trace_ids):
        groups.setdefault(_morphology_id(trace_id), []).append(index)
    if len(groups) <= 1:
        return np.arange(len(trace_ids), dtype=int)
    ordered: list[int] = []
    names = sorted(groups)
    for offset in range(max(len(indices) for indices in groups.values())):
        for name in names:
            if offset < len(groups[name]):
                ordered.append(groups[name][offset])
    return np.asarray(ordered, dtype=int)


def _load_intervention_directory(path: Path, *, max_samples: int) -> dict[str, np.ndarray]:
    paths = sorted(path.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no paired intervention NPZ files found in {path}")
    groups: dict[str, list[Path]] = {}
    for pair_path in paths:
        groups.setdefault(_pair_morphology_id(pair_path), []).append(pair_path)
    names = sorted(groups)
    base, extra = divmod(int(max_samples), len(names))
    loaded: list[tuple[str, dict[str, np.ndarray]]] = []
    for position, name in enumerate(names):
        quota = base + int(position < extra)
        if quota > 0:
            loaded.append((name, _load_intervention_paths(groups[name], max_samples=quota)))
    if not loaded:
        raise ValueError("max_samples must be positive")

    required = ("baseline_inputs", "baseline_targets", "intervention_inputs", "intervention_targets")
    result = {name: np.concatenate([item[name] for _, item in loaded], axis=0) for name in required}
    for name in ("baseline_soma_mv", "intervention_soma_mv"):
        if all(name in item for _, item in loaded):
            result[name] = np.concatenate([item[name] for _, item in loaded], axis=0)
    result["morphology_ids"] = np.concatenate(
        [np.repeat(name, item["baseline_inputs"].shape[0]) for name, item in loaded]
    )
    return result


def _load_intervention_paths(paths: list[Path], *, max_samples: int) -> dict[str, np.ndarray]:
    required = ("baseline_inputs", "baseline_targets", "intervention_inputs", "intervention_targets")
    optional = ("baseline_soma_mv", "intervention_soma_mv")
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in (*required, *optional)}
    remaining = int(max_samples)
    for pair_path in paths:
        if remaining <= 0:
            break
        with np.load(pair_path, allow_pickle=False) as payload:
            missing = [name for name in required if name not in payload.files]
            if missing:
                raise ValueError(f"{pair_path} is missing paired arrays: {missing}")
            count = 0
            for name in (*required, *optional):
                if name not in payload.files:
                    continue
                full_array = np.asarray(payload[name])
                if name == "baseline_inputs":
                    count = min(remaining, int(full_array.shape[0]))
                chunks[name].append(np.array(full_array[:count], copy=True))
                del full_array
        remaining -= count
    if not chunks["baseline_inputs"]:
        raise FileNotFoundError("paired intervention files contain no samples")
    result = {name: np.concatenate(values, axis=0) for name, values in chunks.items() if values}
    for name in optional:
        if chunks[name] and sum(chunk.shape[0] for chunk in chunks[name]) != result["baseline_inputs"].shape[0]:
            result.pop(name, None)
    return result


def _pair_morphology_id(path: Path) -> str:
    match = re.match(r"(specimen_[^_]+)", path.stem)
    return match.group(1) if match else "unknown"


def _predict_in_batches(
    model_fn: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    for start in range(0, inputs.shape[0], batch_size):
        batch = inputs[start : start + batch_size]
        prediction = np.asarray(model_fn(batch))
        if prediction.shape[:2] != batch.shape[:2] or prediction.shape[-1] != 2:
            raise ValueError(
                "model prediction must have shape (batch, time, 2), "
                f"got {prediction.shape} for inputs {batch.shape}"
            )
        predictions.append(prediction.astype(np.float32, copy=False))
    return np.concatenate(predictions, axis=0)


def _ordinary_attention_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    inputs_score: np.ndarray | None,
    raw_soma_mv: np.ndarray | None,
    threshold: float,
    f1_by_step: dict[int, dict[str, Any]],
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    pred_soma = prediction[..., 1] / config.soma_scale + config.soma_bias_mv
    target_soma = raw_soma_mv if raw_soma_mv is not None else target[..., 1] / config.soma_scale + config.soma_bias_mv
    residual = pred_soma - target_soma
    finite = np.isfinite(prediction)
    finite_soma = np.isfinite(pred_soma) & np.isfinite(target_soma)
    finite_residual = residual[finite_soma]
    target_finite = target_soma[finite_soma]
    overall_rmse = _finite_rmse(finite_residual)

    lower, upper = config.physiological_voltage_bounds_mv
    bounds_fraction = _fraction((pred_soma < lower) | (pred_soma > upper), np.isfinite(pred_soma))
    pred_dvdt = np.diff(pred_soma, axis=1) / config.timestep_ms
    slew_fraction = _fraction(
        np.abs(pred_dvdt) > config.physiological_slew_bound_mv_per_ms,
        np.isfinite(pred_dvdt),
    )

    trace_mean = np.nanmean(target_soma, axis=1, keepdims=True)
    null_rmse = _finite_rmse(target_soma - trace_mean)
    voltage_skill = (
        1.0 - (overall_rmse ** 2 / null_rmse ** 2)
        if null_rmse > 0.0
        else (1.0 if overall_rmse == 0.0 else float("-inf"))
    )
    target_std = float(np.std(target_finite)) if target_finite.size else float("nan")
    bias = float(np.mean(finite_residual)) if finite_residual.size else float("nan")
    normalized_bias = abs(bias) / target_std if target_std > 0.0 else (0.0 if bias == 0.0 else float("inf"))
    gain = _least_squares_gain(target_finite, pred_soma[finite_soma])
    lag1 = _lag1_autocorrelation(residual)

    spikes = target[..., 0] > 0.5
    event_load = np.sum(np.abs(inputs_score), axis=-1) if inputs_score is not None else None
    target_dvdt = np.diff(target_soma, axis=1, prepend=target_soma[:, :1]) / config.timestep_ms
    low_input_mask = _low_input_mask(event_load, spikes, radius=5) if event_load is not None else None
    hyper_mask = target_soma <= np.nanquantile(target_soma, 0.10)
    depol_mask = target_soma >= np.nanquantile(target_soma, 0.90)
    dvdt_mask = np.abs(target_dvdt) >= np.nanquantile(np.abs(target_dvdt), 0.90)
    near_spike_mask = _dilate(spikes, 5)
    if event_load is not None:
        high_input = event_load >= np.quantile(event_load, 0.90)
        post_event_mask = _causal_dilate(high_input, 5)
    else:
        post_event_mask = None

    exact_step, tolerant_step = config.selected_f1_steps
    exact_f1_raw = f1_by_step[exact_step]["f1"]
    tolerant_f1_raw = f1_by_step[tolerant_step]["f1"]
    exact_f1 = _numeric(exact_f1_raw)
    tolerant_f1 = _numeric(tolerant_f1_raw)
    timing_gap = tolerant_f1 - exact_f1
    predicted = np.where(np.isfinite(prediction[..., 0]), prediction[..., 0], -np.inf) >= threshold
    default_predicted = (
        np.where(np.isfinite(prediction[..., 0]), prediction[..., 0], -np.inf)
        >= config.default_spike_threshold
    )
    count_nmae = _count_nmae(predicted, spikes)
    timing_bias = _matched_timing_bias_ms(predicted, spikes, tolerant_step, config.timestep_ms)
    target_count = int(spikes.sum())
    default_count_ratio = float(default_predicted.sum() / target_count) if target_count > 0 else float("nan")
    refractory_fraction = _refractory_violation_fraction(predicted, config.refractory_period_ms, config.timestep_ms)

    if target_count == 0 and not default_predicted.any():
        spike_rows = [
            _attention_na("spikes", "spike_exact_fidelity", "< 0.90"),
            _attention_na("spikes", "spike_5ms_fidelity", "< 0.95"),
            _attention_na("spikes", "spike_timing_tolerance_gap", "> 0.05"),
            _attention_na("spikes", "spike_count_error", "> 0.10"),
            _attention_na("spikes", "spike_timing_bias", "|bias| > 0.5 ms"),
            _attention_na("calibration", "default_threshold_count", "outside [0.80, 1.25]"),
        ]
    else:
        spike_rows = [
            _attention_or_na("spikes", "spike_exact_fidelity", exact_f1_raw, "< 0.90", exact_f1 < 0.90, "F1"),
            _attention_or_na("spikes", "spike_5ms_fidelity", tolerant_f1_raw, "< 0.95", tolerant_f1 < 0.95, "F1"),
            _attention("spikes", "spike_timing_tolerance_gap", timing_gap, "> 0.05", timing_gap > 0.05, "F1 delta"),
            _attention("spikes", "spike_count_error", count_nmae, "> 0.10", count_nmae > 0.10, "normalized MAE"),
            _attention_or_na(
                "spikes", "spike_timing_bias", timing_bias, "|bias| > 0.5 ms", abs(timing_bias) > 0.5, "ms"
            ),
            _attention_or_na(
                "calibration", "default_threshold_count", default_count_ratio, "outside [0.80, 1.25]",
                not 0.80 <= default_count_ratio <= 1.25, "ratio",
            ),
        ]

    state_rows = [
        _state_attention("low_input_state_error", low_input_mask, residual, overall_rmse, config),
        _state_attention("hyperpolarized_state_error", hyper_mask, residual, overall_rmse, config),
        _state_attention("depolarized_state_error", depol_mask, residual, overall_rmse, config),
        _state_attention("high_dvdt_error", dvdt_mask, residual, overall_rmse, config),
        _state_attention("near_spike_error", near_spike_mask, residual, overall_rmse, config),
        _state_attention("post_input_burst_error", post_event_mask, residual, overall_rmse, config),
    ]
    return [
        _attention(
            "validity",
            "nonfinite_outputs",
            1.0 - float(finite.mean()),
            "> 0",
            not bool(finite.all()),
            "fraction",
        ),
        _attention("physiology", "voltage_bounds", bounds_fraction, "> 0", bounds_fraction > 0.0, "fraction"),
        _attention("physiology", "voltage_slew", slew_fraction, "> 0", slew_fraction > 0.0, "fraction"),
        _attention("voltage", "voltage_null_skill", voltage_skill, "<= 0", voltage_skill <= 0.0, "fraction"),
        _attention("voltage", "voltage_bias", normalized_bias, "> 0.10 target SD", normalized_bias > 0.10, "ratio"),
        _attention("voltage", "voltage_gain", gain, "outside [0.90, 1.10]", not 0.90 <= gain <= 1.10, "ratio"),
        _attention("dynamics", "residual_lag1", lag1, "|r| > 0.10", abs(lag1) > 0.10, "correlation"),
        *state_rows,
        *spike_rows,
        _attention(
            "physiology",
            "refractory_violations",
            refractory_fraction,
            "> 0",
            refractory_fraction > 0.0,
            "fraction",
        ),
    ]


def _evaluate_intervention(item: PairedInterventionPredictions, config: EvaluationConfig) -> dict[str, Any]:
    base_pred = _as_traces(item.baseline_prediction, "baseline_prediction")[:, config.ignore_start :, :]
    int_pred = _as_traces(item.intervention_prediction, "intervention_prediction")[:, config.ignore_start :, :]
    base_tgt = _as_traces(item.baseline_target, "baseline_target")[:, config.ignore_start :, :]
    int_tgt = _as_traces(item.intervention_target, "intervention_target")[:, config.ignore_start :, :]
    if not (base_pred.shape == int_pred.shape == base_tgt.shape == int_tgt.shape):
        raise ValueError(f"intervention {item.name!r} arrays must have matching shapes")
    base_raw = _crop_intervention_raw(item.baseline_raw_soma_mv, config.ignore_start, base_pred.shape[:2])
    int_raw = _crop_intervention_raw(item.intervention_raw_soma_mv, config.ignore_start, base_pred.shape[:2])
    measured = response_sera(
        base_pred,
        int_pred,
        base_tgt,
        int_tgt,
        baseline_raw_soma_mv=base_raw,
        intervention_raw_soma_mv=int_raw,
        soma_scale=config.soma_scale,
        soma_bias_mv=config.soma_bias_mv,
    )
    zero_response = response_sera(
        base_tgt,
        base_tgt,
        base_tgt,
        int_tgt,
        baseline_raw_soma_mv=base_raw,
        intervention_raw_soma_mv=int_raw,
        soma_scale=config.soma_scale,
        soma_bias_mv=config.soma_bias_mv,
    )
    teacher_base_mv = base_raw if base_raw is not None else base_tgt[..., 1] / config.soma_scale + config.soma_bias_mv
    teacher_int_mv = int_raw if int_raw is not None else int_tgt[..., 1] / config.soma_scale + config.soma_bias_mv
    model_base_mv = base_pred[..., 1] / config.soma_scale + config.soma_bias_mv
    model_int_mv = int_pred[..., 1] / config.soma_scale + config.soma_bias_mv
    teacher_delta = teacher_int_mv - teacher_base_mv
    model_delta = model_int_mv - model_base_mv
    null_sera = float(zero_response["response_sera_mv"])
    skill = 1.0 - float(measured["response_sera_mv"]) / null_sera if null_sera > 0.0 else float("nan")
    direction = _response_direction_agreement(teacher_delta, model_delta)
    gain = _least_squares_gain(teacher_delta.reshape(-1), model_delta.reshape(-1))
    return {
        "name": item.name,
        "response_sera_mv": float(measured["response_sera_mv"]),
        "zero_response_sera_mv": null_sera,
        "response_skill": skill,
        "direction_agreement": direction,
        "response_gain": gain,
    }


def _intervention_attention_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not reports:
        return [
            _attention_na("intervention", "intervention_response_skill", "< 0.80"),
            _attention_na("intervention", "intervention_direction", "< 0.95"),
            _attention_na("intervention", "intervention_gain", "outside [0.80, 1.20]"),
        ]
    skill = min(reports, key=lambda row: _nan_to(row["response_skill"], -np.inf))
    direction = min(reports, key=lambda row: _nan_to(row["direction_agreement"], -np.inf))
    gain = max(reports, key=lambda row: abs(_nan_to(row["response_gain"], np.inf) - 1.0))
    return [
        _attention(
            "intervention", "intervention_response_skill", skill["response_skill"], "< 0.80",
            skill["response_skill"] < 0.80, "skill", suffix=skill["name"],
        ),
        _attention(
            "intervention", "intervention_direction", direction["direction_agreement"], "< 0.95",
            direction["direction_agreement"] < 0.95, "fraction", suffix=direction["name"],
        ),
        _attention(
            "intervention", "intervention_gain", gain["response_gain"], "outside [0.80, 1.20]",
            not 0.80 <= gain["response_gain"] <= 1.20, "ratio", suffix=gain["name"],
        ),
    ]


def _morphology_attention_row(
    prediction: np.ndarray,
    target: np.ndarray,
    raw_soma_mv: np.ndarray | None,
    trace_ids: list[str],
    config: EvaluationConfig,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, trace_id in enumerate(trace_ids):
        morphology = _morphology_id(trace_id)
        groups.setdefault(morphology, []).append(index)
    if len(groups) < 2:
        return _attention_na("transfer", "morphology_error_disparity", "> 1.25x median")
    pred_mv = prediction[..., 1] / config.soma_scale + config.soma_bias_mv
    target_mv = raw_soma_mv if raw_soma_mv is not None else target[..., 1] / config.soma_scale + config.soma_bias_mv
    errors = {
        name: _finite_rmse(pred_mv[indices] - target_mv[indices])
        for name, indices in groups.items()
    }
    finite_errors = np.asarray([value for value in errors.values() if np.isfinite(value)], dtype=float)
    if not finite_errors.size:
        return _attention_na("transfer", "morphology_error_disparity", "> 1.25x median")
    median = float(np.median(finite_errors))
    worst_name, worst_error = max(errors.items(), key=lambda item: _nan_to(item[1], -np.inf))
    ratio = worst_error / median if median > 0.0 else (1.0 if worst_error == 0.0 else float("inf"))
    return _attention(
        "transfer", "morphology_error_disparity", ratio, "> 1.25x median", ratio > 1.25,
        "ratio", suffix=worst_name,
    )


def _state_attention(
    name: str,
    mask: np.ndarray | None,
    residual: np.ndarray,
    overall_rmse: float,
    config: EvaluationConfig,
) -> dict[str, Any]:
    if mask is None or not np.any(mask):
        return _attention_na("states", name, f"> {config.state_error_ratio_watch:.2f}x overall")
    state_rmse = _finite_rmse(residual[mask])
    ratio = state_rmse / overall_rmse if overall_rmse > 0.0 else (1.0 if state_rmse == 0.0 else float("inf"))
    watch = state_rmse > config.state_error_floor_mv and ratio > config.state_error_ratio_watch
    return _attention(
        "states", name, ratio, f"> {config.state_error_ratio_watch:.2f}x & > {config.state_error_floor_mv:g} mV",
        watch, "RMSE ratio",
    )


def _attention(
    area: str,
    identifier: str,
    value: float,
    trigger: str,
    watch: bool,
    unit: str,
    *,
    suffix: str | None = None,
) -> dict[str, Any]:
    primary = _format_value(value, unit)
    if suffix:
        primary = f"{primary} [{suffix}]"
    return {
        "area": area,
        "id": identifier,
        "attention": "YES" if bool(watch) else "NO",
        "value": float(value),
        "unit": unit,
        "primary": primary,
        "trigger": trigger,
        "basis": ATTENTION_BASES[identifier],
    }


def _attention_na(area: str, identifier: str, trigger: str) -> dict[str, Any]:
    return {
        "area": area,
        "id": identifier,
        "attention": "NA",
        "value": None,
        "unit": "",
        "primary": "NA",
        "trigger": trigger,
        "basis": ATTENTION_BASES[identifier],
    }


def _attention_or_na(
    area: str,
    identifier: str,
    value: float | None,
    trigger: str,
    watch: bool,
    unit: str,
) -> dict[str, Any]:
    if value is None or not np.isfinite(value):
        return _attention_na(area, identifier, trigger)
    return _attention(area, identifier, float(value), trigger, watch, unit)


def _metric(identifier: str, value: float | None, unit: str, meaning: str) -> dict[str, Any]:
    return {"id": identifier, "value": value, "unit": unit, "meaning": meaning}


def _as_traces(array: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(array, dtype=np.float64)
    if out.ndim == 2:
        out = out[None, ...]
    if out.ndim != 3:
        raise ValueError(f"{name} must have shape (trace, time, feature), got {out.shape}")
    return out


def _slice_optional_2d(
    array: np.ndarray | None,
    trace_slice: slice,
    ignore_start: int,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    if array is None:
        return None
    out = np.asarray(array, dtype=np.float64)
    if out.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {out.shape}")
    return out[trace_slice, ignore_start:]


def _slice_optional_inputs(
    inputs: np.ndarray | None,
    trace_slice: slice,
    ignore_start: int,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if inputs is None:
        return None
    out = np.asarray(inputs)
    if out.ndim != 3 or out.shape[:2] != expected_shape:
        raise ValueError(f"inputs must begin with shape {expected_shape}, got {out.shape}")
    return out[trace_slice, ignore_start:, :]


def _crop_intervention_raw(
    array: np.ndarray | None,
    ignore_start: int,
    expected_cropped: tuple[int, int],
) -> np.ndarray | None:
    if array is None:
        return None
    out = np.asarray(array, dtype=np.float64)
    cropped = out[:, ignore_start:]
    if cropped.shape != expected_cropped:
        raise ValueError(f"intervention raw soma shape after masking must be {expected_cropped}, got {cropped.shape}")
    return cropped


def _finite_rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    return float(np.sqrt(np.mean(np.square(arr[finite])))) if finite.any() else float("nan")


def _least_squares_gain(source: np.ndarray, response: np.ndarray) -> float:
    x = np.asarray(source, dtype=np.float64).reshape(-1)
    y = np.asarray(response, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if not x.size:
        return float("nan")
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return 1.0 if np.allclose(x, y) else float("nan")
    return float(np.dot(x_centered, y - y.mean()) / denominator)


def _lag1_autocorrelation(residual: np.ndarray) -> float:
    values = []
    for trace in np.asarray(residual):
        left, right = trace[:-1], trace[1:]
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.sum() < 3 or np.std(left[finite]) == 0.0 or np.std(right[finite]) == 0.0:
            continue
        values.append(float(np.corrcoef(left[finite], right[finite])[0, 1]))
    return float(np.mean(values)) if values else 0.0


def _low_input_mask(event_load: np.ndarray, spikes: np.ndarray, *, radius: int) -> np.ndarray:
    load = np.asarray(event_load, dtype=float)
    low_drive = load <= np.quantile(load, 0.10)
    return low_drive & ~_dilate(spikes, radius)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    out = np.zeros_like(source)
    for offset in range(-radius, radius + 1):
        source_start = max(0, -offset)
        source_stop = source.shape[1] - max(0, offset)
        target_start = max(0, offset)
        target_stop = target_start + (source_stop - source_start)
        out[:, target_start:target_stop] |= source[:, source_start:source_stop]
    return out


def _causal_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    out = np.zeros_like(source)
    for offset in range(radius + 1):
        if offset == 0:
            out |= source
        else:
            out[:, offset:] |= source[:, :-offset]
    return out


def _count_nmae(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_count = np.sum(predicted, axis=1)
    target_count = np.sum(target, axis=1)
    denominator = max(float(np.mean(target_count)), 1.0)
    return float(np.mean(np.abs(pred_count - target_count)) / denominator)


def _matched_timing_bias_ms(
    predicted: np.ndarray,
    target: np.ndarray,
    tolerance_steps: int,
    timestep_ms: float,
) -> float:
    lags: list[int] = []
    for pred_trace, target_trace in zip(predicted, target, strict=True):
        pred_idx = np.flatnonzero(pred_trace)
        target_idx = np.flatnonzero(target_trace)
        used: set[int] = set()
        for target_time in target_idx:
            candidates = [
                (abs(int(pred_idx[index]) - int(target_time)), index)
                for index in range(pred_idx.size)
                if index not in used and abs(int(pred_idx[index]) - int(target_time)) <= tolerance_steps
            ]
            if not candidates:
                continue
            _, best = min(candidates)
            used.add(best)
            lags.append(int(pred_idx[best]) - int(target_time))
    return float(np.median(lags) * timestep_ms) if lags else float("nan")


def _refractory_violation_fraction(predicted: np.ndarray, refractory_ms: float, timestep_ms: float) -> float:
    minimum_steps = int(np.ceil(refractory_ms / timestep_ms))
    intervals: list[int] = []
    for trace in predicted:
        indices = np.flatnonzero(trace)
        intervals.extend(np.diff(indices).tolist())
    if not intervals:
        return 0.0
    arr = np.asarray(intervals)
    return float(np.mean(arr < minimum_steps))


def _response_direction_agreement(teacher_delta: np.ndarray, model_delta: np.ndarray) -> float:
    teacher = np.asarray(teacher_delta, dtype=float)
    model = np.asarray(model_delta, dtype=float)
    finite = np.isfinite(teacher) & np.isfinite(model)
    material = finite & (np.abs(teacher) >= 0.5)
    if not material.any():
        return float("nan")
    return float(np.mean(np.sign(teacher[material]) == np.sign(model[material])))


def _morphology_id(trace_id: str) -> str:
    value = str(trace_id)
    for separator in (":", "/", "@"):
        if separator in value:
            return value.split(separator, 1)[0]
    return value


def _fraction(mask: np.ndarray, valid: np.ndarray) -> float:
    valid_count = int(np.sum(valid))
    return float(np.sum(mask & valid) / valid_count) if valid_count else float("nan")


def _root_mean_square(values: list[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else None


def _numeric(value: Any) -> float:
    return float(value) if value is not None and np.isfinite(value) else 0.0


def _nan_to(value: float, replacement: float) -> float:
    return float(value) if np.isfinite(value) else replacement


def _format_value(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    rendered = _format_number(value)
    if unit in {"mV", "ms"}:
        return f"{rendered} {unit}"
    return rendered


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    magnitude = abs(float(value))
    if magnitude != 0.0 and (magnitude < 1e-3 or magnitude >= 1e4):
        return f"{value:.2e}"
    if magnitude < 1.0:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _ascii_table(headers: list[str], rows: list[list[str]], *, widths: tuple[int, ...]) -> list[str]:
    def cell(value: str, width: int) -> str:
        text = str(value)
        return text if len(text) <= width else text[: max(1, width - 1)] + "~"

    rendered = [[cell(value, width) for value, width in zip(row, widths, strict=True)] for row in [headers, *rows]]
    actual_widths = [max(len(row[index]) for row in rendered) for index in range(len(widths))]
    lines = [" | ".join(value.ljust(actual_widths[index]) for index, value in enumerate(rendered[0]))]
    lines.append("-+-".join("-" * width for width in actual_widths))
    lines.extend(
        " | ".join(value.ljust(actual_widths[index]) for index, value in enumerate(row))
        for row in rendered[1:]
    )
    return lines
