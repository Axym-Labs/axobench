from __future__ import annotations

from typing import Any

import numpy as np


def soma_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_soma = np.asarray(prediction, dtype=np.float64)[..., 1]
    target_soma = np.asarray(target, dtype=np.float64)[..., 1]
    return float(np.sqrt(np.mean((pred_soma - target_soma) ** 2)))


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must have same shape, got {scores.shape} and {labels.shape}")

    positive = labels > 0.5
    negative = ~positive
    pos_count = int(positive.sum())
    neg_count = int(negative.sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_positive = positive[order].astype(np.float64, copy=False)
    tie_starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_scores)) + 1))
    tie_ends = np.concatenate((tie_starts[1:], [len(scores)]))
    positive_per_tie = np.add.reduceat(sorted_positive, tie_starts)
    average_ranks = (tie_starts + 1 + tie_ends) / 2.0
    pos_rank_sum = float(np.dot(positive_per_tie, average_ranks))
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def spike_auc(prediction: np.ndarray, target: np.ndarray) -> float:
    return binary_auc(np.asarray(prediction)[..., 0], np.asarray(target)[..., 0])


def prevalence_matched_threshold(scores: np.ndarray, positive_count: int) -> float:
    """Return the score threshold that predicts `positive_count` events."""
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if positive_count <= 0:
        return float("inf")
    if positive_count >= flat.size:
        return float("-inf")
    return float(np.partition(flat, flat.size - positive_count)[flat.size - positive_count])


def voltage_relevance(
    target_soma_mv: np.ndarray,
    spike_labels: np.ndarray | None = None,
    *,
    spike_radius_steps: int = 5,
    voltage_tail_start: float = 0.80,
    voltage_tail_stop: float = 0.995,
    hyperpolar_tail_start: float = 0.90,
    hyperpolar_tail_stop: float = 0.995,
    dvdt_tail_start: float = 0.85,
    dvdt_tail_stop: float = 0.995,
) -> np.ndarray:
    """Teacher-defined relevance for voltage-trace error.

    The returned value is in `[0, 1]` and is high near teacher spikes, at
    voltage extrema, and during high absolute dV/dt.  It is a weighting
    function for evaluation only; it does not clip or transform the target.
    """
    soma = _as_2d(target_soma_mv)
    if spike_radius_steps < 0:
        raise ValueError("spike_radius_steps must be non-negative")
    relevance = np.zeros_like(soma, dtype=np.float64)
    if spike_labels is not None:
        spikes = _as_2d(spike_labels) > 0.5
        if spikes.shape != soma.shape:
            raise ValueError(f"spike_labels shape must match target_soma_mv, got {spikes.shape} and {soma.shape}")
        relevance = np.maximum(relevance, _dilate_spikes(spikes, spike_radius_steps).astype(np.float64))

    relevance = np.maximum(
        relevance,
        _tail_relevance(soma, start_quantile=voltage_tail_start, stop_quantile=voltage_tail_stop),
    )
    relevance = np.maximum(
        relevance,
        _tail_relevance(-soma, start_quantile=hyperpolar_tail_start, stop_quantile=hyperpolar_tail_stop),
    )
    dvdt = np.diff(soma, axis=1, prepend=soma[:, :1])
    relevance = np.maximum(
        relevance,
        _tail_relevance(np.abs(dvdt), start_quantile=dvdt_tail_start, stop_quantile=dvdt_tail_stop),
    )
    max_relevance = float(np.max(relevance)) if relevance.size else 0.0
    if max_relevance <= 0.0:
        return np.ones_like(relevance, dtype=np.float64)
    return np.clip(relevance / max_relevance, 0.0, 1.0)


def sera(
    error: np.ndarray,
    relevance: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    return_curve: bool = False,
) -> float | dict[str, Any]:
    """Support-normalized squared-error relevance area.

    For each relevance threshold, compute MSE on timesteps at or above that
    threshold and integrate those values over the threshold interval. Support
    normalization makes the value independent of dataset length. If relevance
    is constant one, the result equals MSE.
    """
    err = np.asarray(error, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)
    if err.shape != rel.shape:
        raise ValueError(f"error and relevance must have same shape, got {err.shape} and {rel.shape}")
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty 1D array")
    if np.any(np.diff(thresholds) < 0):
        raise ValueError("thresholds must be sorted in ascending order")

    squared_error = err ** 2
    mse_values = []
    support = []
    for threshold in thresholds:
        mask = rel >= float(threshold) - 1e-12
        support.append(int(mask.sum()))
        if mask.any():
            mse_values.append(float(np.mean(squared_error[mask])))
        else:
            mse_values.append(float("nan"))
    mse = np.asarray(mse_values, dtype=np.float64)
    finite = np.isfinite(mse)
    if not finite.any():
        value = float("nan")
    elif finite.sum() == 1:
        value = float(mse[finite][0])
    else:
        integral = float(_trapezoid(mse[finite], thresholds[finite]))
        span = float(thresholds[finite][-1] - thresholds[finite][0])
        if span > 0.0:
            integral /= span
        value = float(max(0.0, integral))
    if not return_curve:
        return value
    return {
        "value": value,
        "thresholds": [float(x) for x in thresholds],
        "mse": [float(x) if np.isfinite(x) else None for x in mse],
        "support": support,
    }


def root_sera(
    error: np.ndarray,
    relevance: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    return_curve: bool = False,
) -> float | dict[str, Any]:
    """Square root of support-normalized SERA for legacy mV reporting."""

    result = sera(
        error,
        relevance,
        thresholds=thresholds,
        return_curve=return_curve,
    )
    if not return_curve:
        return float(np.sqrt(result))
    curve = dict(result)
    curve["sera_value"] = float(curve["value"])
    curve["value"] = float(np.sqrt(curve["value"]))
    return curve


def voltage_sera(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    relevance_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute raw/teacher-relevance voltage SERA and companion RMSE."""
    pred, tgt, raw = _prepare_prediction_target(prediction, target, raw_soma_mv=raw_soma_mv, ignore_start=ignore_start)
    pred_soma_mv = pred[..., 1] / soma_scale + soma_bias_mv
    target_soma_mv = raw if raw is not None else tgt[..., 1] / soma_scale + soma_bias_mv
    target_spikes = tgt[..., 0] > 0.5
    if pred_soma_mv.shape != target_soma_mv.shape:
        raise ValueError(f"prediction and target soma shapes must match, got {pred_soma_mv.shape} and {target_soma_mv.shape}")
    relevance = voltage_relevance(target_soma_mv, target_spikes, **(relevance_kwargs or {}))
    error = pred_soma_mv - target_soma_mv
    sera_curve = sera(error, relevance, return_curve=True)
    root_sera_mv = float(np.sqrt(sera_curve["value"]))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {
        "voltage_sera_mv2": float(sera_curve["value"]),
        "voltage_root_sera_mv": root_sera_mv,
        "voltage_sera_mv": root_sera_mv,
        "voltage_rmse_mv": rmse,
        "voltage_sera_over_rmse": float(root_sera_mv / rmse) if rmse > 0.0 else None,
        "relevance_mean": float(np.mean(relevance)),
        "relevance_support_ge_0_5": int((relevance >= 0.5).sum()),
        "relevance_support_ge_0_9": int((relevance >= 0.9).sum()),
        "sera_curve": sera_curve,
    }


def dynamics_sera(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    timestep_ms: float = 1.0,
    ignore_start: int = 0,
    relevance_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute teacher-relevance SERA on soma-voltage transitions."""
    if timestep_ms <= 0.0:
        raise ValueError("timestep_ms must be positive")
    pred, tgt, raw = _prepare_prediction_target(
        prediction,
        target,
        raw_soma_mv=raw_soma_mv,
        ignore_start=ignore_start,
    )
    if pred.shape[1] < 2:
        raise ValueError("Dynamics SERA requires at least two scored timesteps")
    pred_soma_mv = pred[..., 1] / soma_scale + soma_bias_mv
    target_soma_mv = raw if raw is not None else tgt[..., 1] / soma_scale + soma_bias_mv
    target_spikes = tgt[..., 0] > 0.5
    relevance = voltage_relevance(target_soma_mv, target_spikes, **(relevance_kwargs or {}))
    transition_relevance = np.maximum(relevance[:, :-1], relevance[:, 1:])
    pred_rate = np.diff(pred_soma_mv, axis=1) / timestep_ms
    target_rate = np.diff(target_soma_mv, axis=1) / timestep_ms
    error = pred_rate - target_rate
    sera_curve = sera(error, transition_relevance, return_curve=True)
    root_sera_rate = float(np.sqrt(sera_curve["value"]))
    return {
        "dynamics_sera_mv2_per_ms2": float(sera_curve["value"]),
        "dynamics_root_sera_mv_per_ms": root_sera_rate,
        "dynamics_sera_mv_per_ms": root_sera_rate,
        "relevance_mean": float(np.mean(transition_relevance)),
        "relevance_support_ge_0_5": int((transition_relevance >= 0.5).sum()),
        "relevance_support_ge_0_9": int((transition_relevance >= 0.9).sum()),
        "sera_curve": sera_curve,
    }


def spike_event_confusion(
    predicted_spikes: np.ndarray,
    target_spikes: np.ndarray,
    *,
    tolerance_steps: int,
) -> dict[str, int]:
    """One-to-one event confusion under a symmetric timing tolerance."""
    if tolerance_steps < 0:
        raise ValueError("tolerance_steps must be non-negative")
    pred = _as_2d(predicted_spikes) > 0.5
    tgt = _as_2d(target_spikes) > 0.5
    if pred.shape != tgt.shape:
        raise ValueError(f"predicted_spikes and target_spikes must match, got {pred.shape} and {tgt.shape}")
    tp = 0
    pred_total = int(pred.sum())
    target_total = int(tgt.sum())
    for trace_index in range(pred.shape[0]):
        p_idx = np.flatnonzero(pred[trace_index])
        t_idx = np.flatnonzero(tgt[trace_index])
        i = j = 0
        while i < p_idx.size and j < t_idx.size:
            delta = int(p_idx[i] - t_idx[j])
            if abs(delta) <= tolerance_steps:
                tp += 1
                i += 1
                j += 1
            elif p_idx[i] < t_idx[j] - tolerance_steps:
                i += 1
            else:
                j += 1
    return {"tp": tp, "fp": pred_total - tp, "fn": target_total - tp}


def coincidence_factor(
    predicted_spikes: np.ndarray,
    target_spikes: np.ndarray,
    *,
    delta_steps: int = 2,
) -> dict[str, float | int]:
    """Coincidence factor Gamma (Kistler et al. 1997; Jolivet et al. 2008).

    Gamma = (N_coinc - <N_coinc>) / (0.5 (N_pred + N_target) (1 - 2 nu delta)),
    where coincidences are one-to-one within +-delta_steps and <N_coinc> =
    2 nu delta N_target is the count expected from a Poisson train at the
    predicted rate nu. Terms are pooled over traces, each with its own rate.
    Gamma is 1 for a perfect prediction and about 0 for chance.
    """
    if delta_steps < 0:
        raise ValueError("delta_steps must be non-negative")
    pred = _as_2d(predicted_spikes) > 0.5
    tgt = _as_2d(target_spikes) > 0.5
    if pred.shape != tgt.shape:
        raise ValueError(f"predicted_spikes and target_spikes must match, got {pred.shape} and {tgt.shape}")
    coincidences = spike_event_confusion(pred, tgt, tolerance_steps=delta_steps)["tp"]
    n_pred = pred.sum(axis=1).astype(np.float64)
    n_target = tgt.sum(axis=1).astype(np.float64)
    rate = n_pred / pred.shape[1]
    expected = float(np.sum(2.0 * rate * delta_steps * n_target))
    norm = float(np.sum((1.0 - 2.0 * rate * delta_steps) * 0.5 * (n_pred + n_target)))
    value = (coincidences - expected) / norm if norm > 0.0 else float("nan")
    return {
        "gamma": float(value),
        "coincidences": int(coincidences),
        "expected_coincidences": expected,
        "predicted_spikes": int(n_pred.sum()),
        "target_spikes": int(n_target.sum()),
        "delta_steps": int(delta_steps),
    }


def spike_f1_curve(
    spike_scores: np.ndarray,
    target_spikes: np.ndarray,
    *,
    threshold: float = 0.0,
    tolerances_steps: tuple[int, ...] | list[int] = (0, 1, 2, 3, 4, 5),
) -> dict[str, Any]:
    """Discrete spike-event F1 over a timing-tolerance interval."""
    if len(tolerances_steps) == 0:
        raise ValueError("tolerances_steps must contain at least one tolerance")
    tolerances = tuple(int(tolerance) for tolerance in tolerances_steps)
    if any(tolerance < 0 for tolerance in tolerances):
        raise ValueError("tolerances_steps must be non-negative")
    if any(
        right <= left
        for left, right in zip(tolerances, tolerances[1:])
    ):
        raise ValueError("tolerances_steps must be strictly increasing")
    scores = _as_2d(spike_scores)
    target = _as_2d(target_spikes) > 0.5
    if scores.shape != target.shape:
        raise ValueError(f"spike_scores and target_spikes must match, got {scores.shape} and {target.shape}")
    predicted = scores >= threshold
    rows = []
    for tolerance in tolerances:
        confusion = spike_event_confusion(predicted, target, tolerance_steps=tolerance)
        tp = confusion["tp"]
        fp = confusion["fp"]
        fn = confusion["fn"]
        precision = tp / (tp + fp) if tp + fp > 0 else None
        recall = tp / (tp + fn) if tp + fn > 0 else None
        if precision is None or recall is None or precision + recall == 0.0:
            f1 = 0.0 if (tp + fp + fn) > 0 else None
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        rows.append(
            {
                "tolerance_steps": tolerance,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                **confusion,
            }
        )
    f1_values = np.asarray([0.0 if row["f1"] is None else float(row["f1"]) for row in rows], dtype=np.float64)
    mean_f1 = float(np.mean(f1_values))
    if len(tolerances) == 1:
        area = float(f1_values[0])
    else:
        span = float(tolerances[-1] - tolerances[0])
        area = float(_trapezoid(f1_values, tolerances) / span) if span > 0 else float(np.mean(f1_values))
    return {
        "threshold": float(threshold),
        "t_min_steps": int(tolerances[0]),
        "t_max_steps": int(tolerances[-1]),
        "mean_f1": mean_f1,
        "f1_area": area,
        "rows": rows,
    }


def response_sera(
    baseline_prediction: np.ndarray,
    intervention_prediction: np.ndarray,
    baseline_target: np.ndarray,
    intervention_target: np.ndarray,
    *,
    baseline_raw_soma_mv: np.ndarray | None = None,
    intervention_raw_soma_mv: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    relevance_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute SERA on paired model-vs-teacher intervention response deltas."""
    base_pred, base_tgt, base_raw = _prepare_prediction_target(
        baseline_prediction,
        baseline_target,
        raw_soma_mv=baseline_raw_soma_mv,
        ignore_start=ignore_start,
    )
    int_pred, int_tgt, int_raw = _prepare_prediction_target(
        intervention_prediction,
        intervention_target,
        raw_soma_mv=intervention_raw_soma_mv,
        ignore_start=ignore_start,
    )
    if base_pred.shape != int_pred.shape or base_tgt.shape != int_tgt.shape:
        raise ValueError("baseline and intervention arrays must have matching prediction/target shapes")
    base_pred_mv = base_pred[..., 1] / soma_scale + soma_bias_mv
    int_pred_mv = int_pred[..., 1] / soma_scale + soma_bias_mv
    base_tgt_mv = base_raw if base_raw is not None else base_tgt[..., 1] / soma_scale + soma_bias_mv
    int_tgt_mv = int_raw if int_raw is not None else int_tgt[..., 1] / soma_scale + soma_bias_mv
    teacher_delta = int_tgt_mv - base_tgt_mv
    model_delta = int_pred_mv - base_pred_mv
    error = model_delta - teacher_delta
    base_rel = voltage_relevance(base_tgt_mv, base_tgt[..., 0] > 0.5, **(relevance_kwargs or {}))
    int_rel = voltage_relevance(int_tgt_mv, int_tgt[..., 0] > 0.5, **(relevance_kwargs or {}))
    delta_rel = _tail_relevance(np.abs(teacher_delta), start_quantile=0.80, stop_quantile=0.995)
    relevance = np.maximum.reduce([base_rel, int_rel, delta_rel])
    if float(np.max(relevance)) <= 0.0:
        relevance = np.ones_like(relevance, dtype=np.float64)
    sera_curve = sera(error, relevance, return_curve=True)
    root_sera_mv = float(np.sqrt(sera_curve["value"]))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {
        "response_sera_mv2": float(sera_curve["value"]),
        "response_root_sera_mv": root_sera_mv,
        "response_sera_mv": root_sera_mv,
        "response_rmse_mv": rmse,
        "response_sera_over_rmse": float(root_sera_mv / rmse) if rmse > 0.0 else None,
        "teacher_response_abs_mean_mv": float(np.mean(np.abs(teacher_delta))),
        "relevance_mean": float(np.mean(relevance)),
        "relevance_support_ge_0_5": int((relevance >= 0.5).sum()),
        "relevance_support_ge_0_9": int((relevance >= 0.9).sum()),
        "sera_curve": sera_curve,
    }


def compute_core_metric_set(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    spike_threshold: float | None = None,
    spike_threshold_policy: str = "prevalence",
    spike_tolerances_steps: tuple[int, ...] | list[int] = (0, 1, 2, 3, 4, 5),
    relevance_kwargs: dict[str, Any] | None = None,
    timestep_ms: float = 1.0,
    intervention_prediction: np.ndarray | None = None,
    intervention_target: np.ndarray | None = None,
    intervention_raw_soma_mv: np.ndarray | None = None,
) -> dict[str, Any]:
    """Easy entrypoint for the recommended AxoBench core metrics."""
    pred, tgt, raw = _prepare_prediction_target(prediction, target, raw_soma_mv=raw_soma_mv, ignore_start=ignore_start)
    voltage = voltage_sera(
        pred,
        tgt,
        raw_soma_mv=raw,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
        relevance_kwargs=relevance_kwargs,
    )
    dynamics = dynamics_sera(
        pred,
        tgt,
        raw_soma_mv=raw,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
        timestep_ms=timestep_ms,
        relevance_kwargs=relevance_kwargs,
    )
    target_spikes = tgt[..., 0] > 0.5
    if spike_threshold is None:
        if spike_threshold_policy == "prevalence":
            spike_threshold = prevalence_matched_threshold(pred[..., 0], int(target_spikes.sum()))
            threshold_kind = "prevalence_matched"
        elif spike_threshold_policy == "zero":
            spike_threshold = 0.0
            threshold_kind = "zero"
        else:
            raise ValueError("spike_threshold_policy must be 'prevalence' or 'zero'")
    else:
        threshold_kind = "calibrated"
    spike = spike_f1_curve(
        pred[..., 0],
        target_spikes,
        threshold=float(spike_threshold),
        tolerances_steps=spike_tolerances_steps,
    )
    spike["threshold_kind"] = threshold_kind
    result = {
        "voltage": voltage,
        "dynamics": dynamics,
        "spike": spike,
        "voltage_sera_mv2": voltage["voltage_sera_mv2"],
        "voltage_sera_mv": voltage["voltage_sera_mv"],
        "dynamics_sera_mv2_per_ms2": dynamics["dynamics_sera_mv2_per_ms2"],
        "dynamics_sera_mv_per_ms": dynamics["dynamics_sera_mv_per_ms"],
        "spike_mean_f1": spike["mean_f1"],
        "spike_f1_area": spike["f1_area"],
        "spike_threshold": float(spike_threshold),
        "spike_threshold_kind": threshold_kind,
    }
    if (intervention_prediction is None) != (intervention_target is None):
        raise ValueError("intervention_prediction and intervention_target must be supplied together")
    if intervention_prediction is not None and intervention_target is not None:
        response = response_sera(
            prediction,
            intervention_prediction,
            target,
            intervention_target,
            baseline_raw_soma_mv=raw_soma_mv,
            intervention_raw_soma_mv=intervention_raw_soma_mv,
            soma_scale=soma_scale,
            soma_bias_mv=soma_bias_mv,
            ignore_start=ignore_start,
            relevance_kwargs=relevance_kwargs,
        )
        result["response"] = response
        result["response_sera_mv2"] = response["response_sera_mv2"]
        result["response_sera_mv"] = response["response_sera_mv"]
    return result


def _prepare_prediction_target(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None,
    ignore_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = _as_trace_array(np.asarray(prediction, dtype=np.float64))
    tgt = _as_trace_array(np.asarray(target, dtype=np.float64))
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes must match, got {pred.shape} and {tgt.shape}")
    if ignore_start < 0:
        raise ValueError("ignore_start must be non-negative")
    if ignore_start > 0:
        pred = pred[:, ignore_start:, :]
        tgt = tgt[:, ignore_start:, :]
    if pred.shape[1] == 0:
        raise ValueError("ignore_start leaves no timesteps to evaluate")
    raw = None
    if raw_soma_mv is not None:
        raw = _as_2d(np.asarray(raw_soma_mv, dtype=np.float64))
        if ignore_start > 0:
            raw = raw[:, ignore_start:]
        if raw.shape != pred.shape[:2]:
            raise ValueError(f"raw_soma_mv must match trace/time shape after mask, got {raw.shape} and {pred.shape[:2]}")
    return pred, tgt, raw


def _as_trace_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim == 3:
        return array
    if array.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got {array.ndim}")
    leading = int(np.prod(array.shape[:-2]))
    return array.reshape(leading, array.shape[-2], array.shape[-1])


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array
    if array.ndim < 1:
        raise ValueError(f"expected at least 1 dimension, got {array.ndim}")
    leading = int(np.prod(array.shape[:-1]))
    return array.reshape(leading, array.shape[-1])


def _dilate_spikes(spikes: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros_like(spikes, dtype=bool)
    for trace_index in range(spikes.shape[0]):
        for spike_index in np.flatnonzero(spikes[trace_index]):
            start = max(0, int(spike_index) - radius)
            stop = min(spikes.shape[1], int(spike_index) + radius + 1)
            out[trace_index, start:stop] = True
    return out


def _tail_relevance(
    values: np.ndarray,
    *,
    start_quantile: float,
    stop_quantile: float,
) -> np.ndarray:
    if not 0.0 <= start_quantile <= stop_quantile <= 1.0:
        raise ValueError("tail quantiles must satisfy 0 <= start <= stop <= 1")
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    lo = float(np.quantile(flat, start_quantile))
    hi = float(np.quantile(flat, stop_quantile))
    if hi <= lo:
        return (arr >= hi).astype(np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))
