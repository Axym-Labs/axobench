"""Mechanistic response diagnostics for neuron surrogate traces.

These metrics ask whether a surrogate preserves voltage dynamics and input
response laws that are easy for global RMSE/AUC to hide.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_mechanistic_response_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    inputs: np.ndarray | None = None,
    *,
    target_soma_mv: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 500,
    event_window_pre: int = 5,
    event_window_post: int = 40,
    event_max_per_trace: int = 128,
    history_window: int = 50,
) -> dict[str, Any]:
    """Compute phase-plane, event-response, and history-conditioned metrics.

    `target_soma_mv` lets callers score against raw teacher soma voltage even
    when `target[..., 1]` is the historical clipped/normalized view.
    """
    pred = _apply_time_mask(_as_trace_array(np.asarray(prediction)), ignore_start)
    tgt = _apply_time_mask(_as_trace_array(np.asarray(target)), ignore_start)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes must match, got {pred.shape} and {tgt.shape}")
    if pred.shape[1] == 0:
        raise ValueError("ignore_start leaves no timesteps to evaluate")

    inp = _apply_time_mask(_as_trace_array(np.asarray(inputs)), ignore_start) if inputs is not None else None
    pred_soma_mv = pred[..., 1] / soma_scale + soma_bias_mv
    if target_soma_mv is None:
        tgt_soma_mv = tgt[..., 1] / soma_scale + soma_bias_mv
    else:
        raw = np.asarray(target_soma_mv)
        if raw.ndim == 1:
            raw = raw[None, :]
        tgt_soma_mv = raw[:, ignore_start:]
        if tgt_soma_mv.shape != pred_soma_mv.shape:
            raise ValueError(
                "target_soma_mv shape must match prediction time shape after mask, "
                f"got {tgt_soma_mv.shape} and {pred_soma_mv.shape}"
            )
    spike_labels = tgt[..., 0] > 0.5

    return {
        "kind": "mechanistic_response",
        "phase_plane": _phase_plane_metrics(pred_soma_mv, tgt_soma_mv, spike_labels),
        "event_triggered_response": (
            _event_triggered_response_metrics(
                pred_soma_mv,
                tgt_soma_mv,
                inp,
                window_pre=event_window_pre,
                window_post=event_window_post,
                max_events_per_trace=event_max_per_trace,
            )
            if inp is not None
            else {}
        ),
        "history_conditioned_error": _history_conditioned_metrics(
            pred_soma_mv,
            tgt_soma_mv,
            spike_labels,
            inp,
            history_window=history_window,
        ),
        "parameters": {
            "ignore_start": int(ignore_start),
            "event_window_pre": int(event_window_pre),
            "event_window_post": int(event_window_post),
            "event_max_per_trace": int(event_max_per_trace),
            "history_window": int(history_window),
            "target_soma_view": "raw_soma_mv" if target_soma_mv is not None else "target_channel",
        },
        "what_failure_means": (
            "High phase-plane or event-response error means the surrogate is "
            "not preserving the teacher's voltage dynamics or causal response "
            "to synaptic events, even if aggregate trace RMSE or spike AUC is good."
        ),
    }


def _phase_plane_metrics(
    pred_soma_mv: np.ndarray,
    tgt_soma_mv: np.ndarray,
    spike_labels: np.ndarray,
) -> dict[str, Any]:
    pred_dvdt = _per_trace_diff(pred_soma_mv)
    tgt_dvdt = _per_trace_diff(tgt_soma_mv)
    residual = pred_soma_mv - tgt_soma_mv
    dvdt_residual = pred_dvdt - tgt_dvdt
    high_dvdt = np.abs(tgt_dvdt) >= np.quantile(np.abs(tgt_dvdt), 0.95)
    near_spike = _near_spike_mask(spike_labels, radius=3)
    return {
        "voltage_rmse_mv": _rmse(residual),
        "dvdt_rmse_mv_per_step": _rmse(dvdt_residual),
        "dvdt_correlation": _safe_corr(pred_dvdt.reshape(-1), tgt_dvdt.reshape(-1)),
        "phase_distribution_jsd": _phase_jsd(pred_soma_mv, pred_dvdt, tgt_soma_mv, tgt_dvdt),
        "high_dvdt_rmse_mv_per_step": _rmse(dvdt_residual[high_dvdt]) if bool(np.any(high_dvdt)) else None,
        "high_dvdt_sign_agreement_fraction": (
            float(np.mean(np.sign(pred_dvdt[high_dvdt]) == np.sign(tgt_dvdt[high_dvdt])))
            if bool(np.any(high_dvdt))
            else None
        ),
        "near_spike_phase_rmse_mv": _rmse(residual[near_spike]) if bool(np.any(near_spike)) else None,
        "near_spike_dvdt_rmse_mv_per_step": _rmse(dvdt_residual[near_spike]) if bool(np.any(near_spike)) else None,
        "near_spike_support": int(np.sum(near_spike)),
    }


def _event_triggered_response_metrics(
    pred_soma_mv: np.ndarray,
    tgt_soma_mv: np.ndarray,
    inputs: np.ndarray,
    *,
    window_pre: int,
    window_post: int,
    max_events_per_trace: int,
) -> dict[str, dict[str, Any]]:
    exc_events = np.sum(inputs > 0, axis=2)
    inh_events = np.sum(inputs < 0, axis=2)
    return {
        "exc": _one_event_response(
            pred_soma_mv,
            tgt_soma_mv,
            exc_events,
            event_kind="exc",
            window_pre=window_pre,
            window_post=window_post,
            max_events_per_trace=max_events_per_trace,
        ),
        "inh": _one_event_response(
            pred_soma_mv,
            tgt_soma_mv,
            inh_events,
            event_kind="inh",
            window_pre=window_pre,
            window_post=window_post,
            max_events_per_trace=max_events_per_trace,
        ),
    }


def _one_event_response(
    pred_soma_mv: np.ndarray,
    tgt_soma_mv: np.ndarray,
    event_count: np.ndarray,
    *,
    event_kind: str,
    window_pre: int,
    window_post: int,
    max_events_per_trace: int,
) -> dict[str, Any]:
    offsets = np.arange(-window_pre, window_post + 1, dtype=int)
    pred_curves = []
    tgt_curves = []
    for trace_index in range(event_count.shape[0]):
        candidates = np.flatnonzero(event_count[trace_index] > 0)
        candidates = candidates[(candidates >= window_pre) & (candidates < event_count.shape[1] - window_post)]
        if candidates.size == 0:
            continue
        if candidates.size > max_events_per_trace:
            positions = np.linspace(0, candidates.size - 1, max_events_per_trace).astype(int)
            candidates = candidates[positions]
        for event_time in candidates:
            baseline_index = max(0, event_time - 1)
            window = event_time + offsets
            pred_curves.append(pred_soma_mv[trace_index, window] - pred_soma_mv[trace_index, baseline_index])
            tgt_curves.append(tgt_soma_mv[trace_index, window] - tgt_soma_mv[trace_index, baseline_index])
    if not pred_curves:
        return {
            "event_count": 0,
            "response_curve_rmse_mv": None,
            "target_peak_delta_mv": None,
            "model_peak_delta_mv": None,
            "response_gain": None,
            "latency_error_steps": None,
            "peak_sign_agreement": None,
        }
    pred_mean = np.mean(np.stack(pred_curves), axis=0)
    tgt_mean = np.mean(np.stack(tgt_curves), axis=0)
    response_slice = offsets >= 0
    selector = np.argmax if event_kind == "exc" else np.argmin
    target_local = int(selector(tgt_mean[response_slice]))
    model_local = int(selector(pred_mean[response_slice]))
    response_offsets = offsets[response_slice]
    target_peak = float(tgt_mean[response_slice][target_local])
    model_peak = float(pred_mean[response_slice][model_local])
    gain = None if abs(target_peak) < 1e-6 else float(model_peak / target_peak)
    return {
        "event_count": int(len(pred_curves)),
        "response_curve_rmse_mv": _rmse(pred_mean - tgt_mean),
        "target_peak_delta_mv": target_peak,
        "model_peak_delta_mv": model_peak,
        "response_gain": gain,
        "latency_error_steps": int(response_offsets[model_local] - response_offsets[target_local]),
        "peak_sign_agreement": float(np.sign(model_peak) == np.sign(target_peak)),
    }


def _history_conditioned_metrics(
    pred_soma_mv: np.ndarray,
    tgt_soma_mv: np.ndarray,
    spike_labels: np.ndarray,
    inputs: np.ndarray | None,
    *,
    history_window: int,
) -> dict[str, dict[str, Any]]:
    residual = pred_soma_mv - tgt_soma_mv
    dvdt_residual = _per_trace_diff(pred_soma_mv) - _per_trace_diff(tgt_soma_mv)
    recent_spikes = _causal_rolling_count(spike_labels.astype(float), history_window)
    masks: dict[str, np.ndarray] = {
        "recent_spike_history": (recent_spikes > 0) & ~spike_labels,
    }
    if inputs is not None:
        recent_exc = _causal_rolling_count(np.sum(inputs > 0, axis=2).astype(float), history_window)
        recent_inh = _causal_rolling_count(np.sum(inputs < 0, axis=2).astype(float), history_window)
        exc_threshold = _positive_quantile(recent_exc, 0.9)
        inh_threshold = _positive_quantile(recent_inh, 0.9)
        moderate_exc = _positive_quantile(recent_exc, 0.75)
        moderate_inh = _positive_quantile(recent_inh, 0.75)
        masks.update(
            {
                "high_recent_exc_drive": (recent_exc >= exc_threshold) & (recent_exc > recent_inh),
                "high_recent_inh_drive": (recent_inh >= inh_threshold) & (recent_inh > recent_exc),
                "balanced_high_drive": (recent_exc >= moderate_exc) & (recent_inh >= moderate_inh),
                "quiet_history": (recent_exc <= np.median(recent_exc)) & (recent_inh <= np.median(recent_inh)) & (recent_spikes == 0),
            }
        )
    else:
        masks["quiet_history"] = recent_spikes == 0
    return {name: _masked_error_summary(mask, residual, dvdt_residual) for name, mask in masks.items()}


def _masked_error_summary(mask: np.ndarray, residual: np.ndarray, dvdt_residual: np.ndarray) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {
            "support": 0,
            "fraction": 0.0,
            "voltage_rmse_mv": None,
            "voltage_mae_mv": None,
            "dvdt_mae_mv_per_step": None,
        }
    return {
        "support": count,
        "fraction": float(count / mask.size),
        "voltage_rmse_mv": _rmse(residual[mask]),
        "voltage_mae_mv": float(np.mean(np.abs(residual[mask]))),
        "dvdt_mae_mv_per_step": float(np.mean(np.abs(dvdt_residual[mask]))),
    }


def _phase_jsd(
    pred_soma_mv: np.ndarray,
    pred_dvdt: np.ndarray,
    tgt_soma_mv: np.ndarray,
    tgt_dvdt: np.ndarray,
    *,
    bins: int = 32,
) -> float:
    v_values = np.concatenate([pred_soma_mv.reshape(-1), tgt_soma_mv.reshape(-1)])
    d_values = np.concatenate([pred_dvdt.reshape(-1), tgt_dvdt.reshape(-1)])
    v_range = _robust_range(v_values)
    d_range = _robust_range(d_values)
    pred_hist, _, _ = np.histogram2d(pred_soma_mv.reshape(-1), pred_dvdt.reshape(-1), bins=bins, range=[v_range, d_range])
    tgt_hist, _, _ = np.histogram2d(tgt_soma_mv.reshape(-1), tgt_dvdt.reshape(-1), bins=bins, range=[v_range, d_range])
    return _js_divergence(pred_hist.reshape(-1), tgt_hist.reshape(-1))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def _robust_range(values: np.ndarray) -> list[float]:
    lo = float(np.quantile(values, 0.005))
    hi = float(np.quantile(values, 0.995))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.02 * (hi - lo)
    return [lo - pad, hi + pad]


def _causal_rolling_count(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("history window must be positive")
    values = np.asarray(values, dtype=np.float64)
    shifted = np.pad(values[:, :-1], ((0, 0), (1, 0)), mode="constant")
    cumsum = np.cumsum(shifted, axis=1)
    padded = np.pad(cumsum, ((0, 0), (1, 0)), mode="constant")
    start = np.arange(values.shape[1])
    stop = start + 1
    start = np.maximum(0, stop - window)
    return padded[:, stop] - padded[:, start]


def _positive_quantile(values: np.ndarray, quantile: float) -> float:
    positive = np.asarray(values)[np.asarray(values) > 0]
    if positive.size == 0:
        return float("inf")
    return float(np.quantile(positive, quantile))


def _near_spike_mask(spikes: np.ndarray, *, radius: int) -> np.ndarray:
    out = np.zeros_like(spikes, dtype=bool)
    for trace in range(spikes.shape[0]):
        for idx in np.flatnonzero(spikes[trace]):
            start = max(0, idx - radius)
            stop = min(spikes.shape[1], idx + radius + 1)
            out[trace, start:stop] = True
    return out


def _as_trace_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        leading = int(np.prod(array.shape[:-2]))
        return array.reshape(leading, array.shape[-2], array.shape[-1])
    return array


def _apply_time_mask(array: np.ndarray, ignore_start: int) -> np.ndarray:
    return array[:, ignore_start:, :] if ignore_start > 0 else array


def _per_trace_diff(values: np.ndarray) -> np.ndarray:
    return np.diff(values, axis=1, prepend=values[:, :1])


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
