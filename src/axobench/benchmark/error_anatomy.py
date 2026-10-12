"""Error anatomy: where a model's voltage and spike errors come from.

:func:`error_anatomy` breaks down the errors behind the evaluator's metrics
on the same scored traces, with the same spike threshold, predicted spike
events, action-potential windows, relevance and subthreshold masks as
:func:`~axobench.benchmark.iteration_evaluator.evaluate_predictions`:

- spike events: teacher spikes hit or missed and false predicted spikes,
  F1 per timing tolerance, and the timing errors of matched events;
- spike-triggered average voltage of teacher and model around isolated hit
  and missed teacher spikes and around false predicted spikes (a model that
  averages away action potentials shows a low model peak even on hits);
- subthreshold error by time since the last and to the next teacher spike;
- voltage SERA, subthreshold and spike skill by teacher firing-rate bin and
  by morphology; subthreshold skill by frequency band.

Skill against the dataset's own reference depends on its spike rate (the
SERA ceiling falls and post-spike error grows with rate), so compare
datasets with :func:`reweight_to_rate_distribution`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from axobench.benchmark.iteration_evaluator import (
    EvaluationConfig,
    _as_traces,
    _f1_metrics_from_trace_statistics,
    _f1_trace_statistics,
    _morphology_id,
    _slice_optional_2d,
    _subthreshold_ap_windows,
    calibrate_spike_threshold,
    sera_reference_voltage,
)
from axobench.metrics import (
    sera_skill_trace_terms,
    sera_skills,
    spike_event_matches,
    spike_events,
    subthreshold_voltage_point_terms,
)

HIT_TOLERANCE_STEPS = 2
STA_LAGS_STEPS = (-10, 40)
RATE_BINS_HZ = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, np.inf))
SPIKE_DISTANCE_BINS_MS = ((0.0, 5.0), (5.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 100.0), (100.0, np.inf))
FREQUENCY_BANDS_HZ = ((0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 100.0), (100.0, np.inf))
_SUM_KEYS = ("voltage", "voltage_reference", "voltage_ceiling", "subthreshold_error", "subthreshold_null")


def error_anatomy(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None = None,
    trace_ids: list[str] | None = None,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Break a prediction's errors down by spike outcome, spike distance, rate, band and morphology.

    Arrays and `config` are those of ``evaluate_predictions``; the result is
    JSON-serialisable. Times are in steps (``*_steps``) or ms; voltages in mV.
    """
    cfg = config or EvaluationConfig()
    pred = _as_traces(prediction, "prediction")
    tgt = _as_traces(target, "target")
    if pred.shape != tgt.shape or pred.shape[-1] != 2 or pred.shape[0] < 2:
        raise ValueError(f"prediction and target must match with shape (trace>=2, time, 2), got {pred.shape} and {tgt.shape}")
    if cfg.ignore_start >= pred.shape[1] - 1:
        raise ValueError("ignore_start leaves too few timesteps to evaluate")
    ids = [str(i) for i in trace_ids] if trace_ids is not None else [str(i) for i in range(pred.shape[0])]
    if len(ids) != pred.shape[0]:
        raise ValueError("trace_ids length must match trace count")
    n_calibration, threshold = calibrate_spike_threshold(pred, tgt, cfg)
    scored, start = slice(n_calibration, None), cfg.ignore_start
    raw_scored = _slice_optional_2d(raw_soma_mv, scored, 0, pred.shape[:2], "raw_soma_mv")

    scores = pred[scored, start:, 0]
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    predicted = spike_events(scores, threshold) > 0.5
    teacher = tgt[scored, start:, 0] > 0.5
    teacher_mv = (
        raw_scored[:, start:] if raw_scored is not None else tgt[scored, start:, 1] / cfg.soma_scale + cfg.soma_bias_mv
    )
    model_mv = pred[scored, start:, 1] / cfg.soma_scale + cfg.soma_bias_mv
    sera_terms = sera_skill_trace_terms(
        pred[scored, start:],
        tgt[scored, start:],
        reference_mv=sera_reference_voltage(tgt, raw_soma_mv, n_calibration, cfg),
        raw_soma_mv=None if raw_scored is None else raw_scored[:, start:],
        soma_scale=cfg.soma_scale,
        soma_bias_mv=cfg.soma_bias_mv,
        timestep_ms=cfg.timestep_ms,
    )
    points = subthreshold_voltage_point_terms(
        pred[scored],
        tgt[scored],
        predicted_spikes=predicted,
        raw_soma_mv=raw_scored,
        ap_post_steps=_subthreshold_ap_windows(tgt, raw_soma_mv, ids, cfg)["post_steps"][scored],
        soma_scale=cfg.soma_scale,
        soma_bias_mv=cfg.soma_bias_mv,
        ignore_start=start,
    )
    tolerances = tuple(int(t) for t in cfg.f1_tolerances_steps)
    f1_stats = _f1_trace_statistics(scores, teacher, threshold=threshold, tolerances=tolerances)
    trace_terms = {
        **{key: sera_terms[key] for key in ("voltage", "voltage_reference", "voltage_ceiling")},
        "subthreshold_error": np.sum(points["error"] ** 2, axis=1),
        "subthreshold_null": np.sum(points["deviation"] ** 2, axis=1),
    }
    rate_hz = teacher.sum(axis=1) / (teacher.shape[1] * cfg.timestep_ms / 1000.0)

    def summary(indices: np.ndarray) -> dict[str, Any]:
        return _group_summary(indices, sera_terms, trace_terms, f1_stats, tolerances, rate_hz)

    all_traces = np.arange(teacher.shape[0])
    full_teacher = tgt[scored, :, 0] > 0.5  # spikes before ignore_start count for spike distances
    morphologies = np.array([_morphology_id(i) for i in ids[n_calibration:]])
    return {
        "config": asdict(cfg),
        "calibration_samples": int(n_calibration),
        "scored_samples": int(teacher.shape[0]),
        "spike_threshold": float(threshold),
        "overall": summary(all_traces),
        "spike_events": _spike_event_section(predicted, teacher, f1_stats, tolerances),
        "spike_triggered_average": _spike_triggered_section(predicted, teacher, teacher_mv, model_mv),
        "subthreshold_by_spike_distance": {
            "since_last_teacher_spike": _distance_section(_steps_since_last(full_teacher)[:, start:], points, cfg),
            "to_next_teacher_spike": _distance_section(
                _steps_since_last(full_teacher[:, ::-1])[:, ::-1][:, start:], points, cfg
            ),
        },
        "by_teacher_rate": [
            {"bin": _label(low, high), "low_hz": low, "high_hz": _finite_or_none(high), **summary(
                all_traces[(rate_hz >= low) & (rate_hz < high)]
            )}
            for low, high in RATE_BINS_HZ
        ],
        "subthreshold_by_frequency": _frequency_section(points, cfg),
        "by_morphology": {str(name): summary(all_traces[morphologies == name]) for name in sorted(set(morphologies))},
    }


def reweight_to_rate_distribution(
    anatomy: dict[str, Any], reference: dict[str, Any] | Mapping[str, float]
) -> dict[str, Any]:
    """Skills of `anatomy` with its traces reweighted to a reference rate-bin distribution.

    `reference` is another :func:`error_anatomy` result (its scored-trace
    shares per rate bin) or a mapping from bin label (``"0-1"``, ``"1-3"``,
    ``"3-6"``, ``"6+"`` Hz) to share. Each trace is weighted by reference
    share / own share of its bin; bins the reference lacks get weight zero,
    so the comparison covers their common support. Relevance and references
    stay those of `anatomy`'s own block.
    """
    rows = anatomy["by_teacher_rate"]
    shares = (
        {row["bin"]: row["trace_share"] for row in reference["by_teacher_rate"]}
        if "by_teacher_rate" in reference
        else {str(key): float(value) for key, value in reference.items()}
    )
    sums = dict.fromkeys(_SUM_KEYS, 0.0)
    confusion = None
    for row in rows:
        share = shares.get(row["bin"], 0.0)
        if share <= 0.0:
            continue
        if row["traces"] == 0:
            raise ValueError(f"rate bin {row['bin']} Hz has reference weight but no traces")
        weight = share / row["trace_share"]
        for key in sums:
            sums[key] += weight * row["sums"][key]
        counts = weight * np.asarray(row["spike_confusion"], dtype=np.float64)
        confusion = counts if confusion is None else confusion + counts
    if confusion is None:
        raise ValueError("reference shares no rate bin with anatomy")
    skills = _skills_from_sums(sums)
    f1 = [2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0 for tp, fp, fn in confusion]
    return {**skills, "mean_f1": float(np.mean(f1)), "bins": [b for b in shares if shares[b] > 0]}


def _group_summary(
    indices: np.ndarray,
    sera_terms: dict[str, np.ndarray],
    trace_terms: dict[str, np.ndarray],
    f1_stats: np.ndarray,
    tolerances: tuple[int, ...],
    rate_hz: np.ndarray,
) -> dict[str, Any]:
    traces = int(indices.size)
    sums = {key: float(value[indices].sum()) for key, value in trace_terms.items()}
    out: dict[str, Any] = {
        "traces": traces,
        "trace_share": traces / rate_hz.size,
        "mean_rate_hz": float(rate_hz[indices].mean()) if traces else None,
        "teacher_spikes": int(sera_terms["teacher_spikes"][indices].sum()),
    }
    if not traces:
        return {**out, **dict.fromkeys(("voltage_sera_skill", "voltage_sera_ceiling_skill", "voltage_sera_skill_of_ceiling", "dynamics_sera_skill", "subthreshold_skill", "mean_f1"))}
    out.update(sera_skills(sera_terms, indices))
    out["subthreshold_skill"] = _skills_from_sums(sums)["subthreshold_skill"]
    out["mean_f1"] = _f1_metrics_from_trace_statistics(f1_stats, indices, tolerances)["mean"]
    out["sums"] = sums
    out["spike_confusion"] = f1_stats[indices].sum(axis=0).tolist()
    return out


def _skills_from_sums(sums: dict[str, float]) -> dict[str, float | None]:
    def ratio(a: str, b: str) -> float | None:
        return sums[a] / sums[b] if sums[b] > 0.0 else None

    voltage = None if ratio("voltage", "voltage_reference") is None else 1.0 - ratio("voltage", "voltage_reference")
    ceiling = None if ratio("voltage_ceiling", "voltage_reference") is None else 1.0 - ratio("voltage_ceiling", "voltage_reference")
    sub = ratio("subthreshold_error", "subthreshold_null")
    return {
        "voltage_sera_skill": voltage,
        "voltage_sera_ceiling_skill": ceiling,
        "voltage_sera_skill_of_ceiling": voltage / ceiling if voltage is not None and ceiling else None,
        "subthreshold_skill": None if sub is None else 1.0 - float(np.sqrt(sub)),
    }


def _spike_event_section(
    predicted: np.ndarray, teacher: np.ndarray, f1_stats: np.ndarray, tolerances: tuple[int, ...]
) -> dict[str, Any]:
    matches = spike_event_matches(predicted, teacher, tolerance_steps=HIT_TOLERANCE_STEPS)
    widest = spike_event_matches(predicted, teacher, tolerance_steps=max(tolerances))
    offsets = np.arange(-max(tolerances), max(tolerances) + 1)
    counts = np.bincount(widest[:, 1] - widest[:, 2] + max(tolerances), minlength=offsets.size)
    by_tolerance = []
    for tolerance, (tp, fp, fn) in zip(tolerances, f1_stats.sum(axis=0).tolist(), strict=True):
        denominator = 2 * tp + fp + fn
        by_tolerance.append(
            {"tolerance_steps": tolerance, "tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / denominator if denominator else None}
        )
    return {
        "tolerance_steps": HIT_TOLERANCE_STEPS,
        "teacher_spikes": int(teacher.sum()),
        "predicted_spikes": int(predicted.sum()),
        "hit": int(len(matches)),
        "missed": int(teacher.sum()) - len(matches),
        "false": int(predicted.sum()) - len(matches),
        "f1_by_tolerance": by_tolerance,
        "timing_error": {
            "tolerance_steps": int(max(tolerances)),
            "offset_steps": offsets.tolist(),
            "matched_events": counts.tolist(),
            "description": "predicted minus teacher event step of one-to-one matches",
        },
    }


def _spike_triggered_section(
    predicted: np.ndarray, teacher: np.ndarray, teacher_mv: np.ndarray, model_mv: np.ndarray
) -> dict[str, Any]:
    matches = spike_event_matches(predicted, teacher, tolerance_steps=HIT_TOLERANCE_STEPS)
    hit = np.zeros_like(teacher)
    hit[matches[:, 0], matches[:, 2]] = True
    true_positive = np.zeros_like(predicted)
    true_positive[matches[:, 0], matches[:, 1]] = True
    lags = np.arange(STA_LAGS_STEPS[0], STA_LAGS_STEPS[1] + 1)
    out: dict[str, Any] = {
        "lags_steps": lags.tolist(),
        "description": (
            f"mean unclipped voltage around isolated events (no other event of the same train within "
            f"lags {STA_LAGS_STEPS[0]}..+{STA_LAGS_STEPS[1]}); hit = teacher spike matched within "
            f"+-{HIT_TOLERANCE_STEPS} steps"
        ),
    }
    for name, events, train in (
        ("hit", hit, teacher),
        ("missed", teacher & ~hit, teacher),
        ("false", predicted & ~true_positive, predicted),
    ):
        rows, times = _isolated_events(events, train, lags)
        windows = times[:, None] + lags
        teacher_mean = teacher_mv[rows[:, None], windows].mean(axis=0) if rows.size else None
        model_mean = model_mv[rows[:, None], windows].mean(axis=0) if rows.size else None
        out[name] = {
            "events": int(rows.size),
            "teacher_mv": None if teacher_mean is None else teacher_mean.tolist(),
            "model_mv": None if model_mean is None else model_mean.tolist(),
            "teacher_peak_mv": None if teacher_mean is None else float(teacher_mean.max()),
            "model_peak_mv": None if model_mean is None else float(model_mean.max()),
        }
    return out


def _isolated_events(events: np.ndarray, train: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Events whose window lies inside the trace with no other event of `train` in it."""
    rows, times = np.nonzero(events)
    inside = (times + lags[0] >= 0) & (times + lags[-1] < events.shape[1])
    rows, times = rows[inside], times[inside]
    count = np.concatenate([np.zeros((train.shape[0], 1)), np.cumsum(train, axis=1)], axis=1)
    in_window = count[rows, times + lags[-1] + 1] - count[rows, times + lags[0]]
    return rows[in_window == 1], times[in_window == 1]


def _steps_since_last(spikes: np.ndarray) -> np.ndarray:
    steps = np.arange(spikes.shape[1])
    last = np.maximum.accumulate(np.where(spikes, steps, -1), axis=1)
    return np.where(last >= 0, steps - last, np.inf)


def _distance_section(distance_steps: np.ndarray, points: dict[str, np.ndarray], cfg: EvaluationConfig) -> list[dict[str, Any]]:
    keep = points["keep"]
    error2, null2 = points["error"] ** 2, points["deviation"] ** 2
    total_points, total_error, total_null = keep.sum(), error2.sum(), null2.sum()
    distance_ms = distance_steps * cfg.timestep_ms
    bins = [(_label(low, high), (distance_ms >= low) & (distance_ms < high)) for low, high in SPIKE_DISTANCE_BINS_MS]
    bins.append(("none", np.isinf(distance_ms)))
    out = []
    for label, mask in bins:
        mask = mask & keep
        error, null = float(error2[mask].sum()), float(null2[mask].sum())
        out.append(
            {
                "bin_ms": label,
                "point_share": float(mask.sum() / total_points) if total_points else None,
                "error_share": error / total_error if total_error > 0 else None,
                "null_share": null / total_null if total_null > 0 else None,
                "skill": 1.0 - float(np.sqrt(error / null)) if null > 0 else None,
            }
        )
    return out


def _frequency_section(points: dict[str, np.ndarray], cfg: EvaluationConfig) -> list[dict[str, Any]]:
    """Band power of the zero-filled error and teacher deviation (Parseval, one-sided)."""
    steps = points["error"].shape[1]
    weights = np.full(steps // 2 + 1, 2.0)
    weights[0] = 1.0
    if steps % 2 == 0:
        weights[-1] = 1.0
    error_power = (np.abs(np.fft.rfft(points["error"], axis=1)) ** 2).sum(axis=0) * weights
    null_power = (np.abs(np.fft.rfft(points["deviation"], axis=1)) ** 2).sum(axis=0) * weights
    frequency = np.fft.rfftfreq(steps, cfg.timestep_ms / 1000.0)
    out = []
    for low, high in FREQUENCY_BANDS_HZ:
        band = (frequency >= low) & (frequency < high)
        error, null = float(error_power[band].sum()), float(null_power[band].sum())
        out.append(
            {
                "band_hz": _label(low, high),
                "error_share": error / float(error_power.sum()) if error_power.sum() > 0 else None,
                "teacher_share": null / float(null_power.sum()) if null_power.sum() > 0 else None,
                "skill": 1.0 - float(np.sqrt(error / null)) if null > 0 else None,
            }
        )
    return out


def _label(low: float, high: float) -> str:
    return f"{low:g}+" if np.isinf(high) else f"{low:g}-{high:g}"


def _finite_or_none(value: float) -> float | None:
    return None if np.isinf(value) else float(value)
