from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from axobench.relevance import iron_relevance, iron_threshold_weights

AP_PRE_STEPS = 2
AP_POST_STEPS = 4  # fallback window end
AP_FLAT_SLOPE_MV_PER_MS = 1.0
AP_WAVEFORM_PRE_STEPS = 6
AP_ISOLATION_STEPS = 30
AP_MIN_ISOLATED_SPIKES = 20
SOMA_CLIP_MV = -55.0


def soma_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_soma = np.asarray(prediction, dtype=np.float64)[..., 1]
    target_soma = np.asarray(target, dtype=np.float64)[..., 1]
    return float(np.sqrt(np.mean((pred_soma - target_soma) ** 2)))


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and labels must have same shape, got {scores.shape} and {labels.shape}"
        )

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
    return float(
        np.partition(flat, flat.size - positive_count)[flat.size - positive_count]
    )


def spike_event_peaks(scores: np.ndarray, *, refractory_steps: int = 2, min_score: float = -np.inf) -> np.ndarray:
    """Candidate spike events: local maxima of each trace's score, one per refractory window.

    A timestep is a candidate if its score exceeds the previous one and is not
    below the next one (the first step of a plateau). Candidates are then kept
    greedily from the highest score down, dropping any within
    `refractory_steps` of a kept one. Candidates below `min_score` are dropped
    first; since lower peaks never suppress higher ones, this does not change
    which peaks at or above `min_score` are kept. Returns a boolean
    (traces, time) array.
    """
    if refractory_steps < 0:
        raise ValueError("refractory_steps must be non-negative")
    s = _as_2d(scores)
    left = np.concatenate([np.full((s.shape[0], 1), -np.inf), s[:, :-1]], axis=1)
    right = np.concatenate([s[:, 1:], np.full((s.shape[0], 1), -np.inf)], axis=1)
    candidate = (s > left) & (s >= right) & (s >= min_score)
    if refractory_steps == 0:
        return candidate
    kept = np.zeros_like(candidate)
    for i in range(s.shape[0]):
        idx = np.flatnonzero(candidate[i])
        blocked = np.zeros(s.shape[1], dtype=bool)
        for t in idx[np.argsort(-s[i, idx], kind="stable")]:
            if not blocked[t]:
                kept[i, t] = True
                blocked[max(0, t - refractory_steps) : t + refractory_steps + 1] = True
    return kept


def spike_events(scores: np.ndarray, threshold: float, *, refractory_steps: int | None = 2) -> np.ndarray:
    """Binary spike events from a spike score.

    With `refractory_steps` set, events are the score peaks of `spike_event_peaks`
    at or above `threshold`, so a run of supra-threshold timesteps is one event.
    With `refractory_steps=None`, every timestep at or above threshold is an event.
    """
    s = _as_2d(scores)
    if refractory_steps is None:
        return s >= threshold
    return spike_event_peaks(s, refractory_steps=refractory_steps, min_score=threshold)


def event_count_matched_threshold(scores: np.ndarray, positive_count: int, *, refractory_steps: int | None = 2) -> float:
    """Threshold at which `spike_events` yields `positive_count` events."""
    if refractory_steps is None:
        return prevalence_matched_threshold(scores, positive_count)
    s = _as_2d(scores)
    heights = s[spike_event_peaks(s, refractory_steps=refractory_steps)]
    if positive_count <= 0:
        return float("inf")
    if positive_count >= heights.size:
        return float("-inf")
    return float(np.sort(heights)[::-1][positive_count - 1])


def spike_event_matches(
    predicted_spikes: np.ndarray,
    target_spikes: np.ndarray,
    *,
    tolerance_steps: int,
) -> np.ndarray:
    """One-to-one event matches under a symmetric timing tolerance.

    Events of each trace are paired in time order: a predicted and a target
    event within `tolerance_steps` of each other are matched and both are
    consumed. Returns an integer (matches, 3) array of (trace, predicted step,
    target step).
    """
    if tolerance_steps < 0:
        raise ValueError("tolerance_steps must be non-negative")
    pred = _as_2d(predicted_spikes) > 0.5
    tgt = _as_2d(target_spikes) > 0.5
    if pred.shape != tgt.shape:
        raise ValueError(
            f"predicted_spikes and target_spikes must match, got {pred.shape} and {tgt.shape}"
        )
    matches = []
    for trace_index in range(pred.shape[0]):
        p_idx = np.flatnonzero(pred[trace_index])
        t_idx = np.flatnonzero(tgt[trace_index])
        i = j = 0
        while i < p_idx.size and j < t_idx.size:
            if abs(int(p_idx[i] - t_idx[j])) <= tolerance_steps:
                matches.append((trace_index, int(p_idx[i]), int(t_idx[j])))
                i += 1
                j += 1
            elif p_idx[i] < t_idx[j] - tolerance_steps:
                i += 1
            else:
                j += 1
    return np.asarray(matches, dtype=np.int64).reshape(-1, 3)


def spike_event_confusion(
    predicted_spikes: np.ndarray,
    target_spikes: np.ndarray,
    *,
    tolerance_steps: int,
) -> dict[str, int]:
    """One-to-one event confusion under a symmetric timing tolerance (:func:`spike_event_matches`)."""
    tp = len(spike_event_matches(predicted_spikes, target_spikes, tolerance_steps=tolerance_steps))
    return {
        "tp": tp,
        "fp": int((_as_2d(predicted_spikes) > 0.5).sum()) - tp,
        "fn": int((_as_2d(target_spikes) > 0.5).sum()) - tp,
    }


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
    refractory_steps: int | None = 2,
) -> dict[str, Any]:
    """Discrete spike-event F1 over a timing-tolerance interval.

    Predicted events come from `spike_events`: score peaks at or above
    `threshold`, at most one per `refractory_steps` window (or every
    supra-threshold timestep with `refractory_steps=None`).
    """
    if len(tolerances_steps) == 0:
        raise ValueError("tolerances_steps must contain at least one tolerance")
    tolerances = tuple(int(tolerance) for tolerance in tolerances_steps)
    if any(tolerance < 0 for tolerance in tolerances):
        raise ValueError("tolerances_steps must be non-negative")
    if any(right <= left for left, right in itertools.pairwise(tolerances)):
        raise ValueError("tolerances_steps must be strictly increasing")
    scores = _as_2d(spike_scores)
    target = _as_2d(target_spikes) > 0.5
    if scores.shape != target.shape:
        raise ValueError(f"spike_scores and target_spikes must match, got {scores.shape} and {target.shape}")
    predicted = spike_events(scores, threshold, refractory_steps=refractory_steps)
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
    f1_values = np.asarray(
        [0.0 if row["f1"] is None else float(row["f1"]) for row in rows],
        dtype=np.float64,
    )
    mean_f1 = float(np.mean(f1_values))
    if len(tolerances) == 1:
        area = float(f1_values[0])
    else:
        span = float(tolerances[-1] - tolerances[0])
        area = (
            float(_trapezoid(f1_values, tolerances) / span)
            if span > 0
            else float(np.mean(f1_values))
        )
    return {
        "threshold": float(threshold),
        "refractory_steps": refractory_steps,
        "t_min_steps": int(tolerances[0]),
        "t_max_steps": int(tolerances[-1]),
        "mean_f1": mean_f1,
        "f1_area": area,
        "rows": rows,
    }


def iron_voltage_sera(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
) -> dict[str, Any]:
    """IRon-default voltage SERA per teacher spike.

    Relevance is IRon's default ``phi`` on the teacher somatic voltage
    (adjusted-boxplot extremes; :mod:`axobench.relevance`), derived from the
    pooled scored voltages. SERA is IRon's threshold area of summed squared
    errors (Ribeiro and Moniz, 2020), divided by the number of teacher spikes
    in the scored region.
    """
    pred, tgt, raw = _prepare_prediction_target(
        prediction, target, raw_soma_mv=raw_soma_mv, ignore_start=ignore_start
    )
    pred_mv = pred[..., 1] / soma_scale + soma_bias_mv
    target_mv = raw if raw is not None else tgt[..., 1] / soma_scale + soma_bias_mv
    spikes = int((tgt[..., 0] > 0.5).sum())
    weights = iron_threshold_weights(iron_relevance(target_mv))
    area = float(np.sum((pred_mv - target_mv) ** 2 * weights))
    return {
        "voltage_sera_iron_mv2": area,
        "voltage_sera_iron_per_spike_mv2": area / spikes if spikes else None,
        "teacher_spikes": spikes,
        "relevance_mean": float(np.mean(weights)),
    }


def iron_dynamics_sera(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    timestep_ms: float = 1.0,
    ignore_start: int = 0,
) -> dict[str, Any]:
    """IRon-default SERA of the voltage rate of change, per teacher spike.

    Relevance is IRon's default ``phi`` on the teacher dV/dt; the summed
    relevance-area of squared rate errors is divided by the number of teacher
    spikes in the scored region.
    """
    if timestep_ms <= 0.0:
        raise ValueError("timestep_ms must be positive")
    pred, tgt, raw = _prepare_prediction_target(
        prediction, target, raw_soma_mv=raw_soma_mv, ignore_start=ignore_start
    )
    if pred.shape[1] < 2:
        raise ValueError("Dynamics SERA requires at least two scored timesteps")
    pred_mv = pred[..., 1] / soma_scale + soma_bias_mv
    target_mv = raw if raw is not None else tgt[..., 1] / soma_scale + soma_bias_mv
    spikes = int((tgt[..., 0] > 0.5).sum())
    pred_rate = np.diff(pred_mv, axis=1) / timestep_ms
    target_rate = np.diff(target_mv, axis=1) / timestep_ms
    weights = iron_threshold_weights(iron_relevance(target_rate))
    area = float(np.sum((pred_rate - target_rate) ** 2 * weights))
    return {
        "dynamics_sera_iron_mv2_per_ms2": area,
        "dynamics_sera_iron_per_spike_mv2_per_ms2": area / spikes if spikes else None,
        "teacher_spikes": spikes,
        "relevance_mean": float(np.mean(weights)),
    }


def sera_skill_trace_terms(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    reference_mv: float,
    raw_soma_mv: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    timestep_ms: float = 1.0,
    ignore_start: int = 0,
) -> dict[str, np.ndarray]:
    """Per-trace IRon SERA areas behind the SERA skill scores.

    Relevance is fit once on the pooled scored teacher, as in
    :func:`iron_voltage_sera` and :func:`iron_dynamics_sera`. For each trace
    the result holds the summed relevance area ``sum(e**2 * A(phi))`` of the
    voltage error of the prediction (``voltage``), of the constant reference
    `reference_mv` (``voltage_reference``), and of the teacher clipped at
    `SOMA_CLIP_MV` (``voltage_ceiling``), the dV/dt areas of the prediction
    (``dynamics``) and of the constant reference, whose rate is zero
    (``dynamics_reference``), and the teacher spike count. Sums over any
    subset of traces give that subset's skills (:func:`sera_skills`).
    """
    pred, tgt, raw = _prepare_prediction_target(
        prediction, target, raw_soma_mv=raw_soma_mv, ignore_start=ignore_start
    )
    if pred.shape[1] < 2:
        raise ValueError("SERA skill requires at least two scored timesteps")
    pred_mv = pred[..., 1] / soma_scale + soma_bias_mv
    target_mv = raw if raw is not None else tgt[..., 1] / soma_scale + soma_bias_mv
    weights = iron_threshold_weights(iron_relevance(target_mv))
    pred_rate = np.diff(pred_mv, axis=1) / timestep_ms
    target_rate = np.diff(target_mv, axis=1) / timestep_ms
    rate_weights = iron_threshold_weights(iron_relevance(target_rate))
    return {
        "voltage": np.sum((pred_mv - target_mv) ** 2 * weights, axis=1),
        "voltage_reference": np.sum((reference_mv - target_mv) ** 2 * weights, axis=1),
        "voltage_ceiling": np.sum(
            (np.minimum(target_mv, SOMA_CLIP_MV) - target_mv) ** 2 * weights, axis=1
        ),
        "dynamics": np.sum((pred_rate - target_rate) ** 2 * rate_weights, axis=1),
        "dynamics_reference": np.sum(target_rate**2 * rate_weights, axis=1),
        "teacher_spikes": (tgt[..., 0] > 0.5).sum(axis=1).astype(np.int64),
    }


def sera_skills(terms: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, float | None]:
    """SERA skill scores from (a subset of) :func:`sera_skill_trace_terms`.

    ``voltage_sera_skill = 1 - SERA / SERA(reference)``; SERA per spike shares
    its relevance and spike count with the reference, so the ratio is a ratio
    of summed areas. ``voltage_sera_ceiling_skill`` is the skill of the
    clipped teacher, the best prediction that stays at or below
    `SOMA_CLIP_MV` (no action potentials);
    ``voltage_sera_skill_of_ceiling`` is their ratio. ``dynamics_sera_skill``
    is the dV/dt analogue with the same constant reference.
    """
    sums = {key: float(np.sum(value if indices is None else value[indices])) for key, value in terms.items()}

    def skill(area: str, reference: str) -> float | None:
        return 1.0 - sums[area] / sums[reference] if sums[reference] > 0.0 else None

    voltage = skill("voltage", "voltage_reference")
    ceiling = skill("voltage_ceiling", "voltage_reference")
    return {
        "voltage_sera_skill": voltage,
        "voltage_sera_ceiling_skill": ceiling,
        "voltage_sera_skill_of_ceiling": (
            voltage / ceiling if voltage is not None and ceiling else None
        ),
        "dynamics_sera_skill": skill("dynamics", "dynamics_reference"),
    }


def ap_window_mask(spikes: np.ndarray, post_steps: np.ndarray | None = None) -> np.ndarray:
    """Steps from `AP_PRE_STEPS` before to `post_steps` after each spike (inclusive).

    `post_steps` holds one window end per trace (default `AP_POST_STEPS`).
    """
    spk = _as_2d(spikes) > 0.5
    ends = np.broadcast_to(
        np.asarray(AP_POST_STEPS if post_steps is None else post_steps, dtype=np.int64),
        (spk.shape[0],),
    )
    mask = np.zeros_like(spk)
    steps = spk.shape[1]
    for offset in range(-AP_PRE_STEPS, int(ends.max(initial=0)) + 1):
        if abs(offset) >= steps:
            continue
        rows = ends >= offset
        if offset >= 0:
            mask[rows, offset:] |= spk[rows, : steps - offset]
        else:
            mask[rows, :offset] |= spk[rows, -offset:]
    return mask


def ap_window_ends(
    raw_soma_mv: np.ndarray | None,
    teacher_spikes: np.ndarray,
    *,
    trace_ids: list[str] | None = None,
    timestep_ms: float = 1.0,
) -> dict[str, Any]:
    """Cell-type-specific end of the action-potential window, from the teacher.

    Traces are grouped by cell type, the trace-id prefix before the first
    ``:`` (traces without ids or without ``:`` form one group). A group's end
    is the first lag ``L >= 1`` at which its mean isolated teacher spike
    waveform (lags -6 to +30; no other teacher spike in the following 30
    steps) is flat, ``|V(L) - V(L - 1)| < AP_FLAT_SLOPE_MV_PER_MS * timestep_ms``. Groups
    with fewer than `AP_MIN_ISOLATED_SPIKES` isolated spikes, no flat lag, or
    no raw voltage fall back to `AP_POST_STEPS`. Returns the per-trace
    ``post_steps``, ``end_steps_by_cell_type``, and ``fallback_cell_types``.
    """
    spk = _as_2d(teacher_spikes) > 0.5
    traces, steps = spk.shape
    if trace_ids is not None and len(trace_ids) != traces:
        raise ValueError("trace_ids length must match trace count")
    cells = np.array(
        [
            str(i).split(":", 1)[0] if ":" in str(i) else "all"
            for i in (trace_ids if trace_ids is not None else ["all"] * traces)
        ]
    )
    lags = np.arange(-AP_WAVEFORM_PRE_STEPS, AP_ISOLATION_STEPS + 1)
    rows, times = np.nonzero(spk)
    inside = (times + lags[0] >= 0) & (times + lags[-1] < steps)
    rows, times = rows[inside], times[inside]
    count = np.concatenate([np.zeros((traces, 1)), np.cumsum(spk, axis=1)], axis=1)
    following = count[rows, times + AP_ISOLATION_STEPS + 1] - count[rows, times + 1]
    rows, times = rows[following == 0], times[following == 0]
    voltage = None if raw_soma_mv is None else _as_2d(raw_soma_mv)
    if voltage is not None and voltage.shape != spk.shape:
        raise ValueError(
            f"raw_soma_mv must match teacher spike shape, got {voltage.shape} and {spk.shape}"
        )
    ends: dict[str, int] = {}
    fallback: list[str] = []
    for cell in sorted(set(cells.tolist())):
        selected = cells[rows] == cell
        flat = []
        if voltage is not None and selected.sum() >= AP_MIN_ISOLATED_SPIKES:
            waveform = voltage[rows[selected, None], times[selected, None] + lags].mean(axis=0)
            slope = np.abs(np.diff(waveform))
            flat = lags[1:][(lags[1:] >= 1) & (slope < AP_FLAT_SLOPE_MV_PER_MS * timestep_ms)]
        if len(flat):
            ends[cell] = int(flat[0])
        else:
            ends[cell] = AP_POST_STEPS
            fallback.append(cell)
    return {
        "post_steps": np.array([ends[cell] for cell in cells], dtype=np.int64),
        "end_steps_by_cell_type": ends,
        "fallback_cell_types": fallback,
    }


def subthreshold_voltage_point_terms(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    predicted_spikes: np.ndarray,
    raw_soma_mv: np.ndarray | None = None,
    trace_ids: list[str] | None = None,
    ap_post_steps: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    timestep_ms: float = 1.0,
) -> dict[str, np.ndarray]:
    """Per-step terms of the subthreshold voltage error, on the scored steps.

    `prediction` and `target` cover the whole trace; `predicted_spikes` are the
    predicted spike events of the scored steps (``>= ignore_start``). Scored
    steps inside the action-potential window (:func:`ap_window_mask`) of a
    teacher or a predicted spike are excluded; the per-trace window ends
    `ap_post_steps` default to :func:`ap_window_ends` of the scored teacher
    steps of these traces. Both voltages are clipped from above at
    `SOMA_CLIP_MV`. The result holds the boolean ``keep`` mask of retained
    steps, the clipped ``error`` (prediction minus teacher) and the clipped
    teacher ``deviation`` from the null prediction (the trace's mean clipped
    teacher voltage over its retained steps); both are zero where not kept.
    """
    pred, tgt, raw = _prepare_prediction_target(
        prediction, target, raw_soma_mv=raw_soma_mv, ignore_start=0
    )
    if not 0 <= ignore_start < pred.shape[1]:
        raise ValueError("ignore_start leaves no timesteps to evaluate")
    predicted = _as_2d(predicted_spikes)
    if predicted.shape != (pred.shape[0], pred.shape[1] - ignore_start):
        raise ValueError(
            "predicted_spikes must cover the scored steps, got "
            f"{predicted.shape} for {(pred.shape[0], pred.shape[1] - ignore_start)}"
        )
    if ap_post_steps is None:
        ap_post_steps = ap_window_ends(
            None if raw is None else raw[:, ignore_start:],
            tgt[:, ignore_start:, 0],
            trace_ids=trace_ids,
            timestep_ms=timestep_ms,
        )["post_steps"]
    excluded = ap_window_mask(tgt[..., 0], ap_post_steps)[:, ignore_start:] | ap_window_mask(
        predicted, ap_post_steps
    )
    keep = ~excluded
    pred_mv = pred[:, ignore_start:, 1] / soma_scale + soma_bias_mv
    target_mv = (
        raw[:, ignore_start:]
        if raw is not None
        else tgt[:, ignore_start:, 1] / soma_scale + soma_bias_mv
    )
    pred_mv = np.minimum(pred_mv, SOMA_CLIP_MV)
    target_mv = np.minimum(target_mv, SOMA_CLIP_MV)
    null_mv = np.where(keep, target_mv, 0.0).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
    return {
        "keep": keep,
        "error": np.where(keep, pred_mv - target_mv, 0.0),
        "deviation": np.where(keep, target_mv - null_mv[:, None], 0.0),
    }


def subthreshold_voltage_trace_terms(
    prediction: np.ndarray,
    target: np.ndarray,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Per-trace terms of the subthreshold voltage error.

    Takes the arguments of :func:`subthreshold_voltage_point_terms`. For each
    trace the result holds the summed squared error, the same sum for the
    null prediction (the trace's mean clipped teacher voltage over its
    retained steps), and the retained step count.
    """
    points = subthreshold_voltage_point_terms(prediction, target, **kwargs)
    return {
        "squared_error": np.sum(points["error"] ** 2, axis=1),
        "null_squared_error": np.sum(points["deviation"] ** 2, axis=1),
        "retained_points": points["keep"].sum(axis=1).astype(np.int64),
    }


def subthreshold_voltage_error(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    predicted_spikes: np.ndarray,
    raw_soma_mv: np.ndarray | None = None,
    trace_ids: list[str] | None = None,
    ap_post_steps: np.ndarray | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    timestep_ms: float = 1.0,
) -> dict[str, Any]:
    """RMSE of the somatic voltage outside action-potential windows.

    Voltage SERA is dominated by spike-timing errors near action potentials,
    and spike F1 scores only event times. This metric covers what both miss:
    voltage offset, gain, and post-spike (afterhyperpolarization or
    afterdepolarization) errors. Steps from 2 steps before to the cell type's
    window end after any teacher or predicted spike are excluded, so
    spike-timing errors are not counted twice; the end is where the teacher's
    mean isolated spike waveform flattens (:func:`ap_window_ends`), because
    the repolarization and fast afterhyperpolarization last longer in some
    cells than in others. Both voltages are clipped at `SOMA_CLIP_MV`. The
    RMSE pools squared errors over the retained steps of all traces; skill is
    ``1 - RMSE / RMSE(null)`` for the per-trace mean clipped teacher voltage
    on the same steps. See :func:`subthreshold_voltage_trace_terms`.
    """
    terms = subthreshold_voltage_trace_terms(
        prediction,
        target,
        predicted_spikes=predicted_spikes,
        raw_soma_mv=raw_soma_mv,
        trace_ids=trace_ids,
        ap_post_steps=ap_post_steps,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
        ignore_start=ignore_start,
        timestep_ms=timestep_ms,
    )
    error = float(np.sum(terms["squared_error"]))
    null_error = float(np.sum(terms["null_squared_error"]))
    points = int(np.sum(terms["retained_points"]))
    return {
        "subthreshold_rmse_mv": float(np.sqrt(error / points)) if points else None,
        "subthreshold_skill": (
            float(1.0 - np.sqrt(error / null_error)) if null_error > 0.0 else None
        ),
        "null_subthreshold_rmse_mv": float(np.sqrt(null_error / points)) if points else None,
        "retained_points": points,
    }


def response_sera_trace_terms(
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
) -> dict[str, np.ndarray]:
    """Per-pair terms of the IRon-default Response SERA.

    The error is the predicted voltage change minus the teacher voltage change
    (intervention minus baseline, mV). Relevance is IRon's default ``phi`` fit
    once on the pooled teacher change of all supplied pairs. For each pair the
    result holds the summed relevance area ``sum(e**2 * A(phi))``, the same sum
    for the zero-response control (predicted change zero), and the teacher
    spike count of its baseline plus intervention traces. Ratios of sums of
    these terms give Response SERA per spike for any subset of pairs with the
    relevance held fixed.
    """
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
        raise ValueError(
            "baseline and intervention arrays must have matching prediction/target shapes"
        )
    base_pred_mv = base_pred[..., 1] / soma_scale + soma_bias_mv
    int_pred_mv = int_pred[..., 1] / soma_scale + soma_bias_mv
    base_tgt_mv = (
        base_raw if base_raw is not None else base_tgt[..., 1] / soma_scale + soma_bias_mv
    )
    int_tgt_mv = (
        int_raw if int_raw is not None else int_tgt[..., 1] / soma_scale + soma_bias_mv
    )
    teacher_delta = int_tgt_mv - base_tgt_mv
    error = (int_pred_mv - base_pred_mv) - teacher_delta
    weights = iron_threshold_weights(iron_relevance(teacher_delta))
    spikes = (base_tgt[..., 0] > 0.5).sum(axis=1) + (int_tgt[..., 0] > 0.5).sum(axis=1)
    return {
        "area": np.sum(error**2 * weights, axis=1),
        "zero_response_area": np.sum(teacher_delta**2 * weights, axis=1),
        "teacher_spikes": spikes.astype(np.int64),
        "error": error,
        "teacher_delta": teacher_delta,
        "weights": weights,
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
) -> dict[str, Any]:
    """IRon-default SERA of paired intervention responses, per teacher spike.

    See :func:`response_sera_trace_terms`. The value is the summed relevance
    area of the response error divided by the number of teacher spikes in the
    baseline plus intervention traces. The zero-response control scores a
    model that predicts no change, with the same relevance and spike count.
    """
    terms = response_sera_trace_terms(
        baseline_prediction,
        intervention_prediction,
        baseline_target,
        intervention_target,
        baseline_raw_soma_mv=baseline_raw_soma_mv,
        intervention_raw_soma_mv=intervention_raw_soma_mv,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
        ignore_start=ignore_start,
    )
    area = float(np.sum(terms["area"]))
    null_area = float(np.sum(terms["zero_response_area"]))
    spikes = int(np.sum(terms["teacher_spikes"]))
    error = terms["error"]
    return {
        "response_sera_iron_mv2": area,
        "response_sera_iron_per_spike_mv2": area / spikes if spikes else None,
        "zero_response_sera_iron_mv2": null_area,
        "zero_response_sera_iron_per_spike_mv2": null_area / spikes if spikes else None,
        "teacher_spikes": spikes,
        "response_rmse_mv": float(np.sqrt(np.mean(error**2))),
        "teacher_response_abs_mean_mv": float(np.mean(np.abs(terms["teacher_delta"]))),
        "relevance_mean": float(np.mean(terms["weights"])),
    }


EVENT_WINDOW_STEPS = 5
EVENT_QUANTILE = 0.995


def spike_window_mask(spikes: np.ndarray, window_steps: int = EVENT_WINDOW_STEPS) -> np.ndarray:
    """Steps within `window_steps` of a true spike, per trace."""
    spk = _as_2d(spikes) > 0.5
    mask = spk.copy()
    for k in range(1, min(window_steps, spk.shape[1] - 1) + 1):
        mask[:, :-k] |= spk[:, k:]
        mask[:, k:] |= spk[:, :-k]
    return mask


def _absolute_change(voltage_mv: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(voltage_mv, axis=1, prepend=voltage_mv[:, :1]))


def voltage_event_thresholds(
    voltage_mv: np.ndarray,
    spikes: np.ndarray,
    *,
    quantile: float = EVENT_QUANTILE,
    window_steps: int = EVENT_WINDOW_STEPS,
) -> tuple[float, float, float]:
    """(high, low, change) thresholds of one neuron from its unchanged test traces.

    They are the `quantile` and ``1 - quantile`` quantiles of the true voltage
    and the `quantile` quantile of its absolute change from the previous step,
    over the steps outside the spike windows. Pass the evaluated steps only
    (AxoBench drops the first 500 ms of each trace).
    """
    v = _as_2d(np.asarray(voltage_mv, dtype=np.float64))
    rest = ~spike_window_mask(spikes, window_steps)
    return (
        float(np.quantile(v[rest], quantile)),
        float(np.quantile(v[rest], 1.0 - quantile)),
        float(np.quantile(_absolute_change(v)[rest], quantile)),
    )


def voltage_event_steps(
    voltage_mv: np.ndarray,
    spikes: np.ndarray,
    thresholds: tuple[float, float, float] | None = None,
    *,
    window_steps: int = EVENT_WINDOW_STEPS,
) -> np.ndarray:
    """Event steps of the true output: in a spike window, or voltage at or
    beyond the high or low threshold, or absolute change at or above the change
    threshold. Without `thresholds`, the traces are taken to be the neuron's
    unchanged test traces and the thresholds are computed from them. The
    prediction plays no role.
    """
    v = _as_2d(np.asarray(voltage_mv, dtype=np.float64))
    high, low, fast = (
        voltage_event_thresholds(v, spikes, window_steps=window_steps)
        if thresholds is None
        else thresholds
    )
    return spike_window_mask(spikes, window_steps) | (v >= high) | (v <= low) | (_absolute_change(v) >= fast)


def event_and_subthreshold_error(
    predicted_mv: np.ndarray, target_mv: np.ndarray, event_steps: np.ndarray
) -> dict[str, float | None]:
    """Event error and subthreshold error in mV, pooled over all given steps.

    The event error is the RMSE of the predicted against the true unclipped
    voltage on the event steps, the subthreshold error the RMSE on all other
    steps, so ``rmse^2 = share * event^2 + (1 - share) * subthreshold^2`` with
    ``share`` the fraction of event steps.
    """
    squared = (
        _as_2d(np.asarray(predicted_mv, dtype=np.float64))
        - _as_2d(np.asarray(target_mv, dtype=np.float64))
    ) ** 2
    events = _as_2d(event_steps).astype(bool)
    if squared.shape != events.shape:
        raise ValueError(f"voltages and event_steps must match, got {squared.shape} and {events.shape}")
    return {
        "event_error_mv": float(np.sqrt(squared[events].mean())) if events.any() else None,
        "subthreshold_error_mv": float(np.sqrt(squared[~events].mean())) if (~events).any() else None,
        "event_share": float(events.mean()),
    }


def metric_set(
    spike_scores: np.ndarray,
    predicted_mv: np.ndarray,
    target_spikes: np.ndarray,
    target_mv: np.ndarray,
    *,
    spike_threshold: float,
    event_thresholds: tuple[float, float, float] | None = None,
    ignore_start: int = 500,
) -> dict[str, Any]:
    """The AxoBench metric set for traces of one neuron at 1 ms: Mean F1, event
    error and subthreshold error.

    `spike_threshold` is fitted on validation traces
    (:func:`event_count_matched_threshold`). `event_thresholds` are the
    neuron's thresholds from its unchanged test traces
    (:func:`voltage_event_thresholds`); leave them out only when the given
    traces are those traces. The first `ignore_start` steps are not evaluated.
    """
    scores = _as_2d(np.asarray(spike_scores, dtype=np.float64))[:, ignore_start:]
    spikes = _as_2d(target_spikes)[:, ignore_start:] > 0.5
    predicted = _as_2d(np.asarray(predicted_mv, dtype=np.float64))[:, ignore_start:]
    target = _as_2d(np.asarray(target_mv, dtype=np.float64))[:, ignore_start:]
    if event_thresholds is None:
        event_thresholds = voltage_event_thresholds(target, spikes)
    spike = spike_f1_curve(scores, spikes, threshold=float(spike_threshold))
    voltage = event_and_subthreshold_error(
        predicted, target, voltage_event_steps(target, spikes, event_thresholds)
    )
    return {
        "mean_f1": spike["mean_f1"],
        **voltage,
        "event_thresholds": tuple(float(t) for t in event_thresholds),
        "spike_f1_curve": spike["rows"],
    }


def compute_core_metric_set(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None = None,
    trace_ids: list[str] | None = None,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    spike_threshold: float | None = None,
    spike_threshold_policy: str = "prevalence",
    spike_tolerances_steps: tuple[int, ...] | list[int] = (0, 1, 2, 3, 4, 5),
    spike_refractory_steps: int | None = 2,
    timestep_ms: float = 1.0,
    intervention_prediction: np.ndarray | None = None,
    intervention_target: np.ndarray | None = None,
    intervention_raw_soma_mv: np.ndarray | None = None,
) -> dict[str, Any]:
    """Easy entrypoint for the recommended AxoBench core metrics.

    The core set is Voltage SERA, subthreshold voltage RMSE, spike F1, and
    (with an intervention pair) Response SERA. Dynamics SERA is returned as a
    secondary metric.
    """
    pred, tgt, raw = _prepare_prediction_target(
        prediction, target, raw_soma_mv=raw_soma_mv, ignore_start=ignore_start
    )
    voltage = iron_voltage_sera(
        pred,
        tgt,
        raw_soma_mv=raw,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
    )
    dynamics = iron_dynamics_sera(
        pred,
        tgt,
        raw_soma_mv=raw,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
        timestep_ms=timestep_ms,
    )
    target_spikes = tgt[..., 0] > 0.5
    if spike_threshold is None:
        if spike_threshold_policy == "prevalence":
            spike_threshold = event_count_matched_threshold(
                pred[..., 0], int(target_spikes.sum()), refractory_steps=spike_refractory_steps
            )
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
        refractory_steps=spike_refractory_steps,
    )
    spike["threshold_kind"] = threshold_kind
    subthreshold = subthreshold_voltage_error(
        prediction,
        target,
        predicted_spikes=spike_events(
            pred[..., 0], float(spike_threshold), refractory_steps=spike_refractory_steps
        ),
        raw_soma_mv=raw_soma_mv,
        trace_ids=trace_ids,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
        ignore_start=ignore_start,
        timestep_ms=timestep_ms,
    )
    result = {
        "voltage": voltage,
        "dynamics": dynamics,
        "subthreshold": subthreshold,
        "spike": spike,
        "voltage_sera_iron_per_spike_mv2": voltage["voltage_sera_iron_per_spike_mv2"],
        "subthreshold_rmse_mv": subthreshold["subthreshold_rmse_mv"],
        "subthreshold_skill": subthreshold["subthreshold_skill"],
        "dynamics_sera_iron_per_spike_mv2_per_ms2": dynamics[
            "dynamics_sera_iron_per_spike_mv2_per_ms2"
        ],
        "spike_mean_f1": spike["mean_f1"],
        "spike_f1_area": spike["f1_area"],
        "spike_threshold": float(spike_threshold),
        "spike_threshold_kind": threshold_kind,
    }
    if (intervention_prediction is None) != (intervention_target is None):
        raise ValueError(
            "intervention_prediction and intervention_target must be supplied together"
        )
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
        )
        result["response"] = response
        result["response_sera_iron_per_spike_mv2"] = response[
            "response_sera_iron_per_spike_mv2"
        ]
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
        raise ValueError(
            f"prediction and target shapes must match, got {pred.shape} and {tgt.shape}"
        )
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
            raise ValueError(
                f"raw_soma_mv must match trace/time shape after mask, got {raw.shape} and {pred.shape[:2]}"
            )
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


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))
