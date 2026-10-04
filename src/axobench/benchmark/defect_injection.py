"""Controlled defects injected into teacher traces, for validating metrics.

A metric that measures simulator fidelity should fall monotonically as a known
defect grows. `inject_defect` corrupts a teacher's spike train and soma voltage
with one defect at a given dose, and `defect_dose_rows` scores the corrupted
traces against the teacher with the historical NeuronIO scores (timestep AUC,
clipped and rescaled RMSE) and the AxoBench scores (event F1, Mean F1, the
coincidence factor, Voltage SERA, Dynamics SERA).

Spike-timing defects move the action-potential waveform together with the
spike, so the voltage stays consistent with the spike train. Voltage defects
leave the spike train exact.
"""

from __future__ import annotations

import zlib
from typing import Any, Iterable, Mapping

import numpy as np

from axobench.metrics import (
    binary_auc,
    coincidence_factor,
    dynamics_sera,
    spike_f1_curve,
    voltage_sera,
)

AP_PRE_STEPS = 2
AP_POST_STEPS = 4
SOMA_CLIP_MV = -55.0

DEFECTS: dict[str, str] = {
    "spike_jitter": "Gaussian spike-time jitter, standard deviation in steps",
    "spike_deletion": "fraction of spikes deleted",
    "spike_insertion": "spurious spikes added, as a fraction of the spike count",
    "global_lag": "whole output delayed by this many steps",
    "ap_clip": "fraction of the voltage range above -55 mV removed by clipping (1 = the NeuronIO target)",
    "ap_smoothing": "Gaussian smoothing of the action-potential window, sigma in steps",
    "voltage_lowpass": "Gaussian low-pass of the whole voltage trace, sigma in steps",
    "voltage_offset": "constant voltage offset in mV",
    "voltage_gain": "fractional gain error around each trace's mean",
    "post_spike": "fraction by which the 50 steps after each action potential are replaced by a straight line",
}

DEFAULT_DOSES: dict[str, tuple[float, ...]] = {
    "spike_jitter": (0, 0.5, 1, 2, 4, 8),
    "spike_deletion": (0, 0.1, 0.25, 0.5, 0.75, 1.0),
    "spike_insertion": (0, 0.1, 0.25, 0.5, 1.0, 2.0),
    "global_lag": (0, 1, 2, 3, 5, 10),
    "ap_clip": (0, 0.25, 0.5, 0.75, 0.9, 1.0),
    "ap_smoothing": (0, 0.5, 1, 2, 4),
    "voltage_lowpass": (0, 0.5, 1, 2, 4, 8),
    "voltage_offset": (0, 0.5, 1, 2, 4, 8),
    "voltage_gain": (0, 0.05, 0.1, 0.25, 0.5),
    "post_spike": (0, 0.25, 0.5, 0.75, 1.0),
}


def inject_defect(
    spikes: np.ndarray,
    soma_mv: np.ndarray,
    defect: str,
    dose: float,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (spikes, soma_mv) of a teacher corrupted by one defect at `dose`.

    `spikes` is a binary (traces, time) array and `soma_mv` the matching raw
    soma voltage. Dose 0 returns copies of the inputs.
    """
    if defect not in DEFECTS:
        raise ValueError(f"unknown defect {defect!r}; expected one of {sorted(DEFECTS)}")
    rng = np.random.default_rng() if rng is None else rng
    spk = (np.atleast_2d(np.asarray(spikes)) > 0.5).astype(np.float32)
    v = np.atleast_2d(np.asarray(soma_mv, dtype=np.float64)).copy()
    if spk.shape != v.shape:
        raise ValueError(f"spikes and soma_mv must match, got {spk.shape} and {v.shape}")
    if dose == 0:
        return spk.copy(), v

    if defect in ("spike_jitter", "spike_deletion", "spike_insertion"):
        return _move_spikes(spk, v, defect, float(dose), rng)
    if defect == "global_lag":
        k = int(round(dose))
        out_s = np.zeros_like(spk)
        out_s[:, k:] = spk[:, : spk.shape[1] - k]
        out_v = np.concatenate([np.repeat(v[:, :1], k, axis=1), v[:, : v.shape[1] - k]], axis=1)
        return out_s, out_v
    if defect == "ap_clip":
        level = SOMA_CLIP_MV + (1.0 - float(dose)) * max(float(v.max()) - SOMA_CLIP_MV, 0.0)
        return spk.copy(), np.minimum(v, level)
    if defect == "ap_smoothing":
        window = _ap_window_mask(spk)
        smooth = _gaussian_filter(v, float(dose))
        return spk.copy(), np.where(window, smooth, v)
    if defect == "voltage_lowpass":
        return spk.copy(), _gaussian_filter(v, float(dose))
    if defect == "voltage_offset":
        return spk.copy(), v + float(dose)
    if defect == "voltage_gain":
        mean = v.mean(axis=1, keepdims=True)
        return spk.copy(), mean + (1.0 - float(dose)) * (v - mean)
    if defect == "post_spike":
        return spk.copy(), _flatten_post_spike(v, spk, float(dose))
    raise AssertionError(defect)


def smoothed_spike_score(spikes: np.ndarray, sigma_steps: float = 1.0) -> np.ndarray:
    """Spike train smoothed with a Gaussian, used as a ranking score for AUC."""
    return _gaussian_filter(np.atleast_2d(np.asarray(spikes, dtype=np.float64)), sigma_steps)


def historical_scores(
    pred_spikes: np.ndarray,
    pred_soma_mv: np.ndarray,
    target_spikes: np.ndarray,
    target_soma_mv: np.ndarray,
) -> dict[str, float]:
    """NeuronIO-protocol scores: timestep AUC and clipped RMSE, aligned and unaligned.

    Both voltages are clipped at -55 mV, as for a model trained on the clipped
    target; the aligned RMSE rescales the prediction to the mean and standard
    deviation of the clipped target, as both public pipelines do.
    """
    target_clip = np.minimum(target_soma_mv, SOMA_CLIP_MV)
    pred_clip = np.minimum(pred_soma_mv, SOMA_CLIP_MV)
    std = pred_clip.std()
    aligned = (pred_clip - pred_clip.mean()) / std * target_clip.std() + target_clip.mean() if std > 0 else pred_clip
    return {
        "auc": binary_auc(smoothed_spike_score(pred_spikes), target_spikes),
        "clipped_rmse_aligned_mv": float(np.sqrt(np.mean((aligned - target_clip) ** 2))),
        "clipped_rmse_mv": float(np.sqrt(np.mean((pred_clip - target_clip) ** 2))),
    }


def axobench_scores(
    pred_spikes: np.ndarray,
    pred_soma_mv: np.ndarray,
    target_spikes: np.ndarray,
    target_soma_mv: np.ndarray,
    *,
    relevance_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """AxoBench scores of a prediction against raw teacher voltage and spikes.

    Voltage and Dynamics skill are 1 - SERA / SERA of the per-trace mean of
    the teacher voltage.
    """
    bias, scale = -67.7, 0.1
    stack = lambda s, x: np.stack([s, (x - bias) * scale], axis=-1)  # noqa: E731
    target = stack(target_spikes, np.minimum(target_soma_mv, SOMA_CLIP_MV))
    null_v = np.repeat(target_soma_mv.mean(axis=1, keepdims=True), target_soma_mv.shape[1], axis=1)
    pred = stack(pred_spikes, pred_soma_mv)
    null = stack(np.zeros_like(pred_spikes), null_v)
    kw = dict(relevance_kwargs or {})
    out: dict[str, Any] = {}
    for name, fn, key in (
        ("voltage", voltage_sera, "voltage_sera_mv2"),
        ("dynamics", dynamics_sera, "dynamics_sera_mv2_per_ms2"),
    ):
        model = fn(pred, target, raw_soma_mv=target_soma_mv, relevance_kwargs=kw)[key]
        reference = fn(null, target, raw_soma_mv=target_soma_mv, relevance_kwargs=kw)[key]
        out[f"{name}_sera"] = float(model)
        out[f"{name}_skill"] = float(1.0 - model / reference)
    curve = spike_f1_curve(pred_spikes, target_spikes, threshold=0.5, tolerances_steps=(0, 1, 2, 3, 4, 5), refractory_steps=None)
    out["f1_curve"] = [0.0 if r["f1"] is None else float(r["f1"]) for r in curve["rows"]]
    out["mean_f1"] = curve["mean_f1"]
    out["gamma_2"] = coincidence_factor(pred_spikes, target_spikes, delta_steps=2)["gamma"]
    return out


def defect_dose_rows(
    spikes: np.ndarray,
    soma_mv: np.ndarray,
    doses: Mapping[str, Iterable[float]] | None = None,
    *,
    seed: int = 0,
    ignore_start: int = 0,
    relevance_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Score every defect at every dose against the uncorrupted teacher."""
    doses = DEFAULT_DOSES if doses is None else doses
    spk = (np.atleast_2d(np.asarray(spikes)) > 0.5).astype(np.float32)
    v = np.atleast_2d(np.asarray(soma_mv, dtype=np.float64))
    t_spk, t_v = spk[:, ignore_start:], v[:, ignore_start:]
    rows = []
    for defect, values in doses.items():
        for dose in values:
            rng = np.random.default_rng([seed, zlib.crc32(defect.encode())])
            p_spk, p_v = inject_defect(spk, v, defect, dose, rng=rng)
            p_spk, p_v = p_spk[:, ignore_start:], p_v[:, ignore_start:]
            rows.append(
                {
                    "defect": defect,
                    "dose": float(dose),
                    "predicted_spikes": int(p_spk.sum()),
                    "target_spikes": int(t_spk.sum()),
                    **historical_scores(p_spk, p_v, t_spk, t_v),
                    **axobench_scores(p_spk, p_v, t_spk, t_v, relevance_kwargs=relevance_kwargs),
                }
            )
    return rows


def _ap_window_mask(spikes: np.ndarray) -> np.ndarray:
    mask = np.zeros(spikes.shape, dtype=bool)
    for i, t in zip(*np.nonzero(spikes > 0.5)):
        mask[i, max(0, t - AP_PRE_STEPS) : t + AP_POST_STEPS + 1] = True
    return mask


def _cut_ap(trace: np.ndarray, t: int) -> tuple[int, int, np.ndarray]:
    """Replace the AP window around t by a straight line; return the removed deviation."""
    lo, hi = max(0, t - AP_PRE_STEPS), min(trace.size - 1, t + AP_POST_STEPS)
    line = np.linspace(trace[lo], trace[hi], hi - lo + 1)
    waveform = trace[lo : hi + 1] - line
    trace[lo : hi + 1] = line
    return lo - t, hi - t, waveform


def _paste_ap(trace: np.ndarray, t: int, start: int, waveform: np.ndarray) -> None:
    lo = t + start
    for k, value in enumerate(waveform):
        if 0 <= lo + k < trace.size:
            trace[lo + k] += value


def _move_spikes(spk, v, defect, dose, rng):
    out_s = np.zeros_like(spk)
    out_v = v.copy()
    pool = []
    for i in range(spk.shape[0]):
        for t in np.flatnonzero(spk[i]):
            pool.append(_cut_ap(v[i].copy(), int(t))[::2])
    for i in range(spk.shape[0]):
        times = np.flatnonzero(spk[i])
        cut = []
        for t in times[::-1]:
            cut.append((int(t), *_cut_ap(out_v[i], int(t))))
        cut = cut[::-1]
        new = []
        for t, start, _, waveform in cut:
            if defect == "spike_deletion" and rng.random() < dose:
                continue
            if defect == "spike_jitter":
                t = int(np.clip(t + np.rint(rng.normal(0.0, dose)), 0, spk.shape[1] - 1))
            new.append((t, start, waveform))
        if defect == "spike_insertion" and pool:
            n_extra = int(round(dose * times.size))
            for t in rng.integers(0, spk.shape[1], size=n_extra):
                start, waveform = pool[rng.integers(len(pool))]
                new.append((int(t), start, waveform))
        for t, start, waveform in new:
            out_s[i, t] = 1.0
            _paste_ap(out_v[i], t, start, waveform)
    return out_s, out_v


def _flatten_post_spike(v, spk, dose, span: int = 50):
    out = v.copy()
    for i in range(v.shape[0]):
        for t in np.flatnonzero(spk[i]):
            lo, hi = int(t) + AP_POST_STEPS, min(v.shape[1] - 1, int(t) + span)
            if hi <= lo:
                continue
            line = np.linspace(v[i, lo], v[i, hi], hi - lo + 1)
            out[i, lo : hi + 1] = (1.0 - dose) * out[i, lo : hi + 1] + dose * line
    return out


def _gaussian_filter(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float64, copy=True)
    radius = int(np.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x.astype(np.float64), ((0, 0), (radius, radius)), mode="edge")
    out = np.empty(x.shape, dtype=np.float64)
    for i in range(x.shape[0]):
        out[i] = np.convolve(padded[i], k, mode="valid")
    return out
