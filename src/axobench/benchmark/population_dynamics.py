"""Population-level diagnostics for connected AxoBench simulations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PopulationDynamicsConfig:
    """Fixed analysis settings for reproducible population comparisons."""

    step_ms: float = 1.0
    tail_fraction: float = 0.25
    correlation_bin_ms: int = 20
    correlation_sample_neurons: int = 256
    branching_max_lag_ms: int = 64
    avalanche_subsamples: tuple[int, ...] = (64, 128, 256, 512)
    seed: int = 20260727

    def __post_init__(self) -> None:
        if self.step_ms <= 0:
            raise ValueError("step_ms must be positive")
        if not 0.0 < self.tail_fraction <= 0.5:
            raise ValueError("tail_fraction must be in (0, 0.5]")
        if self.correlation_bin_ms <= 0:
            raise ValueError("correlation_bin_ms must be positive")
        if self.correlation_sample_neurons <= 1:
            raise ValueError("correlation sample must contain multiple neurons")
        if self.branching_max_lag_ms < 2:
            raise ValueError("branching horizon must be at least 2 ms")
        if any(size <= 1 for size in self.avalanche_subsamples):
            raise ValueError("avalanche subsamples must exceed one neuron")


def _as_spike_matrix(spikes: np.ndarray) -> np.ndarray:
    values = np.asarray(spikes)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("spikes must have shape (time, population)")
    if values.dtype != bool:
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("spikes must be binary")
        values = values.astype(bool)
    return values


def _lag_correlation(values: np.ndarray, lag: int) -> float:
    if lag >= values.size:
        return float("nan")
    left = values[:-lag].astype(np.float64)
    right = values[lag:].astype(np.float64)
    if left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.dot(ranks, values) / total - values.size - 1)
        / values.size
    )


def _binned_spikes(spikes: np.ndarray, bin_steps: int) -> np.ndarray:
    bins = spikes.shape[0] // bin_steps
    if bins < 2:
        raise ValueError("recording is too short for the requested bin size")
    trimmed = spikes[: bins * bin_steps]
    return trimmed.reshape(bins, bin_steps, spikes.shape[1]).sum(axis=1)


def _pairwise_count_correlations(
    spikes: np.ndarray,
    *,
    bin_steps: int,
    sample_neurons: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(spikes.sum(axis=0) > 1)
    selected_count = min(sample_neurons, active.size)
    if selected_count < 2:
        return {
            "sampled_neurons": selected_count,
            "mean": float("nan"),
            "mean_absolute": float("nan"),
            "p95_absolute": float("nan"),
        }
    selected = rng.choice(active, size=selected_count, replace=False)
    counts = _binned_spikes(spikes[:, selected], bin_steps)
    correlations = np.corrcoef(counts, rowvar=False)
    upper = correlations[np.triu_indices(selected_count, k=1)]
    upper = upper[np.isfinite(upper)]
    if not upper.size:
        return {
            "sampled_neurons": selected_count,
            "mean": float("nan"),
            "mean_absolute": float("nan"),
            "p95_absolute": float("nan"),
        }
    return {
        "sampled_neurons": selected_count,
        "mean": float(upper.mean()),
        "mean_absolute": float(np.abs(upper).mean()),
        "p95_absolute": float(np.quantile(np.abs(upper), 0.95)),
    }


def _isi_cv_distribution(
    spikes: np.ndarray,
    *,
    step_ms: float,
) -> dict[str, float | int]:
    values = []
    for neuron in range(spikes.shape[1]):
        times = np.flatnonzero(spikes[:, neuron])
        if times.size < 5:
            continue
        intervals = np.diff(times).astype(np.float64) * step_ms
        mean = float(intervals.mean())
        if mean > 0.0:
            values.append(float(intervals.std(ddof=1) / mean))
    if not values:
        return {
            "eligible_neurons": 0,
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    array = np.asarray(values)
    return {
        "eligible_neurons": int(array.size),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def multistep_branching_ratio(
    population_counts: np.ndarray,
    *,
    max_lag_steps: int,
    step_ms: float = 1.0,
) -> dict[str, object]:
    """Estimate branching by fitting r_k = b m^k across multiple lags."""

    counts = np.asarray(population_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size <= max_lag_steps + 2:
        raise ValueError("population counts are too short")
    variance = float(counts.var())
    if variance <= 0.0:
        return {
            "branching_ratio": float("nan"),
            "intrinsic_timescale_ms": float("nan"),
            "fit_r2": float("nan"),
            "lags_steps": [],
            "regression_coefficients": [],
        }
    centered = counts - counts.mean()
    lags = np.arange(1, max_lag_steps + 1)
    coefficients = np.asarray(
        [
            np.dot(centered[:-lag], centered[lag:])
            / centered[:-lag].size
            / variance
            for lag in lags
        ]
    )
    valid = np.isfinite(coefficients) & (coefficients > 0.0)
    if valid.sum() < 3:
        return {
            "branching_ratio": float("nan"),
            "intrinsic_timescale_ms": float("nan"),
            "fit_r2": float("nan"),
            "lags_steps": lags[valid].tolist(),
            "regression_coefficients": coefficients[valid].tolist(),
        }
    x = lags[valid].astype(np.float64)
    y = np.log(coefficients[valid])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.square(y - fitted).sum())
    total = float(np.square(y - y.mean()).sum())
    branching = float(np.exp(slope))
    timescale = (
        float(-step_ms / math.log(branching))
        if 0.0 < branching < 1.0
        else float("inf")
    )
    return {
        "branching_ratio": branching,
        "intrinsic_timescale_ms": timescale,
        "fit_r2": float(1.0 - residual / total) if total > 0.0 else 1.0,
        "lags_steps": lags[valid].tolist(),
        "regression_coefficients": coefficients[valid].tolist(),
    }


def _extract_avalanches(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(counts) > 0
    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    sizes = np.asarray(
        [counts[start:stop].sum() for start, stop in zip(starts, stops)],
        dtype=np.int64,
    )
    return sizes, (stops - starts).astype(np.int64)


def _tail_model_comparison(
    values: np.ndarray,
    *,
    minimum: int = 2,
) -> dict[str, float | int]:
    tail = np.asarray(values, dtype=np.float64)
    tail = tail[tail >= minimum]
    if tail.size < 20:
        return {
            "tail_events": int(tail.size),
            "power_law_alpha": float("nan"),
            "power_vs_exponential_log_likelihood_per_event": float("nan"),
        }
    alpha = 1.0 + tail.size / np.log(
        tail / (minimum - 0.5)
    ).sum()
    power_log_likelihood = (
        tail.size * math.log((alpha - 1.0) / (minimum - 0.5))
        - alpha * np.log(tail / (minimum - 0.5)).sum()
    )
    shifted = tail - minimum
    rate = 1.0 / max(float(shifted.mean()), 1e-12)
    exponential_log_likelihood = (
        tail.size * math.log(rate) - rate * shifted.sum()
    )
    return {
        "tail_events": int(tail.size),
        "power_law_alpha": float(alpha),
        "power_vs_exponential_log_likelihood_per_event": float(
            (power_log_likelihood - exponential_log_likelihood) / tail.size
        ),
    }


def _avalanche_panel(
    spikes: np.ndarray,
    sample_sizes: Iterable[int],
    *,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows = []
    for requested_size in sample_sizes:
        sample_size = min(int(requested_size), spikes.shape[1])
        selected = rng.choice(
            spikes.shape[1],
            size=sample_size,
            replace=False,
        )
        counts = spikes[:, selected].sum(axis=1)
        sizes, durations = _extract_avalanches(counts)
        rows.append(
            {
                "sample_neurons": sample_size,
                "avalanches": int(sizes.size),
                "mean_size": (
                    float(sizes.mean()) if sizes.size else float("nan")
                ),
                "maximum_size": int(sizes.max()) if sizes.size else 0,
                "mean_duration_ms": (
                    float(durations.mean())
                    if durations.size
                    else float("nan")
                ),
                "size_tail": _tail_model_comparison(sizes),
                "duration_tail": _tail_model_comparison(durations),
            }
        )
    return rows


def analyze_population_spikes(
    spikes: np.ndarray,
    *,
    inhibitory: np.ndarray | None = None,
    morphology_ids: np.ndarray | None = None,
    config: PopulationDynamicsConfig = PopulationDynamicsConfig(),
) -> dict[str, object]:
    """Measure firing, heterogeneity, synchrony, and criticality proxies."""

    spikes = _as_spike_matrix(spikes)
    duration_steps, population = spikes.shape
    if inhibitory is not None:
        inhibitory = np.asarray(inhibitory, dtype=bool)
        if inhibitory.shape != (population,):
            raise ValueError("inhibitory must match the population axis")
    if morphology_ids is not None:
        morphology_ids = np.asarray(morphology_ids)
        if morphology_ids.shape != (population,):
            raise ValueError("morphology_ids must match the population axis")

    rates = spikes.mean(axis=0) * 1000.0 / config.step_ms
    counts = spikes.sum(axis=1)
    tail_start = int(round(duration_steps * (1.0 - config.tail_fraction)))
    tail_rates = spikes[tail_start:].mean(axis=0) * 1000.0 / config.step_ms
    positive_rates = rates[rates > 0.0]
    log_rates = np.log(positive_rates) if positive_rates.size else np.asarray([])
    correlation_bin_steps = max(
        1,
        round(config.correlation_bin_ms / config.step_ms),
    )
    branching_max_lag = max(
        2,
        round(config.branching_max_lag_ms / config.step_ms),
    )

    firing = {
        "duration_ms": float(duration_steps * config.step_ms),
        "population_neurons": population,
        "overall_rate_hz": float(rates.mean()),
        "tail_rate_hz": float(tail_rates.mean()),
        "active_neuron_fraction": float((rates > 0.0).mean()),
        "rate_gini": _gini(rates),
        "rate_quantiles_hz": {
            str(quantile): float(np.quantile(rates, quantile))
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        "positive_log_rate_std": (
            float(log_rates.std()) if log_rates.size else float("nan")
        ),
    }
    if inhibitory is not None:
        firing.update(
            {
                "inhibitory_fraction": float(inhibitory.mean()),
                "excitatory_rate_hz": float(rates[~inhibitory].mean()),
                "inhibitory_rate_hz": float(rates[inhibitory].mean()),
            }
        )
    if morphology_ids is not None:
        firing["morphology_rate_hz"] = {
            str(morphology): float(rates[morphology_ids == morphology].mean())
            for morphology in np.unique(morphology_ids)
        }

    pairwise = _pairwise_count_correlations(
        spikes,
        bin_steps=correlation_bin_steps,
        sample_neurons=config.correlation_sample_neurons,
        seed=config.seed,
    )
    isi = _isi_cv_distribution(spikes, step_ms=config.step_ms)
    branching = multistep_branching_ratio(
        counts,
        max_lag_steps=min(branching_max_lag, duration_steps // 4),
        step_ms=config.step_ms,
    )
    count_mean = float(counts.mean())
    temporal = {
        "population_count_mean_per_step": count_mean,
        "population_count_cv": float(
            counts.std() / max(count_mean, 1e-12)
        ),
        "population_count_fano": float(
            counts.var() / max(count_mean, 1e-12)
        ),
        "count_lag_correlation": {
            str(lag): _lag_correlation(counts, lag)
            for lag in (1, 4, 16, 64, 256)
            if lag < duration_steps
        },
        "pairwise_20ms_count_correlation": pairwise,
        "isi_cv": isi,
    }
    stationarity_ratio = float(
        firing["tail_rate_hz"] / max(firing["overall_rate_hz"], 1e-12)
    )
    asynchronous = (
        math.isfinite(float(pairwise["mean_absolute"]))
        and float(pairwise["mean_absolute"]) <= 0.1
        and math.isfinite(float(isi["median"]))
        and 0.5 <= float(isi["median"]) <= 2.0
    )
    reverberating = (
        math.isfinite(float(branching["branching_ratio"]))
        and 0.8 <= float(branching["branching_ratio"]) < 1.0
        and float(branching["fit_r2"]) >= 0.8
    )
    gates = {
        "noncollapsed_activity": (
            0.5 <= float(firing["active_neuron_fraction"]) <= 1.0
            and 0.1 <= float(firing["overall_rate_hz"]) <= 30.0
        ),
        "stationary_rate": 0.8 <= stationarity_ratio <= 1.2,
        "heterogeneous_rates": (
            float(firing["rate_gini"]) >= 0.2
            and float(firing["positive_log_rate_std"]) >= 0.5
        ),
        "asynchronous_irregular": asynchronous,
        "reverberating": reverberating,
    }
    if inhibitory is not None:
        gates["eighty_twenty_composition"] = bool(
            np.isclose(float(inhibitory.mean()), 0.2, atol=0.005)
        )
    return {
        "firing": firing,
        "temporal": temporal,
        "branching": branching,
        "avalanches": _avalanche_panel(
            spikes,
            config.avalanche_subsamples,
            seed=config.seed + 1,
        ),
        "gates": gates,
        "passes_broad_population_gate": all(gates.values()),
        "criticality_status": (
            "not_established_without_paired_perturbation_and_controls"
        ),
    }


def analyze_population_perturbation(
    reference_spikes: np.ndarray,
    perturbed_spikes: np.ndarray,
    *,
    intervention_step: int,
    step_ms: float = 1.0,
    macroscopic_bin_ms: float = 100.0,
) -> dict[str, float | bool]:
    """Separate microscopic sensitivity from macroscopic rate stability."""

    reference = _as_spike_matrix(reference_spikes)
    perturbed = _as_spike_matrix(perturbed_spikes)
    if reference.shape != perturbed.shape:
        raise ValueError("paired spike matrices must have equal shape")
    if not 0 <= intervention_step < reference.shape[0] - 1:
        raise ValueError("intervention_step lies outside the recording")
    if macroscopic_bin_ms <= 0.0:
        raise ValueError("macroscopic_bin_ms must be positive")
    disagreement = np.not_equal(reference, perturbed).mean(axis=1)
    response = disagreement[intervention_step:]
    peak_index = int(np.argmax(response))
    peak = float(response[peak_index])
    final = float(response[-1])
    recovery_threshold = peak / math.e
    recovery_rows = np.flatnonzero(
        response[peak_index:] <= recovery_threshold
    )
    recovery_ms = (
        float(recovery_rows[0] * step_ms)
        if recovery_rows.size
        else float("inf")
    )
    early_steps = min(response.size, max(3, round(20.0 / step_ms)))
    early = np.maximum(response[:early_steps], 1.0 / reference.shape[1])
    growth_slope = float(
        np.polyfit(
            np.arange(early_steps, dtype=np.float64) * step_ms,
            np.log(early),
            1,
        )[0]
    )
    injected_fraction = 1.0 / reference.shape[1]
    microscopic_amplification = peak / injected_fraction
    exact_trajectory_recovery = (
        final <= max(peak * 0.2, injected_fraction)
        and math.isfinite(recovery_ms)
    )
    reference_counts = reference[intervention_step:].sum(axis=1)
    perturbed_counts = perturbed[intervention_step:].sum(axis=1)
    mean_reference_count = max(float(reference_counts.mean()), 1e-12)
    bin_steps = max(1, round(macroscopic_bin_ms / step_ms))
    bin_count = reference_counts.size // bin_steps
    if bin_count:
        reference_binned = reference_counts[: bin_count * bin_steps].reshape(
            bin_count,
            bin_steps,
        ).sum(axis=1)
        perturbed_binned = perturbed_counts[: bin_count * bin_steps].reshape(
            bin_count,
            bin_steps,
        ).sum(axis=1)
        binned_relative_deviation = (
            np.abs(perturbed_binned - reference_binned)
            / max(mean_reference_count * bin_steps, 1e-12)
        )
        maximum_binned_deviation = float(binned_relative_deviation.max())
        mean_binned_deviation = float(binned_relative_deviation.mean())
    else:
        maximum_binned_deviation = float("nan")
        mean_binned_deviation = float("nan")
    tail_steps = max(bin_steps, reference_counts.size // 4)
    reference_tail = float(reference_counts[-tail_steps:].mean())
    perturbed_tail = float(perturbed_counts[-tail_steps:].mean())
    tail_rate_change = abs(perturbed_tail - reference_tail) / max(
        reference_tail,
        1e-12,
    )
    overall_rate_change = abs(
        float(perturbed_counts.mean()) - mean_reference_count
    ) / mean_reference_count
    macroscopic_stability = (
        math.isfinite(maximum_binned_deviation)
        and maximum_binned_deviation <= 0.15
        and mean_binned_deviation <= 0.05
        and overall_rate_change <= 0.05
        and tail_rate_change <= 0.05
    )
    bounded = (
        microscopic_amplification > 1.0
        and peak < 0.25
        and macroscopic_stability
    )
    return {
        "injected_disagreement_fraction": injected_fraction,
        "peak_disagreement_fraction": peak,
        "microscopic_amplification": microscopic_amplification,
        "time_to_peak_ms": float(peak_index * step_ms),
        "response_area_fraction_ms": float(response.sum() * step_ms),
        "final_disagreement_fraction": final,
        "early_log_growth_per_ms": growth_slope,
        "recovery_to_one_over_e_ms": recovery_ms,
        "exact_trajectory_recovery": exact_trajectory_recovery,
        "macroscopic_bin_ms": float(bin_steps * step_ms),
        "maximum_binned_rate_deviation_fraction": maximum_binned_deviation,
        "mean_binned_rate_deviation_fraction": mean_binned_deviation,
        "overall_rate_change_fraction": overall_rate_change,
        "tail_rate_change_fraction": tail_rate_change,
        "macroscopic_rate_stability": macroscopic_stability,
        "bounded_amplification": bounded,
    }
