from __future__ import annotations

import numpy as np
import pytest

from axobench.benchmark.population_dynamics import (
    PopulationDynamicsConfig,
    analyze_population_perturbation,
    analyze_population_spikes,
    multistep_branching_ratio,
)


def test_multistep_branching_recovers_exponential_memory() -> None:
    rng = np.random.default_rng(7)
    values = np.empty(20_000)
    values[0] = 0.0
    for index in range(1, values.size):
        values[index] = 0.94 * values[index - 1] + rng.normal()

    result = multistep_branching_ratio(
        values,
        max_lag_steps=30,
    )

    assert result["branching_ratio"] == pytest.approx(0.94, abs=0.02)
    assert result["fit_r2"] > 0.95
    assert result["intrinsic_timescale_ms"] > 10.0


def test_population_analysis_reports_heterogeneity_and_subsampling() -> None:
    rng = np.random.default_rng(11)
    duration = 8_000
    population = 1_000
    rates_hz = np.exp(rng.normal(np.log(3.0), 0.9, size=population))
    spikes = rng.random((duration, population)) < rates_hz[None, :] / 1000.0
    inhibitory = np.zeros(population, dtype=bool)
    inhibitory[rng.choice(population, size=200, replace=False)] = True

    result = analyze_population_spikes(
        spikes,
        inhibitory=inhibitory,
        config=PopulationDynamicsConfig(
            correlation_sample_neurons=64,
            avalanche_subsamples=(32, 64),
        ),
    )

    assert result["firing"]["inhibitory_fraction"] == pytest.approx(0.2)
    assert result["firing"]["rate_gini"] > 0.2
    assert result["gates"]["eighty_twenty_composition"]
    assert len(result["avalanches"]) == 2
    assert result["avalanches"][0]["avalanches"] > 20
    assert "power_law_alpha" in result["avalanches"][0]["size_tail"]
    assert result["criticality_status"].startswith("not_established")


def test_population_analysis_rejects_nonbinary_values() -> None:
    with pytest.raises(ValueError, match="binary"):
        analyze_population_spikes(np.asarray([[0.0, 0.5], [1.0, 0.0]]))


def test_paired_perturbation_measures_bounded_recovery() -> None:
    reference = np.zeros((400, 100), dtype=bool)
    reference[:, ::20] = True
    perturbed = reference.copy()
    perturbed[20, 1:20] = True
    perturbed[21, 1:10] = True
    perturbed[22, 1:5] = True
    perturbed[23, 1:2] = True

    result = analyze_population_perturbation(
        reference,
        perturbed,
        intervention_step=20,
    )

    assert result["peak_disagreement_fraction"] == pytest.approx(0.19)
    assert result["recovery_to_one_over_e_ms"] == pytest.approx(2.0)
    assert result["microscopic_amplification"] == pytest.approx(19.0)
    assert result["exact_trajectory_recovery"]
    assert result["macroscopic_rate_stability"]
    assert result["bounded_amplification"]


def test_paired_perturbation_distinguishes_micro_and_macro_recovery() -> None:
    reference = np.zeros((1_000, 200), dtype=bool)
    reference[:, ::25] = True
    perturbed = reference.copy()
    perturbed[100:, 0] = False
    perturbed[100:, 1] = True

    result = analyze_population_perturbation(
        reference,
        perturbed,
        intervention_step=100,
    )

    assert result["peak_disagreement_fraction"] == pytest.approx(0.01)
    assert not result["exact_trajectory_recovery"]
    assert result["macroscopic_rate_stability"]
    assert result["bounded_amplification"]


def test_paired_perturbation_requires_equal_shapes() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        analyze_population_perturbation(
            np.zeros((10, 4), dtype=bool),
            np.zeros((10, 5), dtype=bool),
            intervention_step=2,
        )
