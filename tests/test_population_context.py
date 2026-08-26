import numpy as np
import pytest

from axobench.generation.population_context import (
    PopulationContextRoute,
    balanced_cell_types,
    build_population_context_route,
    route_population_spikes,
    summarize_population_context,
)


def test_balanced_cell_types_preserves_exact_eighty_twenty_split():
    cell_types = balanced_cell_types(1000, seed=17)

    assert cell_types.dtype == np.bool_
    assert int(cell_types.sum()) == 200
    assert np.array_equal(cell_types, balanced_cell_types(1000, seed=17))
    assert not np.array_equal(cell_types, balanced_cell_types(1000, seed=18))


def test_population_route_preserves_source_type_and_delay_contract():
    cell_types = balanced_cell_types(1000, seed=4)
    route = build_population_context_route(
        cell_types,
        [3, 17, 81],
        excitatory_channels=12,
        inhibitory_channels=5,
        excitatory_contacts_per_target=24,
        inhibitory_contacts_per_target=10,
        local_fraction=0.8,
        seed=9,
    )

    assert route.source_indices.shape == (3, 34)
    source_types = cell_types[route.source_indices]
    excitatory = route.channel_indices < 12
    assert not source_types[excitatory].any()
    assert source_types[~excitatory].all()
    assert np.all(np.isin(route.delays_ms, (4, 5, 6, 7)))
    assert not np.any(route.source_indices == route.target_indices[:, None])
    for row in route.channel_indices:
        assert set(row[:24]) == set(range(12))
        assert set(row[24:]) == set(range(12, 17))
    route.validate()


def test_population_spikes_are_delayed_without_coarse_graining():
    cell_types = np.array([False, False, True, True])
    route = PopulationContextRoute(
        source_indices=np.array([[0, 1, 2]], dtype=np.int64),
        channel_indices=np.array([[0, 0, 1]], dtype=np.int64),
        delays_ms=np.array([[2, 2, 3]], dtype=np.int16),
        contact_is_local=np.array([[True, True, False]]),
        source_is_inhibitory=cell_types,
        target_indices=np.array([3], dtype=np.int64),
        excitatory_channels=1,
        inhibitory_channels=1,
    )
    spikes = np.zeros((8, 4), dtype=bool)
    spikes[1, 0] = True
    spikes[1, 1] = True
    spikes[2, 2] = True

    signed = route_population_spikes(spikes, route)

    assert signed.shape == (1, 8, 2)
    assert signed[0, 3, 0] == 1
    assert signed[0, 5, 1] == -1
    assert np.count_nonzero(signed) == 2


def test_population_route_is_reproducible_and_summarized():
    cell_types = balanced_cell_types(100, seed=5)
    first = build_population_context_route(
        cell_types,
        [0, 1],
        excitatory_channels=8,
        inhibitory_channels=4,
        seed=12,
    )
    second = build_population_context_route(
        cell_types,
        [0, 1],
        excitatory_channels=8,
        inhibitory_channels=4,
        seed=12,
    )
    spikes = np.zeros((20, 100), dtype=bool)
    spikes[::4, ~cell_types] = True
    spikes[::2, cell_types] = True

    assert np.array_equal(first.source_indices, second.source_indices)
    assert np.array_equal(first.delays_ms, second.delays_ms)
    summary = summarize_population_context(spikes, first)
    assert summary["inhibitory_fraction"] == pytest.approx(0.2)
    assert summary["excitatory_source_rate_hz"] == pytest.approx(250.0)
    assert summary["inhibitory_source_rate_hz"] == pytest.approx(500.0)


def test_population_route_respects_local_regions_and_long_tail():
    cell_types = balanced_cell_types(1000, seed=7)
    regions = np.arange(1000) % 20
    route = build_population_context_route(
        cell_types,
        [21, 522],
        excitatory_channels=8,
        inhibitory_channels=4,
        excitatory_contacts_per_target=80,
        inhibitory_contacts_per_target=20,
        source_regions=regions,
        local_region_radius=2,
        local_fraction=0.9,
        seed=19,
    )

    target_regions = regions[route.target_indices]
    source_regions = regions[route.source_indices]
    distance = np.abs(source_regions - target_regions[:, None])
    distance = np.minimum(distance, 20 - distance)
    assert np.all(distance[route.contact_is_local] <= 2)
    assert np.all(distance[~route.contact_is_local] > 2)
    assert route.contact_is_local.mean() == pytest.approx(0.9)
    route.validate()
