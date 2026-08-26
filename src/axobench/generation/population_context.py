"""Deterministic population spike routing into AxoBench signed inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PopulationContextRoute:
    """Fixed presynaptic identities and delays for target neurons."""

    source_indices: np.ndarray
    channel_indices: np.ndarray
    delays_ms: np.ndarray
    contact_is_local: np.ndarray
    source_is_inhibitory: np.ndarray
    target_indices: np.ndarray
    excitatory_channels: int
    inhibitory_channels: int
    source_regions: np.ndarray | None = None
    local_region_radius: int | None = None

    @property
    def input_channels(self) -> int:
        return self.excitatory_channels + self.inhibitory_channels

    @property
    def contacts_per_target(self) -> int:
        return self.source_indices.shape[1]

    def validate(self) -> None:
        if self.source_indices.ndim != 2:
            raise ValueError("source_indices must have shape (targets, contacts)")
        expected_shape = self.source_indices.shape
        if expected_shape[0] != self.target_indices.size:
            raise ValueError(
                "source_indices target axis does not match target_indices"
            )
        if self.channel_indices.shape != expected_shape:
            raise ValueError(
                "channel_indices must have shape "
                f"{expected_shape}, got {self.channel_indices.shape}"
            )
        if self.contact_is_local.shape != expected_shape:
            raise ValueError(
                "contact_is_local must have shape "
                f"{expected_shape}, got {self.contact_is_local.shape}"
            )
        if self.delays_ms.shape != expected_shape:
            raise ValueError(
                f"delays_ms must have shape {expected_shape}, "
                f"got {self.delays_ms.shape}"
            )
        if self.source_is_inhibitory.ndim != 1:
            raise ValueError("source_is_inhibitory must be one-dimensional")
        if np.any(self.source_indices < 0) or np.any(
            self.source_indices >= self.source_is_inhibitory.size
        ):
            raise ValueError("source index outside the declared population")
        if np.any(self.delays_ms < 1):
            raise ValueError("population routes must use causal positive delays")
        if np.any(self.source_indices == self.target_indices[:, None]):
            raise ValueError("population routes must not contain self-edges")
        if np.any(self.channel_indices < 0) or np.any(
            self.channel_indices >= self.input_channels
        ):
            raise ValueError("channel index outside the AxoBench input schema")
        excitatory_contacts = (
            self.channel_indices < self.excitatory_channels
        )
        source_types = self.source_is_inhibitory[self.source_indices]
        if np.any(source_types[excitatory_contacts]):
            raise ValueError("excitatory channels contain inhibitory sources")
        if np.any(~source_types[~excitatory_contacts]):
            raise ValueError("inhibitory channels contain excitatory sources")
        if self.source_regions is not None:
            if self.source_regions.shape != self.source_is_inhibitory.shape:
                raise ValueError(
                    "source_regions must match the source population"
                )
            if self.local_region_radius is None:
                raise ValueError(
                    "local_region_radius is required with source_regions"
                )
            region_count = int(self.source_regions.max()) + 1
            target_regions = self.source_regions[self.target_indices]
            source_regions = self.source_regions[self.source_indices]
            distance = np.abs(source_regions - target_regions[:, None])
            distance = np.minimum(distance, region_count - distance)
            if np.any(
                distance[self.contact_is_local]
                > self.local_region_radius
            ):
                raise ValueError("local contact leaves its regional radius")
            if np.any(
                distance[~self.contact_is_local]
                <= self.local_region_radius
            ):
                raise ValueError("long contact remains in the local region")


def balanced_cell_types(
    population_size: int,
    *,
    inhibitory_fraction: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Return a reproducible, randomly placed E/I population assignment."""

    if population_size < 2:
        raise ValueError("population_size must be at least two")
    if not 0.0 < inhibitory_fraction < 1.0:
        raise ValueError("inhibitory_fraction must be between zero and one")
    inhibitory_count = int(round(population_size * inhibitory_fraction))
    inhibitory_count = min(max(inhibitory_count, 1), population_size - 1)
    assignments = np.zeros(population_size, dtype=bool)
    assignments[:inhibitory_count] = True
    np.random.default_rng(seed).shuffle(assignments)
    return assignments


def build_population_context_route(
    source_is_inhibitory: np.ndarray,
    target_indices: Iterable[int],
    *,
    excitatory_channels: int = 639,
    inhibitory_channels: int = 639,
    excitatory_contacts_per_target: int | None = None,
    inhibitory_contacts_per_target: int | None = None,
    local_fraction: float = 0.9,
    local_delays_ms: tuple[int, ...] = (4, 5),
    long_delays_ms: tuple[int, ...] = (6, 7),
    source_regions: np.ndarray | None = None,
    local_region_radius: int = 4,
    seed: int = 0,
) -> PopulationContextRoute:
    """Build fixed E/I-preserving fan-in for biological teacher traces."""

    source_is_inhibitory = np.asarray(source_is_inhibitory, dtype=bool)
    targets = np.asarray(tuple(target_indices), dtype=np.int64)
    if source_is_inhibitory.ndim != 1:
        raise ValueError("source_is_inhibitory must be one-dimensional")
    if targets.ndim != 1 or not targets.size:
        raise ValueError("target_indices must contain at least one target")
    if np.any(targets < 0) or np.any(targets >= source_is_inhibitory.size):
        raise ValueError("target index outside the declared population")
    if excitatory_channels < 1 or inhibitory_channels < 1:
        raise ValueError("both E and I channel counts must be positive")
    excitatory_contacts = (
        excitatory_channels
        if excitatory_contacts_per_target is None
        else excitatory_contacts_per_target
    )
    inhibitory_contacts = (
        inhibitory_channels
        if inhibitory_contacts_per_target is None
        else inhibitory_contacts_per_target
    )
    if excitatory_contacts < excitatory_channels:
        raise ValueError(
            "excitatory contacts must cover every excitatory channel"
        )
    if inhibitory_contacts < inhibitory_channels:
        raise ValueError(
            "inhibitory contacts must cover every inhibitory channel"
        )
    if not 0.0 <= local_fraction <= 1.0:
        raise ValueError("local_fraction must be between zero and one")
    if not local_delays_ms or not long_delays_ms:
        raise ValueError("local and long delay sets must not be empty")
    if min((*local_delays_ms, *long_delays_ms)) < 1:
        raise ValueError("all delays must be positive")
    if source_regions is not None:
        source_regions = np.asarray(source_regions, dtype=np.int64)
        if source_regions.shape != source_is_inhibitory.shape:
            raise ValueError(
                "source_regions must match the source population"
            )
        if np.any(source_regions < 0):
            raise ValueError("source regions must be nonnegative")
        unique_regions = np.unique(source_regions)
        if not np.array_equal(
            unique_regions,
            np.arange(unique_regions.size),
        ):
            raise ValueError("source regions must be contiguous from zero")
        if local_region_radius < 0:
            raise ValueError("local_region_radius must be nonnegative")

    excitatory_pool = np.flatnonzero(~source_is_inhibitory)
    inhibitory_pool = np.flatnonzero(source_is_inhibitory)
    if not excitatory_pool.size or not inhibitory_pool.size:
        raise ValueError("source population must contain both cell types")

    rng = np.random.default_rng(seed)
    input_channels = excitatory_channels + inhibitory_channels
    total_contacts = excitatory_contacts + inhibitory_contacts
    source_indices = np.empty((targets.size, total_contacts), dtype=np.int64)
    channel_indices = np.empty_like(source_indices)
    delays = np.empty_like(source_indices, dtype=np.int16)
    contact_is_local = np.empty_like(source_indices, dtype=bool)
    local_excitatory = int(round(excitatory_contacts * local_fraction))
    local_inhibitory = int(round(inhibitory_contacts * local_fraction))

    for row, target in enumerate(targets):
        excitatory_sources, excitatory_local = _sample_typed_contacts(
            rng=rng,
            pool=excitatory_pool,
            target=int(target),
            contacts=excitatory_contacts,
            local_contacts=local_excitatory,
            source_regions=source_regions,
            local_region_radius=local_region_radius,
        )
        inhibitory_sources, inhibitory_local = _sample_typed_contacts(
            rng=rng,
            pool=inhibitory_pool,
            target=int(target),
            contacts=inhibitory_contacts,
            local_contacts=local_inhibitory,
            source_regions=source_regions,
            local_region_radius=local_region_radius,
        )
        source_indices[row, :excitatory_contacts] = excitatory_sources
        source_indices[row, excitatory_contacts:] = inhibitory_sources
        contact_is_local[row, :excitatory_contacts] = excitatory_local
        contact_is_local[row, excitatory_contacts:] = inhibitory_local
        channel_indices[row, :excitatory_contacts] = _contact_channels(
            rng,
            excitatory_channels,
            excitatory_contacts,
        )
        channel_indices[row, excitatory_contacts:] = (
            _contact_channels(
                rng,
                inhibitory_channels,
                inhibitory_contacts,
            )
            + excitatory_channels
        )
        delays[row, contact_is_local[row]] = rng.choice(
            local_delays_ms,
            size=int(contact_is_local[row].sum()),
        )
        delays[row, ~contact_is_local[row]] = rng.choice(
            long_delays_ms,
            size=int((~contact_is_local[row]).sum()),
        )

    route = PopulationContextRoute(
        source_indices=source_indices,
        channel_indices=channel_indices,
        delays_ms=delays,
        contact_is_local=contact_is_local,
        source_is_inhibitory=source_is_inhibitory.copy(),
        target_indices=targets,
        excitatory_channels=excitatory_channels,
        inhibitory_channels=inhibitory_channels,
        source_regions=(
            None if source_regions is None else source_regions.copy()
        ),
        local_region_radius=(
            None if source_regions is None else local_region_radius
        ),
    )
    route.validate()
    return route


def route_population_spikes(
    source_spikes: np.ndarray,
    route: PopulationContextRoute,
) -> np.ndarray:
    """Materialize delayed source spikes in the existing signed-input schema."""

    route.validate()
    source_spikes = np.asarray(source_spikes, dtype=bool)
    if source_spikes.ndim != 2:
        raise ValueError("source_spikes must have shape (time, population)")
    if source_spikes.shape[1] != route.source_is_inhibitory.size:
        raise ValueError(
            "source_spikes population axis does not match the route"
        )

    duration_ms = source_spikes.shape[0]
    excitatory_events = np.zeros(
        (route.target_indices.size, duration_ms, route.input_channels),
        dtype=bool,
    )
    for target_row in range(route.target_indices.size):
        for delay in np.unique(route.delays_ms[target_row]):
            if delay >= duration_ms:
                continue
            contact_mask = route.delays_ms[target_row] == delay
            sources = route.source_indices[target_row, contact_mask]
            channels = route.channel_indices[target_row, contact_mask]
            gathered = source_spikes[: duration_ms - delay, sources]
            times = np.arange(delay, duration_ms)[:, None]
            np.logical_or.at(
                excitatory_events[target_row],
                (times, channels[None, :]),
                gathered,
            )
    signed = excitatory_events.astype(np.int8)
    signed[:, :, route.excitatory_channels :] *= -1
    return signed


def summarize_population_context(
    source_spikes: np.ndarray,
    route: PopulationContextRoute,
) -> dict[str, float | int]:
    """Return the fixed statistical contract attached to a routed context."""

    route.validate()
    source_spikes = np.asarray(source_spikes, dtype=bool)
    inhibitory = route.source_is_inhibitory
    return {
        "population_size": int(inhibitory.size),
        "inhibitory_neurons": int(inhibitory.sum()),
        "inhibitory_fraction": float(inhibitory.mean()),
        "excitatory_source_rate_hz": float(
            source_spikes[:, ~inhibitory].mean() * 1000.0
        ),
        "inhibitory_source_rate_hz": float(
            source_spikes[:, inhibitory].mean() * 1000.0
        ),
        "targets": int(route.target_indices.size),
        "excitatory_channels_per_target": int(route.excitatory_channels),
        "inhibitory_channels_per_target": int(route.inhibitory_channels),
        "contacts_per_target": int(route.contacts_per_target),
        "excitatory_contacts_per_target": int(
            (
                route.channel_indices < route.excitatory_channels
            ).sum(axis=1)[0]
        ),
        "inhibitory_contacts_per_target": int(
            (
                route.channel_indices >= route.excitatory_channels
            ).sum(axis=1)[0]
        ),
        "mean_delay_ms": float(route.delays_ms.mean()),
        "long_range_fraction": float((~route.contact_is_local).mean()),
        "regions": (
            0
            if route.source_regions is None
            else int(route.source_regions.max()) + 1
        ),
    }


def _sample_sources(
    rng: np.random.Generator,
    pool: np.ndarray,
    target: int,
    count: int,
) -> np.ndarray:
    eligible = pool[pool != target]
    if not eligible.size:
        raise ValueError("cannot exclude target from its only source pool")
    return rng.choice(
        eligible,
        size=count,
        replace=eligible.size < count,
    )


def _sample_typed_contacts(
    *,
    rng: np.random.Generator,
    pool: np.ndarray,
    target: int,
    contacts: int,
    local_contacts: int,
    source_regions: np.ndarray | None,
    local_region_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    if source_regions is None:
        sources = _sample_sources(rng, pool, target, contacts)
        local = np.zeros(contacts, dtype=bool)
        local[:local_contacts] = True
        rng.shuffle(local)
        return sources, local

    region_count = int(source_regions.max()) + 1
    target_region = int(source_regions[target])
    distance = np.abs(source_regions[pool] - target_region)
    distance = np.minimum(distance, region_count - distance)
    local_pool = pool[distance <= local_region_radius]
    long_pool = pool[distance > local_region_radius]
    local_sources = _sample_sources(
        rng,
        local_pool,
        target,
        local_contacts,
    )
    long_sources = _sample_sources(
        rng,
        long_pool,
        target,
        contacts - local_contacts,
    )
    sources = np.concatenate((local_sources, long_sources))
    local = np.zeros(contacts, dtype=bool)
    local[:local_contacts] = True
    order = rng.permutation(contacts)
    return sources[order], local[order]


def _contact_channels(
    rng: np.random.Generator,
    channels: int,
    contacts: int,
) -> np.ndarray:
    assignments = np.arange(contacts, dtype=np.int64) % channels
    rng.shuffle(assignments)
    return assignments
