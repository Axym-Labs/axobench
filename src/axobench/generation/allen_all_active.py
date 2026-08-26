"""Run Allen Cell Types all-active neurons on AxoBench signed inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from axobench.generation.generate_hay_neuron_dataset import (
    SOMA_BIAS_MV,
    SOMA_CLIP_THRESHOLD_MV,
    SOMA_SCALE,
    local_peak_spike_labels,
)
from axobench.generation.run_coreneuron_hay_probe import (
    load_probe_mechanisms,
)


@dataclass(frozen=True)
class AllenAllActiveCell:
    """Global-section Allen cell loaded in an isolated NEURON process."""

    soma: tuple[Any, ...]
    axon: tuple[Any, ...]
    dend: tuple[Any, ...]
    apic: tuple[Any, ...]
    all_sections: tuple[Any, ...]
    model_root: Path
    specimen_id: int
    model_id: int
    v_init_mv: float
    celsius: float


def load_allen_all_active_cell(
    h: Any,
    model_root: str | Path,
    *,
    mechanism_root: str | Path | None = None,
    axon_type: str = "truncated",
) -> AllenAllActiveCell:
    """Load one official Allen all-active model into a fresh process."""
    root = Path(model_root).resolve()
    mechanisms = (
        Path(mechanism_root).resolve()
        if mechanism_root is not None
        else root
    )
    parameters = json.loads(
        (root / "fit_parameters.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (root / "model_metadata.json").read_text(encoding="utf-8")
    )

    load_probe_mechanisms(h, mechanisms)
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    reader = h.Import3d_SWC_read()
    reader.input(str(root / "reconstruction.swc"))
    importer = h.Import3d_GUI(reader, 0)
    h("objref axobench_allen_cell")
    importer.instantiate(h.axobench_allen_cell)

    soma = _sections(h, "soma")
    if len(soma) != 1:
        raise ValueError(f"expected one soma section, found {len(soma)}")
    _configure_axon(h, soma[0], axon_type)
    groups = {
        name: _sections(h, name)
        for name in ("soma", "axon", "dend", "apic")
    }
    for section in h.allsec():
        section.nseg = 1 + 2 * int(float(section.L) / 40.0)
    _apply_parameters(h, groups, parameters)

    conditions = parameters["conditions"][0]
    h.v_init = float(conditions["v_init"])
    h.celsius = float(conditions["celsius"])
    return AllenAllActiveCell(
        soma=groups["soma"],
        axon=groups["axon"],
        dend=groups["dend"],
        apic=groups["apic"],
        all_sections=tuple(h.allsec()),
        model_root=root,
        specimen_id=int(metadata["specimen_id"]),
        model_id=int(metadata["id"]),
        v_init_mv=float(conditions["v_init"]),
        celsius=float(conditions["celsius"]),
    )


def collect_allen_dendritic_segments(
    cell: AllenAllActiveCell,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """Return branch segments using AxoBench basal/apical routing labels."""
    sections = cell.dend + cell.apic
    section_types = (
        ("basal",) * len(cell.dend) + ("apical",) * len(cell.apic)
    )
    segments: list[Any] = []
    segment_types: list[str] = []
    segment_lengths: list[float] = []
    for section, section_type in zip(sections, section_types):
        for segment in section:
            segments.append(segment)
            segment_types.append(section_type)
            segment_lengths.append(float(section.L) / float(section.nseg))
    if not segments:
        raise ValueError("Allen model has no dendritic segments")
    return (
        segments,
        np.asarray(segment_types, dtype=str),
        np.asarray(segment_lengths, dtype=np.float64),
    )


def simulate_allen_signed_trace(
    *,
    h: Any,
    cell: AllenAllActiveCell,
    signed_inputs: np.ndarray,
    source_route: dict[str, Any],
    exc_gmax_scale: float = 0.925,
    inh_gmax_scale: float = 1.0,
    input_routing: str = "normalized_by_type",
) -> dict[str, Any]:
    """Simulate one exact-millisecond AxoBench trace on an Allen cell."""
    duration_ms = int(signed_inputs.shape[0])
    segments, segment_types, segment_lengths = (
        collect_allen_dendritic_segments(cell)
    )
    (
        routed_segments,
        inputs_exc,
        inputs_inh,
        routing,
    ) = route_source_sites_to_segments(
        signed_inputs,
        segments,
        segment_types,
        segment_lengths,
        source_route,
        policy=input_routing,
    )
    netcons, synapses, events = _attach_synapses(
        h,
        routed_segments,
        inputs_exc,
        inputs_inh,
        exc_gmax_scale=exc_gmax_scale,
        inh_gmax_scale=inh_gmax_scale,
    )
    t_vec = h.Vector()
    soma_vec = h.Vector()
    t_vec.record(h._ref_t)
    soma_vec.record(cell.soma[0](0.5)._ref_v)

    h.dt = 0.025
    h.steps_per_ms = 40
    h.secondorder = 0
    h.CVode().active(0)
    h.finitialize(cell.v_init_mv)
    for netcon, times in events:
        for event_time in times:
            netcon.event(float(event_time))
    h.continuerun(duration_ms)

    t = np.asarray(t_vec, dtype=np.float64)
    soma = np.asarray(soma_vec, dtype=np.float64)
    low_time = np.arange(0, duration_ms, dtype=np.float64)
    high_time = np.arange(0, duration_ms, 0.125, dtype=np.float64)
    low_soma = np.interp(low_time, t, soma).astype(np.float32)
    high_soma = np.interp(high_time, t, soma)
    spike_labels = local_peak_spike_labels(
        high_time,
        high_soma,
        duration_ms,
    )
    clipped_soma = np.minimum(low_soma, SOMA_CLIP_THRESHOLD_MV)
    normalized_soma = (
        (clipped_soma - SOMA_BIAS_MV) * SOMA_SCALE
    ).astype(np.float32)
    targets = np.stack(
        [spike_labels, normalized_soma],
        axis=1,
    ).astype(np.float32)
    result = {
        "targets": targets,
        "soma_mv": low_soma,
        "input_routing": routing,
        "teacher": {
            "family": "allen_cell_types_all_active",
            "model_id": cell.model_id,
            "specimen_id": cell.specimen_id,
            "cell_role": "inhibitory",
            "transgenic_line": "Sst-IRES-Cre",
        },
    }
    del netcons, synapses, events, t_vec, soma_vec
    return result


def route_source_sites_to_segments(
    signed_inputs: np.ndarray,
    segments: list[Any],
    target_types: np.ndarray,
    target_lengths: np.ndarray,
    source_route: dict[str, Any],
    *,
    policy: str = "source_site_normalized_arclength",
) -> tuple[list[Any], np.ndarray, np.ndarray, dict[str, Any]]:
    """Preserve every source contact while mapping it onto a target segment."""
    if policy not in {
        "normalized_by_type",
        "source_site_normalized_arclength",
    }:
        raise ValueError(f"unsupported Allen input routing {policy!r}")
    if signed_inputs.ndim != 2 or signed_inputs.shape[1] % 2:
        raise ValueError(
            "signed_inputs must have shape (time, 2 * source_sites)"
        )
    source_sites = signed_inputs.shape[1] // 2
    source_types = np.asarray(source_route["types"], dtype=str)
    source_positions = np.asarray(
        source_route["positions"],
        dtype=np.float64,
    )
    if source_sites != source_types.shape[0]:
        raise ValueError(
            "source route and signed inputs disagree: "
            f"{source_types.shape[0]} != {source_sites}"
        )
    if len(segments) != len(target_types):
        raise ValueError("target segment metadata has inconsistent lengths")
    target_positions = _normalized_positions(
        target_types,
        target_lengths,
    )
    target_indices = np.empty(source_sites, dtype=np.int64)
    fallback_contacts = 0
    for source_index, (label, position) in enumerate(
        zip(source_types, source_positions)
    ):
        candidates = np.flatnonzero(target_types == label)
        if candidates.size == 0:
            candidates = np.arange(len(segments), dtype=np.int64)
            fallback_contacts += 1
        nearest = candidates[
            np.argmin(np.abs(target_positions[candidates] - position))
        ]
        target_indices[source_index] = int(nearest)
    routed_segments = [
        segments[int(index)] for index in target_indices
    ]
    inputs_exc = (signed_inputs[:, :source_sites] > 0).T
    inputs_inh = (signed_inputs[:, source_sites:] < 0).T
    contacts_per_segment = np.bincount(
        target_indices,
        minlength=len(segments),
    )
    return routed_segments, inputs_exc, inputs_inh, {
        "policy": "source_site_preserving_normalized_arclength",
        "source_sites_per_sign": int(source_sites),
        "morphology_segments": int(len(segments)),
        "used_sites_per_sign": int(source_sites),
        "unique_source_sites_per_sign": int(source_sites),
        "unused_source_sites_per_sign": 0,
        "target_segments_receiving_contacts": int(
            np.count_nonzero(contacts_per_segment)
        ),
        "max_contacts_per_target_segment": int(
            contacts_per_segment.max(initial=0)
        ),
        "mean_contacts_per_target_segment": float(
            contacts_per_segment.mean()
        ),
        "cross_compartment_fallback_contacts": int(
            fallback_contacts
        ),
        "target_count_by_type": _count_by_type(target_types),
        "source_count_by_type": _count_by_type(source_types),
        "defensibility_note": (
            "Every original signed source-site contact remains a distinct "
            "synapse. Contacts are placed on the nearest same-compartment "
            "target segment by normalized arclength; source compartments "
            "missing from the target morphology fall back to all dendrites."
        ),
    }


def _sections(h: Any, name: str) -> tuple[Any, ...]:
    if not hasattr(h, name):
        return ()
    return tuple(getattr(h, name))


def _normalized_positions(
    types: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(types, dtype=str)
    values = np.asarray(lengths, dtype=np.float64)
    positions = np.zeros(labels.shape[0], dtype=np.float64)
    for label in np.unique(labels):
        mask = labels == label
        selected = np.maximum(values[mask], 1e-12)
        cumulative = np.cumsum(selected) - 0.5 * selected
        positions[mask] = cumulative / max(
            float(selected.sum()),
            1e-12,
        )
    return positions


def _count_by_type(types: np.ndarray) -> dict[str, int]:
    values = np.asarray(types, dtype=str)
    return {
        str(label): int(np.count_nonzero(values == label))
        for label in np.unique(values)
    }


def _configure_axon(h: Any, soma: Any, axon_type: str) -> None:
    if axon_type not in {"stub", "truncated"}:
        raise ValueError(f"unknown Allen axon type {axon_type!r}")
    original = _sections(h, "axon")
    diameters = [1.0, 1.0]
    if axon_type == "truncated" and original:
        diameters = [float(original[0].diam), float(original[0].diam)]
        h.distance(sec=soma)
        for section in original:
            if float(h.distance(0.5, sec=section)) > 60.0:
                diameters[1] = float(section.diam)
                break
    for section in original:
        h.delete_section(sec=section)
    h("create axon[2]")
    for index, section in enumerate(h.axon):
        section.L = 30.0
        section.diam = diameters[index]
        section.nseg = 1 + 2 * int(float(section.L) / 40.0)
    h.axon[0].connect(soma, 1.0, 0.0)
    h.axon[1].connect(h.axon[0], 1.0, 0.0)
    h.define_shape()


def _apply_parameters(
    h: Any,
    groups: dict[str, tuple[Any, ...]],
    parameters: dict[str, Any],
) -> None:
    passive = parameters["passive"][0]
    for section in h.allsec():
        section.Ra = float(passive["ra"])
        section.insert("pas")
    for item in parameters["genome"]:
        group_name = str(item["section"])
        value = float(item["value"])
        if group_name == "glob":
            setattr(h, str(item["name"]), value)
            continue
        for section in groups.get(group_name, ()):
            mechanism = str(item["mechanism"])
            if mechanism and not h.ismembrane(mechanism, sec=section):
                section.insert(mechanism)
            setattr(section, str(item["name"]), value)
    for item in parameters["conditions"][0]["erev"]:
        for section in groups.get(str(item["section"]), ()):
            if h.ismembrane("k_ion", sec=section):
                section.ek = float(item["ek"])
            if h.ismembrane("na_ion", sec=section):
                section.ena = float(item["ena"])


def _attach_synapses(
    h: Any,
    segments: list[Any],
    inputs_exc: np.ndarray,
    inputs_inh: np.ndarray,
    *,
    exc_gmax_scale: float,
    inh_gmax_scale: float,
) -> tuple[list[Any], list[Any], list[tuple[Any, np.ndarray]]]:
    netcons: list[Any] = []
    synapses: list[Any] = []
    events: list[tuple[Any, np.ndarray]] = []
    for index, segment in enumerate(segments):
        ex_syn = h.DetAMPANMDA(segment)
        ex_syn.tau_r_AMPA = 0.3
        ex_syn.tau_d_AMPA = 3.0
        ex_syn.tau_r_NMDA = 2.0
        ex_syn.tau_d_NMDA = 70.0
        ex_syn.gmax = 0.0004 * exc_gmax_scale
        ex_syn.e = 0.0
        ex_nc = h.NetCon(None, ex_syn)
        ex_nc.delay = 0.0
        ex_nc.weight[0] = 1.0
        netcons.append(ex_nc)
        synapses.append(ex_syn)
        events.append(
            (
                ex_nc,
                np.flatnonzero(inputs_exc[index]).astype(np.float64),
            )
        )

        inh_syn = h.DetGABA(segment)
        inh_syn.tau_r = 0.2
        inh_syn.tau_d = 8.0
        inh_syn.gmax = 0.001 * inh_gmax_scale
        inh_syn.e = -80.0
        inh_nc = h.NetCon(None, inh_syn)
        inh_nc.delay = 0.0
        inh_nc.weight[0] = 1.0
        netcons.append(inh_nc)
        synapses.append(inh_syn)
        events.append(
            (
                inh_nc,
                np.flatnonzero(inputs_inh[index]).astype(np.float64),
            )
        )
    return netcons, synapses, events
