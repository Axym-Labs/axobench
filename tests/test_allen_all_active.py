import pytest

from axobench.generation.allen_all_active import (
    _apply_parameters,
    _sections,
    route_source_sites_to_segments,
)
import numpy as np


class FakeSection:
    def __init__(self) -> None:
        self.inserted = {"k_ion", "na_ion"}

    def insert(self, mechanism: str) -> None:
        self.inserted.add(mechanism)


class FakeHoc:
    def __init__(self) -> None:
        self.soma = [FakeSection()]
        self.dend = [FakeSection()]

    def allsec(self):
        return iter(self.soma + self.dend)

    def ismembrane(self, mechanism: str, *, sec: FakeSection) -> bool:
        return mechanism in sec.inserted


def test_sections_treats_missing_compartment_as_empty() -> None:
    h = FakeHoc()

    assert _sections(h, "soma") == tuple(h.soma)
    assert _sections(h, "apic") == ()


def test_all_active_parameters_apply_only_to_present_compartments() -> None:
    h = FakeHoc()
    parameters = {
        "passive": [{"ra": 100.0}],
        "genome": [
            {
                "section": "soma",
                "name": "cm",
                "value": "2.5",
                "mechanism": "",
            },
            {
                "section": "dend",
                "name": "gbar_Test",
                "value": "0.4",
                "mechanism": "Test",
            },
            {
                "section": "apic",
                "name": "cm",
                "value": "9.0",
                "mechanism": "",
            },
        ],
        "conditions": [
            {
                "erev": [
                    {
                        "section": "soma",
                        "ek": -107.0,
                        "ena": 53.0,
                    }
                ]
            }
        ],
    }

    _apply_parameters(
        h,
        {
            "soma": tuple(h.soma),
            "dend": tuple(h.dend),
            "apic": (),
            "axon": (),
        },
        parameters,
    )

    assert h.soma[0].Ra == pytest.approx(100.0)
    assert h.soma[0].cm == pytest.approx(2.5)
    assert h.soma[0].ek == pytest.approx(-107.0)
    assert h.soma[0].ena == pytest.approx(53.0)
    assert h.dend[0].gbar_Test == pytest.approx(0.4)
    assert "Test" in h.dend[0].inserted
    assert "pas" in h.soma[0].inserted


def test_source_site_routing_preserves_every_contact() -> None:
    segments = [object(), object(), object()]
    signed_inputs = np.asarray(
        [
            [1, 0, 1, 0, -1, 0],
            [0, 1, 0, -1, 0, -1],
        ],
        dtype=np.int8,
    )
    source_route = {
        "types": np.asarray(["basal", "apical", "basal"]),
        "positions": np.asarray([0.1, 0.5, 0.9]),
    }

    routed, excitatory, inhibitory, summary = (
        route_source_sites_to_segments(
            signed_inputs,
            segments,
            np.asarray(["basal", "basal", "basal"]),
            np.asarray([1.0, 1.0, 1.0]),
            source_route,
        )
    )

    assert len(routed) == 3
    assert excitatory.shape == (3, 2)
    assert inhibitory.shape == (3, 2)
    assert summary["used_sites_per_sign"] == 3
    assert summary["unused_source_sites_per_sign"] == 0
    assert summary["cross_compartment_fallback_contacts"] == 1


def test_merging_sites_sums_event_counts_per_segment() -> None:
    from axobench.generation.allen_all_active import _event_times, merge_sites_per_segment

    class Sec:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

    class Seg:
        def __init__(self, sec, x):
            self.sec, self.x = sec, x

    a, b = Sec("dend[0]"), Sec("apic[3]")
    segments = [Seg(a, 0.5), Seg(b, 0.25), Seg(a, 0.5), Seg(b, 0.75)]
    exc = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
    inh = np.zeros_like(exc)
    unique, merged_exc, merged_inh = merge_sites_per_segment(segments, exc, inh)
    assert [(s.sec.name(), s.x) for s in unique] == [("dend[0]", 0.5), ("apic[3]", 0.25), ("apic[3]", 0.75)]
    np.testing.assert_array_equal(merged_exc, [[2, 1, 1], [0, 1, 0], [0, 0, 1]])
    assert merged_inh.sum() == 0
    np.testing.assert_array_equal(_event_times(merged_exc[0]), [0.0, 0.0, 1.0, 2.0])
