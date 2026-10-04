import argparse

import numpy as np
import pytest

from axobench.generation.run_coreneuron_event_dropout_pair import _route_inputs_to_segments, _segment_route
from axobench.generation import run_v1_parallel_generation as gen


def source_route():
    types = np.array(["basal"] * 4 + ["apical"] * 6)
    return _segment_route(types, np.ones(10))


def signed_inputs(steps=5, sites=10):
    x = np.zeros((steps, 2 * sites), dtype=np.int8)
    x[:, :sites] = 1
    x[:, sites:] = -1
    return x


def test_all_sites_routing_connects_every_source_site():
    target_types = np.array(["basal"] * 2 + ["apical"] * 3)
    exc, inh, summary = _route_inputs_to_segments(
        signed_inputs(), target_types, np.ones(5), routing="all_sites_by_type", source_route=source_route()
    )
    assert exc.shape == (10, 5) and inh.shape == (10, 5)
    assert summary["unused_source_sites_per_sign"] == 0
    idx = summary["site_segment_indices"]
    assert idx.shape == (10,)
    assert set(idx[:4]) <= {0, 1} and set(idx[4:]) <= {2, 3, 4}  # same compartment type
    assert summary["max_sites_per_segment"] >= 2


def test_normalized_routing_still_drops_sites_on_small_morphologies():
    target_types = np.array(["basal"] * 2 + ["apical"] * 3)
    _, _, summary = _route_inputs_to_segments(
        signed_inputs(), target_types, np.ones(5), routing="normalized_by_type", source_route=source_route()
    )
    assert summary["unused_source_sites_per_sign"] == 5


def test_identical_morphology_routes_each_site_to_itself():
    types = np.array(["basal"] * 4 + ["apical"] * 6)
    _, _, summary = _route_inputs_to_segments(
        signed_inputs(), types, np.ones(10), routing="all_sites_by_type", source_route=source_route()
    )
    np.testing.assert_array_equal(summary["site_segment_indices"], np.arange(10))


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_data_version_presets_fill_unset_options(version):
    args = gen.apply_data_version(gen.build_parser().parse_args(["--data-version", version]))
    for key, value in gen.DATA_VERSIONS[version].items():
        assert getattr(args, key) == value


def test_explicit_options_override_the_preset():
    args = gen.apply_data_version(gen.build_parser().parse_args(["--data-version", "v2", "--synapse-delay-ms", "1"]))
    assert args.synapse_delay_ms == 1.0 and args.input_routing == "all_sites_by_type"


def test_v2_adds_the_neuronio_cell_and_heldout_morphologies_get_no_training_data(tmp_path):
    args = gen.apply_data_version(gen.build_parser().parse_args(
        ["--data-version", "v2", "--train-samples-per-morphology", "2", "--val-samples-per-morphology", "1",
         "--test-samples-per-morphology", "1", "--intervention-samples-per-morphology", "0",
         "--heldout-morphology", gen.HAY_CELL_ID]))
    morphologies = gen.selected_morphologies(None, include_hay_cell=True)
    assert morphologies[0]["id"] == gen.HAY_CELL_ID and morphologies[0]["path"].endswith("cell1.asc")
    tasks = gen.build_tasks(args, tmp_path, morphologies, ())
    hay = [t for t in tasks if t["morphology_id"] == gen.HAY_CELL_ID]
    assert {t["split"] for t in hay} == {"test"}
    others = [t for t in tasks if t["morphology_id"] != gen.HAY_CELL_ID]
    assert {"train", "val", "test"} <= {t["split"] for t in others}
    assert all(t["synapse_delay_ms"] == 0.0 and t["input_routing"] == "all_sites_by_type" for t in tasks)


def test_v1_shards_without_generator_record_count_as_v1_only():
    task = {"data_version": "v1", "synapse_delay_ms": 1.0, "input_routing": "normalized_by_type",
            "nseg_policy": "hay40", "soma_samples_per_ms": 0}
    assert gen._generator_matches({}, task)
    assert not gen._generator_matches({}, {**task, "data_version": "v2", "synapse_delay_ms": 0.0})
