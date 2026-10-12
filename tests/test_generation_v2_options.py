import argparse
from pathlib import Path

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


def _allen_root(tmp_path, specimens=(111, 222)):
    rows = []
    for specimen in specimens:
        model_root = tmp_path / f"specimen_{specimen}" / "model"
        model_root.mkdir(parents=True)
        rows.append({"specimen_id": specimen, "model_id": specimen + 1, "model_root": str(model_root)})
    (tmp_path / "models.json").write_text(__import__("json").dumps(rows))
    return tmp_path


def test_v2_uses_allen_all_active_teachers_and_keeps_the_hay_cell(tmp_path):
    root = _allen_root(tmp_path / "allen")
    args = gen.apply_data_version(gen.build_parser().parse_args(
        ["--data-version", "v2", "--allen-all-active-root", str(root), "--val-samples-per-morphology", "1",
         "--intervention-samples-per-morphology", "1", "--intervention", "inh_dropout"]))
    assert args.excitatory_teacher == "allen_all_active"
    morphologies = gen.selected_morphologies(None, include_hay_cell=True, excitatory_teacher=args.excitatory_teacher,
                                             allen_root=root)
    assert [m["id"] for m in morphologies] == [gen.HAY_CELL_ID, "specimen_111", "specimen_222"]
    assert [m["teacher"] for m in morphologies] == ["hay", "allen_all_active", "allen_all_active"]
    out = tmp_path / "out"
    tasks = gen.build_tasks(args, out, morphologies, ("inh_dropout",))
    for task in tasks:
        allen = task["morphology_id"] != gen.HAY_CELL_ID
        assert task["teacher"] == ("allen_all_active" if allen else "hay")
        assert task["nseg_policy"] == (gen.ALLEN_NSEG_POLICY if allen else "d_lambda")
        assert task["source_route"] == str(out / "source_route.npz")


def test_allen_morphology_selection_by_specimen(tmp_path):
    root = _allen_root(tmp_path)
    rows = gen.selected_morphologies(["222"], excitatory_teacher="allen_all_active", allen_root=root)
    assert [r["id"] for r in rows] == ["specimen_222"]


def test_v1_preset_matches_the_released_shards():
    args = gen.apply_data_version(gen.build_parser().parse_args(["--data-version", "v1"]))
    assert args.exc_gmax_scale == 1.0 and args.synapse_delay_ms == 1.0 and args.excitatory_teacher == "hay"


def test_teacher_change_invalidates_existing_shards():
    task = {"data_version": "v2", "synapse_delay_ms": 0.0, "input_routing": "all_sites_by_type",
            "nseg_policy": gen.ALLEN_NSEG_POLICY, "soma_samples_per_ms": 4, "teacher": "allen_all_active"}
    gen_meta = {k: task[k] for k in ("synapse_delay_ms", "input_routing", "nseg_policy", "soma_samples_per_ms")}
    assert gen._generator_matches({"generator": {**gen_meta, "teacher": "allen_all_active"}}, task)
    assert not gen._generator_matches({"generator": gen_meta}, task)


def test_calibrated_gain_multiplies_the_excitatory_scale_of_allen_cells_only(tmp_path):
    root = _allen_root(tmp_path / "allen")
    cal = tmp_path / "gains.json"
    cal.write_text(__import__("json").dumps({"gains": {"specimen_111": 0.5, "specimen_222": 3.0}}))
    args = gen.apply_data_version(gen.build_parser().parse_args(
        ["--data-version", "v2", "--allen-all-active-root", str(root), "--val-samples-per-morphology", "1",
         "--intervention-samples-per-morphology", "0", "--exc-gain-calibration", str(cal)]))
    morphologies = gen.selected_morphologies(None, include_hay_cell=True, excitatory_teacher="allen_all_active",
                                             allen_root=root)
    scales = {t["morphology_id"]: t["exc_gmax_scale"] for t in gen.build_tasks(args, tmp_path / "out", morphologies, ())}
    assert scales == {gen.HAY_CELL_ID: 1.0, "specimen_111": 0.5, "specimen_222": 3.0}
    cal.write_text(__import__("json").dumps({"gains": {"specimen_111": 0.5}}))
    with pytest.raises(ValueError, match="specimen_222"):
        gen.build_tasks(args, tmp_path / "out", morphologies, ())


def test_morphology_index_lists_cells_and_copies_allen_reconstructions(tmp_path):
    import json

    root = _allen_root(tmp_path / "allen", specimens=(111,))
    model_root = root / "specimen_111" / "model"
    (model_root / "reconstruction.swc").write_text("1 1 0 0 0 5 -1\n")
    (model_root / "model_metadata.json").write_text(json.dumps(
        {"id": 112, "specimen_id": 111, "name": "Biophysical - all active_Rorb-IRES2-Cre-D;Ai14-1",
         "specimen": {"structure": {"acronym": "VISp5"}}}))
    args = gen.apply_data_version(gen.build_parser().parse_args(
        ["--data-version", "v2", "--allen-all-active-root", str(root), "--heldout-morphology", "111"]))
    morphologies = gen.selected_morphologies(None, include_hay_cell=True, excitatory_teacher="allen_all_active",
                                             allen_root=root)
    out = tmp_path / "out"
    out.mkdir()
    index = json.loads(gen.write_morphology_index(args, out, morphologies).read_text())["morphologies"]
    hay, allen = index
    assert hay["id"] == gen.HAY_CELL_ID and not hay["heldout"] and "swc" not in hay
    assert allen["heldout"] and allen["transgenic_line"] == "Rorb-IRES2-Cre-D" and allen["allen_model_id"] == 112
    assert (out / allen["swc"]).read_text().startswith("1 1")


def test_v2_generates_each_intervention_at_three_doses(tmp_path):
    root = _allen_root(tmp_path / "allen", specimens=(111,))
    args = gen.apply_data_version(gen.build_parser().parse_args(
        ["--data-version", "v2", "--allen-all-active-root", str(root), "--val-samples-per-morphology", "0",
         "--intervention-samples-per-morphology", "1", "--intervention", "inh_dropout", "--intervention", "temporal_jitter"]))
    variants = gen.intervention_variants(args, ("inh_dropout", "temporal_jitter"))
    assert [v[0] for v in variants] == ["inh_dropout_p10", "inh_dropout_p35", "inh_dropout_p70",
                                        "temporal_jitter_j1", "temporal_jitter_j2", "temporal_jitter_j8"]
    morphologies = gen.selected_morphologies(["111"], excitatory_teacher="allen_all_active", allen_root=root)
    tasks = gen.build_tasks(args, tmp_path / "out", morphologies, ("inh_dropout", "temporal_jitter"))
    assert {(t["intervention"], t["intervention_dose"], Path(t["output"]).parent.name) for t in tasks} == {
        ("inh_dropout", 0.1, "inh_dropout_p10"), ("inh_dropout", 0.35, "inh_dropout_p35"),
        ("inh_dropout", 0.7, "inh_dropout_p70"), ("temporal_jitter", 1.0, "temporal_jitter_j1"),
        ("temporal_jitter", 2.0, "temporal_jitter_j2"), ("temporal_jitter", 8.0, "temporal_jitter_j8")}
    assert gen.dose_kwargs("temporal_jitter", 8.0)["jitter_steps"] == 8
    assert gen.dose_kwargs("inh_dropout", 0.7)["dropout_probability"] == 0.7


def test_v1_keeps_one_default_dose_per_intervention():
    args = gen.apply_data_version(gen.build_parser().parse_args(["--data-version", "v1"]))
    assert gen.intervention_variants(args, ("inh_dropout",)) == [("inh_dropout", "inh_dropout", 0.35)]


def test_dose_change_invalidates_existing_pair_shards():
    task = {"kind": "intervention", "intervention": "inh_dropout", "intervention_dose": 0.7, "synapse_delay_ms": 0.0,
            "input_routing": "all_sites_by_type", "nseg_policy": "d_lambda", "soma_samples_per_ms": 4}
    gen_meta = {k: task[k] for k in ("synapse_delay_ms", "input_routing", "nseg_policy", "soma_samples_per_ms")}
    assert gen._generator_matches({"generator": gen_meta, "intervention_dose": 0.7}, task)
    assert not gen._generator_matches({"generator": gen_meta, "intervention_dose": 0.35}, task)


def _layout_args(root, *extra):
    return gen.apply_data_version(gen.build_parser().parse_args(
        ["--data-version", "v2", "--allen-all-active-root", str(root), "--samples-per-shard", "2",
         "--val-samples-per-morphology", "2", "--test-samples-per-morphology", "2",
         "--intervention-samples-per-morphology", "2", "--intervention", "inh_dropout",
         "--heldout-morphology", "222", *extra]))


def _layout(tmp_path, *extra):
    root = tmp_path / "allen"
    if not (root / "models.json").exists():
        _allen_root(root, specimens=(111, 222, 333))
    args = _layout_args(root, *extra)
    morphologies = gen.selected_morphologies(None, include_hay_cell=True, excitatory_teacher="allen_all_active",
                                             allen_root=root)
    tasks = gen.build_tasks(args, tmp_path / "out", morphologies, ("inh_dropout",))
    return args, {t["output"]: t for t in tasks}


def test_extension_keeps_every_shard_and_seed_of_the_base_layout(tmp_path):
    _, base = _layout(tmp_path, "--train-samples-per-morphology", "4")
    _, larger = _layout(tmp_path, "--train-samples-per-morphology", "10", "--layout-train-samples-per-morphology", "4")
    for output, task in base.items():
        assert larger[output]["seed"] == task["seed"] and larger[output]["shard_index"] == task["shard_index"]
    extension = [t for t in larger.values() if t["extension"]]
    # 3 extension shards (6 traces) for each of the three training cells; held-out 222 gets none
    assert len(extension) == 9 and {t["split"] for t in extension} == {"train"}
    assert "specimen_222" not in {t["morphology_id"] for t in extension}
    indices = sorted(t["shard_index"] for t in extension)
    assert indices[0] >= gen.EXTENSION_BASE and len(set(indices)) == len(indices)
    assert not {t["seed"] for t in extension} & {t["seed"] for t in base.values()}


def test_without_a_layout_option_the_layout_is_unchanged(tmp_path):
    _, explicit = _layout(tmp_path, "--train-samples-per-morphology", "4", "--layout-train-samples-per-morphology", "4")
    _, default = _layout(tmp_path, "--train-samples-per-morphology", "4")
    assert {o: t["seed"] for o, t in explicit.items()} == {o: t["seed"] for o, t in default.items()}


def test_generate_morphology_selects_tasks_without_renumbering(tmp_path):
    args, full = _layout(tmp_path, "--train-samples-per-morphology", "10", "--layout-train-samples-per-morphology", "4",
                         "--generate-morphology", "333", "--generate-morphology", gen.HAY_CELL_ID, "--extension-only")
    chosen = gen.select_tasks(list(full.values()), args.generate_morphology, args.extension_only)
    assert {t["morphology_id"] for t in chosen} == {"specimen_333", gen.HAY_CELL_ID}
    assert all(t["extension"] and t["split"] == "train" for t in chosen) and len(chosen) == 6
    assert all(full[t["output"]]["seed"] == t["seed"] for t in chosen)


def test_layout_base_must_divide_into_shards(tmp_path):
    with pytest.raises(ValueError):
        _layout(tmp_path, "--train-samples-per-morphology", "10", "--layout-train-samples-per-morphology", "3")
