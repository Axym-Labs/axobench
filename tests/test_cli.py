from pathlib import Path
import json

import numpy as np

from axobench.cli import main


def test_cli_smoke_writes_demo_data(tmp_path):
    data_dir = tmp_path / "data"

    assert main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "4", "--input-dim", "6"]) == 0

    assert len(list(data_dir.glob("*.npz"))) == 1


def test_cli_prints_benchmark_axis_profile(capsys):
    assert main(["benchmark-profile", "--name", "paper-default", "--plan"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["profile"]["name"] == "paper-default"
    assert payload["diagnostic_state_evaluation"]["mode"] == "dense"
    assert (
        payload["morphology_intervention_sanity"]["selected_pairs"]
        < payload["morphology_intervention_sanity"]["full_cartesian_pairs"]
    )
    assert payload["capacity_estimate"]["within_target_wallclock"] is True


def test_cli_prints_benchmark_profile_cost(capsys):
    assert main(["benchmark-profile", "--name", "paper-default", "--cost"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["total_hours"] > 0
    assert payload["teacher_total_seconds"] > payload["model_eval_seconds"]


def test_cli_prints_dataset_schema(capsys):
    assert main(["dataset-schema", "--kind", "manifest"]) == 0
    manifest = json.loads(capsys.readouterr().out)

    assert manifest["schema_version"] == 2
    assert manifest["morphology_count"] == 5
    assert "interventions" in {entry["name"] for entry in manifest["categories"]}


def test_cli_prints_diagnostic_catalog(capsys):
    assert main(["dataset-schema", "--kind", "diagnostics"]) == 0
    catalog = json.loads(capsys.readouterr().out)

    assert catalog["plotter_policy"]["shape"] == "variable_length_list"
    assert "clipping_distortion" in catalog["suites"]["paper-summary"]
    assert "metric_decoupling" in catalog["suites"]["metric-decoupling"]


def test_cli_writes_artifact_derived_diagnostic_rows(tmp_path, capsys):
    teacher_rows = tmp_path / "teacher.jsonl"
    output = tmp_path / "audit_rows.jsonl"
    summary = tmp_path / "audit_summary.json"
    teacher_rows.write_text(
        json.dumps(
            {
                "model_id": "branch_elm_best",
                "regime": "canonical_test",
                "spike_auc": 0.995,
                "event_f1_2ms": 0.34,
                "clipped_rmse_mv": 0.55,
                "raw_soma_rmse_mv": 1.9,
                "near_spike_clipped_fraction": 0.52,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = main(
        [
            "diagnostic-audits",
            "--teacher-grounded-rows",
            str(teacher_rows),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--suites",
            "metric-decoupling,target-view-audit,paper-summary",
            "--teacher-id",
            "neuronio",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] > 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["diagnostic"] for row in rows} >= {"metric_decoupling", "target_view_audit"}
    assert any(row["metric"] == "spike_auc_mean" for row in rows)
    assert summary.exists()


def test_cli_writes_diagnostic_plot_and_table_artifacts(tmp_path, capsys):
    canonical = tmp_path / "canonical.json"
    mechanistic = tmp_path / "mechanistic.json"
    sparse = tmp_path / "sparse.json"
    points = tmp_path / "points.jsonl"
    tcn = tmp_path / "tcn.json"
    morphology = tmp_path / "morphology.jsonl"
    neuronio = tmp_path / "neuronio_interventions.jsonl"
    trace_root = tmp_path / "neuronio_trace_root"
    output = tmp_path / "plots"

    canonical.write_text(json.dumps(_canonical_plot_fixture()), encoding="utf-8")
    mechanistic.write_text(json.dumps(_mechanistic_plot_fixture()), encoding="utf-8")
    sparse.write_text(json.dumps(_sparse_plot_fixture()), encoding="utf-8")
    points.write_text("\n".join(json.dumps(row) for row in _point_plot_fixture()) + "\n", encoding="utf-8")
    tcn.write_text(json.dumps(_tcn_plot_fixture("tcn_dwt7_128_153", raw=226.0)), encoding="utf-8")
    morphology.write_text("\n".join(json.dumps(row) for row in _morphology_plot_fixture()) + "\n", encoding="utf-8")
    neuronio.write_text("\n".join(json.dumps(row) for row in _neuronio_intervention_plot_fixture()) + "\n", encoding="utf-8")
    _write_neuronio_trace_root_fixture(trace_root)

    result = main(
        [
            "diagnostic-plots",
            "--canonical-summary",
            str(canonical),
            "--mechanistic-summary",
            str(mechanistic),
            "--sparse-event-summary",
            str(sparse),
            "--point-rows",
            str(points),
            "--tcn-summary",
            str(tcn),
            "--morphology-contrast-rows",
            str(morphology),
            "--neuronio-intervention-rows",
            str(neuronio),
            "--neuronio-intervention-root",
            str(trace_root),
            "--output-dir",
            str(output),
            "--formats",
            "svg,png",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["figures"]) == 8
    assert len(payload["tables"]) == 6
    assert (output / "fig01_spike_metric_identification.svg").exists()
    assert (output / "fig05_morphology_intervention_response.png").exists()
    assert (output / "fig07_neuronio_matched_intervention_response.svg").exists()
    assert (output / "fig08_neuronio_matched_trace_examples.svg").exists()
    assert "AUC Versus Event Timing" in (output / "table02_metric_decoupling.md").read_text(encoding="utf-8")
    assert "TCN Voltage RMSE" in (output / "table03_target_and_adapter_protocol.md").read_text(encoding="utf-8")
    assert "event_dropout" in (output / "table06_neuronio_matched_interventions.md").read_text(encoding="utf-8")


def test_cli_writes_teacher_qc_benchmark_rows(tmp_path, capsys):
    data_dir = tmp_path / "v1"
    output = tmp_path / "rows.jsonl"
    assert main(["make-demo-data", "--output", str(data_dir / "val"), "--samples", "2", "--time-steps", "8", "--input-dim", "4"]) == 0

    result = main(
        [
            "benchmark-rows",
            "--mode",
            "teacher-qc",
            "--dataset-root",
            str(data_dir),
            "--selector",
            "val",
            "--output",
            str(output),
            "--ignore-start",
            "0",
            "--morphology-id",
            "unit-morph",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] > 0
    assert payload["output"] == str(output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert any(row["diagnostic"] == "teacher_distribution_qc" for row in rows)
    assert all(row["morphology_id"] == "unit-morph" for row in rows)


def test_cli_writes_intervention_benchmark_rows_from_pair_path(tmp_path, capsys):
    pair_path = tmp_path / "intervention_pair.npz"
    output = tmp_path / "intervention_rows.jsonl"
    baseline_inputs = np.zeros((1, 8, 4), dtype=np.int8)
    intervention_inputs = baseline_inputs.copy()
    intervention_inputs[:, 2, 0] = 1
    baseline_targets = np.zeros((1, 8, 2), dtype=np.float32)
    intervention_targets = baseline_targets.copy()
    baseline_targets[..., 1] = 1.0
    intervention_targets[..., 1] = 0.5
    np.savez_compressed(
        pair_path,
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=intervention_targets,
        sample_ids=np.asarray(["trace-a"]),
    )

    result = main(
        [
            "benchmark-rows",
            "--mode",
            "intervention",
            "--dataset-root",
            str(tmp_path / "unused-v1-root"),
            "--selector",
            "intervention",
            "--intervention",
            "event_dropout",
            "--pair-path",
            str(pair_path),
            "--output",
            str(output),
            "--ignore-start",
            "0",
            "--suites",
            "teacher-qc,intervention-response,paper-summary",
            "--morphology-id",
            "unit-morph",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] > 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert any(row["diagnostic"] == "intervention_response" for row in rows)
    assert any(row["metric"] == "teacher_direction" for row in rows)
    assert all(row["morphology_id"] == "unit-morph" for row in rows)
    assert all(row["condition"] == "interventions/event_dropout" for row in rows)
    assert all(row["intervention"] == "event_dropout" for row in rows)


def test_cli_runs_teacher_only_v1_benchmark_bundle(tmp_path, capsys):
    data_dir = tmp_path / "v1"
    output_dir = tmp_path / "bundle"
    pair_dir = tmp_path / "pilot" / "specimen_488683425_event_dropout_simbase8"
    pair_path = pair_dir / "intervention_pair.npz"
    assert main(["make-demo-data", "--output", str(data_dir / "val"), "--samples", "1", "--time-steps", "8", "--input-dim", "4"]) == 0
    pair_dir.mkdir(parents=True)
    baseline_inputs = np.zeros((1, 8, 4), dtype=np.int8)
    intervention_inputs = baseline_inputs.copy()
    intervention_inputs[:, 2, 0] = 1
    baseline_targets = np.zeros((1, 8, 2), dtype=np.float32)
    intervention_targets = baseline_targets.copy()
    intervention_targets[..., 1] = 0.5
    np.savez_compressed(
        pair_path,
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=intervention_targets,
        sample_ids=np.asarray(["trace-a"]),
    )

    result = main(
        [
            "benchmark-v1",
            "--dataset-root",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--suite",
            "teacher-qc",
            "--intervention-pair-glob",
            str(tmp_path / "pilot" / "*" / "intervention_pair.npz"),
            "--max-val-samples",
            "1",
            "--max-intervention-samples",
            "1",
            "--ignore-start",
            "0",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suite"] == "teacher-qc"
    assert len(payload["artifacts"]) == 3
    assert (output_dir / "teacher_qc_val.md").exists()
    assert (output_dir / "morphology_routing_qc.md").exists()
    assert (output_dir / "intervention_specimen_488683425_event_dropout.md").exists()


def test_cli_convert_neuronio_requires_existing_raw_path(tmp_path):
    output_dir = tmp_path / "out"

    result = main(["convert-neuronio", "--input", str(tmp_path / "missing"), "--output", str(output_dir)])

    assert result == 2


def test_cli_repack_shards_writes_sliceable_dataset(tmp_path):
    data_dir = tmp_path / "data"
    repacked_dir = tmp_path / "repacked"

    main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "4", "--input-dim", "6"])

    result = main(["repack-shards", "--input", str(data_dir), "--output", str(repacked_dir)])

    assert result == 0
    assert (repacked_dir / "manifest.json").exists()
    assert len(list(repacked_dir.glob("*_inputs.npy"))) == 1


def _canonical_plot_fixture():
    models = []
    for idx, model_id in enumerate(("branch_elm_best", "branch_elm_m1", "branch_elm_m30", "branch_elm_m100")):
        models.append(
            {
                "model_id": model_id,
                "rmse_to_clipped_valid_mv": 0.55 + 0.1 * idx,
                "rmse_to_raw_valid_mv": 1.9 + 0.2 * idx,
                "rmse_to_clipped_near_spike_mv": 3.1 + 0.1 * idx,
                "rmse_to_raw_near_spike_mv": 28.0 - idx,
                "threshold_logit_zero": {"predicted_spike_count": 3 if idx == 0 else 0},
            }
        )
    return {
        "validation": {
            "models": models,
            "target": {
                "target_spike_count": 2817,
                "target_distortion": {
                    "clipped_fraction_valid": 0.003,
                    "near_spike_clipped_fraction": 0.52,
                    "rmse_mv_near_spike": 26.5,
                    "max_abs_distortion_mv": 88.6,
                },
            },
        }
    }


def _mechanistic_plot_fixture():
    reports = {}
    for idx, model_id in enumerate(("branch_elm_best", "branch_elm_m1", "branch_elm_m30", "branch_elm_m100")):
        reports[model_id] = {
            "phase_plane": {
                "near_spike_phase_rmse_mv": 24.0 + idx,
                "high_dvdt_sign_agreement_fraction": 0.96 - 0.03 * idx,
            },
            "history_conditioned_error": {
                "recent_spike_history": {"voltage_rmse_mv": 10.0 + idx},
                "quiet_history": {"voltage_rmse_mv": 0.6 + 0.1 * idx},
            },
            "event_triggered_response": {
                "exc": {"latency_error_steps": -4 + idx},
                "inh": {"latency_error_steps": 6 - idx, "response_gain": 0.52 + 0.08 * idx},
            },
        }
    return {"reports": reports}


def _sparse_plot_fixture():
    ratios = []
    for budget in (8, 32, 128):
        for group in ("quiet_or_low_spike", "spike_rich"):
            for label, targeted, control in (
                ("global", "gradient_sparse_global", "random_sparse_global"),
                ("time_local_8", "gradient_sparse_time_local_8", "random_sparse_same_time_8"),
                ("time_local_32", "gradient_sparse_time_local_32", "random_sparse_same_time_32"),
            ):
                base = 3.0 + budget / 16.0 + (2.0 if group == "spike_rich" else 0.0)
                ratios.append(
                    {
                        "budget": budget,
                        "window_group": group,
                        "targeted": targeted,
                        "control": control,
                        "median_soma_delta_ratio": base + (0.5 if label == "time_local_8" else 0.0),
                        "changed_fraction_of_events_median": 0.001 * budget / 8.0,
                    }
                )
    return {"ratios": ratios, "headline": {"global_ratio_median": 8.0}}


def _point_plot_fixture():
    points = []
    for regime, auc_base, f1_base in (
        ("canonical_test", 0.994, 0.34),
        ("ood_rates", 0.990, 0.29),
        ("combined_ergodic", 0.989, 0.31),
    ):
        for idx, model_id in enumerate(("branch_elm_best", "branch_elm_m30", "branch_elm_m100")):
            points.append(
                {
                    "model_id": model_id,
                    "model_regime": f"{model_id}:{regime}",
                    "regime": regime,
                    "spike_auc": auc_base - 0.001 * idx,
                    "event_f1_2ms": f1_base - 0.02 * idx,
                }
            )
    return [
        {"point_id": 1, "metric": "points", "value": json.dumps(points)},
        {"point_id": 2, "metric": "delta_min", "value": -0.003},
        {"point_id": 2, "metric": "delta_mean", "value": 0.007},
        {"point_id": 2, "metric": "delta_max", "value": 0.020},
    ]


def _tcn_plot_fixture(model, *, raw):
    return {
        "model": model,
        "rows": [
            {"view": "raw_tcn", "rmse_mv": raw},
            {"view": "same_slice_target_mean_baseline", "rmse_mv": 3.48},
            {"view": "same_slice_mean_std_match", "rmse_mv": 3.6},
            {"view": "same_slice_linear_fit", "rmse_mv": 3.1},
        ],
    }


def _morphology_plot_fixture():
    rows = []
    values = {
        "event_dropout": {"specimen_1": -10.0, "specimen_2": -15.0},
        "exc_dropout": {"specimen_1": -25.0, "specimen_2": -35.0},
        "inh_dropout": {"specimen_1": 20.0, "specimen_2": 28.0},
        "site_silence": {"specimen_1": -5.0, "specimen_2": -12.0},
        "temporal_jitter": {"specimen_1": -1.0, "specimen_2": 2.0},
    }
    for intervention, morphology_values in values.items():
        rows.append(
            {
                "diagnostic": "morphology_contrast",
                "metric": "intervention_response.spike_count_delta_teacher_mean.mean_across_morphologies",
                "intervention": intervention,
                "morphology_values": morphology_values,
            }
        )
    return rows


def _neuronio_intervention_plot_fixture():
    rows = []
    values = {
        "event_dropout": (-0.0038, 1.26, 0.18, -1.08, -2.50, 2.59, "lower_event_drive/lower_spiking/hyperpolarized"),
        "exc_dropout": (-0.0021, 1.26, 0.04, -1.22, -4.94, 4.96, "lower_event_drive/lower_spiking/hyperpolarized"),
        "inh_dropout": (-0.0017, 1.26, 4.12, 2.86, 3.09, 3.39, "lower_event_drive/higher_spiking/depolarized"),
        "site_silence": (-0.0033, 1.26, 0.42, -0.84, -3.56, 3.63, "lower_event_drive/lower_spiking/hyperpolarized"),
        "temporal_jitter": (-0.0001, 1.26, 1.21, -0.05, -0.04, 0.35, "lower_event_drive/lower_spiking/hyperpolarized"),
    }
    metric_names = (
        "event_fraction_delta",
        "baseline_spike_rate_per_1000_steps",
        "intervention_spike_rate_per_1000_steps",
        "delta_spike_rate_per_1000_steps",
        "delta_soma_mean_mv",
        "delta_soma_abs_mean_mv",
        "teacher_direction",
    )
    for intervention, metric_values in values.items():
        for metric, value in zip(metric_names, metric_values):
            rows.append(
                {
                    "diagnostic": "intervention_response",
                    "intervention": intervention,
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def _write_neuronio_trace_root_fixture(root):
    time_steps = 620
    time = np.arange(time_steps, dtype=np.float32)
    baseline_soma = -70.0 + 0.4 * np.sin(time / 20.0)
    baseline_targets = np.zeros((1, time_steps, 2), dtype=np.float32)
    intervention_targets = np.zeros((1, time_steps, 2), dtype=np.float32)
    baseline_targets[..., 1] = (baseline_soma + 67.7) * 0.1
    for idx, intervention in enumerate(("exc_dropout", "inh_dropout", "temporal_jitter", "morphology-3")):
        folder = root / "interventions" / intervention
        folder.mkdir(parents=True)
        intervention_soma = baseline_soma.copy()
        intervention_soma[520:590] += (idx + 1) * 2.0
        intervention_targets[..., 1] = (intervention_soma + 67.7) * 0.1
        baseline_targets[..., 0] = 0
        intervention_targets[..., 0] = 0
        baseline_targets[0, 540, 0] = 1
        intervention_targets[0, 550 + idx, 0] = 1
        np.savez_compressed(
            folder / "intervention_pair_00000.npz",
            baseline_soma_mv=baseline_soma[None, :].astype(np.float32),
            intervention_soma_mv=intervention_soma[None, :].astype(np.float32),
            baseline_targets=baseline_targets.copy(),
            intervention_targets=intervention_targets.copy(),
            baseline_inputs=np.zeros((1, time_steps, 4), dtype=np.int8),
            intervention_inputs=np.zeros((1, time_steps, 4), dtype=np.int8),
            sample_ids=np.asarray([f"{intervention}-trace"]),
        )
