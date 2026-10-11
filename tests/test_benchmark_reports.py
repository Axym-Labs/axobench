import json

from axobench.benchmark.reports import (
    core_metric_comparison_rows,
    summarize_diagnostic_rows,
    write_core_metric_comparison_table,
    write_diagnostic_summary,
)


def _rows():
    return [
        {
            "diagnostic": "trace_core",
            "suite": "trace-core|paper-summary",
            "model_id": "model-a",
            "teacher_id": "teacher",
            "morphology_id": "morph-a",
            "split": "val",
            "condition": "val",
            "intervention": None,
            "trace_id": "trace-a",
            "state": None,
            "metric": "soma_rmse_mv",
            "value": 0.8,
            "unit": "mV",
            "n": 10,
        },
        {
            "diagnostic": "intervention_response",
            "suite": "teacher-qc|intervention-response|paper-summary",
            "model_id": "model-a",
            "teacher_id": "teacher",
            "morphology_id": "morph-a",
            "split": "",
            "condition": "interventions/event_dropout",
            "intervention": "event_dropout",
            "trace_id": None,
            "state": None,
            "metric": "teacher_direction",
            "value": "hyperpolarizing",
            "unit": "label",
            "n": None,
        },
    ]


def test_summarize_diagnostic_rows_groups_numeric_and_label_metrics():
    summary = summarize_diagnostic_rows(_rows())

    assert summary["row_count"] == 2
    assert summary["diagnostics"] == ["intervention_response", "trace_core"]
    assert summary["trace_ids"] == ["trace-a"]
    assert summary["numeric_metrics"][0]["mean"] == 0.8
    assert summary["label_metrics"][0]["counts"] == {"hyperpolarizing": 1}


def test_write_diagnostic_summary_supports_json_and_markdown(tmp_path):
    json_path = tmp_path / "summary.json"
    md_path = tmp_path / "summary.md"

    write_diagnostic_summary(_rows(), json_path)
    write_diagnostic_summary(_rows(), md_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["row_count"] == 2
    assert "Benchmark Diagnostic Summary" in md_path.read_text(encoding="utf-8")


def test_core_metric_comparison_rows_compute_inflation_factors(tmp_path):
    models = [
        {
            "model_id": "model-a",
            "voltage_sera_iron_per_spike_mv2": 20.0,
            "voltage_rmse_mv": 2.0,
            "spike_f1_area": 0.2,
            "spike_f1_area_0_5ms": 0.4,
            "response_sera_iron_per_spike_mv2": 8.0,
        }
    ]
    old = {"model-a": {"clipped_rmse_mv": 0.5, "spike_auc": 0.9}}

    rows = core_metric_comparison_rows(models, old)

    assert rows[0]["voltage_sera_iron_per_spike_mv2"] == 20.0
    assert rows[0]["response_sera_iron_per_spike_mv2"] == 8.0
    assert "voltage_root_sera_inflation" not in rows[0]
    assert rows[0]["raw_voltage_rmse_inflation"] == 4.0
    assert rows[0]["auc_over_exact_f1"] == 4.5
    assert rows[0]["auc_over_f1_area_0_5ms"] == 2.25

    table_path = tmp_path / "core.md"
    write_core_metric_comparison_table(models, table_path, old)
    table = table_path.read_text(encoding="utf-8")
    assert "| 20 |" in table
    assert "0.2 (4.5x)" in table
