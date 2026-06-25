import json

import pytest

from dendrobench.benchmark.diagnostic_audits import (
    DiagnosticAuditSources,
    adapter_fairness_diagnostic_rows,
    diagnostic_audit_rows,
    diagnostic_audit_rows_from_sources,
)
from dendrobench.benchmark.diagnostic_rows import DiagnosticContext, validate_diagnostic_rows


def test_artifact_audits_emit_regular_diagnostic_rows():
    teacher_rows = [
        {
            "model_id": "branch_elm_best",
            "regime": "canonical_test",
            "morphology_id": "hay",
            "source_file": "valid_0.npz",
            "spike_auc": 0.995,
            "val_threshold_event_f1_2ms": 0.34,
            "regime_prevalence_event_f1_2ms": 0.35,
            "zero_threshold_predicted_spikes": 3,
            "target_spikes": 2817,
            "clipped_rmse_mv": 0.55,
            "raw_soma_rmse_mv": 1.92,
            "near_spike_raw_rmse_mv": 28.2,
            "near_spike_clipped_rmse_mv": 3.1,
            "clipped_fraction": 0.002,
            "near_spike_clipped_fraction": 0.52,
            "distortion_max_mv": 50.0,
        },
        {
            "model_id": "branch_elm_best",
            "regime": "ood_rates",
            "morphology_id": "hay",
            "source_file": "valid_1.npz",
            "spike_auc": 0.990,
            "val_threshold_event_f1_2ms": 0.37,
            "regime_prevalence_event_f1_2ms": 0.36,
            "clipped_rmse_mv": 0.85,
            "raw_soma_rmse_mv": 2.4,
            "intervention": "rate_shift",
        },
    ]
    local_rows = [
        {
            "model_id": "branch_elm_m100",
            "kind": "local_linearity",
            "direction": "gradient_sign",
            "sign_flip_mean": 0.29,
        },
        {
            "model_id": "branch_elm_m30",
            "kind": "local_linearity",
            "direction": "random_gaussian",
            "sign_flip_mean": 0.49,
        },
    ]
    perturbation_rows = [
        {"kind": "targeted_event_edit", "soma_delta_mv": 4.0},
        {"kind": "random_event_edit", "soma_delta_mv": 0.5},
    ]
    pretrained_rows = [
        {"model_id": "tcn_dwt7", "raw_soma_rmse_mv": 91.0, "spike_auc": 0.47},
        {"model_id": "branch_elm_best", "soma_rmse_mv": 3.4, "spike_auc": 0.99},
    ]

    rows = diagnostic_audit_rows(
        teacher_grounded_rows=teacher_rows,
        local_stability_rows=local_rows,
        perturbation_rows=perturbation_rows,
        pretrained_summary_rows=pretrained_rows,
        teacher_protocol={
            "source_dataset": "NeuronIO",
            "cell_template": "Hay L5PC",
            "morphology_count": 1,
        },
        context=DiagnosticContext(
            suites=(
                "metric-decoupling",
                "target-view-audit",
                "robustness-directionality",
                "protocol-coverage",
                "adapter-fairness",
                "paper-summary",
            ),
            model_id="analysis",
            teacher_id="neuronio",
            morphology_id="mixed",
        ),
    )

    validate_diagnostic_rows(rows)
    diagnostics = {row["diagnostic"] for row in rows}
    assert diagnostics == {
        "adapter_fairness",
        "metric_decoupling",
        "protocol_coverage",
        "robustness_directionality",
        "target_view_audit",
    }
    assert all("point" not in row["diagnostic"] for row in rows)

    metrics = {(row["diagnostic"], row["metric"]): row for row in rows}
    assert metrics[("metric_decoupling", "zero_threshold_predicted_fraction")]["value"] > 0.0
    assert metrics[("metric_decoupling", "ood_rmse_delta_mv")]["value"] == pytest.approx(0.3)
    assert metrics[("target_view_audit", "near_spike_raw_rmse_mv_mean")]["value"] == 28.2
    assert metrics[("robustness_directionality", "targeted_random_delta_ratio")]["value"] == 8.0
    assert metrics[("protocol_coverage", "teacher_grounded_row_count")]["value"] == 2
    assert metrics[("adapter_fairness", "large_scale_mismatch_models")]["value"] == json.dumps(["tcn_dwt7"])


def test_artifact_audits_can_load_jsonl_sources(tmp_path):
    teacher_path = tmp_path / "teacher.jsonl"
    teacher_path.write_text(
        json.dumps(
            {
                "model_id": "branch_elm_best",
                "regime": "canonical_test",
                "spike_auc": 0.995,
                "event_f1_2ms": 0.34,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = diagnostic_audit_rows_from_sources(
        DiagnosticAuditSources(teacher_grounded_rows=teacher_path),
        context=DiagnosticContext(suites=("metric-decoupling",), model_id="analysis"),
    )

    validate_diagnostic_rows(rows)
    assert {row["diagnostic"] for row in rows} == {"metric_decoupling"}
    assert any(row["metric"] == "spike_auc_mean" and row["value"] == 0.995 for row in rows)


def test_adapter_fairness_rows_are_available_independently():
    rows = adapter_fairness_diagnostic_rows(
        [
            {"model_id": "tcn_dwt8", "soma_rmse_mv": 129.0, "spike_auc": 0.5},
            {"model_id": "branch_elm_m30", "soma_rmse_mv": 3.5, "spike_auc": 0.99},
        ],
        context=DiagnosticContext(suites=("adapter-fairness",), model_id="analysis"),
    )

    validate_diagnostic_rows(rows)
    metrics = {row["metric"]: row["value"] for row in rows}
    assert metrics["tcn_model_count"] == 1
    assert metrics["official_model_count"] == 1
    assert metrics["tcn_rmse_mv_max"] == 129.0
