"""NEURO-BENCH: neuron surrogate benchmark utilities.

The package exports the public benchmark helpers lazily so lightweight
dataframe/TCN evaluation can run in environments that intentionally omit
PyTorch.
"""

from __future__ import annotations

from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "compute_regime_stratified_metrics": ("dendrobench.benchmark.regime_stratified", "compute_regime_stratified_metrics"),
    "compute_state_conditioned_metrics": ("dendrobench.benchmark.state_metrics", "compute_state_conditioned_metrics"),
    "build_neuron_state_masks": ("dendrobench.benchmark.state_metrics", "build_neuron_state_masks"),
    "compute_trace_shape_metrics": ("dendrobench.benchmark.trace_shape", "compute_trace_shape_metrics"),
    "compute_perturbation_stability": ("dendrobench.benchmark.perturbation_stability", "compute_perturbation_stability"),
    "compute_paired_event_intervention_metrics": (
        "dendrobench.benchmark.structured_intervention",
        "compute_paired_event_intervention_metrics",
    ),
    "make_event_dropout_inputs": ("dendrobench.benchmark.structured_intervention", "make_event_dropout_inputs"),
    "make_selective_event_dropout_inputs": (
        "dendrobench.benchmark.structured_intervention",
        "make_selective_event_dropout_inputs",
    ),
    "make_site_silence_inputs": ("dendrobench.benchmark.structured_intervention", "make_site_silence_inputs"),
    "make_structured_intervention_inputs": (
        "dendrobench.benchmark.structured_intervention",
        "make_structured_intervention_inputs",
    ),
    "make_temporal_jitter_inputs": ("dendrobench.benchmark.structured_intervention", "make_temporal_jitter_inputs"),
    "MorphologyTransferEvaluator": ("dendrobench.benchmark.morphology_transfer", "MorphologyTransferEvaluator"),
    "load_available_morphologies": ("dendrobench.benchmark.morphology_transfer", "load_available_morphologies"),
    "adapt_branch_count": ("dendrobench.benchmark.branch_adapter", "adapt_branch_count"),
    "evaluate_with_branch_adaptation": ("dendrobench.benchmark.branch_adapter", "evaluate_with_branch_adaptation"),
    "BenchmarkProfile": ("dendrobench.benchmark.profiles", "BenchmarkProfile"),
    "BenchmarkCostModel": ("dendrobench.benchmark.profiles", "BenchmarkCostModel"),
    "LOCAL_CORENEURON_HAY_COST_MODEL": ("dendrobench.benchmark.profiles", "LOCAL_CORENEURON_HAY_COST_MODEL"),
    "build_axis_plan": ("dendrobench.benchmark.profiles", "build_axis_plan"),
    "estimate_profile_cost": ("dendrobench.benchmark.profiles", "estimate_profile_cost"),
    "get_benchmark_profile": ("dendrobench.benchmark.profiles", "get_benchmark_profile"),
    "list_benchmark_profiles": ("dendrobench.benchmark.profiles", "list_benchmark_profiles"),
    "DIAGNOSTIC_SUITES": ("dendrobench.benchmark.dataset_schema", "DIAGNOSTIC_SUITES"),
    "INTERVENTION_CONDITIONS": ("dendrobench.benchmark.dataset_schema", "INTERVENTION_CONDITIONS"),
    "ORDINARY_SPLITS": ("dendrobench.benchmark.dataset_schema", "ORDINARY_SPLITS"),
    "V1_MORPHOLOGY_COUNT": ("dendrobench.benchmark.dataset_schema", "V1_MORPHOLOGY_COUNT"),
    "build_diagnostic_suite_catalog": ("dendrobench.benchmark.dataset_schema", "build_diagnostic_suite_catalog"),
    "build_v1_dataset_manifest_template": ("dendrobench.benchmark.dataset_schema", "build_v1_dataset_manifest_template"),
    "default_diagnostic_specs": ("dendrobench.benchmark.dataset_schema", "default_diagnostic_specs"),
    "default_target_views": ("dendrobench.benchmark.dataset_schema", "default_target_views"),
    "provisional_v1_morphologies": ("dendrobench.benchmark.dataset_schema", "provisional_v1_morphologies"),
    "validate_v1_manifest_template": ("dendrobench.benchmark.dataset_schema", "validate_v1_manifest_template"),
    "DiagnosticContext": ("dendrobench.benchmark.diagnostic_rows", "DiagnosticContext"),
    "evaluate_model_diagnostic_rows": ("dendrobench.benchmark.diagnostic_rows", "evaluate_model_diagnostic_rows"),
    "morphology_contrast_diagnostic_rows": ("dendrobench.benchmark.diagnostic_rows", "morphology_contrast_diagnostic_rows"),
    "morphology_routing_qc_rows": ("dendrobench.benchmark.diagnostic_rows", "morphology_routing_qc_rows"),
    "paired_intervention_diagnostic_rows": ("dendrobench.benchmark.diagnostic_rows", "paired_intervention_diagnostic_rows"),
    "teacher_qc_diagnostic_rows": ("dendrobench.benchmark.diagnostic_rows", "teacher_qc_diagnostic_rows"),
    "validate_diagnostic_rows": ("dendrobench.benchmark.diagnostic_rows", "validate_diagnostic_rows"),
    "write_diagnostic_rows": ("dendrobench.benchmark.diagnostic_rows", "write_diagnostic_rows"),
    "DiagnosticAuditSources": ("dendrobench.benchmark.diagnostic_audits", "DiagnosticAuditSources"),
    "adapter_fairness_diagnostic_rows": ("dendrobench.benchmark.diagnostic_audits", "adapter_fairness_diagnostic_rows"),
    "diagnostic_audit_rows": ("dendrobench.benchmark.diagnostic_audits", "diagnostic_audit_rows"),
    "diagnostic_audit_rows_from_sources": ("dendrobench.benchmark.diagnostic_audits", "diagnostic_audit_rows_from_sources"),
    "load_artifact_rows": ("dendrobench.benchmark.diagnostic_audits", "load_artifact_rows"),
    "metric_decoupling_diagnostic_rows": ("dendrobench.benchmark.diagnostic_audits", "metric_decoupling_diagnostic_rows"),
    "protocol_coverage_diagnostic_rows": ("dendrobench.benchmark.diagnostic_audits", "protocol_coverage_diagnostic_rows"),
    "robustness_directionality_diagnostic_rows": (
        "dendrobench.benchmark.diagnostic_audits",
        "robustness_directionality_diagnostic_rows",
    ),
    "target_view_audit_diagnostic_rows": ("dendrobench.benchmark.diagnostic_audits", "target_view_audit_diagnostic_rows"),
    "DatasetSelector": ("dendrobench.benchmark.selectors", "DatasetSelector"),
    "V1DatasetLayout": ("dendrobench.benchmark.selectors", "V1DatasetLayout"),
    "load_intervention_pair_npz": ("dendrobench.benchmark.runner", "load_intervention_pair_npz"),
    "load_selector_batch": ("dendrobench.benchmark.runner", "load_selector_batch"),
    "run_model_diagnostics_on_selector": ("dendrobench.benchmark.runner", "run_model_diagnostics_on_selector"),
    "run_paired_intervention_on_npz": ("dendrobench.benchmark.runner", "run_paired_intervention_on_npz"),
    "run_paired_intervention_on_selector": ("dendrobench.benchmark.runner", "run_paired_intervention_on_selector"),
    "run_teacher_qc_on_selector": ("dendrobench.benchmark.runner", "run_teacher_qc_on_selector"),
    "summarize_diagnostic_rows": ("dendrobench.benchmark.reports", "summarize_diagnostic_rows"),
    "write_diagnostic_summary": ("dendrobench.benchmark.reports", "write_diagnostic_summary"),
    "run_v1_benchmark_bundle": ("dendrobench.benchmark.bundle", "run_v1_benchmark_bundle"),
    "infer_pair_metadata": ("dendrobench.benchmark.bundle", "infer_pair_metadata"),
    "BenchmarkSuite": ("dendrobench.benchmark.suite", "BenchmarkSuite"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
