"""Audit-style diagnostic rows derived from benchmark artifacts.

These helpers turn comparison artifacts into the same dataframe-compatible row
schema as the rest of the benchmark. They deliberately use neutral diagnostic
families rather than paper-claim labels, so downstream reports can combine them
with trace, spike, state, morphology, and intervention diagnostics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dendrobench.benchmark.dataset_schema import default_diagnostic_specs
from dendrobench.benchmark.diagnostic_rows import DiagnosticContext, validate_diagnostic_rows


@dataclass(frozen=True)
class DiagnosticAuditSources:
    """Optional artifact sources for derived benchmark audit diagnostics."""

    teacher_grounded_rows: Path | None = None
    local_stability_rows: Path | None = None
    perturbation_rows: Path | None = None
    pretrained_summary_rows: Path | None = None
    teacher_protocol: Path | None = None


def load_artifact_rows(path: str | Path | None) -> list[dict[str, Any]]:
    """Load row-like JSON, JSONL, or CSV artifacts."""
    if path is None:
        return []
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("rows", "summary_rows", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        raise ValueError(f"unsupported JSON row payload in {source}")
    if source.suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("artifact row path must end in .json, .jsonl, or .csv")


def diagnostic_audit_rows_from_sources(
    sources: DiagnosticAuditSources,
    *,
    context: DiagnosticContext | None = None,
) -> list[dict[str, Any]]:
    """Load artifact sources and emit all derived audit diagnostics."""
    teacher_grounded_rows = load_artifact_rows(sources.teacher_grounded_rows)
    local_stability_rows = load_artifact_rows(sources.local_stability_rows)
    perturbation_rows = load_artifact_rows(sources.perturbation_rows)
    pretrained_summary_rows = load_artifact_rows(sources.pretrained_summary_rows)
    teacher_protocol = _load_protocol(sources.teacher_protocol)
    return diagnostic_audit_rows(
        teacher_grounded_rows=teacher_grounded_rows,
        local_stability_rows=local_stability_rows,
        perturbation_rows=perturbation_rows,
        pretrained_summary_rows=pretrained_summary_rows,
        teacher_protocol=teacher_protocol,
        context=context,
    )


def diagnostic_audit_rows(
    *,
    teacher_grounded_rows: Iterable[dict[str, Any]] = (),
    local_stability_rows: Iterable[dict[str, Any]] = (),
    perturbation_rows: Iterable[dict[str, Any]] = (),
    pretrained_summary_rows: Iterable[dict[str, Any]] = (),
    teacher_protocol: dict[str, Any] | None = None,
    context: DiagnosticContext | None = None,
) -> list[dict[str, Any]]:
    """Return all derived audit diagnostics in the standard row schema."""
    context = context or DiagnosticContext(suites=("paper-summary",), model_id="analysis", morphology_id="mixed")
    teacher_grounded_rows = list(teacher_grounded_rows)
    local_stability_rows = list(local_stability_rows)
    perturbation_rows = list(perturbation_rows)
    pretrained_summary_rows = list(pretrained_summary_rows)

    rows: list[dict[str, Any]] = []
    rows.extend(
        metric_decoupling_diagnostic_rows(
            teacher_grounded_rows,
            local_stability_rows=local_stability_rows,
            context=context,
        )
    )
    rows.extend(
        target_view_audit_diagnostic_rows(
            teacher_grounded_rows,
            teacher_protocol=teacher_protocol,
            context=context,
        )
    )
    rows.extend(
        robustness_directionality_diagnostic_rows(
            list(teacher_grounded_rows) + list(local_stability_rows) + list(perturbation_rows),
            context=context,
        )
    )
    rows.extend(
        protocol_coverage_diagnostic_rows(
            teacher_grounded_rows,
            local_stability_rows=local_stability_rows,
            perturbation_rows=perturbation_rows,
            teacher_protocol=teacher_protocol,
            context=context,
        )
    )
    rows.extend(adapter_fairness_diagnostic_rows(pretrained_summary_rows, context=context))
    validate_diagnostic_rows(rows)
    return rows


def metric_decoupling_diagnostic_rows(
    teacher_grounded_rows: Iterable[dict[str, Any]],
    *,
    local_stability_rows: Iterable[dict[str, Any]] = (),
    context: DiagnosticContext | None = None,
) -> list[dict[str, Any]]:
    """Summarize cases where headline metrics and behavioral metrics separate."""
    context = context or DiagnosticContext(suites=("paper-summary",), model_id="analysis", morphology_id="mixed")
    rows = [dict(row) for row in teacher_grounded_rows]
    stability = [dict(row) for row in local_stability_rows]
    out: list[dict[str, Any]] = []

    auc = _numeric_values(rows, "spike_auc")
    event_f1 = _numeric_values(rows, "val_threshold_event_f1_2ms", "event_f1_2ms", "f1_at_2steps")
    prevalence_f1 = _numeric_values(rows, "regime_prevalence_event_f1_2ms", "prevalence_event_f1_2ms")
    zero_pred = _numeric_values(rows, "zero_threshold_predicted_spikes", "zero_predicted_spikes")
    target_spikes = _numeric_values(rows, "target_spikes", "target_spike_count")

    out.extend(_summary_rows("metric_decoupling", context, "spike_auc", auc, unit="fraction"))
    out.extend(_summary_rows("metric_decoupling", context, "event_f1_2ms", event_f1, unit="fraction"))
    if event_f1 and prevalence_f1:
        deltas = [prevalence_f1[i] - event_f1[i] for i in range(min(len(event_f1), len(prevalence_f1)))]
        out.extend(_summary_rows("metric_decoupling", context, "prevalence_minus_validation_f1_2ms", deltas, unit="fraction"))
    if zero_pred and target_spikes:
        out.append(_row("metric_decoupling", context, "zero_threshold_predicted_spikes_total", sum(zero_pred), unit="count", n=len(zero_pred)))
        out.append(_row("metric_decoupling", context, "target_spikes_total", sum(target_spikes), unit="count", n=len(target_spikes)))
        out.append(
            _row(
                "metric_decoupling",
                context,
                "zero_threshold_predicted_fraction",
                sum(zero_pred) / max(1.0, sum(target_spikes)),
                unit="fraction",
                n=len(zero_pred),
            )
        )

    by_model_regime = _rows_by_model_regime(rows)
    for model_id, regimes in sorted(by_model_regime.items()):
        canonical = regimes.get("canonical_test") or regimes.get("canonical") or regimes.get("val")
        for regime_name, regime_row in sorted(regimes.items()):
            if canonical is None or regime_row is canonical or not regime_name:
                continue
            base_rmse = _to_float(canonical.get("clipped_rmse_mv") or canonical.get("soma_rmse_mv"))
            test_rmse = _to_float(regime_row.get("clipped_rmse_mv") or regime_row.get("soma_rmse_mv"))
            base_auc = _to_float(canonical.get("spike_auc"))
            test_auc = _to_float(regime_row.get("spike_auc"))
            row_context = _replace_context(context, model_id=str(model_id), condition=str(regime_name))
            if base_rmse is not None and test_rmse is not None:
                out.append(_row("metric_decoupling", row_context, "ood_rmse_delta_mv", test_rmse - base_rmse, unit="mV", n=1))
            if base_auc is not None and test_auc is not None:
                out.append(_row("metric_decoupling", row_context, "ood_auc_delta", test_auc - base_auc, unit="fraction", n=1))

    smoothest = _best_by_metric(stability, "sign_flip_mean", lower_is_better=True)
    best_ood = _best_by_metric([row for row in rows if "ood" in str(row.get("regime", row.get("condition", "")))], "clipped_rmse_mv", lower_is_better=True)
    if smoothest is not None:
        out.append(_row("metric_decoupling", context, "smoothest_model_by_sign_flip", smoothest, unit="label", n=len(stability)))
    if best_ood is not None:
        out.append(_row("metric_decoupling", context, "best_ood_rate_rmse_model", best_ood, unit="label", n=len(rows)))
    return _filter(out, context.suites)


def target_view_audit_diagnostic_rows(
    teacher_grounded_rows: Iterable[dict[str, Any]],
    *,
    teacher_protocol: dict[str, Any] | None = None,
    context: DiagnosticContext | None = None,
) -> list[dict[str, Any]]:
    """Summarize how target views change measured voltage fidelity."""
    context = context or DiagnosticContext(suites=("paper-summary",), model_id="analysis", morphology_id="mixed")
    rows = [dict(row) for row in teacher_grounded_rows]
    out: list[dict[str, Any]] = []
    out.extend(_summary_rows("target_view_audit", context, "clipped_rmse_mv", _numeric_values(rows, "clipped_rmse_mv"), unit="mV"))
    out.extend(_summary_rows("target_view_audit", context, "raw_soma_rmse_mv", _numeric_values(rows, "raw_soma_rmse_mv", "raw_rmse_mv"), unit="mV"))
    out.extend(_summary_rows("target_view_audit", context, "near_spike_raw_rmse_mv", _numeric_values(rows, "near_spike_raw_rmse_mv"), unit="mV"))
    out.extend(_summary_rows("target_view_audit", context, "near_spike_clipped_rmse_mv", _numeric_values(rows, "near_spike_clipped_rmse_mv"), unit="mV"))

    for metric, unit in (
        ("clipped_fraction", "fraction"),
        ("near_spike_clipped_fraction", "fraction"),
        ("near_spike_distortion_rmse_mv", "mV"),
        ("distortion_max_mv", "mV"),
        ("raw_soma_max_mv", "mV"),
    ):
        values = _numeric_values(rows, metric)
        if not values and teacher_protocol:
            values = _numeric_values([teacher_protocol], metric)
        out.extend(_summary_rows("target_view_audit", context, metric, values, unit=unit))

    if teacher_protocol:
        for metric in ("source_dataset", "target_transform", "cell_template", "morphology_policy"):
            value = teacher_protocol.get(metric)
            if value not in (None, ""):
                out.append(_row("target_view_audit", context, metric, str(value), unit="label", n=1))
    return _filter(out, context.suites)


def robustness_directionality_diagnostic_rows(
    source_rows: Iterable[dict[str, Any]],
    *,
    context: DiagnosticContext | None = None,
) -> list[dict[str, Any]]:
    """Summarize perturbation direction sensitivity and sparse-event edits."""
    context = context or DiagnosticContext(suites=("paper-summary",), model_id="analysis", morphology_id="mixed")
    rows = [dict(row) for row in source_rows]
    out: list[dict[str, Any]] = []

    targeted = []
    random = []
    for row in rows:
        kind = str(row.get("kind", row.get("condition", "")))
        direction = str(row.get("direction", row.get("edit_kind", "")))
        if "target" in kind or "target" in direction:
            targeted.append(row)
        elif "random" in kind or "random" in direction:
            random.append(row)
    targeted_delta = _numeric_values(targeted, "soma_delta_mv", "mean_abs_soma_delta_mv", "spike_mass_delta")
    random_delta = _numeric_values(random, "soma_delta_mv", "mean_abs_soma_delta_mv", "spike_mass_delta")
    out.extend(_summary_rows("robustness_directionality", context, "targeted_delta", targeted_delta, unit="unitless"))
    out.extend(_summary_rows("robustness_directionality", context, "random_delta", random_delta, unit="unitless"))
    if targeted_delta and random_delta:
        out.append(
            _row(
                "robustness_directionality",
                context,
                "targeted_random_delta_ratio",
                float(np.mean(np.abs(targeted_delta))) / max(float(np.mean(np.abs(random_delta))), 1e-12),
                unit="ratio",
                n=min(len(targeted_delta), len(random_delta)),
            )
        )

    gradient = [row for row in rows if "gradient" in str(row.get("direction", row.get("kind", "")))]
    random_local = [row for row in rows if "random" in str(row.get("direction", row.get("kind", "")))]
    gradient_values = _numeric_values(gradient, "sign_flip_mean", "linearity_error_mean", "output_delta_mean")
    random_values = _numeric_values(random_local, "sign_flip_mean", "linearity_error_mean", "output_delta_mean")
    out.extend(_summary_rows("robustness_directionality", context, "gradient_direction_change", gradient_values, unit="unitless"))
    out.extend(_summary_rows("robustness_directionality", context, "random_direction_change", random_values, unit="unitless"))
    if gradient_values and random_values:
        out.append(
            _row(
                "robustness_directionality",
                context,
                "gradient_random_change_ratio",
                float(np.mean(np.abs(gradient_values))) / max(float(np.mean(np.abs(random_values))), 1e-12),
                unit="ratio",
                n=min(len(gradient_values), len(random_values)),
            )
        )
    return _filter(out, context.suites)


def protocol_coverage_diagnostic_rows(
    teacher_grounded_rows: Iterable[dict[str, Any]],
    *,
    local_stability_rows: Iterable[dict[str, Any]] = (),
    perturbation_rows: Iterable[dict[str, Any]] = (),
    teacher_protocol: dict[str, Any] | None = None,
    context: DiagnosticContext | None = None,
) -> list[dict[str, Any]]:
    """Report which diagnostics are teacher-grounded and morphology-aware."""
    context = context or DiagnosticContext(suites=("paper-summary",), model_id="analysis", morphology_id="mixed")
    teacher_rows = [dict(row) for row in teacher_grounded_rows]
    local_rows = [dict(row) for row in local_stability_rows]
    perturb_rows = [dict(row) for row in perturbation_rows]
    morphologies = _unique_nonempty(
        row.get("morphology_id") or row.get("morphology") or row.get("specimen_id")
        for row in teacher_rows + perturb_rows
    )
    source_files = _unique_nonempty(
        row.get("source_file") or row.get("source_path") or row.get("file")
        for row in teacher_rows
    )
    intervention_rows = [
        row
        for row in teacher_rows + perturb_rows
        if row.get("intervention") or "intervention" in str(row.get("condition", ""))
    ]
    out = [
        _row("protocol_coverage", context, "teacher_grounded_row_count", len(teacher_rows), unit="count", n=len(teacher_rows)),
        _row("protocol_coverage", context, "model_only_row_count", len(local_rows), unit="count", n=len(local_rows)),
        _row("protocol_coverage", context, "perturbation_row_count", len(perturb_rows), unit="count", n=len(perturb_rows)),
        _row("protocol_coverage", context, "teacher_grounded_intervention_row_count", len(intervention_rows), unit="count", n=len(intervention_rows)),
        _row("protocol_coverage", context, "morphology_count", len(morphologies), unit="count", n=len(morphologies)),
        _row("protocol_coverage", context, "source_file_count", len(source_files), unit="count", n=len(source_files)),
    ]
    if teacher_protocol:
        for metric in ("source_dataset", "cell_template", "neuron_model", "morphology_count", "intervention_policy"):
            value = teacher_protocol.get(metric)
            if value not in (None, ""):
                out.append(_row("protocol_coverage", context, metric, value, unit="label", n=1))
    return _filter(out, context.suites)


def adapter_fairness_diagnostic_rows(
    pretrained_summary_rows: Iterable[dict[str, Any]],
    *,
    context: DiagnosticContext | None = None,
) -> list[dict[str, Any]]:
    """Summarize adapter and preprocessing sanity for literature-model runs."""
    context = context or DiagnosticContext(suites=("paper-summary",), model_id="analysis", morphology_id="mixed")
    rows = [dict(row) for row in pretrained_summary_rows]
    out: list[dict[str, Any]] = []
    tcn_rows = [row for row in rows if "tcn" in str(row.get("model_id", row.get("model", ""))).lower()]
    official_rows = [
        row
        for row in rows
        if "tcn" not in str(row.get("model_id", row.get("model", ""))).lower()
        and str(row.get("model_id", row.get("model", ""))).strip()
    ]
    out.append(_row("adapter_fairness", context, "tcn_model_count", len(_unique_nonempty(_model_ids(tcn_rows))), unit="count", n=len(tcn_rows)))
    out.append(_row("adapter_fairness", context, "official_model_count", len(_unique_nonempty(_model_ids(official_rows))), unit="count", n=len(official_rows)))
    out.extend(_summary_rows("adapter_fairness", context, "tcn_rmse_mv", _numeric_values(tcn_rows, "soma_rmse_mv", "raw_soma_rmse_mv"), unit="mV"))
    out.extend(_summary_rows("adapter_fairness", context, "official_rmse_mv", _numeric_values(official_rows, "soma_rmse_mv", "raw_soma_rmse_mv"), unit="mV"))
    out.extend(_summary_rows("adapter_fairness", context, "tcn_auc", _numeric_values(tcn_rows, "spike_auc"), unit="fraction"))
    suspect = [
        str(row.get("model_id", row.get("model", "")))
        for row in tcn_rows
        if (_to_float(row.get("soma_rmse_mv")) or _to_float(row.get("raw_soma_rmse_mv")) or 0.0) > 50.0
    ]
    if suspect:
        out.append(_row("adapter_fairness", context, "large_scale_mismatch_models", json.dumps(sorted(set(suspect))), unit="json", n=len(suspect)))
    return _filter(out, context.suites)


def _load_protocol(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("teacher protocol artifact must be a JSON object")
    return payload


def _rows_by_model_regime(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        model = str(row.get("model_id", row.get("model", "")))
        regime = str(row.get("regime", row.get("condition", "")))
        if model:
            grouped[model][regime] = row
    return grouped


def _summary_rows(
    diagnostic: str,
    context: DiagnosticContext,
    metric: str,
    values: list[float],
    *,
    unit: str,
) -> list[dict[str, Any]]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    return [
        _row(diagnostic, context, f"{metric}_min", float(np.min(arr)), unit=unit, n=len(values)),
        _row(diagnostic, context, f"{metric}_max", float(np.max(arr)), unit=unit, n=len(values)),
        _row(diagnostic, context, f"{metric}_mean", float(np.mean(arr)), unit=unit, n=len(values)),
    ]


def _numeric_values(rows: Iterable[dict[str, Any]], *keys: str) -> list[float]:
    values = []
    for row in rows:
        for key in keys:
            value = _to_float(row.get(key))
            if value is not None:
                values.append(value)
                break
    return values


def _best_by_metric(rows: Iterable[dict[str, Any]], key: str, *, lower_is_better: bool) -> str | None:
    candidates = []
    for row in rows:
        value = _to_float(row.get(key))
        model = row.get("model_id") or row.get("model")
        if value is not None and model:
            candidates.append((value, str(model)))
    if not candidates:
        return None
    candidates.sort(reverse=not lower_is_better)
    return candidates[0][1]


def _model_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [str(row.get("model_id", row.get("model", ""))) for row in rows if row.get("model_id") or row.get("model")]


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _filter(rows: list[dict[str, Any]], suites: tuple[str, ...]) -> list[dict[str, Any]]:
    specs = {spec.name: set(spec.suites) for spec in default_diagnostic_specs()}
    filtered = []
    for row in rows:
        diagnostic_suites = specs.get(str(row["diagnostic"]), set())
        row_suites = tuple(suite for suite in suites if suite in diagnostic_suites)
        if row_suites:
            item = dict(row)
            item["suite"] = "|".join(row_suites)
            filtered.append(item)
    validate_diagnostic_rows(filtered)
    return filtered


def _row(
    diagnostic: str,
    context: DiagnosticContext,
    metric: str,
    value: Any,
    *,
    state: str | None = None,
    unit: str = "unitless",
    n: int | None = None,
) -> dict[str, Any]:
    return {
        "diagnostic": diagnostic,
        "suite": "",
        "model_id": context.model_id,
        "teacher_id": context.teacher_id,
        "morphology_id": context.morphology_id,
        "split": context.split,
        "condition": context.condition,
        "intervention": context.intervention,
        "trace_id": context.trace_id,
        "state": state,
        "metric": metric,
        "value": _json_scalar(value),
        "unit": unit,
        "n": n,
        "target_view": context.target_view,
        "ignore_start": context.ignore_start,
        "calibrated_spike_threshold": context.calibrated_spike_threshold,
    }


def _replace_context(context: DiagnosticContext, **updates: Any) -> DiagnosticContext:
    values = {
        "suites": context.suites,
        "model_id": context.model_id,
        "teacher_id": context.teacher_id,
        "morphology_id": context.morphology_id,
        "split": context.split,
        "condition": context.condition,
        "intervention": context.intervention,
        "trace_id": context.trace_id,
        "target_view": context.target_view,
        "soma_scale": context.soma_scale,
        "soma_bias_mv": context.soma_bias_mv,
        "ignore_start": context.ignore_start,
        "calibrated_spike_threshold": context.calibrated_spike_threshold,
    }
    values.update(updates)
    return DiagnosticContext(**values)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
