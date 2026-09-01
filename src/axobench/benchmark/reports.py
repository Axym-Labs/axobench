"""Compact reports for dataframe-first benchmark rows."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from axobench.benchmark.diagnostic_rows import validate_diagnostic_rows


def summarize_diagnostic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact JSON-compatible summary of diagnostic rows."""
    validate_diagnostic_rows(rows)
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "diagnostics": sorted({str(row["diagnostic"]) for row in rows}),
        "suites": sorted(
            {suite for row in rows for suite in str(row["suite"]).split("|") if suite}
        ),
        "models": sorted({str(row["model_id"]) for row in rows}),
        "teachers": sorted({str(row["teacher_id"]) for row in rows}),
        "morphologies": sorted({str(row["morphology_id"]) for row in rows}),
        "conditions": sorted({str(row["condition"]) for row in rows}),
        "trace_ids": _sample_trace_ids(rows),
        "numeric_metrics": _numeric_metric_summaries(rows),
        "label_metrics": _label_metric_summaries(rows),
    }
    return summary


def write_diagnostic_summary(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write a compact Markdown or JSON summary for diagnostic rows."""
    summary = summarize_diagnostic_rows(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    if output.suffix == ".md":
        output.write_text(_summary_markdown(summary), encoding="utf-8")
        return
    raise ValueError("diagnostic summary output path must end in .json or .md")


def core_metric_comparison_rows(
    models: list[dict[str, Any]],
    old_metric_set: dict[str, dict[str, Any]] | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Combine old RMSE/AUC rows with the promoted core metric set.

    The helper is intentionally model-agnostic: callers pass the new metric
    summaries produced by AxoBench and, when available, a mapping or list of
    old compatibility metrics keyed by ``model_id``.  Inflation factors are
    computed as new error divided by old clipped RMSE for voltage metrics, and
    old AUC divided by F1 for spike metrics.
    """
    old_by_model = _old_metric_lookup(old_metric_set)
    rows = []
    for model in models:
        model_id = str(model.get("model_id", ""))
        old = old_by_model.get(model_id, {})
        old_rmse = _first_number(
            old, "clipped_rmse_mv", "old_clipped_rmse_mv", "soma_RMSE", "soma_rmse_mv"
        )
        old_auc = _first_number(old, "spike_auc", "old_spike_auc", "AUC", "auc")
        voltage_root_sera = _as_optional_float(model.get("voltage_root_sera_mv"))
        voltage_rmse = _as_optional_float(model.get("voltage_rmse_mv"))
        exact_f1 = _as_optional_float(
            model.get("exact_spike_f1", model.get("spike_f1_area"))
        )
        f1_area = _as_optional_float(model.get("spike_f1_area_0_5ms"))
        rows.append(
            {
                "model_id": model_id,
                "n_parameters": model.get("n_parameters"),
                "old_clipped_rmse_mv": old_rmse,
                "old_spike_auc": old_auc,
                "voltage_root_sera_mv": voltage_root_sera,
                "voltage_root_sera_inflation": _ratio(
                    voltage_root_sera,
                    old_rmse,
                ),
                "raw_voltage_rmse_mv": voltage_rmse,
                "raw_voltage_rmse_inflation": _ratio(voltage_rmse, old_rmse),
                "exact_spike_f1": exact_f1,
                "auc_over_exact_f1": _ratio(old_auc, exact_f1),
                "spike_f1_area_0_5ms": f1_area,
                "auc_over_f1_area_0_5ms": _ratio(old_auc, f1_area),
                "mean_response_root_sera_mv": _as_optional_float(
                    model.get("mean_response_root_sera_mv")
                ),
            }
        )
    return rows


def write_core_metric_comparison_table(
    models: list[dict[str, Any]],
    path: str | Path,
    old_metric_set: dict[str, dict[str, Any]] | list[dict[str, Any]] | None = None,
) -> None:
    """Write a Markdown old-vs-new core-metric comparison table."""
    rows = core_metric_comparison_rows(models, old_metric_set)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        "model",
        "old clipped RMSE mV",
        "old spike AUC",
        "Voltage Root-SERA mV (x old RMSE)",
        "raw RMSE mV (x old RMSE)",
        "exact F1 (AUC/F1)",
        "F1 area 0-5 ms (AUC/F1)",
        "Response Root-SERA mV",
    )
    table_rows = []
    for row in rows:
        table_rows.append(
            (
                row["model_id"],
                _format_optional(row["old_clipped_rmse_mv"]),
                _format_optional(row["old_spike_auc"]),
                _format_value_with_factor(
                    row["voltage_root_sera_mv"],
                    row["voltage_root_sera_inflation"],
                ),
                _format_value_with_factor(
                    row["raw_voltage_rmse_mv"], row["raw_voltage_rmse_inflation"]
                ),
                _format_value_with_factor(
                    row["exact_spike_f1"], row["auc_over_exact_f1"]
                ),
                _format_value_with_factor(
                    row["spike_f1_area_0_5ms"], row["auc_over_f1_area_0_5ms"]
                ),
                _format_optional(row["mean_response_root_sera_mv"]),
            )
        )
    _write_markdown_table(output, headers, table_rows)


def _numeric_metric_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row["value"]
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            grouped[
                (str(row["diagnostic"]), str(row["metric"]), str(row["unit"]))
            ].append(float(value))
    summaries = []
    for (diagnostic, metric, unit), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        summaries.append(
            {
                "diagnostic": diagnostic,
                "metric": metric,
                "unit": unit,
                "n_values": int(array.size),
                "mean": float(np.mean(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }
        )
    return summaries


def _old_metric_lookup(
    old_metric_set: dict[str, dict[str, Any]] | list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if old_metric_set is None:
        return {}
    if isinstance(old_metric_set, dict):
        return {
            str(model_id): dict(values) for model_id, values in old_metric_set.items()
        }
    lookup = {}
    for row in old_metric_set:
        model_id = row.get("model_id")
        if model_id is not None:
            lookup[str(model_id)] = dict(row)
    return lookup


def _first_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            value = _as_optional_float(row[key])
            if value is not None:
                return value
    return None


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e3):
        return f"{value:.3g}"
    return f"{value:.4g}"


def _format_value_with_factor(value: float | None, factor: float | None) -> str:
    if value is None:
        return "NA"
    if factor is None:
        return _format_optional(value)
    return f"{_format_optional(value)} ({factor:.1f}x)"


def _label_metric_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = row["value"]
        if value is None or isinstance(value, (int, float, bool)):
            continue
        grouped[(str(row["diagnostic"]), str(row["metric"]))][str(value)] += 1
    summaries = []
    for (diagnostic, metric), counter in sorted(grouped.items()):
        summaries.append(
            {
                "diagnostic": diagnostic,
                "metric": metric,
                "counts": dict(sorted(counter.items())),
            }
        )
    return summaries


def _sample_trace_ids(rows: list[dict[str, Any]], *, limit: int = 12) -> list[str]:
    seen = []
    for row in rows:
        trace_id = row.get("trace_id")
        if trace_id is None:
            continue
        trace_id = str(trace_id)
        if trace_id not in seen:
            seen.append(trace_id)
        if len(seen) >= limit:
            break
    return seen


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Diagnostic Summary",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- diagnostics: `{', '.join(summary['diagnostics'])}`",
        f"- suites: `{', '.join(summary['suites'])}`",
        f"- models: `{', '.join(summary['models'])}`",
        f"- morphologies: `{', '.join(summary['morphologies'])}`",
        f"- conditions: `{', '.join(summary['conditions'])}`",
    ]
    if summary["trace_ids"]:
        lines.append(f"- sample trace IDs: `{', '.join(summary['trace_ids'])}`")
    lines.extend(
        [
            "",
            "## Numeric Metrics",
            "",
            "| diagnostic | metric | unit | n | mean | min | max |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["numeric_metrics"]:
        lines.append(
            "| {diagnostic} | {metric} | {unit} | {n_values} | {mean:.6g} | {min:.6g} | {max:.6g} |".format(
                **item
            )
        )
    if summary["label_metrics"]:
        lines.extend(
            [
                "",
                "## Label Metrics",
                "",
                "| diagnostic | metric | counts |",
                "| --- | --- | --- |",
            ]
        )
        for item in summary["label_metrics"]:
            counts = ", ".join(
                f"{key}: {value}" for key, value in item["counts"].items()
            )
            lines.append(f"| {item['diagnostic']} | {item['metric']} | {counts} |")
    return "\n".join(lines) + "\n"


def _write_markdown_table(
    path: Path, headers: tuple[str, ...], rows: list[tuple[Any, ...]]
) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
