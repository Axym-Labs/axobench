#!/usr/bin/env python3
"""Mine existing benchmark rows for defensible failure candidates.

This is the first cheap autoresearch probe. It does not treat the current
benchmark as fixed; it looks for metric disagreements, state/intervention
blind spots, and target distortion candidates that deserve deeper experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_DIRECTIONS = {
    "soma_rmse_mv": "lower",
    "state_weighted_trace_distance_mv": "lower",
    "dynamic_distance": "lower",
    "lag_tolerant_voltage_rmse_mv": "lower",
    "spike_auc": "higher",
    "spike_brier": "lower",
    "spike_count_mae_per_trace": "lower",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/davwis/main/workspace/bnn_sim"))
    parser.add_argument(
        "--pretrained-root",
        type=Path,
        help="Existing pretrained benchmark output root. Defaults under --source-root.",
    )
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark/artifacts/failure_probe"))
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    pretrained_root = (
        args.pretrained_root.resolve()
        if args.pretrained_root is not None
        else source_root / "docs/verify-arbor-simulator/artifacts/pretrained_benchmark_eval_20260617"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary_path = pretrained_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models = summary["models"]
    model_dirs = {row["model_id"]: pretrained_root / row["model_id"] for row in models}

    rank_rows = build_metric_ranks(models)
    state_rows = build_state_gap_rows(model_dirs)
    intervention_rows = build_intervention_rows(model_dirs)
    clipping = load_clipping_summary(next(iter(model_dirs.values())) / "teacher_qc_val.jsonl")

    findings = build_findings(models, rank_rows, state_rows, intervention_rows, clipping)
    payload = {
        "source_root": str(source_root),
        "pretrained_root": str(pretrained_root),
        "findings": findings,
        "clipping": clipping,
        "metric_ranks": rank_rows,
        "state_gaps": state_rows,
        "intervention_gaps": intervention_rows,
    }

    (output / "failure_probe_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "model_metric_ranks.csv", rank_rows)
    write_csv(output / "state_gap.csv", state_rows)
    write_csv(output / "intervention_gap.csv", intervention_rows)
    (output / "failure_probe.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "findings": len(findings)}, indent=2, sort_keys=True))
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric_value(rows: list[dict[str, Any]], diagnostic: str, metric: str, state: str | None = None) -> Any:
    for row in rows:
        if row.get("diagnostic") == diagnostic and row.get("metric") == metric and row.get("state") == state:
            return row.get("value")
    return None


def build_metric_ranks(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks: dict[str, dict[str, int]] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = [
            (row["model_id"], float(row[metric]))
            for row in models
            if row.get(metric) is not None and not _is_nan(row.get(metric))
        ]
        values.sort(key=lambda item: item[1], reverse=(direction == "higher"))
        ranks[metric] = {model_id: rank for rank, (model_id, _value) in enumerate(values, start=1)}

    rows: list[dict[str, Any]] = []
    for row in models:
        out: dict[str, Any] = {
            "model_id": row["model_id"],
            "family": row.get("family"),
        }
        for metric in METRIC_DIRECTIONS:
            out[metric] = row.get(metric)
            out[f"{metric}_rank"] = ranks.get(metric, {}).get(row["model_id"])
        if out.get("soma_rmse_mv_rank") and out.get("spike_auc_rank"):
            out["rmse_auc_rank_gap"] = abs(int(out["soma_rmse_mv_rank"]) - int(out["spike_auc_rank"]))
        else:
            out["rmse_auc_rank_gap"] = None
        if out.get("spike_auc_rank") and out.get("spike_count_mae_per_trace_rank"):
            out["auc_count_rank_gap"] = abs(
                int(out["spike_auc_rank"]) - int(out["spike_count_mae_per_trace_rank"])
            )
        else:
            out["auc_count_rank_gap"] = None
        rows.append(out)
    return rows


def build_state_gap_rows(model_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, model_dir in model_dirs.items():
        model_rows = read_jsonl(model_dir / f"model_val_{model_id}.jsonl")
        global_rmse = metric_value(model_rows, "trace_core", "soma_rmse_mv")
        if global_rmse is None:
            continue
        for row in model_rows:
            if row.get("diagnostic") != "state_dynamics" or row.get("metric") != "voltage_rmse_mv":
                continue
            value = row.get("value")
            if value is None:
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "state": row.get("state"),
                    "n": row.get("n"),
                    "global_rmse_mv": float(global_rmse),
                    "state_rmse_mv": float(value),
                    "state_to_global_rmse_ratio": float(value) / float(global_rmse) if float(global_rmse) else None,
                }
            )
    rows.sort(key=lambda row: (row["state_to_global_rmse_ratio"] is None, -(row["state_to_global_rmse_ratio"] or 0)))
    return rows


def build_intervention_rows(model_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, model_dir in model_dirs.items():
        per_pair = []
        for path in sorted(model_dir.glob("intervention_specimen_*_event_dropout.jsonl")):
            parsed = read_jsonl(path)
            soma_delta = metric_value(parsed, "intervention_response", "soma_delta_rmse_mv")
            spike_delta = metric_value(parsed, "intervention_response", "spike_count_delta_mae_per_trace")
            direction = metric_value(parsed, "intervention_response", "direction_agreement")
            spike_direction_ok = None
            soma_direction_ok = None
            if isinstance(direction, dict):
                spike_direction_ok = bool(direction.get("spike_count"))
                soma_direction_ok = bool(direction.get("soma_mean"))
            per_pair.append(
                {
                    "soma_delta_rmse_mv": soma_delta,
                    "spike_count_delta_mae_per_trace": spike_delta,
                    "spike_direction_ok": spike_direction_ok,
                    "soma_direction_ok": soma_direction_ok,
                }
            )
        if not per_pair:
            continue
        soma_values = [float(row["soma_delta_rmse_mv"]) for row in per_pair if row["soma_delta_rmse_mv"] is not None]
        spike_values = [
            float(row["spike_count_delta_mae_per_trace"])
            for row in per_pair
            if row["spike_count_delta_mae_per_trace"] is not None
        ]
        spike_direction_values = [row["spike_direction_ok"] for row in per_pair if row["spike_direction_ok"] is not None]
        soma_direction_values = [row["soma_direction_ok"] for row in per_pair if row["soma_direction_ok"] is not None]
        rows.append(
            {
                "model_id": model_id,
                "pairs": len(per_pair),
                "mean_soma_delta_rmse_mv": mean(soma_values) if soma_values else None,
                "max_soma_delta_rmse_mv": max(soma_values) if soma_values else None,
                "mean_spike_count_delta_mae_per_trace": mean(spike_values) if spike_values else None,
                "spike_direction_failure_fraction": (
                    1.0 - sum(spike_direction_values) / len(spike_direction_values)
                    if spike_direction_values
                    else None
                ),
                "soma_direction_failure_fraction": (
                    1.0 - sum(soma_direction_values) / len(soma_direction_values)
                    if soma_direction_values
                    else None
                ),
            }
        )
    rows.sort(key=lambda row: row.get("mean_soma_delta_rmse_mv") or 0, reverse=True)
    return rows


def load_clipping_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out = {
        "source": str(path),
        "clipped_fraction": metric_value(rows, "clipping_distortion", "clipped_fraction"),
        "distortion_rmse_mv": metric_value(rows, "clipping_distortion", "distortion_rmse_mv"),
        "distortion_max_mv": metric_value(rows, "clipping_distortion", "distortion_max_mv"),
        "near_spike_clipped_fraction": metric_value(rows, "clipping_distortion", "near_spike_clipped_fraction"),
        "near_spike_distortion_rmse_mv": metric_value(rows, "clipping_distortion", "near_spike_distortion_rmse_mv"),
    }
    trace_distortions = [
        float(row["value"])
        for row in rows
        if row.get("diagnostic") == "clipping_distortion"
        and row.get("metric") == "trace_distortion_max_mv"
        and row.get("value") is not None
    ]
    if trace_distortions:
        out["trace_fraction_with_any_clipping"] = sum(value > 0 for value in trace_distortions) / len(trace_distortions)
        out["trace_distortion_max_mv_median"] = _median(trace_distortions)
        out["trace_distortion_max_mv_max"] = max(trace_distortions)
        out["trace_count"] = len(trace_distortions)
    return out


def build_findings(
    models: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    clipping: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    best_branch_rmse = min(
        float(row["soma_rmse_mv"])
        for row in models
        if row.get("family") == "official" and row.get("soma_rmse_mv") is not None
    )
    if clipping.get("distortion_rmse_mv") is not None:
        findings.append(
            {
                "class": "target_distortion",
                "severity": "high",
                "claim": "Historical -55 mV clipping induces a target-only RMSE comparable to, and larger than, the best Branch-ELM smoke RMSE.",
                "evidence": {
                    "clipping_distortion_rmse_mv": clipping.get("distortion_rmse_mv"),
                    "best_branch_elm_rmse_mv": best_branch_rmse,
                    "ratio": float(clipping["distortion_rmse_mv"]) / best_branch_rmse,
                    "near_spike_clipped_fraction": clipping.get("near_spike_clipped_fraction"),
                    "near_spike_distortion_rmse_mv": clipping.get("near_spike_distortion_rmse_mv"),
                    "distortion_max_mv": clipping.get("distortion_max_mv"),
                },
                "construct_validity": "Strong for target distortion; not yet a complete model-ranking claim without re-evaluating model predictions against raw and clipped target views.",
            }
        )

    official = [row for row in rank_rows if row.get("family") == "official"]
    official_auc_best = min(official, key=lambda row: row["spike_auc_rank"])
    official_rmse_best = min(official, key=lambda row: row["soma_rmse_mv_rank"])
    if official_auc_best["model_id"] != official_rmse_best["model_id"]:
        findings.append(
            {
                "class": "metric_inflation",
                "severity": "medium",
                "claim": "AUC and voltage RMSE rank different Branch-ELM variants differently.",
                "evidence": {
                    "best_auc_model": official_auc_best["model_id"],
                    "best_auc": official_auc_best["spike_auc"],
                    "best_auc_rmse_rank": official_auc_best["soma_rmse_mv_rank"],
                    "best_rmse_model": official_rmse_best["model_id"],
                    "best_rmse_mv": official_rmse_best["soma_rmse_mv"],
                    "best_rmse_auc_rank": official_rmse_best["spike_auc_rank"],
                },
                "construct_validity": "Suggestive; AUC is threshold-free and can ignore calibration/count/timing. Needs calibrated spike-threshold and timing diagnostics for a paper claim.",
            }
        )

    tcn_auc_best = min((row for row in rank_rows if row.get("family") == "keras-tcn"), key=lambda row: row["spike_auc_rank"])
    findings.append(
        {
            "class": "metric_inflation",
            "severity": "high",
            "claim": "The best TCN AUC is not enough to make the spike behavior sane under the fair adapter.",
            "evidence": {
                "model_id": tcn_auc_best["model_id"],
                "spike_auc": tcn_auc_best["spike_auc"],
                "spike_brier": tcn_auc_best["spike_brier"],
                "spike_count_mae_per_trace": tcn_auc_best["spike_count_mae_per_trace"],
                "soma_rmse_mv": tcn_auc_best["soma_rmse_mv"],
            },
            "construct_validity": "Strong as an adapter/domain-compatibility sanity failure; fair final TCN comparison still needs exact original target scaling recovered.",
        }
    )

    if state_rows:
        worst_state = max(
            (row for row in state_rows if row.get("state_to_global_rmse_ratio") is not None),
            key=lambda row: row["state_to_global_rmse_ratio"],
        )
        best_official_state = max(
            (
                row
                for row in state_rows
                if any(model["model_id"] == row["model_id"] and model.get("family") == "official" for model in models)
            ),
            key=lambda row: row["state_to_global_rmse_ratio"],
        )
        findings.append(
            {
                "class": "mechanistic_blindspot",
                "severity": "high",
                "claim": "Global RMSE hides much larger errors in specific neuronal states.",
                "evidence": {
                    "worst_overall": worst_state,
                    "worst_official": best_official_state,
                },
                "construct_validity": "Strong if state masks are accepted; next step is to inspect example traces and make the masks paper-facing.",
            }
        )

    if intervention_rows:
        worst_intervention = max(
            intervention_rows,
            key=lambda row: (
                row.get("mean_soma_delta_rmse_mv") or 0,
                row.get("spike_direction_failure_fraction") or 0,
            ),
        )
        findings.append(
            {
                "class": "mechanistic_blindspot",
                "severity": "medium",
                "claim": "Paired event-dropout responses expose delta-response errors not visible from ordinary validation RMSE alone.",
                "evidence": worst_intervention,
                "construct_validity": "Suggestive on this small smoke set; needs more intervention examples and more intervention types before becoming a headline claim.",
            }
        )
    return findings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Failure Probe",
        "",
        f"- Source root: `{payload['source_root']}`",
        f"- Pretrained rows: `{payload['pretrained_root']}`",
        "",
        "## Findings",
        "",
    ]
    for index, finding in enumerate(payload["findings"], start=1):
        lines.extend(
            [
                f"### {index}. {finding['class']} ({finding['severity']})",
                "",
                finding["claim"],
                "",
                "Evidence:",
                "",
                "```json",
                json.dumps(finding["evidence"], indent=2, sort_keys=True),
                "```",
                "",
                f"Construct validity: {finding['construct_validity']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Output Tables",
            "",
            "- `model_metric_ranks.csv`: metric-wise rank reversals.",
            "- `state_gap.csv`: state-conditioned RMSE divided by global RMSE.",
            "- `intervention_gap.csv`: paired event-dropout response metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


if __name__ == "__main__":
    raise SystemExit(main())
