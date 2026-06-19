#!/usr/bin/env python3
"""Run a small all-intervention benchmark sweep for pretrained Branch-ELM."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dendrobench.benchmark.bundle import run_v1_benchmark_bundle
from dendrobench.benchmark.dataset_schema import INTERVENTION_CONDITIONS
from dendrobench.benchmark.diagnostic_rows import DiagnosticContext
from dendrobench.official import create_official_elm, load_official_elm_config, load_official_state_dict


OFFICIAL_MODELS = {
    "branch_elm_m30": {
        "config": ".external/elmneuron/models/num_memory_30/model_config.json",
        "checkpoint": ".external/elmneuron/models/num_memory_30/neuronio_best_model_state.pt",
    },
    "branch_elm_m100": {
        "config": ".external/elmneuron/models/num_memory_100/model_config.json",
        "checkpoint": ".external/elmneuron/models/num_memory_100/neuronio_best_model_state.pt",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/davwis/main/workspace/bnn_sim"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/davwis/main/workspace/bnn_sim/data/v1-neuronio-like-gpu-20260611"),
    )
    parser.add_argument("--model", choices=sorted(OFFICIAL_MODELS), default="branch_elm_m30")
    parser.add_argument(
        "--intervention",
        action="append",
        choices=INTERVENTION_CONDITIONS,
        help="Restrict the stress probe to one or more interventions. Defaults to all available interventions.",
    )
    parser.add_argument("--max-val-samples", type=int, default=2)
    parser.add_argument("--max-intervention-samples", type=int, default=2)
    parser.add_argument(
        "--calibration-samples-per-morphology",
        type=int,
        default=0,
        help="If positive, set the spike threshold by matching validation spike prevalence on this many traces per morphology.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark/artifacts/intervention_stress_probe"))
    args = parser.parse_args()

    output = args.output.resolve()
    rows_dir = output / args.model
    rows_dir.mkdir(parents=True, exist_ok=True)
    model_fn = build_model_fn(args.source_root, args.model, batch_size=args.batch_size, device=args.device)
    selected_pair_paths = selected_intervention_pair_paths(args.dataset_root, args.intervention)
    calibrated_spike_threshold = None
    if args.calibration_samples_per_morphology > 0:
        calibrated_spike_threshold = calibrate_spike_threshold(
            model_fn,
            args.dataset_root / "val",
            samples_per_morphology=args.calibration_samples_per_morphology,
        )
    manifest = run_v1_benchmark_bundle(
        dataset_root=args.dataset_root,
        output_dir=rows_dir,
        base_context=DiagnosticContext(model_id=args.model, calibrated_spike_threshold=calibrated_spike_threshold),
        model_fn=model_fn,
        model_id=args.model,
        intervention_pair_paths=selected_pair_paths,
        suite="paper-summary",
        max_val_samples=args.max_val_samples,
        max_intervention_samples=args.max_intervention_samples,
        cache_shards=1,
        write_summaries=False,
    )
    summary = summarize_interventions(rows_dir)
    payload = {
        "dataset_root": str(args.dataset_root),
        "model_id": args.model,
        "device": args.device,
        "interventions": args.intervention or "all",
        "intervention_pair_paths": [str(path) for path in selected_pair_paths],
        "calibrated_spike_threshold": calibrated_spike_threshold,
        "calibration_samples_per_morphology": args.calibration_samples_per_morphology,
        "max_val_samples": args.max_val_samples,
        "max_intervention_samples_per_morphology": args.max_intervention_samples,
        "manifest": manifest,
        "summary": summary,
    }
    (output / "intervention_stress_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "intervention_stress.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "model": args.model}, indent=2, sort_keys=True))
    return 0


def selected_intervention_pair_paths(dataset_root: Path, interventions: list[str] | None) -> list[Path]:
    if not interventions:
        return []
    paths: list[Path] = []
    for intervention in interventions:
        intervention_root = dataset_root / "interventions" / intervention
        paths.extend(sorted(intervention_root.glob("*_intervention_pair_*.npz")))
        paths.extend(sorted(intervention_root.glob("intervention_pair_*.npz")))
        direct = intervention_root / "intervention_pair.npz"
        if direct.exists():
            paths.append(direct)
    if not paths:
        raise FileNotFoundError(f"no intervention pair shards found for {interventions!r} under {dataset_root}")
    return sorted(set(paths))


def calibrate_spike_threshold(
    model_fn,
    val_root: Path,
    *,
    samples_per_morphology: int,
    ignore_start: int = 500,
) -> float:
    inputs, targets = load_balanced_validation_slice(val_root, samples_per_morphology=samples_per_morphology)
    predictions = np.asarray(model_fn(inputs))
    valid = np.zeros(targets.shape[:2], dtype=bool)
    valid[:, ignore_start:] = True
    labels = (targets[..., 0] > 0.5) & valid
    scores = predictions[..., 0][valid]
    positives = int(labels[valid].sum())
    if positives <= 0:
        return float("inf")
    if positives >= scores.size:
        return float("-inf")
    return float(np.partition(scores, scores.size - positives)[scores.size - positives])


def load_balanced_validation_slice(val_root: Path, *, samples_per_morphology: int) -> tuple[np.ndarray, np.ndarray]:
    files_by_morphology: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(val_root.glob("*.npz")):
        morphology_id = "_".join(path.stem.split("_")[:2])
        files_by_morphology[morphology_id].append(path)

    inputs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    for paths in files_by_morphology.values():
        remaining = samples_per_morphology
        for path in paths:
            if remaining <= 0:
                break
            data = np.load(path)
            take = min(remaining, data["inputs"].shape[0])
            inputs_list.append(data["inputs"][:take].astype(np.float32, copy=False))
            targets_list.append(data["targets"][:take].astype(np.float32, copy=False))
            remaining -= take
    if not inputs_list:
        raise FileNotFoundError(f"no validation shards found under {val_root}")
    return np.concatenate(inputs_list, axis=0), np.concatenate(targets_list, axis=0)


def build_model_fn(source_root: Path, model_id: str, *, batch_size: int, device: str):
    spec = OFFICIAL_MODELS[model_id]
    model = create_official_elm(load_official_elm_config(source_root / spec["config"]))
    model.load_state_dict(load_official_state_dict(source_root / spec["checkpoint"], map_location=device), strict=True)
    model = model.to(device).eval()

    def predict(inputs: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, inputs.shape[0], batch_size):
                batch = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device)
                outputs.append(model(batch).detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    return predict


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_interventions(rows_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in rows_dir.glob("intervention_specimen_*.jsonl"):
        rows = read_jsonl(path)
        if not rows:
            continue
        intervention = rows[0].get("intervention") or path.stem.split("_")[-1]
        metrics = {row["metric"]: row["value"] for row in rows if row.get("diagnostic") == "intervention_response"}
        grouped[intervention].append(metrics)

    summary = []
    for intervention, metrics_list in sorted(grouped.items()):
        soma_delta = _float_values(metrics_list, "soma_delta_rmse_mv")
        spike_delta = _float_values(metrics_list, "spike_count_delta_mae_per_trace")
        teacher_soma = _float_values(metrics_list, "delta_soma_abs_mean_mv")
        teacher_soma_mean = _float_values(metrics_list, "soma_mean_delta_teacher_mv")
        model_soma_mean = _float_values(metrics_list, "soma_mean_delta_model_mv")
        soma_gain = _float_values(metrics_list, "soma_delta_gain")
        soma_corr = _float_values(metrics_list, "soma_delta_correlation")
        teacher_spike_rate = _float_values(metrics_list, "delta_spike_rate_per_1000_steps")
        teacher_spike_delta = _float_values(metrics_list, "spike_count_delta_teacher_mean")
        model_spike_delta = _float_values(metrics_list, "spike_count_delta_model_mean")
        spike_gain = _float_values(metrics_list, "spike_count_delta_gain")
        spike_direction = [
            bool(row.get("direction_agreement", {}).get("spike_count"))
            for row in metrics_list
            if isinstance(row.get("direction_agreement"), dict)
        ]
        soma_direction = [
            bool(row.get("direction_agreement", {}).get("soma_mean"))
            for row in metrics_list
            if isinstance(row.get("direction_agreement"), dict)
        ]
        summary.append(
            {
                "intervention": intervention,
                "morphology_groups": len(metrics_list),
                "teacher_abs_soma_delta_mean_mv": _mean(teacher_soma),
                "teacher_soma_mean_delta_mv": _mean(teacher_soma_mean),
                "model_soma_mean_delta_mv": _mean(model_soma_mean),
                "model_soma_delta_gain_mean": _mean(soma_gain),
                "model_soma_delta_correlation_mean": _mean(soma_corr),
                "teacher_abs_spike_rate_delta_mean_per_1000": _mean_abs(teacher_spike_rate),
                "teacher_spike_count_delta_mean": _mean(teacher_spike_delta),
                "model_spike_count_delta_mean": _mean(model_spike_delta),
                "model_spike_count_delta_gain_mean": _mean(spike_gain),
                "model_soma_delta_rmse_mean_mv": _mean(soma_delta),
                "model_soma_delta_rmse_max_mv": max(soma_delta) if soma_delta else None,
                "model_spike_count_delta_mae_mean": _mean(spike_delta),
                "soma_direction_failure_fraction": _failure_fraction(soma_direction),
                "spike_direction_failure_fraction": _failure_fraction(spike_direction),
            }
        )
    summary.sort(key=lambda row: (row["model_soma_delta_rmse_mean_mv"] or 0), reverse=True)
    return summary


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _mean_abs(values: list[float]) -> float | None:
    return float(np.mean(np.abs(values))) if values else None


def _failure_fraction(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(1.0 - np.mean(values))


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Intervention Stress Probe",
        "",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Model: `{payload['model_id']}`",
        f"- Calibrated spike threshold: `{payload['calibrated_spike_threshold']}`",
        f"- Max intervention samples per morphology: `{payload['max_intervention_samples_per_morphology']}`",
        "",
        "| intervention | groups | teacher soma mean delta | model soma mean delta | soma gain | soma corr | model soma-delta RMSE | teacher spike delta | model spike delta | spike gain | spike delta MAE | soma dir fail | spike dir fail |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {intervention} | {groups} | {teacher_soma_mean:.4f} | {model_soma_mean:.4f} | {soma_gain:.4f} | {soma_corr:.4f} | {soma:.4f} | {teacher_spike:.4f} | {model_spike:.4f} | {spike_gain:.4f} | {spike:.4f} | {soma_fail:.3f} | {spike_fail:.3f} |".format(
                intervention=row["intervention"],
                groups=row["morphology_groups"],
                teacher_soma_mean=row["teacher_soma_mean_delta_mv"] or 0.0,
                model_soma_mean=row["model_soma_mean_delta_mv"] or 0.0,
                soma_gain=row["model_soma_delta_gain_mean"] or 0.0,
                soma_corr=row["model_soma_delta_correlation_mean"] or 0.0,
                soma=row["model_soma_delta_rmse_mean_mv"] or 0.0,
                teacher_spike=row["teacher_spike_count_delta_mean"] or 0.0,
                model_spike=row["model_spike_count_delta_mean"] or 0.0,
                spike_gain=row["model_spike_count_delta_gain_mean"] or 0.0,
                spike=row["model_spike_count_delta_mae_mean"] or 0.0,
                soma_fail=row["soma_direction_failure_fraction"] or 0.0,
                spike_fail=row["spike_direction_failure_fraction"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this per-morphology cap is intended to rank",
            "plausible intervention families and inspect gain/direction failures.",
            "Large model delta errors relative to teacher response magnitude are",
            "candidates for a larger robustness sweep.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
