#!/usr/bin/env python3
"""Probe spike-event timing and threshold calibration for pretrained Branch-ELM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dendrobench.metrics import binary_auc
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
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--model", action="append", choices=sorted(OFFICIAL_MODELS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark/artifacts/spike_timing_probe"))
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, targets, sample_ids, shard_path = load_slice(args.dataset_root / args.split, max_samples=args.max_samples)
    valid = np.zeros(targets.shape[:2], dtype=bool)
    valid[:, args.ignore_start :] = True
    labels = (targets[..., 0] > 0.5) & valid

    rows = []
    for model_id in args.model or sorted(OFFICIAL_MODELS):
        predictions = predict_official_model(
            args.source_root,
            model_id,
            inputs,
            batch_size=args.batch_size,
            device=args.device,
        )
        scores = predictions[..., 0]
        rows.append(
            {
                "model_id": model_id,
                "device": args.device,
                "samples": int(inputs.shape[0]),
                "target_spike_count": int(labels.sum()),
                "spike_auc": binary_auc(scores[valid], labels[valid]),
                "threshold_logit_zero": threshold_report(scores, labels, valid, threshold=0.0),
                "threshold_prevalence_matched": threshold_report(
                    scores,
                    labels,
                    valid,
                    threshold=prevalence_threshold(scores, labels, valid),
                ),
            }
        )

    payload = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "ignore_start": args.ignore_start,
        "shard_path": str(shard_path),
        "sample_ids": [str(item) for item in sample_ids],
        "models": rows,
    }
    (output / "spike_timing_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "spike_timing.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "models": len(rows)}, indent=2, sort_keys=True))
    return 0


def load_slice(split_root: Path, *, max_samples: int):
    shard_path = sorted(split_root.glob("*.npz"))[0]
    data = np.load(shard_path)
    inputs = data["inputs"][:max_samples].astype(np.float32, copy=False)
    targets = data["targets"][:max_samples].astype(np.float32, copy=False)
    sample_ids = data["sample_ids"][:max_samples]
    return inputs, targets, sample_ids, shard_path


def predict_official_model(
    source_root: Path,
    model_id: str,
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    spec = OFFICIAL_MODELS[model_id]
    model = create_official_elm(load_official_elm_config(source_root / spec["config"]))
    model.load_state_dict(load_official_state_dict(source_root / spec["checkpoint"], map_location=device), strict=True)
    model = model.to(device).eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            batch = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def prevalence_threshold(scores: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> float:
    valid_scores = np.asarray(scores[valid], dtype=np.float64)
    positives = int(labels[valid].sum())
    if positives <= 0:
        return float("inf")
    if positives >= valid_scores.size:
        return float("-inf")
    return float(np.partition(valid_scores, valid_scores.size - positives)[valid_scores.size - positives])


def threshold_report(scores: np.ndarray, labels: np.ndarray, valid: np.ndarray, *, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold) & valid
    target_count = int(labels.sum())
    predicted_count = int(predictions.sum())
    report: dict[str, Any] = {
        "threshold": threshold,
        "predicted_spike_count": predicted_count,
        "target_spike_count": target_count,
        "count_error": predicted_count - target_count,
        "count_absolute_error": abs(predicted_count - target_count),
    }
    for tolerance in (0, 1, 2, 5):
        matched, false_positive, false_negative, abs_offsets = match_events(predictions, labels, tolerance=tolerance)
        precision = matched / (matched + false_positive) if matched + false_positive else 0.0
        recall = matched / (matched + false_negative) if matched + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[f"f1_tol{tolerance}"] = f1
        report[f"precision_tol{tolerance}"] = precision
        report[f"recall_tol{tolerance}"] = recall
        report[f"matched_tol{tolerance}"] = matched
        report[f"false_positive_tol{tolerance}"] = false_positive
        report[f"false_negative_tol{tolerance}"] = false_negative
        report[f"mean_abs_timing_error_tol{tolerance}"] = (
            float(np.mean(abs_offsets)) if abs_offsets else None
        )
    return report


def match_events(predictions: np.ndarray, labels: np.ndarray, *, tolerance: int) -> tuple[int, int, int, list[int]]:
    matched = 0
    false_positive = 0
    false_negative = 0
    abs_offsets: list[int] = []
    for pred_trace, label_trace in zip(predictions, labels, strict=True):
        pred_times = list(np.flatnonzero(pred_trace))
        target_times = list(np.flatnonzero(label_trace))
        used_targets: set[int] = set()
        for pred_time in pred_times:
            best_index = None
            best_distance = tolerance + 1
            for index, target_time in enumerate(target_times):
                if index in used_targets:
                    continue
                distance = abs(int(pred_time) - int(target_time))
                if distance <= tolerance and distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is None:
                false_positive += 1
            else:
                used_targets.add(best_index)
                matched += 1
                abs_offsets.append(best_distance)
        false_negative += len(target_times) - len(used_targets)
    return matched, false_positive, false_negative, abs_offsets


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Spike Timing Probe",
        "",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Shard: `{payload['shard_path']}`",
        f"- Ignore start: `{payload['ignore_start']}`",
        "",
        "| model | AUC | target spikes | threshold | predicted spikes | F1 tol0 | F1 tol2 | F1 tol5 | count error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["models"]:
        for label, report in (
            ("logit 0", row["threshold_logit_zero"]),
            ("prevalence", row["threshold_prevalence_matched"]),
        ):
            lines.append(
                "| {model} ({label}) | {auc:.4f} | {target} | {threshold:.6g} | {predicted} | {f10:.4f} | {f12:.4f} | {f15:.4f} | {count_error} |".format(
                    model=row["model_id"],
                    label=label,
                    auc=row["spike_auc"],
                    target=row["target_spike_count"],
                    threshold=report["threshold"],
                    predicted=report["predicted_spike_count"],
                    f10=report["f1_tol0"],
                    f12=report["f1_tol2"],
                    f15=report["f1_tol5"],
                    count_error=report["count_error"],
                )
            )
    lines.extend(
        [
            "",
            "Interpretation: AUC measures ranking of spike scores over individual",
            "timesteps. Event F1 asks whether thresholded spikes occur at the same",
            "times, with small tolerance windows. Prevalence matching is a favorable",
            "calibration oracle; failures there are harder to dismiss as threshold choice.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
