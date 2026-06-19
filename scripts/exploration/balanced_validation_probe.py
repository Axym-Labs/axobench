#!/usr/bin/env python3
"""Run raw/clipped and spike-event metrics on a morphology-balanced val slice."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dendrobench.metrics import binary_auc

from clipping_downstream_probe import (
    OFFICIAL_MODELS,
    SOMA_BIAS_MV,
    SOMA_SCALE,
    build_masks,
    compare_prediction,
    compare_views,
    predict_official_model,
)
from spike_timing_probe import prevalence_threshold, threshold_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/davwis/main/workspace/bnn_sim"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/davwis/main/workspace/bnn_sim/data/v1-neuronio-like-gpu-20260611"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples-per-morphology", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--model", action="append", choices=sorted(OFFICIAL_MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark/artifacts/balanced_validation_probe"))
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, clipped_targets, raw_soma_mv, sample_ids, shard_paths, morphology_ids = load_balanced_slice(
        args.dataset_root / args.split,
        samples_per_morphology=args.samples_per_morphology,
    )
    raw_targets = clipped_targets.copy()
    raw_targets[..., 1] = (raw_soma_mv - SOMA_BIAS_MV) * SOMA_SCALE
    masks = build_masks(clipped_targets, raw_soma_mv, ignore_start=args.ignore_start)
    labels = (clipped_targets[..., 0] > 0.5) & masks["valid"]

    rows: list[dict[str, Any]] = []
    per_morphology: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_id in args.model or sorted(OFFICIAL_MODELS):
        predictions = predict_official_model(
            args.source_root,
            model_id,
            inputs,
            batch_size=args.batch_size,
            device=args.device,
        )
        model_summary = {
            "model_id": model_id,
            "samples": int(inputs.shape[0]),
            "spike_auc": binary_auc(predictions[..., 0][masks["valid"]], labels[masks["valid"]]),
            **compare_prediction(predictions, clipped_targets, raw_targets, masks),
            "spike_timing_logit_zero": threshold_report(predictions[..., 0], labels, masks["valid"], threshold=0.0),
            "spike_timing_prevalence": threshold_report(
                predictions[..., 0],
                labels,
                masks["valid"],
                threshold=prevalence_threshold(predictions[..., 0], labels, masks["valid"]),
            ),
        }
        rows.append(model_summary)
        for morphology_id in sorted(set(morphology_ids)):
            morph_mask = np.asarray([item == morphology_id for item in morphology_ids], dtype=bool)
            morph_time_mask = np.broadcast_to(morph_mask[:, None], masks["valid"].shape)
            morph_masks = {name: mask & morph_time_mask for name, mask in masks.items()}
            morph_labels = labels & morph_time_mask
            per_morphology[model_id].append(
                {
                    "morphology_id": morphology_id,
                    "samples": int(morph_mask.sum()),
                    "target_distortion": compare_views(clipped_targets, raw_targets, morph_masks),
                    "metrics": {
                        "spike_auc": binary_auc(
                            predictions[..., 0][morph_masks["valid"]],
                            morph_labels[morph_masks["valid"]],
                        ),
                        **compare_prediction(predictions, clipped_targets, raw_targets, morph_masks),
                    },
                }
            )

    payload = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "samples_per_morphology": args.samples_per_morphology,
        "samples": int(inputs.shape[0]),
        "morphology_ids": morphology_ids,
        "sample_ids": [str(item) for item in sample_ids],
        "shard_paths": [str(path) for path in shard_paths],
        "target_distortion": compare_views(clipped_targets, raw_targets, masks),
        "models": rows,
        "per_morphology": per_morphology,
    }
    (output / "balanced_validation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "balanced_validation.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": int(inputs.shape[0])}, indent=2, sort_keys=True))
    return 0


def load_balanced_slice(split_root: Path, *, samples_per_morphology: int):
    files_by_morphology: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(split_root.glob("*.npz")):
        morphology_id = "_".join(path.stem.split("_")[:2])
        files_by_morphology[morphology_id].append(path)

    inputs_list = []
    targets_list = []
    soma_list = []
    sample_ids: list[str] = []
    morphology_ids: list[str] = []
    used_shards: list[Path] = []
    for morphology_id, paths in sorted(files_by_morphology.items()):
        remaining = samples_per_morphology
        for path in paths:
            if remaining <= 0:
                break
            data = np.load(path)
            take = min(remaining, data["inputs"].shape[0])
            inputs_list.append(data["inputs"][:take].astype(np.float32, copy=False))
            targets_list.append(data["targets"][:take].astype(np.float32, copy=False))
            soma_list.append(data["soma_mv"][:take].astype(np.float32, copy=False))
            sample_ids.extend(str(item) for item in data["sample_ids"][:take])
            morphology_ids.extend([morphology_id] * take)
            used_shards.append(path)
            remaining -= take
    return (
        np.concatenate(inputs_list, axis=0),
        np.concatenate(targets_list, axis=0),
        np.concatenate(soma_list, axis=0),
        sample_ids,
        used_shards,
        morphology_ids,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Balanced Validation Probe",
        "",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Samples: `{payload['samples']}` (`{payload['samples_per_morphology']}` per morphology)",
        "",
        "## Target Distortion",
        "",
        "```json",
        json.dumps(payload["target_distortion"], indent=2, sort_keys=True),
        "```",
        "",
        "## Model Summary",
        "",
        "| model | AUC | clipped RMSE | raw RMSE | near clipped RMSE | near raw RMSE | logit0 spikes | prevalence F1 tol2 | prevalence F1 tol5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["models"]:
        lines.append(
            "| {model} | {auc:.4f} | {clip:.4f} | {raw:.4f} | {near_clip:.4f} | {near_raw:.4f} | {logit_spikes} | {f12:.4f} | {f15:.4f} |".format(
                model=row["model_id"],
                auc=row["spike_auc"],
                clip=row["rmse_to_clipped_valid_mv"],
                raw=row["rmse_to_raw_valid_mv"],
                near_clip=row["rmse_to_clipped_near_spike_mv"],
                near_raw=row["rmse_to_raw_near_spike_mv"],
                logit_spikes=row["spike_timing_logit_zero"]["predicted_spike_count"],
                f12=row["spike_timing_prevalence"]["f1_tol2"],
                f15=row["spike_timing_prevalence"]["f1_tol5"],
            )
        )
    lines.extend(["", "## Per-Morphology Raw RMSE Delta", ""])
    for model_id, rows in payload["per_morphology"].items():
        lines.append(f"### {model_id}")
        lines.append("")
        lines.append("| morphology | clipped RMSE | raw RMSE | near raw RMSE | AUC |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in rows:
            metrics = row["metrics"]
            lines.append(
                "| {morph} | {clip:.4f} | {raw:.4f} | {near_raw:.4f} | {auc:.4f} |".format(
                    morph=row["morphology_id"],
                    clip=metrics["rmse_to_clipped_valid_mv"],
                    raw=metrics["rmse_to_raw_valid_mv"],
                    near_raw=metrics["rmse_to_raw_near_spike_mv"] or 0.0,
                    auc=metrics["spike_auc"],
                )
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
