#!/usr/bin/env python3
"""Evaluate pretrained Branch-ELM against clipped and raw soma target views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dendrobench.official import create_official_elm, load_official_elm_config, load_official_state_dict


SOMA_BIAS_MV = -67.7
SOMA_SCALE = 0.1
SOMA_CLIP_THRESHOLD_MV = -55.0

OFFICIAL_MODELS = {
    "branch_elm_best": {
        "config": ".external/elmneuron/models/best_elm_neuron/model_config.json",
        "checkpoint": ".external/elmneuron/models/best_elm_neuron/neuronio_best_model_state.pt",
    },
    "branch_elm_m1": {
        "config": ".external/elmneuron/models/num_memory_1/model_config.json",
        "checkpoint": ".external/elmneuron/models/num_memory_1/neuronio_best_model_state.pt",
    },
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
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--model", action="append", choices=sorted(OFFICIAL_MODELS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark/artifacts/clipping_downstream_probe"))
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, clipped_targets, raw_soma_mv, sample_ids, shard_path = load_slice(
        args.dataset_root / args.split,
        max_samples=args.max_samples,
    )
    raw_targets = clipped_targets.copy()
    raw_targets[..., 1] = (raw_soma_mv - SOMA_BIAS_MV) * SOMA_SCALE
    masks = build_masks(clipped_targets, raw_soma_mv, ignore_start=args.ignore_start)
    target_distortion = compare_views(clipped_targets, raw_targets, masks)

    rows: list[dict[str, Any]] = []
    for model_id in args.model or sorted(OFFICIAL_MODELS):
        predictions = predict_official_model(
            args.source_root,
            model_id,
            inputs,
            batch_size=args.batch_size,
            device=args.device,
        )
        rows.append(
            {
                "model_id": model_id,
                "device": args.device,
                "samples": int(inputs.shape[0]),
                "shard_path": str(shard_path),
                "sample_ids": [str(item) for item in sample_ids],
                **compare_prediction(predictions, clipped_targets, raw_targets, masks),
            }
        )

    payload = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "ignore_start": args.ignore_start,
        "target_distortion": target_distortion,
        "models": rows,
    }
    (output / "clipping_downstream_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "clipping_downstream.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "models": len(rows)}, indent=2, sort_keys=True))
    return 0


def load_slice(split_root: Path, *, max_samples: int):
    shard_path = sorted(split_root.glob("*.npz"))[0]
    data = np.load(shard_path)
    inputs = data["inputs"][:max_samples].astype(np.float32, copy=False)
    clipped_targets = data["targets"][:max_samples].astype(np.float32, copy=False)
    raw_soma_mv = data["soma_mv"][:max_samples].astype(np.float32, copy=False)
    sample_ids = data["sample_ids"][:max_samples]
    return inputs, clipped_targets, raw_soma_mv, sample_ids, shard_path


def build_masks(targets: np.ndarray, raw_soma_mv: np.ndarray, *, ignore_start: int) -> dict[str, np.ndarray]:
    valid = np.zeros(targets.shape[:2], dtype=bool)
    valid[:, ignore_start:] = True
    clipped = (raw_soma_mv > SOMA_CLIP_THRESHOLD_MV) & valid
    spikes = (targets[..., 0] > 0.5) & valid
    near_spike = spikes.copy()
    for shift in (-2, -1, 1, 2):
        near_spike |= np.roll(spikes, shift=shift, axis=1)
    near_spike &= valid
    return {
        "valid": valid,
        "clipped": clipped,
        "near_spike": near_spike,
        "nonclipped": valid & ~clipped,
    }


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


def compare_views(clipped_targets: np.ndarray, raw_targets: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    diff_mv = (raw_targets[..., 1] - clipped_targets[..., 1]) / SOMA_SCALE
    return {
        "clipped_fraction_valid": fraction(masks["clipped"], masks["valid"]),
        "near_spike_clipped_fraction": fraction(masks["clipped"] & masks["near_spike"], masks["near_spike"]),
        "rmse_mv_valid": rmse_from_diff(diff_mv, masks["valid"]),
        "rmse_mv_near_spike": rmse_from_diff(diff_mv, masks["near_spike"]),
        "max_abs_distortion_mv": float(np.max(np.abs(diff_mv[masks["valid"]]))),
    }


def compare_prediction(
    predictions: np.ndarray,
    clipped_targets: np.ndarray,
    raw_targets: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    pred_soma = predictions[..., 1]
    clipped_soma = clipped_targets[..., 1]
    raw_soma = raw_targets[..., 1]
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        if name == "valid":
            label = "valid"
        else:
            label = name
        out[f"rmse_to_clipped_{label}_mv"] = rmse_normalized(pred_soma, clipped_soma, mask)
        out[f"rmse_to_raw_{label}_mv"] = rmse_normalized(pred_soma, raw_soma, mask)
        out[f"raw_minus_clipped_rmse_delta_{label}_mv"] = (
            out[f"rmse_to_raw_{label}_mv"] - out[f"rmse_to_clipped_{label}_mv"]
            if out[f"rmse_to_raw_{label}_mv"] is not None and out[f"rmse_to_clipped_{label}_mv"] is not None
            else None
        )
    out["prediction_soma_max_normalized"] = float(np.max(pred_soma[masks["valid"]]))
    out["prediction_soma_min_normalized"] = float(np.min(pred_soma[masks["valid"]]))
    return out


def rmse_normalized(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    diff_mv = (pred[mask] - target[mask]) / SOMA_SCALE
    return float(np.sqrt(np.mean(diff_mv * diff_mv)))


def rmse_from_diff(diff_mv: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    values = diff_mv[mask]
    return float(np.sqrt(np.mean(values * values)))


def fraction(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    denom = int(np.sum(denominator))
    if denom == 0:
        return None
    return float(np.sum(numerator) / denom)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Clipping Downstream Probe",
        "",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Split: `{payload['split']}`",
        f"- Ignore start: `{payload['ignore_start']}`",
        "",
        "## Target View Distortion",
        "",
        "```json",
        json.dumps(payload["target_distortion"], indent=2, sort_keys=True),
        "```",
        "",
        "## Model Metrics",
        "",
        "| model | RMSE clipped valid | RMSE raw valid | delta | RMSE clipped near-spike | RMSE raw near-spike | near delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["models"]:
        lines.append(
            "| {model_id} | {rc:.4f} | {rr:.4f} | {dc:.4f} | {nc:.4f} | {nr:.4f} | {nd:.4f} |".format(
                model_id=row["model_id"],
                rc=row["rmse_to_clipped_valid_mv"],
                rr=row["rmse_to_raw_valid_mv"],
                dc=row["raw_minus_clipped_rmse_delta_valid_mv"],
                nc=row["rmse_to_clipped_near_spike_mv"],
                nr=row["rmse_to_raw_near_spike_mv"],
                nd=row["raw_minus_clipped_rmse_delta_near_spike_mv"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: positive deltas mean the same pretrained model looks worse",
            "when evaluated against raw teacher AP waveforms instead of the historical",
            "clipped target view. The near-spike rows estimate the local AP-shape cost.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
