#!/usr/bin/env python3
"""Probe how TCN soma scaling choices affect validation RMSE."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT / "scripts" / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from evaluate_pretrained_literature_benchmarks import KERAS_TCN_MODELS, _build_keras_tcn_model_fn  # noqa: E402


SOMA_SCALE = 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/davwis/main/workspace/bnn_sim/data/v1-neuronio-like-gpu-20260611"),
    )
    parser.add_argument("--resource-root", type=Path, default=Path("/home/davwis/main/workspace/bnn_sim"))
    parser.add_argument("--model", choices=sorted(KERAS_TCN_MODELS), default="tcn_dwt8_224_217")
    parser.add_argument("--samples-per-morphology", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark/artifacts/tcn_scaling_probe"))
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, targets, morphology_ids = load_balanced_slice(
        args.dataset_root / "val",
        samples_per_morphology=args.samples_per_morphology,
    )
    model_fn, adapter = _build_keras_tcn_model_fn(
        KERAS_TCN_MODELS[args.model],
        resource_root=args.resource_root,
        batch_size=args.batch_size,
    )
    predictions = model_fn(inputs)
    valid = np.zeros(targets.shape[:2], dtype=bool)
    valid[:, args.ignore_start :] = True

    target_soma = targets[..., 1]
    raw_soma = predictions[..., 1]
    rows = [
        summarize_view("raw_tcn", raw_soma, target_soma, valid),
        summarize_view(
            "same_slice_target_mean_baseline",
            np.full_like(target_soma, float(np.mean(target_soma[valid]))),
            target_soma,
            valid,
        ),
        summarize_view(
            "same_slice_mean_std_match",
            match_mean_std(raw_soma, target_soma, valid),
            target_soma,
            valid,
        ),
        summarize_view(
            "same_slice_linear_fit",
            apply_linear_fit(raw_soma, *fit_linear(raw_soma, target_soma, valid)),
            target_soma,
            valid,
        ),
        summarize_view(
            "leave_one_morphology_target_mean_baseline",
            leave_one_morphology_prediction(raw_soma, target_soma, valid, morphology_ids, mode="target_mean"),
            target_soma,
            valid,
        ),
        summarize_view(
            "leave_one_morphology_mean_std_match",
            leave_one_morphology_prediction(raw_soma, target_soma, valid, morphology_ids, mode="mean_std"),
            target_soma,
            valid,
        ),
        summarize_view(
            "leave_one_morphology_linear_fit",
            leave_one_morphology_prediction(raw_soma, target_soma, valid, morphology_ids, mode="linear"),
            target_soma,
            valid,
        ),
    ]
    payload = {
        "dataset_root": str(args.dataset_root),
        "resource_root": str(args.resource_root),
        "model": args.model,
        "adapter": adapter,
        "samples": int(inputs.shape[0]),
        "samples_per_morphology": args.samples_per_morphology,
        "morphology_ids": morphology_ids,
        "raw_soma_summary": summarize_array(raw_soma[valid]),
        "target_soma_summary": summarize_array(target_soma[valid]),
        "rows": rows,
    }
    (output / "tcn_scaling_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "tcn_scaling.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": int(inputs.shape[0])}, indent=2, sort_keys=True))
    return 0


def load_balanced_slice(split_root: Path, *, samples_per_morphology: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files_by_morphology: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(split_root.glob("*.npz")):
        morphology_id = "_".join(path.stem.split("_")[:2])
        files_by_morphology[morphology_id].append(path)

    inputs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    morphology_ids: list[str] = []
    for morphology_id, paths in sorted(files_by_morphology.items()):
        remaining = samples_per_morphology
        for path in paths:
            if remaining <= 0:
                break
            data = np.load(path)
            take = min(remaining, data["inputs"].shape[0])
            inputs_list.append(data["inputs"][:take].astype(np.float32, copy=False))
            targets_list.append(data["targets"][:take].astype(np.float32, copy=False))
            morphology_ids.extend([morphology_id] * take)
            remaining -= take
    if not inputs_list:
        raise FileNotFoundError(f"no validation shards found under {split_root}")
    return np.concatenate(inputs_list, axis=0), np.concatenate(targets_list, axis=0), morphology_ids


def summarize_view(name: str, prediction_soma: np.ndarray, target_soma: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    diff = prediction_soma[valid] - target_soma[valid]
    return {
        "view": name,
        "rmse_normalized": float(np.sqrt(np.mean(diff**2))),
        "rmse_mv": float(np.sqrt(np.mean((diff / SOMA_SCALE) ** 2))),
        "mae_mv": float(np.mean(np.abs(diff / SOMA_SCALE))),
        "prediction_summary": summarize_array(prediction_soma[valid]),
    }


def summarize_array(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
    }


def match_mean_std(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred_values = prediction[mask]
    target_values = target[mask]
    pred_std = float(np.std(pred_values))
    if pred_std <= 1e-12:
        return np.full_like(prediction, float(np.mean(target_values)))
    return (prediction - float(np.mean(pred_values))) / pred_std * float(np.std(target_values)) + float(np.mean(target_values))


def fit_linear(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    x = prediction[mask].astype(np.float64)
    y = target[mask].astype(np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def apply_linear_fit(prediction: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return prediction * slope + intercept


def leave_one_morphology_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    morphology_ids: list[str],
    *,
    mode: str,
) -> np.ndarray:
    out = np.empty_like(prediction)
    morph_arr = np.asarray(morphology_ids)
    for morphology_id in sorted(set(morphology_ids)):
        test_traces = morph_arr == morphology_id
        train_traces = ~test_traces
        train_mask = valid & train_traces[:, None]
        if mode == "target_mean":
            calibrated = np.full_like(prediction, float(np.mean(target[train_mask])))
        elif mode == "mean_std":
            calibrated = match_mean_std(prediction, target, train_mask)
        elif mode == "linear":
            calibrated = apply_linear_fit(prediction, *fit_linear(prediction, target, train_mask))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        out[test_traces] = calibrated[test_traces]
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TCN Scaling Probe",
        "",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Model: `{payload['model']}`",
        f"- Samples: `{payload['samples']}` (`{payload['samples_per_morphology']}` per morphology)",
        "",
        "## Output Ranges",
        "",
        "```json",
        json.dumps(
            {
                "raw_soma_summary": payload["raw_soma_summary"],
                "target_soma_summary": payload["target_soma_summary"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## RMSE By Scaling View",
        "",
        "| view | RMSE mV | MAE mV | pred mean | pred std | pred min | pred max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        summary = row["prediction_summary"]
        lines.append(
            "| {view} | {rmse:.4f} | {mae:.4f} | {mean:.4f} | {std:.4f} | {min_:.4f} | {max_:.4f} |".format(
                view=row["view"],
                rmse=row["rmse_mv"],
                mae=row["mae_mv"],
                mean=summary["mean"],
                std=summary["std"],
                min_=summary["min"],
                max_=summary["max"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: same-slice scaling views use evaluation targets and",
            "are leakage probes, not fair benchmark settings. Leave-one-morphology",
            "views test whether a calibration learned from other morphologies",
            "transfers to a held-out morphology on this slice.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
