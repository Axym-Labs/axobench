#!/usr/bin/env python3
"""Calibrate each Allen all-active teacher's excitatory gain to the NeuronIO cell's firing rate.

On identical NeuronIO inputs the all-active mouse cells fire from 0 to over
40 Hz while the NeuronIO (Hay) cell fires about 2 Hz, because cell size,
input resistance and excitability differ. To keep every cell in NeuronIO's
output regime, the excitatory gmax of each Allen cell is scaled by one factor
per cell, chosen so that its mean firing rate on a fixed calibration input set
matches the Hay cell's mean rate on the same inputs (v2 settings, gain 1.0).
Inhibition and the cell's own biophysics are unchanged.

The calibration inputs use seeds disjoint from the generator's. The same inputs
are used at every gain, so the rate is a deterministic, increasing function of
the gain; the search brackets the target by doubling or halving and then
bisects the gain geometrically until the rate is within `--tolerance` of the
target or the bracket is narrower than 2%. Rates count spikes after the first
500 ms. The output JSON maps morphology id to its gain and search history and
is read by `run_v1_parallel_generation --exc-gain-calibration`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from axobench.generation import run_v1_parallel_generation as gen

CALIBRATION_SEED = 7_700_000  # generator seeds start at 20260606
SCORED_FROM_MS = 500


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allen-all-active-root", type=Path, default=gen.ALLEN_ALL_ACTIVE_ROOT)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--traces", type=int, default=8)
    ap.add_argument("--duration-ms", type=int, default=6000)
    ap.add_argument("--tolerance", type=float, default=0.05, help="relative rate tolerance")
    ap.add_argument("--max-evaluations", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--morphology", action="append", help="specimen ids (default: all prepared models)")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--target-rate-hz", type=float, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(calibrate_one(args)))
        return 0

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    route = out.parent / "source_route.npz"
    gen.write_source_route(argparse.Namespace(python=args.python), out.parent)
    hay = _child(args, "hay", None, route)
    target = hay["rate_hz"]
    print(json.dumps({"morphology": gen.HAY_CELL_ID, **hay}), flush=True)
    cells = gen.allen_all_active_morphologies(args.allen_all_active_root)
    if args.morphology:
        cells = [c for c in cells if c["id"].split("_")[1] in args.morphology]

    def run(cell: dict[str, str]) -> tuple[str, dict[str, Any]]:
        row = _child(args, cell["path"], target, route)
        print(json.dumps({"morphology": cell["id"], "gain": row["gain"], "rate_hz": row["rate_hz"]}), flush=True)
        return cell["id"], row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = dict(pool.map(run, cells))
    result = {
        "target": {"morphology": gen.HAY_CELL_ID, **hay},
        "settings": {"traces": args.traces, "duration_ms": args.duration_ms, "seed": CALIBRATION_SEED,
                     "scored_from_ms": SCORED_FROM_MS, "tolerance": args.tolerance, "data_version": "v2"},
        "gains": {cid: row["gain"] for cid, row in rows.items()},
        "cells": rows,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _child(args: argparse.Namespace, one: str, target: float | None, route: Path) -> dict[str, Any]:
    cmd = [str(args.python), "-m", "axobench.generation.calibrate_exc_gain", "--out", str(args.out), "--one", one,
           "--traces", str(args.traces), "--duration-ms", str(args.duration_ms), "--tolerance", str(args.tolerance),
           "--max-evaluations", str(args.max_evaluations)]
    if target is not None:
        cmd += ["--target-rate-hz", str(target)]
    env = os.environ.copy()
    env["AXOBENCH_CALIBRATION_SOURCE_ROUTE"] = str(route)
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env, cwd=gen.ROOT)
    return json.loads(res.stdout.strip().splitlines()[-1])


def _worker_args(one: str, route: Path) -> argparse.Namespace:
    hay = one == "hay"
    args = gen.apply_data_version(gen.build_parser().parse_args(["--data-version", "v2"]))
    args.worker = True
    args.worker_teacher = "hay" if hay else "allen_all_active"
    args.worker_morphology_path = Path(gen.SOURCE_ROOT / "morphologies" / "cell1.asc" if hay else one)
    args.worker_source_route = route
    if not hay:
        args.nseg_policy = gen.ALLEN_NSEG_POLICY
    args.soma_samples_per_ms = 0  # rates only
    return args


def calibrate_one(args: argparse.Namespace) -> dict[str, Any]:
    wargs = _worker_args(args.one, Path(os.environ["AXOBENCH_CALIBRATION_SOURCE_ROUTE"]))
    setup = gen.prepare_neuron("neuron", False) if args.one == "hay" else gen.prepare_allen_all_active(wargs)
    inputs = np.stack([
        gen.sample_signed_inputs(setup["source_segment_types"], setup["source_segment_lengths"],
                                 duration_ms=args.duration_ms, seed=CALIBRATION_SEED + i)
        for i in range(args.traces)
    ]).astype(np.int8)
    seconds = (args.duration_ms - SCORED_FROM_MS) / 1000.0

    def rate(gain: float) -> float:
        wargs.exc_gmax_scale = gain
        traces = gen.simulate_batch(wargs, setup, inputs)
        return float(np.mean([t["targets"][SCORED_FROM_MS:, 0].sum() / seconds for t in traces]))

    if args.one == "hay":
        return {"gain": 1.0, "rate_hz": rate(1.0), "history": []}
    target = float(args.target_rate_hz)
    history: list[tuple[float, float]] = []

    def evaluate(gain: float) -> float:
        r = rate(gain)
        history.append((gain, r))
        return r

    def done(r: float) -> bool:
        return abs(r - target) <= args.tolerance * target

    lo: float | None = None  # largest gain with rate below target
    hi: float | None = None  # smallest gain with rate at or above target

    def record(gain: float, r: float) -> None:
        nonlocal lo, hi
        if r < target:
            lo = gain if lo is None else max(lo, gain)
        else:
            hi = gain if hi is None else min(hi, gain)

    gain = 1.0
    r = evaluate(gain)
    record(gain, r)
    while not done(r) and (lo is None or hi is None) and len(history) < args.max_evaluations:
        gain = gain * 2.0 if hi is None else gain / 2.0
        r = evaluate(gain)
        record(gain, r)
    while not done(r) and lo is not None and hi is not None and hi / lo > 1.02 and len(history) < args.max_evaluations:
        gain = float(np.sqrt(lo * hi))
        r = evaluate(gain)
        record(gain, r)
    best = min(history, key=lambda item: abs(item[1] - target))
    return {"gain": best[0], "rate_hz": best[1], "target_rate_hz": target, "converged": done(best[1]),
            "history": [{"gain": g, "rate_hz": v} for g, v in history]}


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
