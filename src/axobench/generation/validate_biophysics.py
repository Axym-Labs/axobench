#!/usr/bin/env python3
"""Compare the Hay biophysics on each morphology with the specimen's Allen electrophysiology.

AxoBench places the Hay et al. (2011) L5PC conductances on Allen Cell Types
reconstructions. This tool runs the Allen long-square current-clamp protocol on
each such cell and reports the features the Allen Cell Types Database measured
on the same specimen: resting potential, input resistance (from a -50 pA step),
rheobase (1 s steps, bisection to `--rheobase-resolution-pa`), f-I slope (spikes/s
per pA over the first `--fi-steps` suprathreshold steps of 20 pA), and the first
action potential at rheobase (threshold where dV/dt first exceeds 20 mV/ms, peak,
fast trough, upstroke/downstroke ratio). It also records whether an Allen
"all active" biophysical model exists for the specimen (Gouwens et al. 2018).

The Allen features are fetched once from api.brain-map.org and cached next to
the output. Simulations use the generator's v2 settings (d_lambda compartments,
Hay's 60 um axon stub, 34 degrees C from the Hay biophysics).

With `--allen-all-active-root` the same protocol runs on the Allen all-active
models prepared by `scripts/prepare_allen_all_active.py` instead (Allen's own
compartments, truncated axon, temperature and v_init). Each model loads its own
mechanism library, so each runs in a separate process. Allen fits its models to
recordings that are not corrected for the liquid junction potential (-14 mV in
every fit_parameters.json), so the model voltage minus the junction potential is
compared with the recorded features, as in AllenSDK.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ALLEN_API = "https://api.brain-map.org/api/v2/data/query.json?criteria="
FEATURES = ("vrest", "input_resistance_mohm", "threshold_i_long_square", "f_i_curve_slope", "threshold_v_long_square",
            "peak_v_long_square", "fast_trough_v_long_square", "upstroke_downstroke_ratio_long_square")


def fetch_allen(specimen_id: int, cache: Path) -> dict[str, Any]:
    """Allen ephys features and the list of biophysical models for one specimen (cached)."""
    path = cache / f"allen_{specimen_id}.json"
    if path.exists():
        return json.loads(path.read_text())

    def query(criteria: str) -> list[dict]:
        url = ALLEN_API + criteria.replace("[", "%5B").replace("]", "%5D")
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read())["msg"]

    feats = query(f"model::EphysFeature,rma::criteria,[specimen_id$eq{specimen_id}]")
    models = query(f"model::NeuronalModel,rma::criteria,[specimen_id$eq{specimen_id}],rma::include,neuronal_model_template")
    out = {"specimen_id": specimen_id,
           "features": {k: feats[0].get(k) for k in FEATURES} if feats else None,
           "models": [{"id": m["id"], "template": (m.get("neuronal_model_template") or {}).get("name")} for m in models]}
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


class StepCell:
    """One Hay-template cell with a somatic current clamp."""

    def __init__(self, h, morphology: Path, nseg_policy: str):
        from axobench.generation.run_coreneuron_event_dropout_pair import apply_d_lambda_nseg

        self.h = h
        self.cell = h.L5PCtemplate(str(morphology))
        if nseg_policy == "d_lambda":
            apply_d_lambda_nseg(h, self.cell)
        for section in list(self.cell.dend) + list(self.cell.apic):
            section.vshift_Ih = 0
        self.cell.soma[0].vshift_Ih = 0
        self.stim = h.IClamp(self.cell.soma[0](0.5))
        self.t, self.v = h.Vector(), h.Vector()
        self.t.record(h._ref_t)
        self.v.record(self.cell.soma[0](0.5)._ref_v)

    v_init_mv = -76.0
    junction_potential_mv = 0.0

    def run(self, amp_pa: float, *, delay=1000.0, dur=1000.0, tail=200.0) -> tuple[np.ndarray, np.ndarray]:
        h = self.h
        self.stim.delay, self.stim.dur, self.stim.amp = delay, dur, amp_pa / 1000.0
        h.dt = 0.025
        h.finitialize(self.v_init_mv)
        h.continuerun(delay + dur + tail)
        return np.asarray(self.t), np.asarray(self.v) - self.junction_potential_mv


class AllenStepCell(StepCell):
    """One Allen all-active model with a somatic current clamp."""

    def __init__(self, h, model_root: Path):
        from axobench.generation.allen_all_active import load_allen_all_active_cell

        self.h = h
        self.cell = load_allen_all_active_cell(h, model_root, mechanism_root=model_root / "runtime")
        self.v_init_mv = self.cell.v_init_mv
        fitting = json.loads((Path(model_root) / "fit_parameters.json").read_text())["fitting"][0]
        self.junction_potential_mv = float(fitting["junction_potential"])
        self.stim = h.IClamp(self.cell.soma[0](0.5))
        self.t, self.v = h.Vector(), h.Vector()
        self.t.record(h._ref_t)
        self.v.record(self.cell.soma[0](0.5)._ref_v)


def ap_features(t: np.ndarray, v: np.ndarray, start: float, stop: float) -> dict[str, Any]:
    """Spike count and first-AP features in [start, stop) ms (Allen 20 mV/ms threshold convention)."""
    m = (t >= start) & (t < stop)
    tt, vv = t[m], v[m]
    dvdt = np.gradient(vv, tt)
    up = np.flatnonzero((vv[1:] > -20) & (vv[:-1] <= -20))
    out: dict[str, Any] = {"n_spikes": int(up.size)}
    if up.size:
        i = int(up[0])
        k = i
        while k > 0 and dvdt[k] > 20:
            k -= 1
        peak = i + int(np.argmax(vv[i : i + int(2 / (tt[1] - tt[0]))]))
        end = up[1] if up.size > 1 else vv.size
        trough = peak + int(np.argmin(vv[peak : min(end, peak + int(5 / (tt[1] - tt[0])))]))
        out.update(threshold_v=float(vv[k + 1]), peak_v=float(vv[peak]), fast_trough_v=float(vv[trough]),
                   upstroke_downstroke_ratio=float(dvdt[k:peak + 1].max() / -dvdt[peak:trough + 1].min()))
    return out


def validate(cell: StepCell, *, resolution_pa: float, fi_steps: int) -> dict[str, Any]:
    t, v = cell.run(-50.0)
    vrest = float(v[(t > 900) & (t < 1000)].mean())
    vss = float(v[(t > 1900) & (t < 2000)].mean())
    rin = (vss - vrest) / -50.0 * 1000.0  # MOhm
    lo, hi = 0.0, 50.0
    while ap_features(*cell.run(hi), 1000, 2000)["n_spikes"] == 0:
        lo, hi = hi, hi * 2
        if hi > 5000:
            return {"vrest": vrest, "input_resistance_mohm": rin, "threshold_i_long_square": None}
    while hi - lo > resolution_pa:
        mid = 0.5 * (lo + hi)
        if ap_features(*cell.run(mid), 1000, 2000)["n_spikes"] > 0:
            hi = mid
        else:
            lo = mid
    first = ap_features(*cell.run(hi), 1000, 2000)
    amps = hi + 20.0 * np.arange(fi_steps)
    rates = [ap_features(*cell.run(a), 1000, 2000)["n_spikes"] for a in amps]
    slope = float(np.polyfit(amps, rates, 1)[0]) if fi_steps > 1 else None
    return {"vrest": vrest, "input_resistance_mohm": rin, "threshold_i_long_square": hi, "f_i_curve_slope": slope,
            "threshold_v_long_square": first.get("threshold_v"), "peak_v_long_square": first.get("peak_v"),
            "fast_trough_v_long_square": first.get("fast_trough_v"),
            "upstroke_downstroke_ratio_long_square": first.get("upstroke_downstroke_ratio"),
            "fi_amps_pa": amps.tolist(), "fi_rates_hz": rates}


def main() -> int:
    from axobench.generation import run_v1_parallel_generation as gen

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output JSON; Allen responses are cached next to it")
    ap.add_argument("--morphology", action="append", help="specimen id or neuronio_hay_cell1 (default: all v2 morphologies)")
    ap.add_argument("--nseg-policy", default="d_lambda")
    ap.add_argument("--rheobase-resolution-pa", type=float, default=10.0)
    ap.add_argument("--fi-steps", type=int, default=5)
    ap.add_argument("--allen-all-active-root", type=Path,
                    help="validate the all-active models under this root (specimen_<id>/model) instead")
    ap.add_argument("--one-allen-model", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel Allen models")
    args = ap.parse_args()
    if args.one_allen_model:
        from neuron import h

        result = validate(AllenStepCell(h, args.one_allen_model), resolution_pa=args.rheobase_resolution_pa,
                          fi_steps=args.fi_steps)
        print(json.dumps(result))
        return 0
    if args.allen_all_active_root:
        return validate_all_active(args)

    setup = gen.prepare_neuron("neuron", False)
    h = setup["h"]
    rows = {}
    for m in gen.selected_morphologies(args.morphology, include_hay_cell=True):
        row: dict[str, Any] = {"morphology": m["id"]}
        if m["id"].startswith("specimen_"):
            row["allen"] = fetch_allen(int(m["id"].split("_")[1]), args.out.parent / "allen_cache")
        row["model"] = validate(StepCell(h, Path(m["path"]), args.nseg_policy),
                                resolution_pa=args.rheobase_resolution_pa, fi_steps=args.fi_steps)
        rows[m["id"]] = row
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps({"nseg_policy": args.nseg_policy, "morphologies": rows}, indent=2))
    return 0


def validate_all_active(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    roots = [root for root in sorted(args.allen_all_active_root.glob("specimen_*/model"))
             if not args.morphology or root.parent.name.split("_")[1] in args.morphology]

    def one(model_root: Path) -> dict[str, Any]:
        specimen = int(model_root.parent.name.split("_")[1])
        out = subprocess.run([sys.executable, "-m", "axobench.generation.validate_biophysics", "--out", str(args.out),
                              "--one-allen-model", str(model_root),
                              "--rheobase-resolution-pa", str(args.rheobase_resolution_pa),
                              "--fi-steps", str(args.fi_steps)],
                             capture_output=True, text=True, check=True)
        row = {"morphology": f"specimen_{specimen}", "allen": fetch_allen(specimen, args.out.parent / "allen_cache"),
               "model": json.loads(out.stdout.strip().splitlines()[-1])}
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = {row["morphology"]: row for row in pool.map(one, roots)}
    args.out.write_text(json.dumps({"teacher": "allen_all_active", "morphologies": rows}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
