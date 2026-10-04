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
"""

from __future__ import annotations

import argparse
import json
import os
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

    def run(self, amp_pa: float, *, delay=1000.0, dur=1000.0, tail=200.0) -> tuple[np.ndarray, np.ndarray]:
        h = self.h
        self.stim.delay, self.stim.dur, self.stim.amp = delay, dur, amp_pa / 1000.0
        h.dt = 0.025
        h.finitialize(-76.0)
        h.continuerun(delay + dur + tail)
        return np.asarray(self.t), np.asarray(self.v)


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


def validate(h, morphology: Path, *, nseg_policy: str, resolution_pa: float, fi_steps: int) -> dict[str, Any]:
    cell = StepCell(h, morphology, nseg_policy)
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
    args = ap.parse_args()

    setup = gen.prepare_neuron("neuron", False)
    h = setup["h"]
    rows = {}
    for m in gen.selected_morphologies(args.morphology, include_hay_cell=True):
        row: dict[str, Any] = {"morphology": m["id"]}
        if m["id"].startswith("specimen_"):
            row["allen"] = fetch_allen(int(m["id"].split("_")[1]), args.out.parent / "allen_cache")
        row["model"] = validate(h, Path(m["path"]), nseg_policy=args.nseg_policy,
                                resolution_pa=args.rheobase_resolution_pa, fi_steps=args.fi_steps)
        rows[m["id"]] = row
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps({"nseg_policy": args.nseg_policy, "morphologies": rows}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
