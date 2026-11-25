#!/usr/bin/env python3
"""Generate NeuronIO-like morphology shards with parallel NEURON workers.

`--data-version v1` (default) reproduces the released v1 settings: 1 ms NetCon
delay, `normalized_by_type` routing (one input site per dendritic segment, so
42-59% of NeuronIO's sites stay unconnected), Hay's 40 um compartments, 1 ms
voltage. `--data-version v2` sets delay 0 (as NeuronIO), connects all 1,278
input sites (`all_sites_by_type`), uses d_lambda compartments, stores the raw
soma voltage at 4 kHz as well, adds the original NeuronIO (Hay) cell as a
morphology, and uses excitatory gmax scale 1.0. Every option can be overridden
individually, and every shard sidecar records the complete generator settings.

Long runs can be paused and resumed. Shards are written atomically (temporary
file, then rename), and `--resume` skips every complete shard. Creating the file
`<output-root>/STOP` lets the running shards finish, starts no new ones, writes
the manifests and exits; SIGINT or SIGTERM stops at once and kills the running
workers, losing only their unfinished shards. Tasks are interleaved across
morphologies, so a paused run holds a balanced partial dataset.

`--excitatory-teacher` chooses the biophysics of the Allen cells: `hay` puts
the Hay et al. (2011) densities on the v1 reconstructions; `allen_all_active`
(v2) uses the Allen all-active models (Gouwens et al. 2018) prepared by
`scripts/prepare_allen_all_active.py`, each with its own channels, compartments
and v_init. The NeuronIO (Hay) cell keeps the Hay biophysics in both cases.
Allen and Hay channel files share names, so an Allen worker never loads the Hay
mechanisms; it reads the NeuronIO site map from `source_route.npz`, which the
orchestrator writes once per output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

# CoreNEURON GPU uses the NVHPC/OpenACC runtime. Import it before other
# accelerator-aware libraries so the worker does not later abort during GPU
# discovery.
from neuron import coreneuron as _early_coreneuron  # noqa: F401
from neuron import h as _early_h  # noqa: F401


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(os.environ.get("AXOBENCH_GENERATION_ARTIFACTS", ROOT / "artifacts" / "generation")).resolve()
CORENEURON_PROBE = ARTIFACTS / "coreneuron_hay_port_probe_20260605"
SWC_MECH_ROOT = Path(os.environ.get("AXOBENCH_SWC_MECH_ROOT", ARTIFACTS / "hay_swc_template")).resolve()
SOURCE_ROOT = Path(
    os.environ.get(
        "AXOBENCH_SOURCE_ROOT",
        ARTIFACTS / "source_models" / "neuron_as_deep_net" / "L5PC_NEURON_simulation",
    )
).resolve()
TEMPLATE_HOC = Path(os.environ.get("AXOBENCH_SWC_TEMPLATE_HOC", SWC_MECH_ROOT / "L5PCtemplate_swc.hoc")).resolve()
ALLEN_ALL_ACTIVE_ROOT = Path(os.environ.get("AXOBENCH_ALLEN_ALL_ACTIVE_ROOT", ARTIFACTS / "allen_all_active")).resolve()
DEFAULT_PYTHON = Path(".venv/bin/python")

from axobench.benchmark import make_structured_intervention_inputs
from axobench.benchmark.dataset_schema import (
    INTERVENTION_CONDITIONS,
    OOD_DIR,
    build_v1_dataset_manifest_template,
    v1_morphologies,
)
from axobench.generation.generate_hay_neuron_dataset import SOMA_BIAS_MV, SOMA_CLIP_THRESHOLD_MV, SOMA_SCALE, local_peak_spike_labels
from axobench.generation.run_coreneuron_event_dropout_pair import (
    NSEG_POLICIES,
    _build_source_site_route,
    _segment_route,
    _summarize_input_routing,
    simulate_signed_trace_batch,
)
from axobench.generation.run_coreneuron_hay_probe import load_probe_mechanisms
from axobench.generation.run_hay_neuron_driven_probe import collect_segments, sample_neuronio_inputs


DATA_VERSIONS: dict[str, dict[str, Any]] = {
    "v1": {"synapse_delay_ms": 1.0, "input_routing": "normalized_by_type", "nseg_policy": "hay40",
           "soma_samples_per_ms": 0, "include_hay_cell": False, "intervention_dose": [],
           # The released v1 shards reproduce exactly with 1.0, not the 0.925 the old default said
           # (axobench-internal ta1 results.md, exp6: 8/8 identical traces at 1.0, 0/8 at 0.925).
           "exc_gmax_scale": 1.0,
           "excitatory_teacher": "hay"},
    "v2": {"synapse_delay_ms": 0.0, "input_routing": "all_sites_by_type", "nseg_policy": "d_lambda",
           "soma_samples_per_ms": 4, "include_hay_cell": True, "exc_gmax_scale": 1.0,
           "excitatory_teacher": "allen_all_active",
           # A weak dose, the v1 default, and a dose past the disinhibition breakdown (article A, exp27).
           "intervention_dose": ["event_dropout=0.1,0.35,0.7", "exc_dropout=0.1,0.35,0.7", "inh_dropout=0.1,0.35,0.7",
                                 "temporal_jitter=1,2,8", "site_silence=0.1,0.25,0.5"]},
}
HAY_CELL_ID = "neuronio_hay_cell1"
# Extension train shards (beyond --layout-train-samples-per-morphology) take indices
# EXTENSION_BASE + rank * EXTENSION_STRIDE + k, above every layout index of the v2 variants.
EXTENSION_BASE = 20_000
EXTENSION_STRIDE = 5_000
# Intervention strength: the parameter each condition's dose sets, and its v1 default.
DOSE_PARAMETER = {"event_dropout": "dropout_probability", "exc_dropout": "dropout_probability",
                  "inh_dropout": "dropout_probability", "temporal_jitter": "jitter_steps", "site_silence": "site_fraction"}
DEFAULT_DOSE = {"event_dropout": 0.35, "exc_dropout": 0.35, "inh_dropout": 0.35, "temporal_jitter": 2.0,
                "site_silence": 0.25}
# Allen all-active models keep the compartments they were fitted with (1 + 2 * int(L / 40)).
ALLEN_NSEG_POLICY = "allen_fit"


def apply_data_version(args: argparse.Namespace) -> argparse.Namespace:
    """Fill every generator option left unset with the preset of `--data-version`."""
    for key, value in DATA_VERSIONS[args.data_version].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main() -> int:
    parser = build_parser()
    args = apply_data_version(parser.parse_args())
    validate_positive(args.exc_gmax_scale, "exc-gmax-scale")
    validate_positive(args.inh_gmax_scale, "inh-gmax-scale")
    if args.worker:
        code = worker_main(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return orchestrator_main(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS / "v1_dataset_smoke_20260606")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--samples-per-shard", type=int, default=128)
    parser.add_argument(
        "--intervention-samples-per-shard",
        type=int,
        help="Shard size for paired intervention examples. Defaults to --samples-per-shard.",
    )
    parser.add_argument("--duration-ms", type=int, default=6000)
    parser.add_argument("--backend", choices=["neuron", "coreneuron"], default="neuron")
    parser.add_argument("--coreneuron-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument(
        "--teacher-balance-label",
        default="axobench_ei_balanced",
        help="Human-readable label for the AxoBench teacher calibration stored in metadata.",
    )
    parser.add_argument("--data-version", choices=sorted(DATA_VERSIONS), default="v1",
                        help="Preset for the options below that are not given explicitly.")
    parser.add_argument("--synapse-delay-ms", type=float, help="NetCon delay (v1 1.0, v2 0.0 as NeuronIO).")
    parser.add_argument("--input-routing", choices=["normalized_by_type", "all_sites_by_type"])
    parser.add_argument("--nseg-policy", choices=[*NSEG_POLICIES, ALLEN_NSEG_POLICY],
                        help="Compartments of Hay-biophysics cells; Allen all-active cells always use allen_fit.")
    parser.add_argument("--excitatory-teacher", choices=["hay", "allen_all_active"],
                        help="Biophysics of the Allen cells (v1 hay, v2 allen_all_active).")
    parser.add_argument("--allen-all-active-root", type=Path, default=ALLEN_ALL_ACTIVE_ROOT,
                        help="Output root of scripts/prepare_allen_all_active.py.")
    parser.add_argument("--exc-gain-calibration", type=Path,
                        help="Output of calibrate_exc_gain: per-cell factors applied on top of --exc-gmax-scale. "
                             "Every Allen all-active morphology must be listed.")
    parser.add_argument("--soma-samples-per-ms", type=int, help="Also store raw soma voltage at this rate (0 = 1 ms only).")
    parser.add_argument("--include-hay-cell", action=argparse.BooleanOptionalAction, default=None,
                        help="Add the original NeuronIO (Hay) cell as a morphology.")
    parser.add_argument("--heldout-morphology", action="append", default=[],
                        help="Morphology id (specimen id or neuronio_hay_cell1) that gets no train/val samples.")
    parser.add_argument("--exc-gmax-scale", type=float)
    parser.add_argument("--inh-gmax-scale", type=float, default=1.0)
    parser.add_argument("--train-samples-per-morphology", type=int, default=0)
    parser.add_argument("--layout-train-samples-per-morphology", type=int,
                        help="Train traces per morphology that define the shard layout (default: all of them). "
                             "Traces beyond it get shard indices from a separate extension range, so a larger "
                             "variant keeps every shard, index and seed of the smaller one it extends.")
    parser.add_argument("--generate-morphology", action="append", default=[],
                        help="Specimen id (or the Hay cell id) to generate; repeatable. Unlike --morphology it "
                             "does not change the layout, so machines that split the morphologies write the "
                             "shards a single run would write.")
    parser.add_argument("--extension-only", action="store_true",
                        help="Generate only the train shards beyond --layout-train-samples-per-morphology.")
    parser.add_argument("--val-samples-per-morphology", type=int, default=1)
    parser.add_argument("--test-samples-per-morphology", type=int, default=0)
    parser.add_argument("--private-test-samples-per-morphology", type=int, default=0)
    parser.add_argument("--intervention-samples-per-morphology", type=int, default=1)
    parser.add_argument(
        "--intervention-total-samples-per-morphology",
        type=int,
        help=(
            "Total paired intervention examples per morphology, distributed across the selected "
            "interventions. Prefer this for v1 full generation because the dataset target defines "
            "the intervention category total, not a per-condition count."
        ),
    )
    parser.add_argument("--intervention", action="append", choices=INTERVENTION_CONDITIONS)
    parser.add_argument("--intervention-dose", action="append", metavar="CONDITION=D1,D2,...",
                        help="Generate a condition at several strengths (dropout or site fraction, jitter in ms), each "
                             "in its own folder, e.g. inh_dropout_p70. Without it a condition uses its v1 default.")
    parser.add_argument("--morphology", action="append", help="Specimen id to include; defaults to all v1 morphologies")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip existing valid shard outputs and rebuild manifests/summaries.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-kind", choices=["split", "intervention", "source-route"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-teacher", choices=["hay", "allen_all_active"], default="hay", help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-route", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-morphology-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-morphology-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-split", help=argparse.SUPPRESS)
    parser.add_argument("--worker-intervention", choices=INTERVENTION_CONDITIONS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-intervention-dose", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-input-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-shard-index", type=int, help=argparse.SUPPRESS)
    return parser


def orchestrator_main(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "STOP").exists():
        print(f"removing {output_root / 'STOP'} left by the previous run", flush=True)
        (output_root / "STOP").unlink()

    morphologies = selected_morphologies(args.morphology, include_hay_cell=bool(args.include_hay_cell),
                                         excitatory_teacher=args.excitatory_teacher,
                                         allen_root=args.allen_all_active_root)
    unknown = set(args.heldout_morphology) - {m["id"].replace("specimen_", "") for m in morphologies} - {HAY_CELL_ID}
    if unknown:
        raise ValueError(f"held-out morphologies not among the selected ones: {sorted(unknown)}")
    unknown = set(args.generate_morphology) - {m["id"].replace("specimen_", "") for m in morphologies} - {HAY_CELL_ID}
    if unknown:
        raise ValueError(f"--generate-morphology ids not among the selected ones: {sorted(unknown)}")
    interventions = tuple(args.intervention or INTERVENTION_CONDITIONS)
    variants = intervention_variants(args, interventions)
    intervention_counts = intervention_counts_per_morphology(args, interventions)
    tasks = select_tasks(build_tasks(args, output_root, morphologies, interventions),
                         args.generate_morphology, args.extension_only)
    total_tasks = len(tasks)
    total_samples = sum(task["samples"] for task in tasks)
    skipped_tasks = 0
    skipped_samples = 0
    if args.resume:
        remaining_tasks = []
        for task in tasks:
            if task_is_complete(task):
                skipped_tasks += 1
                skipped_samples += int(task["samples"])
            else:
                remaining_tasks.append(task)
        tasks = remaining_tasks
    started = time.perf_counter()
    if any(task["teacher"] == "allen_all_active" for task in tasks):
        write_source_route(args, output_root)
    finished = run_tasks(interleave_by_morphology(tasks), args.parallelism, stop_file=output_root / "STOP")
    write_split_manifests(output_root)
    write_morphology_index(args, output_root, morphologies)
    summary = {
        "kind": "v1_parallel_generation",
        "output_root": str(output_root),
        "layout_train_samples_per_morphology": args.layout_train_samples_per_morphology,
        "generate_morphology": sorted(args.generate_morphology),
        "extension_only": bool(args.extension_only),
        "backend": args.backend,
        "coreneuron_gpu": bool(args.coreneuron_gpu),
        "duration_ms": int(args.duration_ms),
        "teacher_balance_label": args.teacher_balance_label,
        "input_event_policy": "source_neuronio_sampler_unmodified",
        "synaptic_conductance_scale": {
            "excitatory": float(args.exc_gmax_scale),
            "inhibitory": float(args.inh_gmax_scale),
        },
        "generator": generator_settings(args),
        "excitatory_teacher": args.excitatory_teacher,
        "exc_gain_calibration": None if args.exc_gain_calibration is None else {
            "path": str(args.exc_gain_calibration), "sha256": _sha256(args.exc_gain_calibration),
            "gains": exc_gains(args, morphologies)},
        "heldout_morphologies": sorted(args.heldout_morphology),
        "parallelism": int(args.parallelism),
        "samples_per_shard": int(args.samples_per_shard),
        "intervention_samples_per_shard": int(intervention_samples_per_shard(args)),
        "elapsed_seconds": time.perf_counter() - started,
        "morphologies": [morphology["id"] for morphology in morphologies],
        "interventions": list(interventions),
        "intervention_variants": {name: {"condition": cond, "dose": dose} for name, cond, dose in variants},
        "intervention_samples_per_morphology_by_condition": intervention_counts,
        "intervention_total_samples_per_morphology": sum(intervention_counts.values()),
        "tasks": len(tasks),
        "tasks_finished_this_run": finished,
        "complete": finished == len(tasks),
        "tasks_total": total_tasks,
        "tasks_skipped_existing": skipped_tasks,
        "samples": sum(task["samples"] for task in tasks),
        "samples_total": total_samples,
        "samples_skipped_existing": skipped_samples,
        "source_root": str(SOURCE_ROOT),
        "mechanism_root": str(SWC_MECH_ROOT),
        "template_hoc": str(TEMPLATE_HOC),
        "note": (
            "SWC morphology path uses the Hay-compatible SWC template/mechanisms. "
            "CoreNEURON GPU is supported for actual v1 SWC morphologies when the durable GPU runtime "
            "and SWC mechanism root are on PATH/PYTHONPATH/LD_LIBRARY_PATH."
        ),
    }
    (output_root / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "dataset_manifest_template.json").write_text(
        json.dumps(build_v1_dataset_manifest_template(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def selected_morphologies(
    selected: list[str] | None,
    include_hay_cell: bool = False,
    excitatory_teacher: str = "hay",
    allen_root: Path | None = None,
) -> list[dict[str, str]]:
    wanted_raw = set(selected or [])
    wanted = {int(item) for item in wanted_raw if item != HAY_CELL_ID} if selected else None
    rows = []
    if include_hay_cell and (not selected or HAY_CELL_ID in wanted_raw):
        rows.append({"id": HAY_CELL_ID, "path": str((SOURCE_ROOT / "morphologies" / "cell1.asc").resolve()),
                     "teacher": "hay"})
    if excitatory_teacher == "allen_all_active":
        rows += [row for row in allen_all_active_morphologies(allen_root or ALLEN_ALL_ACTIVE_ROOT)
                 if wanted is None or int(row["id"].split("_")[1]) in wanted]
        if not rows:
            raise ValueError("no selected morphologies")
        return rows
    for morphology in v1_morphologies():
        if wanted is not None and morphology.specimen_id not in wanted:
            continue
        rows.append(
            {
                "id": f"specimen_{morphology.specimen_id}",
                "path": str((ROOT / morphology.swc_path).resolve()),
                "teacher": "hay",
            }
        )
    if not rows:
        raise ValueError("no selected morphologies")
    return rows


def allen_all_active_morphologies(root: Path) -> list[dict[str, str]]:
    """One row per prepared Allen all-active model, in the order of `models.json`."""
    index = root / "models.json"
    if not index.exists():
        raise FileNotFoundError(f"{index} not found; run scripts/prepare_allen_all_active.py first")
    return [
        {"id": f"specimen_{row['specimen_id']}", "path": str(Path(row["model_root"]).resolve()),
         "teacher": "allen_all_active", "model_id": str(row["model_id"])}
        for row in json.loads(index.read_text(encoding="utf-8"))
    ]


def write_morphology_index(args: argparse.Namespace, output_root: Path, morphologies: list[dict[str, str]]) -> Path:
    """Write `morphologies.json` and copy each Allen reconstruction to `morphologies/<id>.swc`.

    Models in the conditioned and zero-shot tracks get each sample's
    morphology id (`supports_morphology_ids`); this index lets them look up the
    cell itself (reconstruction, Cre line, held-out status, excitatory gain).
    The NeuronIO cell's morphology is referenced, not copied: it is part of the
    Hay et al. (2011) model files, which carry no licence.
    """
    gains = exc_gains(args, morphologies)
    heldout = {f"specimen_{item}" if item != HAY_CELL_ID else item for item in args.heldout_morphology}
    rows = []
    (output_root / "morphologies").mkdir(exist_ok=True)
    for morphology in morphologies:
        row: dict[str, Any] = {"id": morphology["id"], "teacher": morphology.get("teacher", "hay"),
                               "heldout": morphology["id"] in heldout,
                               "exc_gmax_scale": float(args.exc_gmax_scale) * gains[morphology["id"]]}
        if morphology["id"] == HAY_CELL_ID:
            row.update(species="rat", source="Hay et al. 2011, ModelDB 139653, morphologies/cell1.asc "
                                            "(as distributed with github.com/SelfishGene/neuron_as_deep_net)")
        elif row["teacher"] == "allen_all_active":
            root = Path(morphology["path"])
            meta = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
            swc = output_root / "morphologies" / f"{morphology['id']}.swc"
            shutil.copy2(root / "reconstruction.swc", swc)
            row.update(species="mouse", allen_specimen_id=int(meta["specimen_id"]), allen_model_id=int(meta["id"]),
                       transgenic_line=str(meta.get("name", "")).split("_", 1)[-1].split(";")[0] or None,
                       structure=((meta.get("specimen") or {}).get("structure") or {}).get("acronym"),
                       swc=str(swc.relative_to(output_root)),
                       source="Allen Cell Types Database, celltypes.brain-map.org (Allen Institute Terms of Use)")
        else:
            swc = output_root / "morphologies" / f"{morphology['id']}.swc"
            shutil.copy2(morphology["path"], swc)
            row.update(species="mouse", swc=str(swc.relative_to(output_root)),
                       source="Allen Cell Types Database, celltypes.brain-map.org (Allen Institute Terms of Use)")
        rows.append(row)
    path = output_root / "morphologies.json"
    path.write_text(json.dumps({"version": 1, "morphologies": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_source_route(args: argparse.Namespace, output_root: Path) -> Path:
    """Write the NeuronIO site map from a process that has the Hay mechanisms loaded."""
    path = output_root / "source_route.npz"
    if not path.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT / 'src'}:{ARTIFACTS}:{CORENEURON_PROBE}:{env.get('PYTHONPATH', '')}"
        subprocess.run([str(args.python), "-m", "axobench.generation.run_v1_parallel_generation", "--worker",
                        "--worker-kind", "source-route", "--worker-output", str(path), "--worker-samples", "0"],
                       cwd=ROOT, env=env, check=True)
    return path


def build_tasks(
    args: argparse.Namespace,
    output_root: Path,
    morphologies: list[dict[str, str]],
    interventions: tuple[str, ...],
) -> list[dict[str, Any]]:
    if args.samples_per_shard <= 0:
        raise ValueError(f"--samples-per-shard must be positive, got {args.samples_per_shard}")
    pair_samples_per_shard = intervention_samples_per_shard(args)
    split_counts = {
        "train": args.train_samples_per_morphology,
        "val": args.val_samples_per_morphology,
        "test": args.test_samples_per_morphology,
        "private-test": args.private_test_samples_per_morphology,
    }
    intervention_counts = intervention_counts_per_morphology(args, interventions)
    gains = exc_gains(args, morphologies)
    layout_train = getattr(args, "layout_train_samples_per_morphology", None)
    if layout_train is None:
        layout_train = args.train_samples_per_morphology
    if layout_train > args.train_samples_per_morphology or (
        layout_train < args.train_samples_per_morphology and layout_train % args.samples_per_shard
    ):
        raise ValueError("--layout-train-samples-per-morphology must be a multiple of --samples-per-shard "
                         "and at most --train-samples-per-morphology")
    extension_shards = -(-(args.train_samples_per_morphology - layout_train) // args.samples_per_shard)
    if extension_shards > EXTENSION_STRIDE:
        raise ValueError(f"at most {EXTENSION_STRIDE * args.samples_per_shard} extension train traces per morphology")
    tasks = []
    shard_index = 0
    heldout = {f"specimen_{item}" if item != HAY_CELL_ID else item for item in getattr(args, "heldout_morphology", [])}
    for rank, morphology in enumerate(morphologies):
        for split, samples in split_counts.items():
            if samples <= 0 or (morphology["id"] in heldout and split in ("train", "val")):
                continue
            layout_samples = layout_train if split == "train" else samples
            for offset in range(0, samples, args.samples_per_shard):
                shard_samples = min(args.samples_per_shard, samples - offset)
                if offset < layout_samples:
                    index = shard_index
                    shard_index += 1
                else:  # extension train shard: index from a range disjoint from the layout's
                    index = EXTENSION_BASE + rank * EXTENSION_STRIDE + (offset - layout_samples) // args.samples_per_shard
                tasks.append(
                    task_payload(
                        args,
                        "split",
                        output_root / split / f"{morphology['id']}_{index:05d}.npz",
                        morphology,
                        split,
                        None,
                        shard_samples,
                        index,
                        exc_gain=gains[morphology["id"]],
                    )
                )
                tasks[-1]["extension"] = offset >= layout_samples
        for name, condition, dose in intervention_variants(args, interventions):
            intervention_samples = intervention_counts[name]
            if intervention_samples <= 0:
                continue
            for offset in range(0, intervention_samples, pair_samples_per_shard):
                shard_samples = min(pair_samples_per_shard, intervention_samples - offset)
                task = task_payload(
                    args,
                    "intervention",
                    output_root / OOD_DIR / name / f"{morphology['id']}_intervention_pair_{shard_index:05d}.npz",
                    morphology,
                    "",
                    condition,
                    shard_samples,
                    shard_index,
                    exc_gain=gains[morphology["id"]],
                )
                task["intervention_dose"] = float(dose)
                task["extension"] = False
                tasks.append(task)
                shard_index += 1
    return tasks


def select_tasks(tasks: list[dict[str, Any]], generate_morphology: list[str], extension_only: bool) -> list[dict[str, Any]]:
    """Restrict a full layout to the morphologies and shards this machine generates."""
    wanted = {item if item == HAY_CELL_ID else f"specimen_{item}" for item in generate_morphology}
    return [task for task in tasks
            if (not wanted or task["morphology_id"] in wanted) and (not extension_only or task["extension"])]


def exc_gains(args: argparse.Namespace, morphologies: list[dict[str, str]]) -> dict[str, float]:
    """Per-morphology excitatory gain from `--exc-gain-calibration` (1.0 without it, and for Hay cells)."""
    path = getattr(args, "exc_gain_calibration", None)
    if path is None:
        return {m["id"]: 1.0 for m in morphologies}
    calibrated = json.loads(Path(path).read_text(encoding="utf-8"))["gains"]
    missing = [m["id"] for m in morphologies if m.get("teacher") == "allen_all_active" and m["id"] not in calibrated]
    if missing:
        raise ValueError(f"no calibrated excitatory gain for {missing} in {path}")
    return {m["id"]: float(calibrated.get(m["id"], 1.0)) if m.get("teacher") == "allen_all_active" else 1.0
            for m in morphologies}


def intervention_samples_per_shard(args: argparse.Namespace) -> int:
    value = args.intervention_samples_per_shard
    if value is None:
        value = args.samples_per_shard
    value = int(value)
    if value <= 0:
        raise ValueError(f"--intervention-samples-per-shard must be positive, got {value}")
    return value


def intervention_variants(args: argparse.Namespace, interventions: tuple[str, ...]) -> list[tuple[str, str, float]]:
    """(folder name, condition, dose) for every requested condition and dose.

    A condition without `--intervention-dose` keeps its v1 folder name and default dose;
    with doses, each gets a folder named by the dose (`_p10` = 10%, `_j8` = up to 8 ms).
    """
    doses: dict[str, list[float]] = {}
    for item in getattr(args, "intervention_dose", None) or []:
        condition, _, values = item.partition("=")
        if condition not in DOSE_PARAMETER or not values:
            raise ValueError(f"--intervention-dose expects CONDITION=D1,D2,... with a known condition, got {item!r}")
        doses[condition] = [float(v) for v in values.split(",")]
    out = []
    for condition in interventions:
        if condition not in doses:
            out.append((condition, condition, DEFAULT_DOSE[condition]))
            continue
        for dose in doses[condition]:
            label = f"j{dose:g}" if condition == "temporal_jitter" else f"p{round(100 * dose):02d}"
            out.append((f"{condition}_{label}", condition, dose))
    return out


def intervention_counts_per_morphology(args: argparse.Namespace, interventions: tuple[str, ...]) -> dict[str, int]:
    """Pairs per morphology for each intervention variant (folder name)."""
    interventions = tuple(name for name, _, _ in intervention_variants(args, interventions))
    if not interventions:
        return {}
    total = args.intervention_total_samples_per_morphology
    if total is None:
        count = int(args.intervention_samples_per_morphology)
        if count < 0:
            raise ValueError(f"--intervention-samples-per-morphology must be non-negative, got {count}")
        return {intervention: count for intervention in interventions}
    if total < 0:
        raise ValueError(f"--intervention-total-samples-per-morphology must be non-negative, got {total}")
    base, remainder = divmod(int(total), len(interventions))
    return {
        intervention: base + (1 if index < remainder else 0)
        for index, intervention in enumerate(interventions)
    }


def task_payload(
    args: argparse.Namespace,
    kind: str,
    output: Path,
    morphology: dict[str, str],
    split: str,
    intervention: str | None,
    samples: int,
    shard_index: int,
    exc_gain: float = 1.0,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "output": str(output),
        "morphology_id": morphology["id"],
        "morphology_path": morphology["path"],
        "teacher": morphology.get("teacher", "hay"),
        "source_route": str(output.parents[2 if kind == "intervention" else 1] / "source_route.npz"),
        "split": split,
        "intervention": intervention,
        "samples": int(samples),
        "shard_index": int(shard_index),
        "duration_ms": int(args.duration_ms),
        "backend": args.backend,
        "coreneuron_gpu": bool(args.coreneuron_gpu),
        "seed": int(args.seed + 1009 * shard_index),
        "teacher_balance_label": args.teacher_balance_label,
        "exc_gmax_scale": float(args.exc_gmax_scale) * float(exc_gain),
        "inh_gmax_scale": float(args.inh_gmax_scale),
        "data_version": args.data_version,
        "synapse_delay_ms": float(args.synapse_delay_ms),
        "input_routing": args.input_routing,
        "nseg_policy": ALLEN_NSEG_POLICY if morphology.get("teacher") == "allen_all_active" else args.nseg_policy,
        "soma_samples_per_ms": int(args.soma_samples_per_ms),
        "python": str(args.python),
    }


def interleave_by_morphology(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin over morphologies, keeping each morphology's task order (seeds depend on shard index only)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        groups.setdefault(task["morphology_id"], []).append(task)
    ordered = []
    for offset in range(max((len(g) for g in groups.values()), default=0)):
        ordered += [g[offset] for g in groups.values() if offset < len(g)]
    return ordered


def run_tasks(tasks: list[dict[str, Any]], parallelism: int, stop_file: Path | None = None) -> int:
    """Run worker tasks; returns the number that finished.

    If `stop_file` appears, no new task starts and the running ones are awaited.
    SIGINT or SIGTERM terminates the running workers and raises KeyboardInterrupt.
    """
    running: list[tuple[subprocess.Popen, dict[str, Any]]] = []
    pending = list(tasks)
    finished = 0

    def stop_now(signum, _frame):
        for proc, _ in running:
            proc.terminate()
        for proc, _ in running:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise KeyboardInterrupt(f"stopped by signal {signum}; rerun with --resume")

    previous = {sig: signal.signal(sig, stop_now) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or running:
            if pending and stop_file is not None and stop_file.exists():
                print(f"{stop_file} found: finishing {len(running)} running tasks, {len(pending)} left for --resume",
                      flush=True)
                pending = []
            while pending and len(running) < parallelism:
                task = pending.pop(0)
                cmd = [
                    task["python"],
                    "-m",
                    "axobench.generation.run_v1_parallel_generation",
                    "--worker",
                    "--worker-kind",
                    task["kind"],
                    "--worker-output",
                    task["output"],
                    "--worker-morphology-id",
                    task["morphology_id"],
                    "--worker-morphology-path",
                    task["morphology_path"],
                    "--worker-split",
                    task["split"],
                    "--worker-samples",
                    str(task["samples"]),
                    "--worker-shard-index",
                    str(task["shard_index"]),
                    "--duration-ms",
                    str(task["duration_ms"]),
                    "--backend",
                    task["backend"],
                    "--seed",
                    str(task["seed"]),
                    "--teacher-balance-label",
                    task["teacher_balance_label"],
                    "--exc-gmax-scale",
                    str(task["exc_gmax_scale"]),
                    "--inh-gmax-scale",
                    str(task["inh_gmax_scale"]),
                    "--data-version",
                    task["data_version"],
                    "--synapse-delay-ms",
                    str(task["synapse_delay_ms"]),
                    "--input-routing",
                    task["input_routing"],
                    "--nseg-policy",
                    task["nseg_policy"],
                    "--soma-samples-per-ms",
                    str(task["soma_samples_per_ms"]),
                    "--worker-teacher",
                    task["teacher"],
                    "--worker-source-route",
                    task["source_route"],
                ]
                if task["intervention"]:
                    cmd += ["--worker-intervention", task["intervention"],
                            "--worker-intervention-dose", str(task.get("intervention_dose", DEFAULT_DOSE[task["intervention"]]))]
                if task.get("input_file"):
                    cmd += ["--worker-input-file", task["input_file"]]
                if task["coreneuron_gpu"]:
                    cmd += ["--coreneuron-gpu"]
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{ROOT / 'src'}:{ARTIFACTS}:{CORENEURON_PROBE}:{env.get('PYTHONPATH', '')}"
                running.append((subprocess.Popen(cmd, cwd=ROOT, env=env), task))
            next_running = []
            for proc, task in running:
                status = proc.poll()
                if status is None:
                    next_running.append((proc, task))
                elif status != 0:
                    raise RuntimeError(f"task failed with status {status}: {task}")
                else:
                    finished += 1
            running = next_running
            if running:
                time.sleep(0.25)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return finished


def task_is_complete(task: dict[str, Any]) -> bool:
    output = Path(task["output"])
    metadata = output.with_suffix(".json")
    if not output.exists() or not metadata.exists():
        return False
    try:
        with np.load(output) as data:
            key = "inputs" if task["kind"] == "split" else "baseline_inputs"
            if key not in data or int(data[key].shape[0]) != int(task["samples"]):
                return False
            if "sample_ids" not in data or int(data["sample_ids"].shape[0]) != int(task["samples"]):
                return False
        meta = json.loads(metadata.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("kind") == task["kind"]
        and meta.get("morphology_id") == task["morphology_id"]
        and int(meta.get("samples", -1)) == int(task["samples"])
        and meta.get("teacher_balance_label") == task["teacher_balance_label"]
        and meta.get("input_event_policy") == task.get("input_event_policy", "source_neuronio_sampler_unmodified")
        and abs(float(meta.get("exc_gmax_scale", float("nan"))) - float(task["exc_gmax_scale"])) < 1e-12
        and abs(float(meta.get("inh_gmax_scale", float("nan"))) - float(task["inh_gmax_scale"])) < 1e-12
        and _generator_matches(meta, task)
    )


def _generator_matches(meta: dict[str, Any], task: dict[str, Any]) -> bool:
    gen = meta.get("generator")
    if gen is None:  # shards written before generator settings were recorded are v1 shards
        return task["data_version"] == "v1" and task["synapse_delay_ms"] == 1.0 and task["input_routing"] == "normalized_by_type" \
            and task["nseg_policy"] == "hay40" and task["soma_samples_per_ms"] == 0
    if task.get("kind") == "intervention" and task.get("intervention"):
        default = DEFAULT_DOSE[task["intervention"]]
        if float(meta.get("intervention_dose") or default) != float(task.get("intervention_dose", default)):
            return False
    return all(gen.get(key) == task[key] for key in ("synapse_delay_ms", "input_routing", "nseg_policy", "soma_samples_per_ms")) \
        and gen.get("teacher", "hay") == task.get("teacher", "hay")


def worker_main(args: argparse.Namespace) -> int:
    if args.worker_samples is None or args.worker_output is None:
        raise ValueError("worker requires --worker-samples and --worker-output")
    started = time.perf_counter()
    worker_output = args.worker_output.resolve()
    if args.worker_kind == "source-route":
        setup = prepare_neuron("neuron", False)
        worker_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(worker_output, types=setup["source_segment_types"], lengths=setup["source_segment_lengths"])
        return 0
    rng = np.random.default_rng(args.seed)
    if args.worker_teacher == "allen_all_active":
        setup = prepare_allen_all_active(args)
    else:
        setup = prepare_neuron(args.backend, args.coreneuron_gpu)
    sample_rows = []
    baseline_inputs = []
    baseline_targets = []
    baseline_soma = []
    intervention_inputs = []
    intervention_targets = []
    intervention_soma = []
    changed_masks = []
    sample_ids = []
    supplied = None
    if args.worker_input_file is not None:
        # Replay supplied inputs (e.g. recorded population contexts) instead of sampling them; every other
        # array in the file is carried through to the shard unchanged.
        if args.worker_kind != "split":
            raise ValueError("--worker-input-file supports split shards only")
        with np.load(args.worker_input_file, allow_pickle=False) as data:
            supplied = {key: data[key] for key in data.files}
        if int(supplied["inputs"].shape[0]) != int(args.worker_samples):
            raise ValueError(f"{args.worker_input_file} holds {supplied['inputs'].shape[0]} traces, "
                             f"expected {args.worker_samples}")
    for sample_index in range(args.worker_samples):
        if supplied is not None:
            baseline_inputs.append(supplied["inputs"][sample_index].astype(np.int8, copy=False))
            sample_ids.append(str(supplied["sample_ids"][sample_index]))
            continue
        sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        signed_inputs = sample_signed_inputs(
            setup["source_segment_types"],
            setup["source_segment_lengths"],
            duration_ms=args.duration_ms,
            seed=sample_seed,
        )
        baseline_inputs.append(signed_inputs)
        condition = args.worker_split or (f"{args.worker_intervention}@{intervention_dose(args):g}"
                                          if args.worker_intervention_dose is not None else args.worker_intervention)
        sample_id = f"{args.worker_morphology_id}:{args.worker_kind}:{condition}:{args.worker_shard_index:05d}:{sample_index:04d}:seed{sample_seed}"
        sample_ids.append(sample_id)
        if args.worker_kind == "intervention":
            intervention_batch, changed_mask, _ = make_structured_intervention_inputs(
                signed_inputs[None, ...],
                intervention=args.worker_intervention,
                seed=args.seed + sample_index,
                site_start_fraction=0.0,
                **dose_kwargs(args.worker_intervention, intervention_dose(args)),
            )
            intervention_inputs.append(intervention_batch[0])
            changed_masks.append(changed_mask[0])

    baseline_traces = simulate_batch(args, setup, np.stack(baseline_inputs).astype(np.int8, copy=False))
    baseline_soma_hr, intervention_soma_hr = [], []
    for baseline in baseline_traces:
        baseline_targets.append(baseline["targets"])
        baseline_soma.append(baseline["soma_mv"])
        sample_rows.append(baseline["input_routing"])
        if "soma_mv_hr" in baseline:
            baseline_soma_hr.append(baseline["soma_mv_hr"])

    if args.worker_kind == "intervention":
        intervention_traces = simulate_batch(args, setup, np.stack(intervention_inputs).astype(np.int8, copy=False))
        for intervention in intervention_traces:
            intervention_targets.append(intervention["targets"])
            intervention_soma.append(intervention["soma_mv"])
            if "soma_mv_hr" in intervention:
                intervention_soma_hr.append(intervention["soma_mv_hr"])

    worker_output.parent.mkdir(parents=True, exist_ok=True)
    high_res = {}
    if baseline_soma_hr:
        high_res["soma_mv_hr" if args.worker_kind == "split" else "baseline_soma_mv_hr"] = np.stack(baseline_soma_hr).astype(np.float32)
    if intervention_soma_hr:
        high_res["intervention_soma_mv_hr"] = np.stack(intervention_soma_hr).astype(np.float32)
    partial = worker_output.with_name(worker_output.name + ".partial")
    if args.worker_kind == "split":
        arrays = dict(
            inputs=np.stack(baseline_inputs).astype(np.int8, copy=False),
            targets=np.stack(baseline_targets).astype(np.float32, copy=False),
            soma_mv=np.stack(baseline_soma).astype(np.float32, copy=False),
            sample_ids=np.asarray(sample_ids),
            **high_res,
            **{key: value for key, value in (supplied or {}).items() if key not in ("inputs", "sample_ids")},
        )
    else:
        arrays = dict(
            **high_res,
            baseline_inputs=np.stack(baseline_inputs).astype(np.int8, copy=False),
            baseline_targets=np.stack(baseline_targets).astype(np.float32, copy=False),
            baseline_soma_mv=np.stack(baseline_soma).astype(np.float32, copy=False),
            intervention_inputs=np.stack(intervention_inputs).astype(np.int8, copy=False),
            intervention_targets=np.stack(intervention_targets).astype(np.float32, copy=False),
            intervention_soma_mv=np.stack(intervention_soma).astype(np.float32, copy=False),
            changed_event_mask=np.stack(changed_masks).astype(bool, copy=False),
            sample_ids=np.asarray(sample_ids),
        )
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, worker_output)
    metadata = {
        "kind": args.worker_kind,
        "backend": args.backend,
        "coreneuron_gpu": bool(args.coreneuron_gpu),
        "duration_ms": int(args.duration_ms),
        "elapsed_seconds": time.perf_counter() - started,
        "teacher_balance_label": args.teacher_balance_label,
        "input_event_policy": "source_neuronio_sampler_unmodified" if supplied is None else "supplied_inputs",
        "input_file_sha256": None if supplied is None else _sha256(args.worker_input_file),
        "exc_gmax_scale": float(args.exc_gmax_scale),
        "inh_gmax_scale": float(args.inh_gmax_scale),
        "morphology_id": args.worker_morphology_id,
        "morphology_path": str(args.worker_morphology_path),
        "samples": int(args.worker_samples),
        "shard_index": int(args.worker_shard_index),
        "intervention": args.worker_intervention,
        "intervention_dose": intervention_dose(args) if args.worker_kind == "intervention" else None,
        "input_routing": _summarize_input_routing(sample_rows),
        "generator": generator_settings(args, morphology_path=args.worker_morphology_path, setup=setup),
    }
    partial_json = worker_output.with_name(worker_output.stem + ".json.partial")
    partial_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial_json, worker_output.with_suffix(".json"))
    return 0


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _code_revision() -> dict[str, Any]:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": rev, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def intervention_dose(args: argparse.Namespace) -> float:
    dose = getattr(args, "worker_intervention_dose", None)
    return float(DEFAULT_DOSE[args.worker_intervention] if dose is None else dose)


def dose_kwargs(condition: str, dose: float) -> dict[str, Any]:
    """Keyword arguments of `make_structured_intervention_inputs` for one dose, other parameters at v1 defaults."""
    kwargs: dict[str, Any] = {"dropout_probability": 0.35, "jitter_steps": 2, "site_fraction": 0.25}
    parameter = DOSE_PARAMETER[condition]
    kwargs[parameter] = int(round(dose)) if parameter == "jitter_steps" else float(dose)
    return kwargs


def simulate_batch(args: argparse.Namespace, setup: dict[str, Any], signed_inputs_batch: np.ndarray) -> list[dict[str, Any]]:
    """Simulate a batch on the worker's cell with the worker's teacher."""
    if args.worker_teacher != "allen_all_active":
        return simulate_signed_trace_batch(
            h=setup["h"],
            source_root=SOURCE_ROOT,
            morphology_path=args.worker_morphology_path.resolve(),
            backend=args.backend,
            signed_inputs_batch=signed_inputs_batch,
            v_init_mv=-76.0,
            exc_gmax_scale=args.exc_gmax_scale,
            inh_gmax_scale=args.inh_gmax_scale,
            input_routing=args.input_routing,
            source_route=setup["source_route"],
            synapse_delay_ms=args.synapse_delay_ms,
            nseg_policy=args.nseg_policy,
            soma_samples_per_ms=args.soma_samples_per_ms or None,
        )
    from axobench.generation.allen_all_active import simulate_allen_signed_trace

    return [
        simulate_allen_signed_trace(
            h=setup["h"],
            cell=setup["cell"],
            signed_inputs=signed_inputs,
            source_route=setup["source_route"],
            exc_gmax_scale=args.exc_gmax_scale,
            inh_gmax_scale=args.inh_gmax_scale,
            input_routing="source_site_normalized_arclength",
            synapse_delay_ms=args.synapse_delay_ms,
            soma_samples_per_ms=args.soma_samples_per_ms or None,
            teacher_info=setup["teacher_info"],
            merge_synapses=True,
        )
        for signed_inputs in signed_inputs_batch
    ]


def prepare_allen_all_active(args: argparse.Namespace) -> dict[str, Any]:
    """Load one Allen all-active model and the cached NeuronIO site map (no Hay mechanisms)."""
    from neuron import h

    from axobench.generation.allen_all_active import load_allen_all_active_cell

    if args.backend != "neuron":
        raise ValueError("Allen all-active teachers run on the NEURON backend only")
    if args.input_routing != "all_sites_by_type":
        raise ValueError("Allen all-active teachers connect every source site; use --input-routing all_sites_by_type")
    if args.worker_source_route is None or not args.worker_source_route.exists():
        raise FileNotFoundError(f"source route not found: {args.worker_source_route}")
    with np.load(args.worker_source_route) as data:
        types = data["types"].astype(str)
        lengths = data["lengths"].astype(np.float64)
    model_root = args.worker_morphology_path.resolve()
    cell = load_allen_all_active_cell(h, model_root, mechanism_root=model_root / "runtime")
    metadata = json.loads((model_root / "model_metadata.json").read_text(encoding="utf-8"))
    line = str(metadata.get("name", "")).split("_", 1)[-1].split(";")[0] or None
    return {
        "h": h,
        "cell": cell,
        "model_root": model_root,
        "source_route": _segment_route(types, lengths),
        "source_segment_types": types,
        "source_segment_lengths": lengths,
        "teacher_info": {"cell_role": "excitatory", "transgenic_line": line},
    }


def generator_settings(args: argparse.Namespace, morphology_path: Path | None = None,
                       setup: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every setting that determines a generated trace, for the shard sidecar and the run summary."""
    import neuron

    mods = sorted(SWC_MECH_ROOT.glob("*.mod"))
    digest = hashlib.sha256()
    for mod in mods:
        digest.update(mod.name.encode() + mod.read_bytes())
    out = {
        "data_version": args.data_version,
        "synapse_delay_ms": float(args.synapse_delay_ms),
        "exc_gmax_scale": float(args.exc_gmax_scale),
        "inh_gmax_scale": float(args.inh_gmax_scale),
        "input_routing": args.input_routing,
        "nseg_policy": args.nseg_policy,
        "soma_samples_per_ms": int(args.soma_samples_per_ms),
        "integrator": "fixed_step",
        "dt_ms": 0.025,
        "v_init_mv": -76.0,
        "synapses": "DetAMPANMDA/DetGABA (deterministic ProbAMPANMDA2/ProbUDFsyn2 at Use=1, Dep=0, Fac=0)",
        "spike_label_rule": "local maxima > -25 mV on the 8 samples/ms trace, labelled floor(t)",
        "input_sampler": "axobench sample_neuronio_inputs (NeuronIO sampler reimplementation)",
        "backend": args.backend,
        "neuron_version": getattr(neuron, "__version__", None),
        "code_revision": _code_revision(),
        "mechanisms_sha256": digest.hexdigest(),
        "mechanism_files": [mod.name for mod in mods],
        "template_hoc_sha256": _sha256(TEMPLATE_HOC),
        "biophysics_hoc_sha256": _sha256(SOURCE_ROOT / "L5PCbiophys5b.hoc"),
    }
    if morphology_path is not None:
        out["morphology_sha256"] = _sha256(morphology_path)
    teacher = getattr(args, "worker_teacher", "hay") if args.worker else "hay"
    out["teacher"] = teacher
    if teacher == "allen_all_active" and setup is not None:
        root, cell = setup["model_root"], setup["cell"]
        mods = sorted((root / "runtime" / "mod").glob("*.mod"))
        digest = hashlib.sha256()
        for mod in mods:
            digest.update(mod.name.encode() + mod.read_bytes())
        out.update(
            mechanisms_sha256=digest.hexdigest(),
            mechanism_files=[mod.name for mod in mods],
            template_hoc_sha256=None,
            biophysics_hoc_sha256=None,
            allen_model_id=cell.model_id,
            allen_specimen_id=cell.specimen_id,
            fit_parameters_sha256=_sha256(root / "fit_parameters.json"),
            morphology_sha256=_sha256(root / "reconstruction.swc"),
            v_init_mv=cell.v_init_mv,
            celsius=cell.celsius,
            # Stored voltages are the model's; Allen's recorded features correspond to v - junction potential.
            junction_potential_mv=float(json.loads((root / "fit_parameters.json").read_text())["fitting"][0]
                                        ["junction_potential"]),
            axon="Allen truncated: two 30 um sections with the diameters of the reconstruction's first 60 um",
            synapse_objects="one DetAMPANMDA and one DetGABA per segment, driven by every site routed to it "
                            "(exact: linear synapse states, shared time constants and gmax)",
        )
    return out


def prepare_neuron(backend: str, coreneuron_gpu: bool) -> dict[str, Any]:
    from neuron import coreneuron, h

    if backend == "coreneuron":
        coreneuron.enable = True
        coreneuron.gpu = bool(coreneuron_gpu)
        if coreneuron_gpu:
            coreneuron.cell_permute = int(os.environ.get("AXOBENCH_CORENEURON_CELL_PERMUTE", "1"))
        if os.environ.get("AXOBENCH_CORENEURON_NUM_GPUS"):
            coreneuron.num_gpus = int(os.environ["AXOBENCH_CORENEURON_NUM_GPUS"])
        coreneuron.verbose = 0
    else:
        coreneuron.enable = False
        coreneuron.gpu = False
    os.chdir(SWC_MECH_ROOT)
    load_probe_mechanisms(h, SWC_MECH_ROOT)
    hoc_lib = Path(os.environ.get("NEURONHOME", "")) / "share" / "nrn" / "lib" / "hoc"
    if hoc_lib.exists():
        h.load_file(str(hoc_lib / "stdrun.hoc"))
        h.load_file(str(hoc_lib / "import3d.hoc"))
    else:
        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
    h.load_file(str(SOURCE_ROOT / "L5PCbiophys5b.hoc"))
    h.load_file(str(TEMPLATE_HOC))
    h.dt = 0.025
    h.steps_per_ms = 40
    h.secondorder = 0
    h.CVode().active(0)
    h.CVode().cache_efficient(1)
    source_cell = h.L5PCtemplate(str(SOURCE_ROOT / "morphologies" / "cell1.asc"))
    _, source_segment_types, source_segment_lengths = collect_segments(source_cell)
    del source_cell
    return {
        "h": h,
        "source_route": _build_source_site_route(h, SOURCE_ROOT),
        "source_segment_types": source_segment_types,
        "source_segment_lengths": source_segment_lengths,
    }


def sample_signed_inputs(
    source_segment_types: np.ndarray,
    source_segment_lengths: np.ndarray,
    *,
    duration_ms: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    inputs_exc, inputs_inh = sample_neuronio_inputs(rng, duration_ms, source_segment_types, source_segment_lengths)
    signed_inputs = np.zeros((duration_ms, 2 * inputs_exc.shape[0]), dtype=np.int8)
    signed_inputs[:, : inputs_exc.shape[0]] = inputs_exc.T.astype(np.int8, copy=False)
    signed_inputs[:, inputs_exc.shape[0] :] = -inputs_inh.T.astype(np.int8, copy=False)
    return signed_inputs


def validate_positive(value: float, name: str) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"--{name} must be positive, got {value}")


def write_split_manifests(output_root: Path) -> None:
    for split in ("train", "val", "test", "private-test"):
        write_npz_manifest(output_root / split)
    interventions_root = output_root / OOD_DIR
    if interventions_root.exists():
        for folder in sorted(path for path in interventions_root.iterdir() if path.is_dir()):
            write_pair_manifest(folder)


def write_npz_manifest(root: Path) -> None:
    if not root.exists():
        return
    entries = []
    for shard in sorted(root.glob("*.npz")):
        with np.load(shard) as data:
            sample_ids = [str(item) for item in data["sample_ids"]]
            entries.append({"file": shard.name, "count": int(data["inputs"].shape[0]), "sample_ids": sample_ids})
    if entries:
        (root / "manifest.json").write_text(json.dumps({"version": 1, "shards": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_pair_manifest(root: Path) -> None:
    if not root.exists():
        return
    entries = []
    for shard in sorted({*root.glob("intervention_pair_*.npz"), *root.glob("*_intervention_pair_*.npz")}):
        with np.load(shard) as data:
            entries.append({"file": shard.name, "count": int(data["baseline_inputs"].shape[0]), "sample_ids": [str(item) for item in data["sample_ids"]]})
    if entries:
        (root / "manifest.json").write_text(json.dumps({"version": 1, "pair_shards": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
