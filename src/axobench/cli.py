from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from axobench.benchmark.bundle import run_v1_benchmark_bundle
from axobench.benchmark.dataset_schema import (
    build_diagnostic_suite_catalog,
    build_v1_dataset_manifest_template,
)
from axobench.benchmark.diagnostic_audits import DiagnosticAuditSources, diagnostic_audit_rows_from_sources
from axobench.benchmark.diagnostic_rows import DiagnosticContext, write_diagnostic_rows
from axobench.benchmark.profiles import (
    build_axis_plan,
    estimate_profile_cost,
    get_benchmark_profile,
    list_benchmark_profiles,
)
from axobench.benchmark.reports import write_diagnostic_summary
from axobench.benchmark.runner import (
    run_paired_intervention_on_npz,
    run_paired_intervention_on_selector,
    run_teacher_qc_on_selector,
)
from axobench.benchmark.selectors import V1DatasetLayout
from axobench.data import repack_shards_as_npy, write_demo_shards
from axobench.generation import ArborHeteroConfig, parse_split_samples, write_arbor_hetero_dataset, write_arbor_hetero_splits
from axobench.hub import POPULATION_SPLITS, download_dataset, download_population_dataset
from axobench.neuronio_raw import convert_neuronio_pickles


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="axobench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-data",
        help="download the public AxoBench datasets from the Hugging Face Hub",
    )
    download.add_argument(
        "--output",
        help="download directory; the data land in <output>/<variant>; defaults to $AXOBENCH_DATA_DIR/v2 "
        "(or .../population with --population)",
    )
    download.add_argument("--variant", default="minimal", help="AxoBench v2 variant (default minimal)")
    download.add_argument("--splits", help="comma-separated splits; default val (train,val with --population)")
    download.add_argument(
        "--interventions",
        default="all",
        help="comma-separated intervention folders (inh_dropout_p35) or conditions (all doses), 'all', or 'none'",
    )
    download.add_argument("--population", action="store_true", help="download the population-context corpus instead")
    download.add_argument("--revision", help="Hub branch, tag, or commit to download")

    demo = subparsers.add_parser("make-demo-data", help="write deterministic demo NPZ shards")
    demo.add_argument("--output", required=True)
    demo.add_argument("--samples", type=int, default=8)
    demo.add_argument("--shards", type=int, default=1)
    demo.add_argument("--time-steps", type=int, default=32)
    demo.add_argument("--input-dim", type=int, default=64)
    demo.add_argument("--seed", type=int, default=0)

    arbor_data = subparsers.add_parser(
        "make-arbor-hetero-data",
        help="write Arbor-backed heterogeneous cable-cell shards for distillation",
    )
    arbor_data.add_argument("--config", help="JSON config with make-arbor-hetero-data option names")
    arbor_data.add_argument("--output")
    arbor_data.add_argument("--samples", type=int, default=128)
    arbor_data.add_argument("--shards", type=int, default=1)
    arbor_data.add_argument("--splits", help="comma-separated NAME:SAMPLES pairs, e.g. train:128,val:32,test:32")
    arbor_data.add_argument("--shards-per-split", type=int, default=1)
    arbor_data.add_argument("--time-steps", type=int, default=6000)
    arbor_data.add_argument("--input-dim", type=int, default=1278)
    arbor_data.add_argument("--branches", type=int, default=45)
    arbor_data.add_argument("--integration-dt-ms", type=float, default=0.025)
    arbor_data.add_argument("--input-protocol", choices=["arbor_poisson", "neuronio_smoothed"], default="arbor_poisson")
    arbor_data.add_argument("--synapse-routing", choices=["branch", "distributed"], default="branch")
    arbor_data.add_argument("--distributed-sites-per-polarity", type=int)
    arbor_data.add_argument("--spike-label-mode", choices=["threshold_crossing", "local_peak"], default="threshold_crossing")
    arbor_data.add_argument("--spike-peak-threshold-mv", type=float, default=-25.0)
    arbor_data.add_argument("--base-event-rate-hz", type=float, default=2.0)
    arbor_data.add_argument("--burst-probability", type=float, default=0.015)
    arbor_data.add_argument("--burst-event-rate-hz", type=float, default=35.0)
    arbor_data.add_argument("--arbor-gpu-id", type=int, default=0, help="GPU id for Arbor context; ignored with --cpu-only")
    arbor_data.add_argument("--cpu-only", action="store_true", help="use Arbor CPU context instead of CUDA")
    arbor_data.add_argument("--arbor-threads", type=int, default=1)
    arbor_data.add_argument("--sim-batch-size", type=int, default=32, help="independent cells per Arbor simulation batch")
    arbor_data.add_argument("--seed", type=int, default=0)
    arbor_data.add_argument(
        "--param-overrides",
        help="JSON object or path to a JSON object overriding Arbor heterogeneity parameters for all samples",
    )
    arbor_data.add_argument("--report", help="optional path for a copy of the generation report")

    convert = subparsers.add_parser("convert-neuronio", help="convert raw NeuronIO pickle files to deterministic shards")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--shard-size", type=int, default=128)
    convert.add_argument("--window-size", type=int)
    convert.add_argument("--window-stride", type=int)
    convert.add_argument("--ignore-start", type=int, default=0)

    repack = subparsers.add_parser("repack-shards", help="repack shards as sliceable .npy arrays")
    repack.add_argument("--input", required=True)
    repack.add_argument("--output", required=True)
    repack.add_argument("--input-dtype", choices=["int8", "float32"], default="int8")

    profile = subparsers.add_parser("benchmark-profile", help="print an AxoBench axis/default profile")
    profile.add_argument("--name", default="paper-default", choices=list_benchmark_profiles())
    profile.add_argument("--plan", action="store_true", help="print the expanded axis-composition plan")
    profile.add_argument("--cost", action="store_true", help="print only the local capacity estimate")

    dataset_schema = subparsers.add_parser(
        "dataset-schema",
        help="print the v1 dataset manifest template or diagnostic suite catalog",
    )
    dataset_schema.add_argument("--kind", choices=["manifest", "diagnostics"], default="manifest")

    benchmark_rows = subparsers.add_parser(
        "benchmark-rows",
        help="write dataframe-first v1 teacher or intervention diagnostic rows",
    )
    benchmark_rows.add_argument("--mode", choices=["teacher-qc", "intervention"], default="teacher-qc")
    benchmark_rows.add_argument("--dataset-root", required=True)
    benchmark_rows.add_argument("--selector", choices=["val", "intervention"], default="val")
    benchmark_rows.add_argument("--intervention", default="event_dropout")
    benchmark_rows.add_argument("--pair-path", help="paired intervention .npz path, for pilot artifacts outside a v1 layout")
    benchmark_rows.add_argument("--output", required=True)
    benchmark_rows.add_argument("--summary-output", help="optional compact benchmark summary path (.md or .json)")
    benchmark_rows.add_argument("--suites", default="teacher-qc")
    benchmark_rows.add_argument("--morphology-id", default="unknown")
    benchmark_rows.add_argument("--teacher-id", default="teacher")
    benchmark_rows.add_argument("--max-samples", type=int)
    benchmark_rows.add_argument("--cache-shards", type=int, default=1)
    benchmark_rows.add_argument("--ignore-start", type=int, default=500)
    benchmark_rows.add_argument("--soma-scale", type=float, default=0.1)
    benchmark_rows.add_argument("--soma-bias-mv", type=float, default=-67.7)
    benchmark_rows.add_argument("--calibrated-spike-threshold", type=float)

    diagnostic_audits = subparsers.add_parser(
        "diagnostic-audits",
        help="derive dataframe-first diagnostic rows from benchmark artifact rows",
    )
    diagnostic_audits.add_argument("--teacher-grounded-rows")
    diagnostic_audits.add_argument("--local-stability-rows")
    diagnostic_audits.add_argument("--perturbation-rows")
    diagnostic_audits.add_argument("--pretrained-summary-rows")
    diagnostic_audits.add_argument("--teacher-protocol")
    diagnostic_audits.add_argument("--output", required=True)
    diagnostic_audits.add_argument("--summary-output")
    diagnostic_audits.add_argument(
        "--suites",
        default="metric-decoupling,target-view-audit,robustness-directionality,protocol-coverage,adapter-fairness,paper-summary",
    )
    diagnostic_audits.add_argument("--model-id", default="analysis")
    diagnostic_audits.add_argument("--teacher-id", default="teacher")
    diagnostic_audits.add_argument("--morphology-id", default="mixed")
    diagnostic_audits.add_argument("--split", default="analysis")
    diagnostic_audits.add_argument("--condition", default="analysis")

    diagnostic_plots = subparsers.add_parser(
        "diagnostic-plots",
        help="write diagnostic-attached figures and tables from audit artifacts",
    )
    diagnostic_plots.add_argument("--canonical-summary", required=True)
    diagnostic_plots.add_argument("--mechanistic-summary", required=True)
    diagnostic_plots.add_argument("--sparse-event-summary", required=True)
    diagnostic_plots.add_argument("--point-rows", required=True)
    diagnostic_plots.add_argument("--tcn-summary", action="append", required=True)
    diagnostic_plots.add_argument("--morphology-contrast-rows")
    diagnostic_plots.add_argument("--neuronio-intervention-rows")
    diagnostic_plots.add_argument("--neuronio-intervention-root")
    diagnostic_plots.add_argument("--output-dir", required=True)
    diagnostic_plots.add_argument("--formats", default="svg,pdf,png")

    anatomy = subparsers.add_parser(
        "error-anatomy",
        help="break a cached prediction's errors down by spike outcome, spike distance, rate, band and morphology",
    )
    anatomy.add_argument("--cache", required=True, help="evaluation cache or NPZ with a 'prediction' array")
    anatomy.add_argument(
        "--targets", help="NPZ with 'target' and optional 'raw_soma_mv' and 'trace_ids' (default: the cache file)"
    )
    anatomy.add_argument("--output", required=True, help="output directory")
    anatomy.add_argument("--ignore-start", type=int, help="default: the cache protocol's, else 500")
    anatomy.add_argument("--calibration-fraction", type=float, help="default: the cache protocol's, else 0.25")
    anatomy.add_argument("--formats", default="pdf,png")

    benchmark_v1 = subparsers.add_parser(
        "benchmark-v1",
        help="run the promoted v1 benchmark bundle on teacher QC plus paired interventions",
    )
    benchmark_v1.add_argument("--dataset-root", required=True)
    benchmark_v1.add_argument("--output-dir", required=True)
    benchmark_v1.add_argument("--suite", choices=["teacher-qc", "paper-summary", "full"], default="paper-summary")
    benchmark_v1.add_argument("--intervention-pair-glob", action="append", default=[])
    benchmark_v1.add_argument("--morphology-id", default="unknown")
    benchmark_v1.add_argument("--teacher-id", default="teacher")
    benchmark_v1.add_argument("--max-val-samples", type=int)
    benchmark_v1.add_argument("--max-intervention-samples", type=int)
    benchmark_v1.add_argument("--cache-shards", type=int, default=1)
    benchmark_v1.add_argument("--ignore-start", type=int, default=500)
    benchmark_v1.add_argument("--soma-scale", type=float, default=0.1)
    benchmark_v1.add_argument("--soma-bias-mv", type=float, default=-67.7)
    benchmark_v1.add_argument("--calibrated-spike-threshold", type=float)
    benchmark_v1.add_argument(
        "--no-summaries",
        action="store_false",
        dest="write_summaries",
        help="write only machine-readable benchmark rows/manifests, not Markdown summaries",
    )
    benchmark_v1.set_defaults(write_summaries=True)

    args = parser.parse_args(argv)
    explicit_dests = _explicit_option_dests(parser, argv, args.command)

    if args.command == "benchmark-profile":
        if args.cost:
            payload = estimate_profile_cost(args.name)
        elif args.plan:
            payload = build_axis_plan(args.name)
        else:
            payload = get_benchmark_profile(args.name).to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "dataset-schema":
        payload = build_v1_dataset_manifest_template() if args.kind == "manifest" else build_diagnostic_suite_catalog()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark-rows":
        try:
            layout = V1DatasetLayout(args.dataset_root)
            selector = (
                layout.default_validation()
                if args.selector == "val"
                else layout.intervention(args.intervention)
            )
            context = DiagnosticContext(
                suites=_parse_suites(args.suites),
                model_id="teacher",
                teacher_id=args.teacher_id,
                morphology_id=args.morphology_id,
                split=selector.split or "",
                condition=selector.condition,
                intervention=selector.intervention,
                ignore_start=args.ignore_start,
                soma_scale=args.soma_scale,
                soma_bias_mv=args.soma_bias_mv,
                calibrated_spike_threshold=args.calibrated_spike_threshold,
            )
            if args.mode == "teacher-qc":
                rows = run_teacher_qc_on_selector(
                    selector,
                    context=context,
                    max_samples=args.max_samples,
                    output=args.output,
                    cache_shards=args.cache_shards,
                )
            elif args.pair_path:
                rows = run_paired_intervention_on_npz(
                    args.pair_path,
                    context=context,
                    max_samples=args.max_samples,
                    output=args.output,
                )
            else:
                rows = run_paired_intervention_on_selector(
                    selector,
                    context=context,
                    max_samples=args.max_samples,
                    output=args.output,
                )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        if args.summary_output:
            try:
                write_diagnostic_summary(rows, args.summary_output)
            except ValueError as error:
                print(str(error))
                return 2
        print(json.dumps({"rows": len(rows), "output": args.output}, indent=2, sort_keys=True))
        return 0

    if args.command == "diagnostic-audits":
        try:
            rows = diagnostic_audit_rows_from_sources(
                DiagnosticAuditSources(
                    teacher_grounded_rows=Path(args.teacher_grounded_rows) if args.teacher_grounded_rows else None,
                    local_stability_rows=Path(args.local_stability_rows) if args.local_stability_rows else None,
                    perturbation_rows=Path(args.perturbation_rows) if args.perturbation_rows else None,
                    pretrained_summary_rows=Path(args.pretrained_summary_rows) if args.pretrained_summary_rows else None,
                    teacher_protocol=Path(args.teacher_protocol) if args.teacher_protocol else None,
                ),
                context=DiagnosticContext(
                    suites=_parse_suites(args.suites),
                    model_id=args.model_id,
                    teacher_id=args.teacher_id,
                    morphology_id=args.morphology_id,
                    split=args.split,
                    condition=args.condition,
                ),
            )
            write_diagnostic_rows(rows, args.output)
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        if args.summary_output:
            try:
                write_diagnostic_summary(rows, args.summary_output)
            except ValueError as error:
                print(str(error))
                return 2
        print(json.dumps({"rows": len(rows), "output": args.output}, indent=2, sort_keys=True))
        return 0

    if args.command == "diagnostic-plots":
        try:
            from axobench.benchmark.plots import write_analysis_audit_artifacts

            manifest = write_analysis_audit_artifacts(
                output_dir=args.output_dir,
                canonical_summary=Path(args.canonical_summary),
                mechanistic_summary=Path(args.mechanistic_summary),
                sparse_event_summary=Path(args.sparse_event_summary),
                point_rows=Path(args.point_rows),
                tcn_summaries=[Path(path) for path in args.tcn_summary],
                morphology_contrast_rows=Path(args.morphology_contrast_rows) if args.morphology_contrast_rows else None,
                neuronio_intervention_rows=Path(args.neuronio_intervention_rows) if args.neuronio_intervention_rows else None,
                neuronio_intervention_root=Path(args.neuronio_intervention_root) if args.neuronio_intervention_root else None,
                formats=_parse_suites(args.formats),
            )
        except (FileNotFoundError, KeyError, ValueError, ImportError) as error:
            print(str(error))
            return 2
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.command == "error-anatomy":
        try:
            summary = _run_error_anatomy(args)
        except (FileNotFoundError, KeyError, ValueError) as error:
            print(str(error))
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark-v1":
        try:
            manifest = run_v1_benchmark_bundle(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                base_context=DiagnosticContext(
                    model_id="teacher",
                    teacher_id=args.teacher_id,
                    morphology_id=args.morphology_id,
                    ignore_start=args.ignore_start,
                    soma_scale=args.soma_scale,
                    soma_bias_mv=args.soma_bias_mv,
                    calibrated_spike_threshold=args.calibrated_spike_threshold,
                ),
                intervention_pair_paths=_expand_globs(args.intervention_pair_glob),
                suite=args.suite,
                max_val_samples=args.max_val_samples,
                max_intervention_samples=args.max_intervention_samples,
                cache_shards=args.cache_shards,
                write_summaries=args.write_summaries,
            )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.command == "download-data":
        try:
            if args.population:
                root = download_population_dataset(
                    args.output,
                    splits=_parse_suites(args.splits) if args.splits else POPULATION_SPLITS,
                    revision=args.revision,
                )
            else:
                interventions = _parse_suites(args.interventions)
                root = download_dataset(
                    args.output,
                    variant=args.variant,
                    splits=_parse_suites(args.splits or "val"),
                    interventions=(
                        None if interventions == ("none",)
                        else "all" if interventions == ("all",)
                        else interventions
                    ),
                    revision=args.revision,
                )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(json.dumps({"dataset_root": str(root)}, indent=2, sort_keys=True))
        return 0

    if args.command == "make-demo-data":
        write_demo_shards(
            args.output,
            shard_count=args.shards,
            samples_per_shard=args.samples,
            time_steps=args.time_steps,
            input_dim=args.input_dim,
            seed=args.seed,
        )
        return 0

    if args.command == "make-arbor-hetero-data":
        try:
            _apply_arbor_data_config(args, explicit_dests)
            if not args.output:
                raise ValueError("--output is required unless provided by --config")
            config = ArborHeteroConfig(
                input_dim=args.input_dim,
                num_branch=args.branches,
                time_steps=args.time_steps,
                integration_dt_ms=args.integration_dt_ms,
                input_protocol=args.input_protocol,
                synapse_routing=args.synapse_routing,
                distributed_sites_per_polarity=args.distributed_sites_per_polarity,
                spike_label_mode=args.spike_label_mode,
                spike_peak_threshold_mv=args.spike_peak_threshold_mv,
                base_event_rate_hz=args.base_event_rate_hz,
                burst_probability=args.burst_probability,
                burst_event_rate_hz=args.burst_event_rate_hz,
                arbor_threads=args.arbor_threads,
                arbor_gpu_id=_resolve_arbor_gpu_id(args, explicit_dests),
                sim_batch_size=args.sim_batch_size,
            )
            if args.splits:
                report = write_arbor_hetero_splits(
                    args.output,
                    split_samples=parse_split_samples(args.splits),
                    shards_per_split=args.shards_per_split,
                    seed=args.seed,
                    config=config,
                    param_overrides=_parse_param_overrides(args.param_overrides),
                )
            else:
                report = write_arbor_hetero_dataset(
                    args.output,
                    samples=args.samples,
                    shards=args.shards,
                    seed=args.seed,
                    config=config,
                    param_overrides=_parse_param_overrides(args.param_overrides),
                )
        except ValueError as error:
            print(str(error))
            return 2
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "convert-neuronio":
        try:
            written = convert_neuronio_pickles(
                args.input,
                args.output,
                shard_size=args.shard_size,
                window_size=args.window_size,
                window_stride=args.window_stride,
                ignore_start=args.ignore_start,
            )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(f"wrote_shards={len(written)}")
        return 0

    if args.command == "repack-shards":
        try:
            input_dtype = np.float32 if args.input_dtype == "float32" else np.int8
            written = repack_shards_as_npy(args.input, args.output, input_dtype=input_dtype)
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(f"wrote_shards={len(written)}")
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def _run_error_anatomy(args) -> dict[str, object]:
    from dataclasses import fields

    from axobench.benchmark.error_anatomy import error_anatomy
    from axobench.benchmark.iteration_evaluator import EvaluationConfig, _array_sha256
    from axobench.benchmark.plots import plot_error_anatomy

    with np.load(args.cache, allow_pickle=False) as cache:
        prediction = np.asarray(cache["prediction"])
        metadata = json.loads(str(cache["metadata_json"].item())) if "metadata_json" in cache.files else {}
    with np.load(args.targets or args.cache, allow_pickle=False) as data:
        if "target" not in data.files:
            raise ValueError("no 'target' array; pass --targets")
        target = np.asarray(data["target"])
        raw = np.asarray(data["raw_soma_mv"]) if "raw_soma_mv" in data.files else None
        trace_ids = [str(i) for i in data["trace_ids"]] if "trace_ids" in data.files else metadata.get("trace_ids")
    if metadata.get("target_sha256") and _array_sha256(target) != metadata["target_sha256"]:
        raise ValueError("targets do not match the cache's target_sha256")
    if metadata.get("trace_ids") and trace_ids is not None and list(trace_ids) != list(metadata["trace_ids"]):
        raise ValueError("target trace_ids do not match the cache's trace_ids")
    names = {field.name for field in fields(EvaluationConfig)}
    stored = {key: tuple(value) if isinstance(value, list) else value
              for key, value in metadata.get("protocol", {}).get("config", {}).items() if key in names}
    if args.ignore_start is not None:
        stored["ignore_start"] = args.ignore_start
    if args.calibration_fraction is not None:
        stored["calibration_fraction"] = args.calibration_fraction
    config = EvaluationConfig(**stored)
    result = error_anatomy(prediction, target, raw_soma_mv=raw, trace_ids=trace_ids, config=config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "error-anatomy.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figures = plot_error_anatomy(result, output / "error-anatomy", formats=_parse_suites(args.formats))
    overall = result["overall"]
    return {
        "output": str(output / "error-anatomy.json"),
        "figures": figures,
        **{key: overall[key] for key in ("voltage_sera_skill", "voltage_sera_ceiling_skill", "voltage_sera_skill_of_ceiling", "subthreshold_skill", "mean_f1")},
    }


def _expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"intervention pair glob matched no files: {pattern}")
        paths.extend(Path(match) for match in matches)
    return paths


def _parse_suites(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _apply_arbor_data_config(args, explicit_dests: set[str]) -> None:
    config_path = getattr(args, "config", None)
    if not config_path:
        return
    path = Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"simulator config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid simulator config JSON in {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("simulator config must be a JSON object")
    allowed = {
        "output",
        "samples",
        "shards",
        "splits",
        "shards_per_split",
        "time_steps",
        "input_dim",
        "branches",
        "integration_dt_ms",
        "input_protocol",
        "synapse_routing",
        "distributed_sites_per_polarity",
        "spike_label_mode",
        "spike_peak_threshold_mv",
        "base_event_rate_hz",
        "burst_probability",
        "burst_event_rate_hz",
        "arbor_gpu_id",
        "cpu_only",
        "arbor_threads",
        "sim_batch_size",
        "seed",
        "param_overrides",
        "report",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown simulator config keys: {', '.join(unknown)}")
    for key, value in config.items():
        if key not in explicit_dests:
            setattr(args, key, value)


def _parse_param_overrides(value: str | dict[str, object] | None) -> dict[str, float] | None:
    if value is None:
        return None
    raw: object
    if isinstance(value, dict):
        raw = value
    else:
        text = str(value)
        path = Path(text)
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("param_overrides must be a JSON object")
    overrides: dict[str, float] = {}
    for key, item in raw.items():
        if not isinstance(key, str):
            raise ValueError("param_overrides keys must be strings")
        try:
            overrides[key] = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"param_overrides value for {key!r} must be numeric") from error
    return overrides or None


def _resolve_arbor_gpu_id(args, explicit_dests: set[str]) -> int | None:
    if args.cpu_only:
        return None
    if "arbor_gpu_id" in explicit_dests:
        return args.arbor_gpu_id
    try:
        import arbor as A
    except ModuleNotFoundError:
        return args.arbor_gpu_id
    if A.config().get("gpu") != "cuda":
        return None
    return args.arbor_gpu_id


def _explicit_option_dests(parser: argparse.ArgumentParser, argv: list[str], command: str) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option_string in getattr(action, "option_strings", ()):
            option_to_dest[option_string] = action.dest
    subparser_action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    for action in subparser_action.choices[command]._actions:
        for option_string in getattr(action, "option_strings", ()):
            option_to_dest[option_string] = action.dest

    explicit_dests: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit_dests.add(dest)
    return explicit_dests


if __name__ == "__main__":
    raise SystemExit(main())
