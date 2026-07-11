# AxoBench

AxoBench is the extracted benchmark and dataset-generation workbench for
single-neuron surrogate evaluation. It owns dataset generation, data/schema
utilities, diagnostic suites, and benchmark reporting.

The package keeps simulator-backed generation code under
`src/axobench/generation/`: Arbor, NEURON/Hay, CoreNEURON/Hay, SWC-template
preparation, morphology-candidate ranking, throughput probes, and the v1
parallel dataset generator. `scripts/` is reserved for thin operational entry
points such as launch scripts and data-download helpers.

Model implementations, training loops, checkpoint formats, and model-specific
experiments stay in the separate AxoSim model repository. AxoBench evaluates
caller-supplied prediction functions or stored prediction/diagnostic artifacts;
it does not ship BranchELM, Mamba, RNN, or training code.

## Iterative Model Evaluation

`evaluate_model` accepts any batched NumPy prediction callable and a v1-style
dataset root. It calibrates the spike threshold on a held-out fraction of the
selected validation traces, computes the core metric set, evaluates available
paired interventions, and returns a fixed 24-row attention panel.

```python
from axobench.benchmark import EvaluationConfig, evaluate_model, format_evaluation_report

report = evaluate_model(
    predict,
    dataset_root="data/axobench-v1",
    model_id="candidate-17",
    config=EvaluationConfig(default_spike_threshold=0.0),
)
print(format_evaluation_report(report))
```

The default iteration profile uses 240 morphology-balanced validation traces,
60 paired traces per intervention, 16 traces per model batch, and four parallel
CPU intervention loaders. These defaults were selected from trace-level,
morphology-stratified 95% bootstrap intervals; all limits remain configurable.

Evaluation only prints by default. Persist a strict JSON or tidy CSV report
when needed:

```python
from axobench.benchmark import write_evaluation_report

write_evaluation_report(report, "runs/candidate-17.csv")
```

The terminal report is bounded to eight metric rows and 24 diagnostic rows.
`attention=YES` is an early watch, `NO` is clear at the stated trigger, and
`NA` makes missing evidence explicit. Each diagnostic row also carries its
mathematical or physiological `basis` in the returned report.

The active exploration arc lives in `docs/benchmark/`, which is ignored by Git
while the benchmark audit is still exploratory.
