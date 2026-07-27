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

The default iteration profile evaluates one morphology with 120 validation
traces, 40 paired traces per intervention, 128 traces per model batch, and four
parallel CPU intervention loaders. Set `multiple_morphologies=True` to
distribute the same 120/40 profile across all available morphologies, or pass
`morphology_id` to choose one specimen. Trace-level uncertainty differs among
specimens, so confirmation runs can raise either limit explicitly.

Evaluation only prints by default. Persist a strict JSON or tidy CSV report
when needed:

```python
from axobench.benchmark import write_evaluation_report

write_evaluation_report(report, "runs/candidate-17.csv")
```

The terminal report is bounded to ten metric rows and 24 diagnostic rows.
The primary metrics are support-normalized Voltage/Dynamics/Response SERA in
squared units and the Squared-Deficit F1 Score over 0--5 ms. The latter is
`1 - mean((1 - F1(tolerance))^2)`, integrated over the fixed tolerance span,
so higher is better and gains in poor-tolerance regions receive more weight.
The report also includes Root-SERA, ordinary F1 area, exact F1, 5 ms F1, and
the lower-is-better squared-deficit area for interpretation and compatibility.
Every report includes the full tolerance curve and calibrated threshold.

`attention=YES` is an early watch, `NO` is clear at the stated trigger, and
`NA` makes missing evidence explicit. Each diagnostic row also carries its
mathematical or physiological `basis` in the returned report.

For iterative comparisons, create and save a baseline with
`create_model_evaluation_cache`, then use `evaluate_model_against_cache` for
each candidate. Paired metric deltas use morphology-stratified 95% bootstrap
intervals without rerunning the baseline. The corresponding `improved`,
`regressed`, and `uncertain` labels are evidence, not a model-promotion rule.

The active exploration arc lives in `docs/benchmark/`, which is ignored by Git
while the benchmark audit is still exploratory.
