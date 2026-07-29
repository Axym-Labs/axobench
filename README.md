# AxoBench

AxoBench is the extracted benchmark and dataset-generation workbench for
single-neuron surrogate evaluation. It owns dataset generation, data/schema
utilities, diagnostic suites, and benchmark reporting. Its publication
contract reports support-normalized SERA for voltage fidelity, Dynamics SERA
for voltage-transition fidelity, and the arithmetic Mean F1 at integer spike
tolerances from 0 through 5 ms.

## Install and verify

```bash
python -m pip install -e .[dev]
python -m pytest -q
```

The package keeps simulator-backed generation code under
`src/axobench/generation/`: Arbor, NEURON/Hay, CoreNEURON/Hay, SWC-template
preparation, morphology-candidate ranking, throughput probes, and the v1
parallel dataset generator. `scripts/` is reserved for thin operational entry
points such as launch scripts and data-download helpers.

Model implementations, training loops, checkpoint formats, and model-specific
experiments stay in the separate AxoSim model repository. AxoBench evaluates
caller-supplied prediction functions or stored prediction/diagnostic artifacts;
it does not ship BranchELM, Mamba, RNN, or training code.

## Population-context extension

The population-context route preserves source identity, E/I role, causal
delay, and destination channel when connected spike histories are converted
into AxoBench inputs. The completed publication corpus contains 50,000
native-1-ms traces from 100 disjoint connectome contexts, split into
40,000/10,000 train/validation traces and exact 40,000/10,000
excitatory/inhibitory roles. Five excitatory Hay/Allen morphologies and one
inhibitory Allen SST teacher provide 1,200-step voltage and spike labels over
the common 1,278-channel input contract.

The corpus itself is not stored in this source repository. Its release
manifest records 600 shards, 602 files, 1,591,623,131 bytes, and 50,000 unique
sample identifiers; independent validation checks every manifest hash, shape,
dtype, finite voltage target, and binary spike target. Connected-model input
histories are replayed through biophysical teachers, so the labels are
counterfactual teacher responses rather than a fully biophysical closed loop.
A fixed training recipe improves held-out trace metrics but does not improve
the connected objective, which is why connected diagnostics remain part of
model selection.

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
The primary metrics are support-normalized Voltage SERA, Dynamics SERA,
Response SERA, and Mean F1 over the declared integer timing tolerances. SERA
has squared error units by definition; its rooted companion is named
Root-SERA. The default spike score is the arithmetic mean of event F1 at
`0, 1, 2, 3, 4, 5 ms`, so it is higher-is-better and each declared tolerance
has equal weight. The report also includes Root-SERA, ordinary trapezoidal F1
area, exact F1, and 5 ms F1. Every report includes the full tolerance curve and
calibrated threshold.

`attention=YES` is an early watch, `NO` is clear at the stated trigger, and
`NA` makes missing evidence explicit. Each diagnostic row also carries its
mathematical or physiological `basis` in the returned report.

For iterative comparisons, create and save a baseline with
`create_model_evaluation_cache`, then use `evaluate_model_against_cache` for
each candidate. Paired metric deltas use morphology-stratified 95% bootstrap
intervals without rerunning the baseline. The corresponding `improved`,
`regressed`, and `uncertain` labels are evidence, not a model-promotion rule.

## Scope

AxoBench defines evaluation and generation contracts; it does not claim that
one scalar metric captures every biologically relevant error. Voltage SERA,
Dynamics SERA, Response SERA, and Mean F1 should be reported with the
tolerance grid, trace selection, calibration split, and uncertainty procedure.
Ordinary trapezoidal F1 area and Root-SERA remain compatibility views rather
than headline metrics.
