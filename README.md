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

## Download the data

The benchmark lives on the Hugging Face Hub as
[`Axym-Labs/axobench-v2`](https://huggingface.co/datasets/Axym-Labs/axobench-v2):
15 layer-5 cells (14 Allen all-active models and the NeuronIO cell, 3 of them
held out), train/val/test splits and paired interventions at three doses each.
Each variant is a folder of that repository; `minimal` (2,000 training traces
per cell, 14.6 GB) is released, and larger variants will sit beside it. The
population-context corpus is
[`Axym-Labs/axobench-population`](https://huggingface.co/datasets/Axym-Labs/axobench-population).
`download-data` fetches the requested folders of a variant together with its
cell index into the dataset-root layout the evaluator reads:

```bash
axobench download-data                          # minimal: val + all interventions (~6 GB)
axobench download-data --splits train,val,test  # minimal training corpus
axobench download-data --interventions inh_dropout,temporal_jitter_j8
axobench download-data --population             # population-context corpus
```

The same is available from Python as `axobench.hub.download_dataset` and
`download_population_dataset`. Data goes to `$AXOBENCH_DATA_DIR/v2/<variant>`
(default `~/.cache/axobench/v2/minimal`) unless `--output` is given. Complete
folders are skipped, so repeated calls work offline, and `--revision` pins a
Hub commit.

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
dataset root; when `dataset_root` is omitted, the validation split and the
requested interventions are downloaded on first use. It calibrates the spike
threshold on a held-out fraction of the selected validation traces, computes
the core metric set, evaluates available paired interventions, and returns a
fixed 24-row attention panel.

```python
from axobench.benchmark import EvaluationConfig, evaluate_model, format_evaluation_report

report = evaluate_model(
    predict,
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

## License

The code is licensed under Apache-2.0 (`LICENSE`). The three NMODL synapse
mechanisms derive from NEURON's `exp2syn.mod` and are BSD-3-Clause
(`LICENSES/`); the bundled Allen morphology is under the Allen Terms of Use.
`NOTICE` lists these files. The released
datasets are CC BY-NC 4.0: every trace is simulated on Allen Cell Types
Database reconstructions, and the Allen Institute Terms of Use permit derived
works for noncommercial purposes only. Data you generate with this code on
Allen reconstructions or models fall under the same restriction; the Apache
License covers the code, not its output.
