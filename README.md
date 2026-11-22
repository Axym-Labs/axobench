# AxoBench

AxoBench is the extracted benchmark and dataset-generation workbench for
single-neuron surrogate evaluation. It owns dataset generation, data/schema
utilities, diagnostic suites, and benchmark reporting. Its publication
contract reports Voltage SERA for voltage fidelity, subthreshold voltage RMSE
outside action potentials, Response SERA for intervention-response fidelity
(both SERA metrics are IRon-default SERA per teacher spike), and the
arithmetic Mean F1 at integer spike tolerances from 0 through 5 ms.

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
[`Axym-Labs/axobench`](https://huggingface.co/datasets/Axym-Labs/axobench):
15 layer-5 cells (14 Allen all-active models and the NeuronIO cell, 3 of them
held out), train/val/test splits and paired interventions at three doses each.
Each variant is a folder of that repository: `minimal` (2,000 training traces
per cell, 14.6 GB) and `full` (8,000 per cell, 35.6 GB). `full` contains
every `minimal` shard, and both share the same val, test and intervention
shards, so their scores are comparable. The
population-context corpus is
[`Axym-Labs/axobench-population`](https://huggingface.co/datasets/Axym-Labs/axobench-population).
`download-data` fetches the requested folders of a variant together with its
cell index into the dataset-root layout the evaluator reads:

```bash
axobench download-data                          # minimal: val + all interventions (~6 GB)
axobench download-data --splits train,val,test  # minimal training corpus
axobench download-data --variant full --splits train  # full training corpus
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
AxoBench Population is available from
[Axym-Labs/axobench-population](https://huggingface.co/datasets/Axym-Labs/axobench-population).
Paired causal perturbations are released separately as
[Axym-Labs/axobench-interventions](https://huggingface.co/datasets/Axym-Labs/axobench-interventions)
so ordinary benchmark and population users do not need to download
intervention shards.
A fixed training recipe improves held-out trace metrics but does not improve
the morphology-conditioned firing-rate criterion on three held-out
connectomes, which is why connected diagnostics remain part of model
selection.

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
For selection followed by untouched confirmation, set
`validation_sample_start=0` for the selection cache and advance it by the
selection-block size for the confirmation cache. The selector preserves
morphology balance and records the start in both the report and cache
protocol, so paired comparisons cannot silently mix blocks.

Evaluation only prints by default. Persist a strict JSON or tidy CSV report
when needed:

```python
from axobench.benchmark import write_evaluation_report

write_evaluation_report(report, "runs/candidate-17.csv")
```

The terminal report is bounded to seven core metric rows, six secondary metric
rows, and 24 diagnostic rows. The primary metrics are Voltage SERA,
subthreshold RMSE, Response SERA, and Mean F1 over the declared integer timing
tolerances. SERA is the squared error-relevance area of Ribeiro and Moniz
(2020) with the IRon default relevance:
`phi` is 1 at the adjusted-boxplot fences of the teacher signal, 0 at its
median, and interpolated by IRon's monotone cubic in between, so extreme
teacher values carry the weight. SERA integrates the summed squared error over
relevance thresholds in [0, 1]; AxoBench divides that sum by the number of
teacher spikes in the scored region, so values are comparable across trace
counts and lengths. The relevance signal is the teacher soma voltage for
Voltage SERA (`voltage_sera_iron_per_spike_mv2`), the teacher dV/dt for
Dynamics SERA (`dynamics_sera_iron_per_spike_mv2_per_ms2`), and the pooled
teacher voltage change (intervention minus baseline) for Response SERA
(`response_sera_iron_per_spike_mv2`), whose error is the predicted minus the
teacher voltage change and whose spike count covers baseline and intervention
traces. Response skill compares Response SERA with a zero-response control.
Subthreshold RMSE (`subthreshold_rmse_mv`) is the soma voltage RMSE pooled
over scored steps outside the action-potential windows of every teacher and
every predicted spike, with both voltages clipped at -55 mV. A window runs from
2 steps before the spike to a cell-type-specific end: the first lag at which the
teacher's mean isolated spike waveform for that morphology (trace-id prefix)
changes by less than 1 mV/ms, computed once from all teacher traces of the
evaluated block (fixed +4 steps when fewer than 20 isolated spikes or no raw
voltage are available). Repolarization and the fast afterhyperpolarization last
until 4 to 9 ms after the spike depending on the cell, so a fixed end would
score part of the spike waveform as subthreshold; the report lists the ends.
The metric measures voltage offset, gain, and post-spike errors that Voltage
SERA, dominated by spike timing near action potentials, and spike F1 miss; excluding predicted-spike windows avoids counting spike-timing errors
twice. Its companion `subthreshold_skill` is `1 - RMSE / RMSE(null)` for the
per-trace mean clipped teacher voltage on the same steps. Subthreshold skill
and Dynamics SERA are secondary rows.
The default spike score is the arithmetic mean of event F1 at
`0, 1, 2, 3, 4, 5 ms`, so it is higher-is-better and each declared tolerance
has equal weight. The report also includes ordinary trapezoidal F1 area, exact
F1, and 5 ms F1. Every report includes the full tolerance curve and calibrated
threshold.

`attention=YES` is an early watch, `NO` is clear at the stated trigger, and
`NA` makes missing evidence explicit. Each diagnostic row also carries its
mathematical or physiological `basis` in the returned report.

For iterative comparisons, create and save a baseline with
`create_model_evaluation_cache`, then use `evaluate_model_against_cache` for
each candidate. Paired metric deltas use morphology-stratified 95% bootstrap
intervals without rerunning the baseline. The corresponding `improved`,
`regressed`, and `uncertain` labels are evidence, not a model-promotion rule.

### SERA skill and the action-potential ceiling

SERA per spike has no natural scale, so four secondary rows express it as skill
against a constant voltage, the mean -55 mV-clipped teacher voltage of the
calibration traces: `voltage_sera_skill` and `dynamics_sera_skill` are
`1 - SERA / SERA(constant)` with the same relevance and spike count. Almost all
voltage SERA sits in action-potential windows. A prediction that stays at or
below -55 mV, as from a model trained on clipped voltage, cannot exceed
`voltage_sera_ceiling_skill`, the skill of the teacher clipped at -55 mV, and
models that do not emit action potentials stay near it. The
ceiling falls as the spike rate rises (0.48 on the AxoBench confirmation block,
about 0.72 on the NeuronIO test set), so raw skill is not comparable across datasets;
`voltage_sera_skill_of_ceiling` divides the ceiling out. All four are ratios of
per-trace sums and are included in the paired bootstrap; the ceiling depends on
the teacher only, so its delta between two models is zero.

### Error anatomy

`error_anatomy` explains where a prediction's errors come from, on the
evaluator's scored traces with its spike threshold, predicted events, AP
windows and relevance:

```python
from axobench.benchmark.error_anatomy import error_anatomy, reweight_to_rate_distribution

anatomy = error_anatomy(prediction, target, raw_soma_mv=raw, trace_ids=ids, config=config)
```

It reports teacher spikes hit, missed and false predicted spikes (within 2 ms)
with the timing errors of matched events; the spike-triggered mean voltage of
teacher and model around isolated hit, missed and false spikes, which shows
whether a model emits action potentials or averages them away; subthreshold
error, null variance and skill by time since the last and until the next
teacher spike; voltage SERA, subthreshold and spike skill by teacher firing rate
(0-1, 1-3, 3-6, 6+ Hz) and by morphology; and subthreshold skill by frequency
band (Parseval on the zero-filled masked traces).
`reweight_to_rate_distribution(anatomy, reference)` reweights traces to another
dataset's rate-bin distribution for cross-dataset comparisons. From the command
line, with a cache written by `create_model_evaluation_cache` (or any NPZ with a
`prediction` array) and the block's targets (`target`, optional `raw_soma_mv`
and `trace_ids`):

```bash
axobench error-anatomy --cache runs/baseline/axobench-evaluation-cache.npz \
  --targets block-targets.npz --output runs/baseline/anatomy
```

This writes `error-anatomy.json`, the figure as PDF and PNG, and
`error-anatomy.tex`, a standalone pgfplots version of the same figure for LaTeX
documents. The evaluation settings come from the cache's protocol, and the
targets must match the cache's target hash.

## Scope

AxoBench defines evaluation and generation contracts; it does not claim that
one scalar metric captures every biologically relevant error. Voltage SERA,
subthreshold RMSE, Response SERA, and Mean F1 should be reported with the
tolerance grid, trace selection, calibration split, and uncertainty procedure.
Ordinary trapezoidal F1 area remains a compatibility view rather than a
headline metric.

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
