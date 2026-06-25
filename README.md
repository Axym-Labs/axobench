# DendroBench

DendroBench is the extracted benchmark and dataset-generation workbench for
single-neuron surrogate evaluation. It owns dataset generation, data/schema
utilities, diagnostic suites, and benchmark reporting.

The package keeps simulator-backed generation code under
`src/dendrobench/generation/`: Arbor, NEURON/Hay, CoreNEURON/Hay, SWC-template
preparation, morphology-candidate ranking, throughput probes, and the v1
parallel dataset generator. `scripts/` is reserved for thin operational entry
points such as launch scripts and data-download helpers.

Model implementations, training loops, checkpoint formats, and model-specific
experiments stay in `bnn_sim`. DendroBench evaluates caller-supplied prediction
functions or stored prediction/diagnostic artifacts; it does not ship BranchELM,
Mamba, RNN, or training code.

The active exploration arc lives in `docs/benchmark/`, which is ignored by Git
while the benchmark audit is still exploratory.
