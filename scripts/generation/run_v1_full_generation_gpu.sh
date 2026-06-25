#!/usr/bin/env bash
set -euo pipefail

ROOT=${DENDROBENCH_ROOT:-/home/davwis/main/workspace/dendrobench}
NRN_GPU=${NRN_GPU:-/home/davwis/.local/neuron-9.0.1-gpu-runtime}
NVHPC=${NVHPC:-/home/davwis/.local/nvidia/Linux_x86_64/26.3}
BUILD_TOOLS=${BUILD_TOOLS:-/home/davwis/.local/neuron-build-tools}
DENDROBENCH_PYTHON=${DENDROBENCH_PYTHON:-/home/davwis/miniconda3/envs/bnn_sim/bin/python}
DENDROBENCH_ENV=${DENDROBENCH_ENV:-/home/davwis/miniconda3/envs/bnn_sim}
GEN_ARTIFACTS=${DENDROBENCH_GENERATION_ARTIFACTS:-$ROOT/artifacts/generation}
SWC_ROOT=${DENDROBENCH_SWC_MECH_ROOT:-$GEN_ARTIFACTS/hay_swc_template_gpu_durable_20260606}
DATA_ROOT=${DENDROBENCH_V1_DATA_ROOT:-$ROOT/data/v1-dendrobench-gpu-ei-balanced-20260619}
LOG_PATH=${DENDROBENCH_V1_GENERATION_LOG:-$GEN_ARTIFACTS/v1_full_generation_gpu.log}

export PYTHONUNBUFFERED=1
export DENDROBENCH_V1_DATA_ROOT="$DATA_ROOT"
export DENDROBENCH_GENERATION_ARTIFACTS="$GEN_ARTIFACTS"
export DENDROBENCH_SWC_MECH_ROOT="$SWC_ROOT"
export PATH="$NRN_GPU/bin:$BUILD_TOOLS/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH"
export PYTHONPATH="$NRN_GPU/lib/python:$ROOT/src:$GEN_ARTIFACTS:$GEN_ARTIFACTS/coreneuron_hay_port_probe_20260605:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SWC_ROOT/x86_64:$NRN_GPU/lib:$BUILD_TOOLS/lib:$DENDROBENCH_ENV/lib:$NVHPC/compilers/lib:$NVHPC/cuda/lib64:${LD_LIBRARY_PATH:-}"
export NEURONHOME="$NRN_GPU"
export NRNHOME="$NRN_GPU"
export NRNHOME_EXEC="$NRN_GPU"
export CORENEURONLIB="$SWC_ROOT/x86_64/libcorenrnmech.so"

mkdir -p "$DATA_ROOT" "$(dirname "$LOG_PATH")"
cd "$ROOT"

{
  echo "v1 full generation start $(date -Iseconds)"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "LOG_PATH=$LOG_PATH"
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true

  "$DENDROBENCH_PYTHON" -m dendrobench.generation.run_v1_parallel_generation \
    --output-root "$DATA_ROOT" \
    --python "$DENDROBENCH_PYTHON" \
    --parallelism 1 \
    --samples-per-shard 128 \
    --intervention-samples-per-shard 64 \
    --duration-ms 6000 \
    --backend coreneuron \
    --coreneuron-gpu \
    --train-samples-per-morphology 20000 \
    --val-samples-per-morphology 2500 \
    --test-samples-per-morphology 2500 \
    --private-test-samples-per-morphology 2500 \
    --intervention-total-samples-per-morphology 2500 \
    --seed 20260611 \
    --teacher-balance-label dendrobench_ei_balanced \
    --exc-gmax-scale 0.925 \
    --inh-gmax-scale 1.0 \
    --resume \
    "$@"

  "$DENDROBENCH_PYTHON" - <<'PY'
from pathlib import Path
import json
import os

root = Path(os.environ["DENDROBENCH_V1_DATA_ROOT"] if "DENDROBENCH_V1_DATA_ROOT" in os.environ else "/home/davwis/main/workspace/dendrobench/data/v1-neuronio-like-gpu-20260611")
summary = {
    "data_root": str(root),
    "ordinary": {},
    "interventions": {},
}
for split in ("train", "val", "test", "private-test"):
    manifest_path = root / split / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["ordinary"][split] = {
            "shards": len(manifest.get("shards", [])),
            "samples": sum(int(row["count"]) for row in manifest.get("shards", [])),
        }
for intervention_root in sorted((root / "interventions").glob("*")) if (root / "interventions").exists() else []:
    manifest_path = intervention_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["interventions"][intervention_root.name] = {
            "pair_shards": len(manifest.get("pair_shards", [])),
            "pairs": sum(int(row["count"]) for row in manifest.get("pair_shards", [])),
        }
(root / "generation_completion_check.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

  echo "v1 full generation done $(date -Iseconds)"
} 2>&1 | tee -a "$LOG_PATH"
