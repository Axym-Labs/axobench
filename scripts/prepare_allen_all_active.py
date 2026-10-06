#!/usr/bin/env python3
"""Download and compile Allen all-active models for the v2 excitatory teachers.

Each specimen's "Biophysical - all active" model (Gouwens et al. 2018) is looked
up through the Allen API, downloaded, unpacked to
`<output-root>/specimen_<id>/model`, and its mechanisms are compiled together
with AxoBench's deterministic synapses into `model/runtime`. Allen's channel
files share names with the Hay mechanisms (Ca_HVA, Ih, Im), so an Allen cell
always runs in a process that loads only its own library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT_ROOT / "src/axobench/generation/assets"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/allen_all_active"
ALLEN_QUERY = "https://api.brain-map.org/api/v2/data/query.json?criteria="
DOWNLOAD_URL = "https://api.brain-map.org/neuronal_model/download/{model_id}"
ALL_ACTIVE_TEMPLATE = "Biophysical - all active"

# Mouse L5 spiny cells with an all-active model and an intact apical dendrite
# (Allen API, 2026-10-06; ta1/TODO.md P0.7c).
V2_SPECIMENS = (
    471129934, 485574832, 488683425,  # Rbp4
    314822529, 473601979, 480169178, 476048909, 480351780, 484559000, 478586295, 480353286,  # Rorb
    354190013, 468120757, 314831019,  # Scnn1a
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--specimen", type=int, action="append", help="default: the 14 v2 specimens")
    args = parser.parse_args()
    rows = [prepare(specimen, args.output_root.resolve()) for specimen in (args.specimen or V2_SPECIMENS)]
    (args.output_root / "models.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def query(criteria: str) -> list[dict]:
    url = ALLEN_QUERY + criteria.replace("[", "%5B").replace("]", "%5D").replace(" ", "%20")
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())["msg"]


def all_active_model_id(specimen_id: int) -> int:
    models = query(
        f"model::NeuronalModel,rma::criteria,[specimen_id$eq{specimen_id}],"
        "rma::include,neuronal_model_template"
    )
    ids = [m["id"] for m in models if (m.get("neuronal_model_template") or {}).get("name") == ALL_ACTIVE_TEMPLATE]
    if len(ids) != 1:
        raise ValueError(f"specimen {specimen_id}: expected one all-active model, found {ids}")
    return int(ids[0])


def prepare(specimen_id: int, output_root: Path) -> dict[str, object]:
    root = output_root / f"specimen_{specimen_id}"
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / "provenance.json"
    if provenance.exists():
        cached = json.loads(provenance.read_text(encoding="utf-8"))
        if Path(str(cached["mechanism_library"])).exists():
            return cached
    model_id = all_active_model_id(specimen_id)
    archive = root / f"neuronal_model_{model_id}.zip"
    model_root = root / "model"
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        with urllib.request.urlopen(DOWNLOAD_URL.format(model_id=model_id), timeout=300) as response:
            with partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        partial.replace(archive)
    if not (model_root / "fit_parameters.json").exists():
        with zipfile.ZipFile(archive) as source:
            source.extractall(model_root)
    metadata = json.loads((model_root / "model_metadata.json").read_text(encoding="utf-8"))
    if int(metadata["id"]) != model_id or int(metadata["specimen_id"]) != specimen_id:
        raise ValueError(f"downloaded model does not match model {model_id}, specimen {specimen_id}")

    runtime_root = model_root / "runtime"
    library = runtime_root / "x86_64/libnrnmech.so"
    if not library.exists():
        runtime_root.mkdir(parents=True, exist_ok=True)
        wrapper = Path(sys.executable).with_name("nrnivmodl")
        if not wrapper.exists():
            raise FileNotFoundError(f"nrnivmodl wrapper not found beside {sys.executable}")
        mods = runtime_root / "mod"
        mods.mkdir(exist_ok=True)
        for mod in [*sorted((model_root / "modfiles").glob("*.mod")), ASSETS / "DetAMPANMDA.mod", ASSETS / "DetGABA.mod"]:
            shutil.copy2(mod, mods / mod.name)
        subprocess.run([sys.executable, str(wrapper), "mod"], cwd=runtime_root, check=True,
                       stdout=subprocess.DEVNULL)
    specimen = metadata.get("specimen") or {}
    run = (metadata.get("neuronal_model_runs") or [{}])[0]
    result = {
        "specimen_id": specimen_id,
        "model_id": model_id,
        "download_url": DOWNLOAD_URL.format(model_id=model_id),
        "model_name": metadata.get("name"),
        "structure": (specimen.get("structure") or {}).get("acronym"),
        "explained_variance_ratio": run.get("explained_variance_ratio"),
        "model_root": str(model_root),
        "mechanism_library": str(library),
    }
    provenance.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
