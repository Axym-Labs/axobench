#!/usr/bin/env python3
"""Download and compile AxoBench's official Allen Sst teacher."""

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
MODEL_ID = 497232373
SPECIMEN_ID = 482690728
DOWNLOAD_URL = (
    "https://api.brain-map.org/neuronal_model/download/"
    f"{MODEL_ID}"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/allen_inhibitory"
    / f"specimen_{SPECIMEN_ID}"
)
ASSETS = (
    PROJECT_ROOT / "src/axobench/generation/assets"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def prepare(output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    archive = output_root / f"neuronal_model_{MODEL_ID}.zip"
    model_root = output_root / "model"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.partial")
        with urllib.request.urlopen(DOWNLOAD_URL) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    if not (model_root / "fit_parameters.json").exists():
        with zipfile.ZipFile(archive) as source:
            source.extractall(model_root)
    metadata = json.loads(
        (model_root / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        int(metadata["id"]) != MODEL_ID
        or int(metadata["specimen_id"]) != SPECIMEN_ID
    ):
        raise ValueError(
            "downloaded Allen model provenance does not match "
            f"model {MODEL_ID}, specimen {SPECIMEN_ID}"
        )

    runtime_root = model_root / "runtime"
    library = runtime_root / "x86_64/libnrnmech.so"
    if not library.exists():
        runtime_root.mkdir(parents=True, exist_ok=True)
        wrapper = Path(sys.executable).with_name("nrnivmodl")
        if not wrapper.exists():
            raise FileNotFoundError(
                f"nrnivmodl wrapper not found beside {sys.executable}"
            )
        subprocess.run(
            [
                sys.executable,
                str(wrapper),
                str(model_root / "modfiles"),
                str(ASSETS / "DetAMPANMDA.mod"),
                str(ASSETS / "DetGABA.mod"),
            ],
            cwd=runtime_root,
            check=True,
        )
    result = {
        "download_url": DOWNLOAD_URL,
        "model_id": MODEL_ID,
        "specimen_id": SPECIMEN_ID,
        "model_root": str(model_root),
        "mechanism_library": str(library),
        "model_name": metadata["name"],
        "structure": metadata["specimen"]["structure"]["acronym"],
        "reported_explained_variance_ratio": metadata[
            "neuronal_model_runs"
        ][0]["explained_variance_ratio"],
    }
    (output_root / "provenance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
