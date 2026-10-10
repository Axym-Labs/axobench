"""Download the public AxoBench datasets from the Hugging Face Hub.

The benchmark corpus and its paired interventions are published as two Hub
repositories. ``download_dataset`` places both into one directory with the
layout that ``V1DatasetLayout`` and ``evaluate_model`` expect::

    <root>/train/  <root>/val/  <root>/test/
    <root>/interventions/<condition>/

Downloads are per split or per intervention condition because each folder's
``manifest.json`` indexes every shard in that folder. Folders whose manifest
and shards are already present are not requested again, so repeated calls work
offline once the data is local.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import sys
from typing import Literal

from axobench.benchmark.dataset_schema import INTERVENTION_CONDITIONS


BENCHMARK_REPO_ID = "Axym-Labs/axobench"
INTERVENTIONS_REPO_ID = "Axym-Labs/axobench-interventions"
POPULATION_REPO_ID = "Axym-Labs/axobench-population"

PUBLIC_SPLITS = ("train", "val", "test")
POPULATION_SPLITS = ("train", "val")
DATA_DIR_ENV = "AXOBENCH_DATA_DIR"

_METADATA_FILES = (
    "README.md",
    "checksums.sha256",
    "dataset_package_manifest.json",
    "package-manifest.json",
)


def default_data_dir() -> Path:
    """Return ``$AXOBENCH_DATA_DIR``, else ``$XDG_CACHE_HOME/axobench``, else ``~/.cache/axobench``."""
    configured = os.environ.get(DATA_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "axobench"


def download_dataset(
    root: str | Path | None = None,
    *,
    splits: Iterable[str] = ("val",),
    interventions: Iterable[str] | Literal["all"] | None = "all",
    revision: str | None = None,
    max_workers: int = 8,
) -> Path:
    """Download benchmark splits and intervention conditions into a v1 dataset root.

    The defaults fetch what ``evaluate_model`` reads: the validation split and
    all paired interventions (about 8.5 GB). Pass ``splits=("train", "val",
    "test")`` for the full training corpus. Returns the dataset root.
    """
    root = default_data_dir() / "v1" if root is None else Path(root)
    split_names = _validated_names(splits, PUBLIC_SPLITS, kind="split")
    if interventions is None:
        condition_names: tuple[str, ...] = ()
    elif interventions == "all":
        condition_names = INTERVENTION_CONDITIONS
    else:
        condition_names = _validated_names(interventions, INTERVENTION_CONDITIONS, kind="intervention")

    _download_folders(BENCHMARK_REPO_ID, root, split_names, revision=revision, max_workers=max_workers)
    _download_folders(
        INTERVENTIONS_REPO_ID,
        root / "interventions",
        condition_names,
        revision=revision,
        max_workers=max_workers,
    )
    return root


def download_population_dataset(
    root: str | Path | None = None,
    *,
    splits: Iterable[str] = POPULATION_SPLITS,
    revision: str | None = None,
    max_workers: int = 8,
) -> Path:
    """Download the population-context corpus (``train``/``val``) and return its root."""
    root = default_data_dir() / "population" if root is None else Path(root)
    split_names = _validated_names(splits, POPULATION_SPLITS, kind="population split")
    _download_folders(POPULATION_REPO_ID, root, split_names, revision=revision, max_workers=max_workers)
    return root


def folder_is_complete(path: str | Path) -> bool:
    """Return whether ``path`` has a manifest and every shard file it lists."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    files = _manifest_files(manifest)
    return bool(files) and all((path / name).is_file() for name in files)


def _download_folders(
    repo_id: str,
    local_dir: Path,
    folders: tuple[str, ...],
    *,
    revision: str | None,
    max_workers: int,
) -> None:
    missing = [folder for folder in folders if not folder_is_complete(local_dir / folder)]
    if not missing:
        return
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:  # pragma: no cover - declared dependency
        raise ModuleNotFoundError(
            "downloading AxoBench data requires huggingface_hub; install it with "
            "`python -m pip install huggingface_hub`"
        ) from error

    print(f"[axobench] downloading {repo_id} {', '.join(missing)} -> {local_dir}", file=sys.stderr, flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=[*_METADATA_FILES, *(f"{folder}/*" for folder in missing)],
        max_workers=max_workers,
    )
    incomplete = [folder for folder in missing if not folder_is_complete(local_dir / folder)]
    if incomplete:
        raise FileNotFoundError(
            f"{repo_id} download left incomplete folders under {local_dir}: {', '.join(incomplete)}"
        )


def _manifest_files(manifest: object) -> list[str]:
    if not isinstance(manifest, dict):
        return []
    files: list[str] = []
    for key in ("shards", "pair_shards"):
        for entry in manifest.get(key, []) or []:
            if isinstance(entry, dict):
                files.extend(
                    str(value)
                    for name, value in entry.items()
                    if (name == "file" or name.endswith("_file")) and isinstance(value, str)
                )
    return files


def _validated_names(names: Iterable[str] | str, allowed: tuple[str, ...], *, kind: str) -> tuple[str, ...]:
    if isinstance(names, str):
        names = (names,)
    resolved = tuple(dict.fromkeys(names))
    unknown = [name for name in resolved if name not in allowed]
    if unknown:
        hint = " (the private test split is not published)" if "private-test" in unknown else ""
        raise ValueError(f"unknown {kind} {unknown}; expected any of {allowed}{hint}")
    return resolved
