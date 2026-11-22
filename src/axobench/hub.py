"""Download the public AxoBench datasets from the Hugging Face Hub.

AxoBench v2 is one Hub repository with one folder per variant: ``minimal/``
(2,000 training traces per cell) and ``full/`` (8,000; the same val, test and
intervention shards). ``download_dataset`` fetches the
requested folders of a variant together with its cell index, giving the layout
that ``V1DatasetLayout`` and ``evaluate_model`` expect::

    <root>/<variant>/train/<cell>/  .../val/<cell>/  .../test/<cell>/
    <root>/<variant>/interventions/<condition>_<dose>/
    <root>/<variant>/morphologies.json  .../morphologies/

Split shards sit in one folder per cell (the Hub allows 10,000 files per
directory). Downloads are per split or per intervention folder because each
folder's ``manifest.json`` indexes every shard in that folder. Folders whose manifest
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


REPO_ID = "Axym-Labs/axobench"
POPULATION_REPO_ID = "Axym-Labs/axobench-population"

VARIANTS = ("minimal", "full")
DEFAULT_VARIANT = "minimal"
INTERVENTION_DOSES: dict[str, tuple[str, ...]] = {
    "event_dropout": ("p10", "p35", "p70"),
    "exc_dropout": ("p10", "p35", "p70"),
    "inh_dropout": ("p10", "p35", "p70"),
    "temporal_jitter": ("j1", "j2", "j8"),
    "site_silence": ("p10", "p25", "p50"),
}
INTERVENTION_FOLDERS = tuple(
    f"{condition}_{dose}" for condition in INTERVENTION_CONDITIONS for dose in INTERVENTION_DOSES[condition]
)

PUBLIC_SPLITS = ("train", "val", "test")
POPULATION_SPLITS = ("train", "val")
DATA_DIR_ENV = "AXOBENCH_DATA_DIR"

_METADATA_FILES = (
    "README.md",
    "checksums.sha256",
    "dataset_package_manifest.json",
    "package-manifest.json",
)
_CELL_INDEX = ("morphologies.json", "morphologies/*")


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
    variant: str = DEFAULT_VARIANT,
    splits: Iterable[str] = ("val",),
    interventions: Iterable[str] | Literal["all"] | None = "all",
    revision: str | None = None,
    max_workers: int = 8,
) -> Path:
    """Download splits and intervention folders of one AxoBench v2 variant.

    The defaults fetch what ``evaluate_model`` reads: the validation split and
    all paired interventions of the minimal variant (about 6 GB). Pass
    ``splits=("train", "val", "test")`` for the training corpus. Interventions
    are named by folder (``inh_dropout_p35``) or by condition, which selects all
    of its doses. The data land in ``<root>/<variant>``, which is returned.
    """
    root = default_data_dir() / "v2" if root is None else Path(root)
    (variant,) = _validated_names(variant, VARIANTS, kind="variant")
    split_names = _validated_names(splits, PUBLIC_SPLITS, kind="split")
    if interventions is None:
        folder_names: tuple[str, ...] = ()
    elif interventions == "all":
        folder_names = INTERVENTION_FOLDERS
    else:
        if isinstance(interventions, str):
            interventions = (interventions,)
        expanded = [
            folder
            for name in interventions
            for folder in (
                [f"{name}_{dose}" for dose in INTERVENTION_DOSES[name]] if name in INTERVENTION_DOSES else [name]
            )
        ]
        folder_names = _validated_names(expanded, INTERVENTION_FOLDERS, kind="intervention")

    folders = (*split_names, *(f"interventions/{name}" for name in folder_names))
    _download_folders(
        REPO_ID,
        root,
        folders,
        prefix=variant,
        extra_patterns=_CELL_INDEX,
        revision=revision,
        max_workers=max_workers,
    )
    return root / variant


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
    prefix: str = "",
    extra_patterns: tuple[str, ...] = (),
    revision: str | None,
    max_workers: int,
) -> None:
    base = local_dir / prefix if prefix else local_dir
    missing = [folder for folder in folders if not folder_is_complete(base / folder)]
    # The first extra pattern is a file whose presence marks the extras as downloaded.
    extras_missing = bool(extra_patterns) and not (base / extra_patterns[0]).is_file()
    if not missing and not extras_missing:
        return
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:  # pragma: no cover - declared dependency
        raise ModuleNotFoundError(
            "downloading AxoBench data requires huggingface_hub; install it with "
            "`python -m pip install huggingface_hub`"
        ) from error

    def in_prefix(pattern: str) -> str:
        return f"{prefix}/{pattern}" if prefix else pattern

    target = f"{repo_id}/{prefix}" if prefix else repo_id
    print(f"[axobench] downloading {target} {', '.join(missing) or 'cell index'} -> {base}", file=sys.stderr, flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=[
            *_METADATA_FILES,
            *(in_prefix(name) for name in (*_METADATA_FILES, *extra_patterns)),
            *(in_prefix(f"{folder}/*") for folder in missing),
        ],
        max_workers=max_workers,
    )
    incomplete = [folder for folder in missing if not folder_is_complete(base / folder)]
    if incomplete:
        raise FileNotFoundError(
            f"{target} download left incomplete folders under {base}: {', '.join(incomplete)}"
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
