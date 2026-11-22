"""Dataset-generation and diagnostic utilities for single-neuron benchmarks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("axobench")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0+unknown"

__all__: list[str] = ["__version__"]
