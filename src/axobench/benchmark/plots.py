"""Plot helpers for dataframe-first benchmark diagnostics."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("SOURCE_DATE_EPOCH", "1782252000")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

PLOT_FORMATS = ("svg", "pdf", "png")
_PAPER_STYLE_ENV = os.environ.get("AXOBENCH_PAPER_STYLE")
PAPER_STYLE = Path(_PAPER_STYLE_ENV).expanduser() if _PAPER_STYLE_ENV else None
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#777777",
    "light": "#F3F4F6",
    "black": "#111111",
}


def apply_paper_plot_style() -> None:
    """Use the compact serif style used by the IEWC paper figures."""
    if PAPER_STYLE is not None and PAPER_STYLE.exists():
        plt.style.use(str(PAPER_STYLE))
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.color": "#D7DCE2",
            "grid.linewidth": 0.55,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "font.family": "serif",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "axobench-diagnostic-plots",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_mechanistic_metric_matrix(
    rows: list[dict[str, Any]], output: str | Path
) -> None:
    """Plot selected mechanistic-response rows as compact metric profiles."""
    apply_paper_plot_style()
    selected = [
        ("phase_plane", "near_spike_phase_rmse_mv", "Near-spike\nphase RMSE"),
        ("phase_plane", "dvdt_rmse_mv_per_step", "dV/dt\nRMSE"),
        ("phase_plane", "phase_distribution_jsd", "Phase\nJSD"),
        ("exc_event_triggered", "response_gain", "Exc.\ngain"),
        ("inh_event_triggered", "response_gain", "Inh.\ngain"),
        ("recent_spike_history", "voltage_rmse_mv", "Recent-spike\nRMSE"),
        ("balanced_high_drive", "voltage_rmse_mv", "High E/I\nRMSE"),
    ]
    values: dict[tuple[str, str, str], float] = {}
    models = []
    for row in rows:
        if row.get("diagnostic") != "mechanistic_response":
            continue
        value = row.get("value")
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            continue
        model = str(row["model_id"])
        models.append(model)
        values[(model, str(row["state"]), str(row["metric"]))] = float(value)
    models = sorted(dict.fromkeys(models))
    if not models:
        return
    values_by_metric = np.full((len(models), len(selected)), np.nan, dtype=float)
    for model_index, model in enumerate(models):
        for metric_index, (state, metric, _label) in enumerate(selected):
            values_by_metric[model_index, metric_index] = values.get(
                (model, state, metric), np.nan
            )

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.4, 3.9))
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
    ]
    x = np.arange(len(models))
    for metric_index, ax in enumerate(axes.flat):
        if metric_index >= len(selected):
            ax.axis("off")
            continue
        _state, _metric, label = selected[metric_index]
        y = values_by_metric[:, metric_index]
        ax.plot(x, y, color="#B8C0CC", linewidth=0.9)
        for model_index, value in enumerate(y):
            if np.isfinite(value):
                ax.scatter(
                    model_index,
                    value,
                    s=21,
                    color=colors[model_index % len(colors)],
                    edgecolor="white",
                    linewidth=0.5,
                )
        ax.set_xticks(
            x,
            labels=[model.replace("branch_elm_", "") for model in models],
            rotation=18,
        )
        ax.text(
            0.02,
            0.94,
            label.replace("\n", " "),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )
        ax.grid(True, axis="y", alpha=0.32)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


def plot_event_response_curves(curves: dict[str, Any], output: str | Path) -> None:
    """Plot teacher/model event-triggered voltage-response curves."""
    apply_paper_plot_style()
    models = sorted(curves)
    if not models:
        return
    colors = {
        "teacher": "#222222",
        "branch_elm_best": "#2f7f75",
        "branch_elm_m1": "#8a3f3f",
        "branch_elm_m30": "#6b5aa6",
        "branch_elm_m100": "#2f5f8f",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=False)
    for ax, event_kind in zip(axes, ("exc", "inh")):
        teacher_plotted = False
        for model in models:
            event = curves[model].get(event_kind, {})
            offsets = np.asarray(event.get("offsets", []), dtype=float)
            if offsets.size == 0:
                continue
            if not teacher_plotted:
                ax.plot(
                    offsets,
                    event["target_mean"],
                    color=colors["teacher"],
                    linewidth=1.6,
                    label="teacher",
                )
                teacher_plotted = True
            ax.plot(
                offsets,
                event["prediction_mean"],
                linewidth=1.1,
                color=colors.get(model, "#555555"),
                alpha=0.9,
                label=model,
            )
        ax.axvline(0, color="#777777", linewidth=0.8, alpha=0.6)
        ax.axhline(0, color="#777777", linewidth=0.6, alpha=0.4)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Time from event (ms)")
        ax.set_ylabel("Soma response (mV)")
        ax.text(
            0.04,
            0.92,
            "Excitatory" if event_kind == "exc" else "Inhibitory",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    axes[1].legend(loc="best", frameon=False)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


def plot_phase_plane_example(example: dict[str, Any], output: str | Path) -> None:
    """Plot a downsampled target/model phase-plane comparison for one model."""
    apply_paper_plot_style()
    target_v = np.asarray(example["target_soma_mv"], dtype=float)
    pred_v = np.asarray(example["prediction_soma_mv"], dtype=float)
    target_d = np.diff(target_v, prepend=target_v[:1])
    pred_d = np.diff(pred_v, prepend=pred_v[:1])
    step = max(1, target_v.size // 4000)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True, sharey=True)
    axes[0].scatter(
        target_v[::step], target_d[::step], s=1.3, alpha=0.32, color="#222222"
    )
    axes[1].scatter(pred_v[::step], pred_d[::step], s=1.3, alpha=0.32, color="#2f7f75")
    axes[0].set_ylabel("dV/dt (mV/ms)")
    for ax, title in zip(axes, ("Teacher", str(example.get("model_id", "model")))):
        ax.set_xlabel("Soma voltage (mV)")
        ax.grid(True, alpha=0.2)
        ax.text(0.04, 0.92, title, transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


def plot_phase_plane_surface(
    *,
    output: str | Path,
    targets: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray] | None = None,
    raw_soma_mv: np.ndarray | None = None,
    statistic: str = "voltage_rmse",
    ignore_start: int = 0,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    bins: int | tuple[int, int] = 52,
    min_bin_count: int = 4,
    formats: tuple[str, ...] = PLOT_FORMATS,
) -> str:
    """Plot a binned teacher phase-plane statistic over voltage and dV/dt.

    Supported statistics are:
    - ``voltage_rmse``: per-bin RMSE for each model in ``predictions_by_model``.
    - ``teacher_spike_rate``: fraction of teacher-spike timesteps per bin.
    - ``sample_fraction``: empirical support per bin.
    """
    apply_paper_plot_style()
    targets = np.asarray(targets)
    if targets.ndim != 3 or targets.shape[-1] < 2:
        raise ValueError(
            f"targets must have shape (samples, time, channels>=2), got {targets.shape}"
        )
    if ignore_start < 0:
        raise ValueError("ignore_start must be non-negative")
    if min_bin_count <= 0:
        raise ValueError("min_bin_count must be positive")
    statistic = statistic.lower().strip()
    if statistic not in {"voltage_rmse", "teacher_spike_rate", "sample_fraction"}:
        raise ValueError(
            "statistic must be 'voltage_rmse', 'teacher_spike_rate', or 'sample_fraction'"
        )
    if statistic == "voltage_rmse" and not predictions_by_model:
        raise ValueError("predictions_by_model is required for voltage_rmse")

    teacher_soma = _soma_mv_from_targets(
        targets,
        raw_soma_mv=raw_soma_mv,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
    )
    trim = slice(ignore_start, None)
    teacher_soma = teacher_soma[:, trim]
    teacher_spikes = targets[:, trim, 0] > 0.5
    teacher_dvdt = np.diff(teacher_soma, axis=1, prepend=teacher_soma[:, :1])
    valid = np.isfinite(teacher_soma) & np.isfinite(teacher_dvdt)
    voltage = teacher_soma[valid]
    dvdt = teacher_dvdt[valid]
    if voltage.size == 0:
        raise ValueError("phase-plane surface has no finite teacher timesteps")

    v_edges, d_edges = _phase_plane_edges(voltage, dvdt, bins=bins)
    panels: list[tuple[str, np.ndarray]] = []
    color_label = ""
    cmap = "magma"
    if statistic == "voltage_rmse":
        assert predictions_by_model is not None
        for model_name, prediction in predictions_by_model.items():
            pred = np.asarray(prediction)
            if pred.shape[:2] != targets.shape[:2] or pred.shape[-1] < 2:
                raise ValueError(
                    f"prediction for {model_name!r} must match target time shape and have >=2 channels"
                )
            pred_soma = pred[:, trim, 1] / soma_scale + soma_bias_mv
            error = pred_soma - teacher_soma
            panel = _phase_plane_rmse(
                voltage=teacher_soma,
                dvdt=teacher_dvdt,
                error=error,
                valid=valid,
                v_edges=v_edges,
                d_edges=d_edges,
                min_bin_count=min_bin_count,
            )
            panels.append((_display_model_name(model_name), panel))
        color_label = "Voltage RMSE (mV)"
        cmap = "magma"
    elif statistic == "teacher_spike_rate":
        panel = _phase_plane_mean(
            voltage=teacher_soma,
            dvdt=teacher_dvdt,
            values=teacher_spikes.astype(float),
            valid=valid,
            v_edges=v_edges,
            d_edges=d_edges,
            min_bin_count=min_bin_count,
        )
        panels.append(("teacher spikes", panel))
        color_label = "Spike fraction"
        cmap = "cividis"
    else:
        count = _phase_plane_count(
            teacher_soma, teacher_dvdt, valid=valid, v_edges=v_edges, d_edges=d_edges
        )
        panel = count / np.nansum(count)
        panel[count < min_bin_count] = np.nan
        panels.append(("teacher support", panel))
        color_label = "Fraction of timesteps"
        cmap = "cividis"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_phase_plane_panels(
        panels,
        output_path,
        v_edges=v_edges,
        d_edges=d_edges,
        color_label=color_label,
        cmap=cmap,
        formats=formats,
    )
    return str(output_path.with_suffix(f".{formats[0]}"))


def plot_state_space_surface(*args: Any, **kwargs: Any) -> str:
    """Compatibility alias for the old phase-plane surface name."""
    return plot_phase_plane_surface(*args, **kwargs)


def plot_state_embedding_surface(
    *,
    output: str | Path,
    targets: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray] | None = None,
    inputs: np.ndarray | None = None,
    raw_soma_mv: np.ndarray | None = None,
    statistic: str = "voltage_rmse",
    reducer: str = "umap",
    ignore_start: int = 0,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    max_points: int = 15_000,
    bins: int | tuple[int, int] = 58,
    min_bin_count: int = 3,
    random_seed: int = 0,
    formats: tuple[str, ...] = PLOT_FORMATS,
) -> str:
    """Plot model error over a low-dimensional embedding of neuron state.

    Unlike ``plot_phase_plane_surface``, which is a voltage/dVdt phase-plane
    diagnostic, this function embeds a richer state descriptor containing
    voltage, voltage history, recent teacher spikes, and recent input drive.
    ``reducer='umap'`` uses UMAP when installed; ``reducer='pca'`` is a
    dependency-free deterministic fallback.
    """
    apply_paper_plot_style()
    targets = np.asarray(targets)
    if targets.ndim != 3 or targets.shape[-1] < 2:
        raise ValueError(
            f"targets must have shape (samples, time, channels>=2), got {targets.shape}"
        )
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    statistic = statistic.lower().strip()
    if statistic not in {"voltage_rmse", "teacher_spike_rate"}:
        raise ValueError("statistic must be 'voltage_rmse' or 'teacher_spike_rate'")
    if statistic == "voltage_rmse" and not predictions_by_model:
        raise ValueError("predictions_by_model is required for voltage_rmse")
    teacher_soma_full = _soma_mv_from_targets(
        targets,
        raw_soma_mv=raw_soma_mv,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
    )
    features, flat = _state_embedding_features(
        targets=targets,
        teacher_soma_mv=teacher_soma_full,
        inputs=None if inputs is None else np.asarray(inputs),
        ignore_start=ignore_start,
        max_points=max_points,
        random_seed=random_seed,
    )
    embedding = _reduce_state_embedding(
        features, reducer=reducer, random_seed=random_seed
    )
    bins_x, bins_y = _embedding_edges(embedding, bins=bins)
    panels: list[tuple[str, np.ndarray]] = []
    color_label = ""
    cmap = "magma"
    if statistic == "voltage_rmse":
        assert predictions_by_model is not None
        for model_name, prediction in predictions_by_model.items():
            pred = np.asarray(prediction)
            if pred.shape[:2] != targets.shape[:2] or pred.shape[-1] < 2:
                raise ValueError(
                    f"prediction for {model_name!r} must match target time shape and have >=2 channels"
                )
            pred_soma = pred[..., 1] / soma_scale + soma_bias_mv
            error = (
                pred_soma[flat["sample_index"], flat["time_index"]]
                - flat["teacher_soma_mv"]
            )
            panel = _embedding_rmse(
                embedding,
                error=error,
                x_edges=bins_x,
                y_edges=bins_y,
                min_bin_count=min_bin_count,
            )
            panels.append((_display_model_name(model_name), panel))
        color_label = "Voltage RMSE (mV)"
        cmap = "magma"
    else:
        panel = _embedding_mean(
            embedding,
            values=flat["teacher_spike"].astype(float),
            x_edges=bins_x,
            y_edges=bins_y,
            min_bin_count=min_bin_count,
        )
        panels.append(("teacher spikes", panel))
        color_label = "Spike fraction"
        cmap = "cividis"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_embedding_panels(
        panels,
        output_path,
        x_edges=bins_x,
        y_edges=bins_y,
        color_label=color_label,
        cmap=cmap,
        reducer=reducer,
        formats=formats,
    )
    return str(output_path.with_suffix(f".{formats[0]}"))


def _soma_mv_from_targets(
    targets: np.ndarray,
    *,
    raw_soma_mv: np.ndarray | None,
    soma_scale: float,
    soma_bias_mv: float,
) -> np.ndarray:
    if raw_soma_mv is not None:
        raw = np.asarray(raw_soma_mv, dtype=float)
        if raw.shape != targets.shape[:2]:
            raise ValueError(
                f"raw_soma_mv must have shape {targets.shape[:2]}, got {raw.shape}"
            )
        return raw
    return np.asarray(targets[..., 1], dtype=float) / soma_scale + soma_bias_mv


def _state_embedding_features(
    *,
    targets: np.ndarray,
    teacher_soma_mv: np.ndarray,
    inputs: np.ndarray | None,
    ignore_start: int,
    max_points: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if ignore_start < 0:
        raise ValueError("ignore_start must be non-negative")
    if ignore_start >= targets.shape[1]:
        raise ValueError("ignore_start leaves no timesteps to embed")
    soma = np.asarray(teacher_soma_mv, dtype=float)
    spikes = targets[..., 0] > 0.5
    dvdt = np.diff(soma, axis=1, prepend=soma[:, :1])
    time_mask = np.zeros(targets.shape[:2], dtype=bool)
    time_mask[:, ignore_start:] = True
    valid = time_mask & np.isfinite(soma) & np.isfinite(dvdt)
    candidate = np.flatnonzero(valid.ravel())
    if candidate.size == 0:
        raise ValueError("state embedding has no valid timesteps")
    flat_spikes = spikes.ravel()
    flat_dvdt = np.abs(dvdt).ravel()
    high_dvdt_threshold = np.nanquantile(flat_dvdt[candidate], 0.95)
    priority_mask = flat_spikes | (flat_dvdt >= high_dvdt_threshold)
    priority = candidate[priority_mask[candidate]]
    rng = np.random.default_rng(random_seed)
    if candidate.size > max_points:
        remaining_budget = max(0, max_points - min(priority.size, max_points // 2))
        priority_keep = (
            rng.choice(priority, size=max_points - remaining_budget, replace=False)
            if priority.size > max_points - remaining_budget
            else priority
        )
        rest = np.setdiff1d(candidate, priority_keep, assume_unique=False)
        if rest.size > remaining_budget:
            rest = rng.choice(rest, size=remaining_budget, replace=False)
        selected = np.concatenate([priority_keep, rest])
        rng.shuffle(selected)
    else:
        selected = candidate
    sample_index, time_index = np.unravel_index(selected, targets.shape[:2])
    feature_columns = [
        soma[sample_index, time_index],
        dvdt[sample_index, time_index],
        _lagged_at(soma, sample_index, time_index, 1),
        _lagged_at(soma, sample_index, time_index, 5),
        _lagged_at(soma, sample_index, time_index, 20),
        _lagged_at(dvdt, sample_index, time_index, 1),
        _lagged_at(dvdt, sample_index, time_index, 5),
        _window_sum(spikes.astype(float), sample_index, time_index, 5),
        _window_sum(spikes.astype(float), sample_index, time_index, 25),
    ]
    if inputs is not None:
        if inputs.shape[:2] != targets.shape[:2]:
            raise ValueError(
                f"inputs must match target sample/time shape {targets.shape[:2]}, got {inputs.shape[:2]}"
            )
        input_activity = np.abs(inputs).sum(axis=-1).astype(float)
        input_positive = (inputs > 0).sum(axis=-1).astype(float)
        input_negative = (inputs < 0).sum(axis=-1).astype(float)
        for activity in (input_activity, input_positive, input_negative):
            feature_columns.extend(
                [
                    _window_sum(activity, sample_index, time_index, 1),
                    _window_sum(activity, sample_index, time_index, 5),
                    _window_sum(activity, sample_index, time_index, 25),
                ]
            )
    features = np.column_stack(feature_columns).astype(float)
    features = _standardize_features(features)
    flat = {
        "sample_index": sample_index,
        "time_index": time_index,
        "teacher_soma_mv": soma[sample_index, time_index],
        "teacher_spike": spikes[sample_index, time_index],
    }
    return features, flat


def _lagged_at(
    values: np.ndarray, sample_index: np.ndarray, time_index: np.ndarray, lag: int
) -> np.ndarray:
    lagged_index = np.maximum(time_index - lag, 0)
    return values[sample_index, lagged_index]


def _window_sum(
    values: np.ndarray, sample_index: np.ndarray, time_index: np.ndarray, window: int
) -> np.ndarray:
    if window <= 0:
        return values[sample_index, time_index]
    cumsum = np.cumsum(values, axis=1)
    end = time_index
    start = np.maximum(time_index - window + 1, 0)
    total = cumsum[sample_index, end]
    before = np.zeros_like(total, dtype=float)
    has_before = start > 0
    before[has_before] = cumsum[sample_index[has_before], start[has_before] - 1]
    return total - before


def _standardize_features(features: np.ndarray) -> np.ndarray:
    median = np.nanmedian(features, axis=0)
    q25, q75 = np.nanquantile(features, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
    standardized = (features - median) / scale
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def _reduce_state_embedding(
    features: np.ndarray, *, reducer: str, random_seed: int
) -> np.ndarray:
    reducer = reducer.lower().strip()
    if reducer == "umap":
        try:
            from umap import UMAP  # type: ignore
        except ImportError as error:
            raise ImportError(
                "UMAP state embeddings require the optional dependency 'umap-learn'"
            ) from error
        embedding = UMAP(
            n_neighbors=35,
            min_dist=0.08,
            metric="euclidean",
            random_state=random_seed,
            low_memory=True,
        ).fit_transform(features)
    elif reducer == "pca":
        embedding = _pca_embedding(features)
    else:
        raise ValueError("reducer must be 'umap' or 'pca'")
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[1] != 2:
        raise ValueError(
            f"state embedding reducer returned shape {embedding.shape}, expected second dimension 2"
        )
    return embedding


def _pca_embedding(features: np.ndarray) -> np.ndarray:
    centered = features - np.mean(features, axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    if components.shape[1] < 2:
        components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))
    return centered @ components


def _embedding_edges(
    embedding: np.ndarray, *, bins: int | tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bins, int):
        if bins <= 1:
            raise ValueError("bins must be greater than 1")
        x_bins = y_bins = bins
    else:
        x_bins, y_bins = bins
        if x_bins <= 1 or y_bins <= 1:
            raise ValueError("both bin counts must be greater than 1")
    x_lo, x_hi = np.nanquantile(embedding[:, 0], [0.005, 0.995])
    y_lo, y_hi = np.nanquantile(embedding[:, 1], [0.005, 0.995])
    x_pad = max(1e-3, 0.03 * float(x_hi - x_lo))
    y_pad = max(1e-3, 0.03 * float(y_hi - y_lo))
    return (
        np.linspace(float(x_lo - x_pad), float(x_hi + x_pad), int(x_bins) + 1),
        np.linspace(float(y_lo - y_pad), float(y_hi + y_pad), int(y_bins) + 1),
    )


def _embedding_rmse(
    embedding: np.ndarray,
    *,
    error: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_bin_count: int,
) -> np.ndarray:
    valid = (
        np.isfinite(error) & np.isfinite(embedding[:, 0]) & np.isfinite(embedding[:, 1])
    )
    count = np.histogram2d(
        embedding[valid, 0], embedding[valid, 1], bins=(x_edges, y_edges)
    )[0]
    squared_sum = np.histogram2d(
        embedding[valid, 0],
        embedding[valid, 1],
        bins=(x_edges, y_edges),
        weights=np.square(error[valid]),
    )[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(squared_sum / count)
    rmse[count < min_bin_count] = np.nan
    return rmse


def _embedding_mean(
    embedding: np.ndarray,
    *,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_bin_count: int,
) -> np.ndarray:
    valid = (
        np.isfinite(values)
        & np.isfinite(embedding[:, 0])
        & np.isfinite(embedding[:, 1])
    )
    count = np.histogram2d(
        embedding[valid, 0], embedding[valid, 1], bins=(x_edges, y_edges)
    )[0]
    total = np.histogram2d(
        embedding[valid, 0],
        embedding[valid, 1],
        bins=(x_edges, y_edges),
        weights=values[valid],
    )[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count < min_bin_count] = np.nan
    return mean


def _plot_embedding_panels(
    panels: list[tuple[str, np.ndarray]],
    output: Path,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    color_label: str,
    cmap: str,
    reducer: str,
    formats: tuple[str, ...],
) -> None:
    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.45 * n_cols + 0.45, 2.3 * n_rows), squeeze=False
    )
    finite_values = [
        panel[np.isfinite(panel)]
        for _label, panel in panels
        if np.any(np.isfinite(panel))
    ]
    values = (
        np.concatenate(finite_values) if finite_values else np.asarray([], dtype=float)
    )
    vmax = float(np.nanpercentile(values, 96)) if values.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    mesh = None
    for ax, (label, panel) in zip(axes.flat, panels, strict=False):
        mesh = ax.pcolormesh(
            x_edges, y_edges, panel.T, cmap=cmap, vmin=0.0, vmax=vmax, shading="auto"
        )
        ax.text(
            0.04, 0.93, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.8
        )
        ax.set_box_aspect(1.0)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{reducer.upper()} state 1")
        ax.set_ylabel(f"{reducer.upper()} state 2")
    for ax in list(axes.flat)[n_panels:]:
        ax.axis("off")
    fig.subplots_adjust(
        left=0.08, right=0.86, bottom=0.14, top=0.96, wspace=0.35, hspace=0.42
    )
    if mesh is not None:
        cax = fig.add_axes([0.89, 0.24, 0.018, 0.60])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(color_label)
    _save_figure(fig, output, formats)
    plt.close(fig)


def _phase_plane_edges(
    voltage: np.ndarray,
    dvdt: np.ndarray,
    *,
    bins: int | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bins, int):
        if bins <= 1:
            raise ValueError("bins must be greater than 1")
        x_bins = y_bins = bins
    else:
        x_bins, y_bins = bins
        if x_bins <= 1 or y_bins <= 1:
            raise ValueError("both bin counts must be greater than 1")
    v_lo, v_hi = np.nanquantile(voltage, [0.002, 0.998])
    d_limit = float(np.nanquantile(np.abs(dvdt), 0.998))
    d_limit = max(d_limit, 1e-3)
    if not np.isfinite(v_hi) or v_hi <= v_lo:
        center = float(np.nanmean(voltage)) if voltage.size else 0.0
        v_lo, v_hi = center - 1.0, center + 1.0
    v_pad = max(0.5, 0.025 * float(v_hi - v_lo))
    return (
        np.linspace(float(v_lo - v_pad), float(v_hi + v_pad), int(x_bins) + 1),
        np.linspace(-d_limit, d_limit, int(y_bins) + 1),
    )


def _phase_plane_count(
    voltage: np.ndarray,
    dvdt: np.ndarray,
    *,
    valid: np.ndarray,
    v_edges: np.ndarray,
    d_edges: np.ndarray,
) -> np.ndarray:
    return np.histogram2d(voltage[valid], dvdt[valid], bins=(v_edges, d_edges))[
        0
    ].astype(float)


def _phase_plane_rmse(
    *,
    voltage: np.ndarray,
    dvdt: np.ndarray,
    error: np.ndarray,
    valid: np.ndarray,
    v_edges: np.ndarray,
    d_edges: np.ndarray,
    min_bin_count: int,
) -> np.ndarray:
    value_valid = valid & np.isfinite(error)
    count = _phase_plane_count(
        voltage, dvdt, valid=value_valid, v_edges=v_edges, d_edges=d_edges
    )
    squared_sum = np.histogram2d(
        voltage[value_valid],
        dvdt[value_valid],
        bins=(v_edges, d_edges),
        weights=np.square(error[value_valid]),
    )[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(squared_sum / count)
    rmse[count < min_bin_count] = np.nan
    return rmse


def _phase_plane_mean(
    *,
    voltage: np.ndarray,
    dvdt: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    v_edges: np.ndarray,
    d_edges: np.ndarray,
    min_bin_count: int,
) -> np.ndarray:
    value_valid = valid & np.isfinite(values)
    count = _phase_plane_count(
        voltage, dvdt, valid=value_valid, v_edges=v_edges, d_edges=d_edges
    )
    total = np.histogram2d(
        voltage[value_valid],
        dvdt[value_valid],
        bins=(v_edges, d_edges),
        weights=values[value_valid],
    )[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count < min_bin_count] = np.nan
    return mean


def _plot_phase_plane_panels(
    panels: list[tuple[str, np.ndarray]],
    output: Path,
    *,
    v_edges: np.ndarray,
    d_edges: np.ndarray,
    color_label: str,
    cmap: str,
    formats: tuple[str, ...],
) -> None:
    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.35 * n_cols + 0.45, 2.25 * n_rows), squeeze=False
    )
    finite_values = [
        panel[np.isfinite(panel)]
        for _label, panel in panels
        if np.any(np.isfinite(panel))
    ]
    values = (
        np.concatenate(finite_values) if finite_values else np.asarray([], dtype=float)
    )
    if values.size:
        vmax = float(np.nanpercentile(values, 96))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.nanmax(values))
    else:
        vmax = 1.0
    vmin = 0.0
    mesh = None
    for ax, (label, panel) in zip(axes.flat, panels, strict=False):
        mesh = ax.pcolormesh(
            v_edges, d_edges, panel.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
        )
        ax.text(
            0.04, 0.93, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.8
        )
        ax.set_box_aspect(1.0)
        ax.grid(False)
        ax.set_xlabel("Teacher soma voltage (mV)")
        ax.set_ylabel(r"Teacher $\Delta V$ (mV/step)")
    for ax in list(axes.flat)[n_panels:]:
        ax.axis("off")
    fig.subplots_adjust(
        left=0.08, right=0.86, bottom=0.14, top=0.96, wspace=0.42, hspace=0.46
    )
    if mesh is not None:
        cax = fig.add_axes([0.89, 0.24, 0.018, 0.60])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(color_label)
    _save_figure(fig, output, formats)
    plt.close(fig)


def _display_model_name(model_name: str) -> str:
    return (
        model_name.replace("branch_elm_", "Branch-ELM ")
        .replace("sera_weighted_", "SERA ")
        .replace("baseline", "uniform MSE")
        .replace("0p5", "0.5")
        .replace("1p0", "1.0")
        .replace("_", " ")
    )


def group_rows_by_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_id"])].append(row)
    return dict(grouped)


def write_core_metric_set_artifacts(
    *,
    summary: dict[str, Any] | str | Path,
    output_dir: str | Path,
    formats: tuple[str, ...] = PLOT_FORMATS,
) -> dict[str, Any]:
    """Write plots and tables for the promoted AxoBench core metric set."""
    apply_paper_plot_style()
    payload = _load_json(summary) if isinstance(summary, (str, Path)) else summary
    models = list(payload.get("models", []))
    if not models:
        raise ValueError("core metric summary contains no models")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = [
        _plot_core_spike_f1_curves(
            models, output / "fig01_spike_f1_tolerance_curve", formats
        ),
        _plot_core_sera_pareto(
            models,
            output / "fig02_sera_model_size_pareto",
            formats,
            x_axis="parameters",
        ),
        _plot_core_sera_pareto(
            models,
            output / "fig03_sera_branch_memory_sweep",
            formats,
            x_axis="branch_memory",
        ),
    ]
    tables = [
        _write_core_metric_table(models, output / "table01_core_metric_set.md"),
        _write_core_metric_configuration_table(
            payload, output / "table02_core_metric_configuration.md"
        ),
    ]
    manifest = {
        "artifact_kind": "axobench_core_metric_set",
        "figures": figures,
        "tables": tables,
        "formats": list(formats),
        "source_summary": str(summary) if isinstance(summary, (str, Path)) else None,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _format_value(value: float) -> str:
    value = float(value)
    if value != 0.0 and (abs(value) < 1e-2 or abs(value) >= 1e3):
        return f"{value:.1e}"
    return f"{value:.2f}"


def _plot_core_spike_f1_curves(
    models: list[dict[str, Any]], output: Path, formats: tuple[str, ...]
) -> str:
    fig, ax = plt.subplots(figsize=(5.9, 3.05))
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
        OKABE_ITO["red"],
    ]
    for index, model in enumerate(models):
        curve = model.get("spike_f1_curve_full") or model.get("spike_f1_curve", [])
        if not curve:
            continue
        x = [int(row["tolerance_steps"]) for row in curve]
        y = [float(row["f1"]) if row.get("f1") is not None else np.nan for row in curve]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.25,
            markersize=3.2,
            color=colors[index % len(colors)],
            label=str(model.get("model_id", f"model_{index}")),
        )
    ax.set_xlabel("Spike timing tolerance (ms)")
    ax.set_ylabel("Discrete event F1")
    ax.set_ylim(
        0.0, min(1.0, max(0.25, _max_model_value(models, "spike_f1_area") * 2.4))
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    _save_figure(fig, output, formats)
    plt.close(fig)
    return str(output.with_suffix(f".{formats[0]}"))


def _plot_core_sera_pareto(
    models: list[dict[str, Any]],
    output: Path,
    formats: tuple[str, ...],
    *,
    x_axis: str = "parameters",
) -> str:
    if x_axis not in {"parameters", "branch_memory"}:
        raise ValueError("x_axis must be 'parameters' or 'branch_memory'")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.05), sharex=True)
    legend_handles: dict[str, Any] = {}
    panels = (
        (axes[0], "voltage_sera_iron_per_spike_mv2", "Voltage SERA (mV²/spike)"),
        (axes[1], "subthreshold_rmse_mv", "Subthreshold RMSE (mV)"),
        (
            axes[2],
            "response_sera_iron_per_spike_mv2",
            "Response SERA (mV²/spike)",
        ),
    )
    for ax, metric, label in panels:
        points = _core_metric_points(models, metric)
        sweep = sorted(
            (point for point in points if point["branch_elm_size"] is not None),
            key=lambda p: int(p["branch_elm_size"]),
        )
        best = [point for point in points if point["model_id"] == "branch_elm_best"]
        other = [
            point
            for point in points
            if point["branch_elm_size"] is None
            and point["model_id"] != "branch_elm_best"
        ]
        use_branch_size_axis = x_axis == "branch_memory"
        max_branch_size = max(
            (int(point["branch_elm_size"]) for point in sweep), default=None
        )
        if sweep:
            ax.scatter(
                [
                    _core_metric_plot_x(point, branch_size_axis=use_branch_size_axis)
                    for point in sweep
                ],
                [point["value"] for point in sweep],
                s=23,
                color=OKABE_ITO["blue"],
                alpha=0.9,
                zorder=3,
                label="Branch-ELM sweep",
            )
            _plot_log_x_smooth_fit(
                ax,
                sweep,
                color=OKABE_ITO["blue"],
                branch_size_axis=use_branch_size_axis,
            )
            for point in sweep:
                size = int(point["branch_elm_size"])
                if _label_branch_elm_size(size):
                    ax.annotate(
                        f"$d_m$={size}",
                        (
                            _core_metric_plot_x(
                                point, branch_size_axis=use_branch_size_axis
                            ),
                            point["value"],
                        ),
                        textcoords="offset points",
                        xytext=(3, 3),
                        fontsize=5.8,
                    )
        for point in best:
            best_x = (
                float(max_branch_size)
                if use_branch_size_axis and max_branch_size is not None
                else point["params"]
            )
            ax.scatter(
                best_x,
                point["value"],
                s=46,
                marker="*",
                color=OKABE_ITO["orange"],
                edgecolor="#222222",
                linewidth=0.35,
                zorder=4,
                label="Best-effort checkpoint",
            )
            ax.annotate(
                "best",
                (best_x, point["value"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=5.8,
            )
        colors = [
            OKABE_ITO["green"],
            OKABE_ITO["purple"],
            OKABE_ITO["red"],
            OKABE_ITO["gray"],
        ]
        if not use_branch_size_axis:
            for index, point in enumerate(other):
                ax.scatter(
                    point["params"],
                    point["value"],
                    s=26,
                    color=colors[index % len(colors)],
                    zorder=3,
                )
                ax.annotate(
                    point["model_id"].replace("branch_elm_", ""),
                    (point["params"], point["value"]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=5.8,
                )
        ax.set_xscale("log")
        ax.set_xlabel(
            r"Branch memory size $d_m$" if use_branch_size_axis else "Parameters"
        )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.35)
        ax.margins(x=0.07, y=0.12)
        handles, labels = ax.get_legend_handles_labels()
        legend_handles.update(dict(zip(labels, handles, strict=False)))
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(3, len(legend_handles)),
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save_figure(fig, output, formats)
    plt.close(fig)
    return str(output.with_suffix(f".{formats[0]}"))


def _core_metric_points(
    models: list[dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        value = model.get(metric)
        params = model.get("n_parameters")
        if value is None or params is None:
            continue
        try:
            value_f = float(value)
            params_f = float(params)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(value_f) and np.isfinite(params_f) and params_f > 0.0):
            continue
        model_id = str(model.get("model_id", f"model_{index}"))
        points.append(
            {
                "model_id": model_id,
                "params": params_f,
                "value": value_f,
                "branch_elm_size": _branch_elm_size(model_id),
            }
        )
    return points


def _branch_elm_size(model_id: str) -> int | None:
    prefix = "branch_elm_m"
    if not model_id.startswith(prefix):
        return None
    suffix = model_id[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _label_branch_elm_size(size: int) -> bool:
    return size in {1, 5, 10, 30, 100}


def _core_metric_plot_x(point: dict[str, Any], *, branch_size_axis: bool) -> float:
    if branch_size_axis and point["branch_elm_size"] is not None:
        return float(point["branch_elm_size"])
    return float(point["params"])


def _plot_log_x_smooth_fit(
    ax: plt.Axes, points: list[dict[str, Any]], *, color: str, branch_size_axis: bool
) -> None:
    if len(points) < 4:
        return
    xs = np.asarray(
        [
            _core_metric_plot_x(point, branch_size_axis=branch_size_axis)
            for point in points
        ],
        dtype=float,
    )
    ys = np.asarray([point["value"] for point in points], dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    log_xs = np.log10(xs)
    if len(np.unique(log_xs)) < 4:
        return
    grid = np.linspace(float(log_xs.min()), float(log_xs.max()), 180)
    fit = _local_linear_smooth(log_xs, ys, grid, fraction=0.45)
    ax.plot(
        10.0**grid, fit, color=color, linewidth=1.15, alpha=0.84, label="Smooth fit"
    )


def _local_linear_smooth(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, *, fraction: float
) -> np.ndarray:
    """Evaluate a LOWESS-style local linear smooth on one-dimensional data."""
    if x.ndim != 1 or y.ndim != 1 or grid.ndim != 1 or x.size != y.size:
        raise ValueError(
            "x, y, and grid must be one-dimensional, with x and y the same length"
        )
    if x.size < 3:
        return np.interp(grid, x, y)
    span_count = max(3, int(np.ceil(float(fraction) * x.size)))
    span_count = min(span_count, x.size)
    smoothed = []
    for point in grid:
        distances = np.abs(x - point)
        bandwidth = float(np.partition(distances, span_count - 1)[span_count - 1])
        if bandwidth <= 0.0:
            positive = distances[distances > 0.0]
            bandwidth = float(np.min(positive)) if positive.size else 1.0
        scaled = np.minimum(distances / bandwidth, 1.0)
        weights = (1.0 - scaled**3) ** 3
        if float(np.sum(weights)) <= 0.0:
            smoothed.append(float(np.interp(point, x, y)))
            continue
        design = np.column_stack([np.ones_like(x), x - point])
        weighted_design = design * weights[:, None]
        weighted_y = y * weights
        coefficients = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        smoothed.append(float(coefficients[0]))
    return np.asarray(smoothed, dtype=float)


def _write_core_metric_table(models: list[dict[str, Any]], output: Path) -> str:
    headers = (
        "model",
        "params",
        "Voltage SERA (mV²/spike)",
        "Subthreshold RMSE (mV)",
        "Response SERA (mV²/spike)",
        "Mean F1 (0-5 ms)",
        "trapezoidal F1 area",
    )
    rows = []
    for model in models:
        rows.append(
            (
                model.get("model_id"),
                model.get("n_parameters"),
                _fmt_table(model.get("voltage_sera_iron_per_spike_mv2")),
                _fmt_table(model.get("subthreshold_rmse_mv")),
                _fmt_table(model.get("response_sera_iron_per_spike_mv2")),
                _fmt_table(model.get("spike_mean_f1")),
                _fmt_table(model.get("spike_f1_area")),
            )
        )
    _write_markdown_table(output, headers, rows)
    return str(output)


def _write_core_metric_configuration_table(
    payload: dict[str, Any], output: Path
) -> str:
    metadata = payload.get("metadata", {})
    rows = [
        ("evaluation split", metadata.get("evaluation_split")),
        ("calibration split", metadata.get("calibration_split")),
        ("ignore start", metadata.get("ignore_start")),
        ("spike tolerance interval", metadata.get("spike_tolerance_interval")),
        ("voltage target view", metadata.get("voltage_target_view")),
        ("response target view", metadata.get("response_target_view")),
        ("non-saturation criterion", metadata.get("non_saturation_criterion")),
    ]
    _write_markdown_table(output, ("setting", "value"), rows)
    return str(output)


def _save_figure(fig: plt.Figure, output: Path, formats: tuple[str, ...]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240)


def _max_model_value(models: list[dict[str, Any]], field: str) -> float:
    values = [
        float(model[field])
        for model in models
        if isinstance(model.get(field), (int, float)) and np.isfinite(model[field])
    ]
    return max(values) if values else 1.0


def _fmt_table(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.4g}"


def write_analysis_audit_artifacts(
    *,
    output_dir: str | Path,
    canonical_summary: str | Path,
    mechanistic_summary: str | Path,
    sparse_event_summary: str | Path,
    point_rows: str | Path,
    tcn_summaries: list[str | Path],
    morphology_contrast_rows: str | Path | None = None,
    neuronio_intervention_rows: str | Path | None = None,
    neuronio_intervention_root: str | Path | None = None,
    formats: tuple[str, ...] = PLOT_FORMATS,
) -> dict[str, Any]:
    """Write paper-facing diagnostic plots and tables from saved audit artifacts.

    This function is intentionally neutral about the paper claim labels. It is
    the package-level plotter export for the normal diagnostics used in the TA1
    audit: metric decoupling, target-view audit, mechanistic response,
    robustness directionality, morphology contrast, protocol coverage, and
    adapter fairness.
    """
    apply_paper_plot_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    canonical = _load_json(canonical_summary)
    mechanistic = _load_json(mechanistic_summary)
    sparse = _load_json(sparse_event_summary)
    point_metrics = _point_metrics(point_rows)
    tcn = [_load_json(path) for path in tcn_summaries]
    morphology_rows = (
        _load_jsonl(morphology_contrast_rows)
        if morphology_contrast_rows is not None
        else []
    )
    neuronio_rows = (
        _load_jsonl(neuronio_intervention_rows)
        if neuronio_intervention_rows is not None
        else []
    )
    intervention_evidence = bool(neuronio_rows)

    figures = [
        _plot_spike_metric_identification(
            point_metrics, output / "fig01_spike_metric_identification", formats
        ),
        _plot_mechanistic_behavior_matrix(
            mechanistic, output / "fig02_mechanistic_behavior_profiles", formats
        ),
        _plot_target_view_distortion_map(
            canonical, output / "fig03_target_view_distortion_profile", formats
        ),
        _plot_sparse_event_sensitivity_profiles(
            sparse, output / "fig04_sparse_event_sensitivity_profiles", formats
        ),
        _plot_morphology_intervention_response(
            morphology_rows, output / "fig05_morphology_intervention_response", formats
        ),
        _plot_tcn_preprocessing_trajectories(
            tcn, output / "fig06_tcn_preprocessing_trajectories", formats
        ),
        (
            _plot_neuronio_matched_intervention_response(
                neuronio_rows,
                output / "fig07_neuronio_matched_intervention_response",
                formats,
            )
            if intervention_evidence
            else _plot_teacher_intervention_contract(
                output / "fig07_teacher_intervention_contract", formats
            )
        ),
    ]
    trace_evidence = False
    if neuronio_intervention_root is not None:
        trace_figure = _plot_neuronio_intervention_trace_examples(
            Path(neuronio_intervention_root),
            output / "fig08_neuronio_matched_trace_examples",
            formats,
        )
        if trace_figure is not None:
            figures.append(trace_figure)
            trace_evidence = True
    tables = [
        _write_point_artifact_table(
            output / "table01_point_artifact_map.md",
            intervention_evidence=intervention_evidence,
            trace_evidence=trace_evidence,
        ),
        _write_metric_decoupling_table(
            point_metrics, canonical, output / "table02_metric_decoupling.md"
        ),
        _write_target_protocol_table(
            canonical, tcn, output / "table03_target_and_adapter_protocol.md"
        ),
        _write_morphology_intervention_table(
            morphology_rows, output / "table04_morphology_intervention_response.md"
        ),
        _write_group_evidence_table(
            output / "table05_group_claim_evidence.md",
            intervention_evidence=intervention_evidence,
            trace_evidence=trace_evidence,
        ),
    ]
    if intervention_evidence:
        tables.append(
            _write_neuronio_intervention_table(
                neuronio_rows, output / "table06_neuronio_matched_interventions.md"
            )
        )
    manifest = {
        "artifact_kind": "axobench_analysis_audit_plots",
        "plotting_requirement": "matplotlib",
        "formats": list(formats),
        "figures": figures,
        "tables": tables,
        "source_artifacts": {
            "canonical_summary": str(canonical_summary),
            "mechanistic_summary": str(mechanistic_summary),
            "sparse_event_summary": str(sparse_event_summary),
            "point_rows": str(point_rows),
            "tcn_summaries": [str(path) for path in tcn_summaries],
            "morphology_contrast_rows": str(morphology_contrast_rows)
            if morphology_contrast_rows
            else None,
            "neuronio_intervention_rows": str(neuronio_intervention_rows)
            if neuronio_intervention_rows
            else None,
            "neuronio_intervention_root": str(neuronio_intervention_root)
            if neuronio_intervention_root
            else None,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _plot_spike_metric_identification(
    point_metrics: dict[tuple[int, str], Any], output: Path, formats: tuple[str, ...]
) -> str:
    points = json.loads(str(point_metrics[(1, "points")]))
    deltas = {
        "min": float(point_metrics[(2, "delta_min")]),
        "mean": float(point_metrics[(2, "delta_mean")]),
        "max": float(point_metrics[(2, "delta_max")]),
    }
    regimes = ("canonical_test", "ood_rates", "combined_ergodic")
    colors = {
        "canonical_test": OKABE_ITO["blue"],
        "ood_rates": OKABE_ITO["orange"],
        "combined_ergodic": OKABE_ITO["green"],
    }
    markers = {"branch_elm_best": "o", "branch_elm_m30": "s", "branch_elm_m100": "^"}
    fig = plt.figure(figsize=(7.4, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.2], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for regime in regimes:
        rows = [row for row in points if row["regime"] == regime]
        for row in rows:
            ax.scatter(
                row["spike_auc"],
                row["event_f1_2ms"],
                marker=markers.get(row["model_id"], "o"),
                s=48,
                color=colors[regime],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
    for model_id, marker in markers.items():
        model_rows = [row for row in points if row["model_id"] == model_id]
        model_rows.sort(key=lambda row: regimes.index(row["regime"]))
        ax.plot(
            [row["spike_auc"] for row in model_rows],
            [row["event_f1_2ms"] for row in model_rows],
            color="#777777",
            linewidth=0.6,
            alpha=0.55,
            zorder=1,
        )
    for regime in regimes:
        ax.scatter(
            [], [], color=colors[regime], marker="o", label=regime.replace("_", " ")
        )
    for model_id, marker in markers.items():
        ax.scatter(
            [],
            [],
            color="#BBBBBB",
            edgecolor="white",
            marker=marker,
            label=_model_short(model_id),
        )
    ax.set_xlabel("spike AUC")
    ax.set_ylabel("event F1 within 2 ms")
    ax.set_xlim(0.985, 0.996)
    ax.set_ylim(0.20, 0.40)
    ax.legend(loc="lower right", ncols=2, handletextpad=0.35, columnspacing=0.8)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axvline(0, color="#222222", linewidth=0.8)
    keys = ["min", "mean", "max"]
    vals = [deltas[key] for key in keys]
    y = np.arange(len(keys))
    ax2.scatter(
        vals,
        y,
        s=45,
        color=[OKABE_ITO["gray"], OKABE_ITO["blue"], OKABE_ITO["gray"]],
        zorder=3,
    )
    for yy, val in zip(y, vals):
        ax2.plot([0, val], [yy, yy], color="#777777", linewidth=1.2)
        ax2.text(
            val + (0.001 if val >= 0 else -0.001),
            yy,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
        )
    ax2.set_yticks(y, labels=keys)
    ax2.set_xlabel("prevalence-matched\nminus validation F1")
    ax2.set_xlim(-0.01, 0.025)
    _add_panel_label(ax, "A")
    _add_panel_label(ax2, "B")
    _save_figure(fig, output, formats)
    return output.name


def _plot_mechanistic_behavior_matrix(
    mechanistic: dict[str, Any], output: Path, formats: tuple[str, ...]
) -> str:
    reports = mechanistic["reports"]
    models = ["branch_elm_best", "branch_elm_m1", "branch_elm_m30", "branch_elm_m100"]
    metrics = [
        (
            "Near-spike phase RMSE",
            "mV",
            lambda r: r["phase_plane"]["near_spike_phase_rmse_mv"],
        ),
        (
            "High-dV/dt sign error",
            "fraction",
            lambda r: 1.0 - r["phase_plane"]["high_dvdt_sign_agreement_fraction"],
        ),
        (
            "Recent/quiet RMSE ratio",
            "x",
            lambda r: (
                r["history_conditioned_error"]["recent_spike_history"][
                    "voltage_rmse_mv"
                ]
                / max(
                    r["history_conditioned_error"]["quiet_history"]["voltage_rmse_mv"],
                    1e-12,
                )
            ),
        ),
        (
            "Excitatory latency error",
            "ms",
            lambda r: abs(r["event_triggered_response"]["exc"]["latency_error_steps"]),
        ),
        (
            "Inhibitory latency error",
            "ms",
            lambda r: abs(r["event_triggered_response"]["inh"]["latency_error_steps"]),
        ),
        (
            "Inhibitory gain error",
            "|1-gain|",
            lambda r: abs(1.0 - r["event_triggered_response"]["inh"]["response_gain"]),
        ),
    ]
    raw = np.asarray(
        [
            [getter(reports[model]) for _label, _unit, getter in metrics]
            for model in models
        ],
        dtype=float,
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.25))
    colors = [
        OKABE_ITO["orange"],
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
    ]
    x = np.arange(len(models))
    for metric_index, ax in enumerate(axes.flat):
        label, unit, _getter = metrics[metric_index]
        y = raw[:, metric_index]
        ax.plot(x, y, color="#B8C0CC", linewidth=0.9, zorder=1)
        for model_index, value in enumerate(y):
            ax.scatter(
                model_index,
                value,
                s=24,
                color=colors[model_index],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.text(
                model_index,
                value,
                _format_value(value),
                ha="center",
                va="bottom",
                fontsize=5.5,
            )
        ax.set_xticks(
            x,
            labels=[_model_short(model).replace("branch_", "") for model in models],
            rotation=18,
        )
        ax.set_ylabel(unit)
        ax.text(
            0.02, 0.94, label, transform=ax.transAxes, ha="left", va="top", fontsize=7.2
        )
        ax.margins(y=0.28)
        ax.grid(True, axis="y", alpha=0.32)
    fig.tight_layout()
    _save_figure(fig, output, formats)
    return output.name


def _plot_target_view_distortion_map(
    canonical: dict[str, Any], output: Path, formats: tuple[str, ...]
) -> str:
    model_rows = canonical["validation"]["models"]
    models = [_model_short(row["model_id"]) for row in model_rows]
    y = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharey=True)
    panels = [
        (
            axes[0],
            "valid timesteps",
            "rmse_to_clipped_valid_mv",
            "rmse_to_raw_valid_mv",
        ),
        (
            axes[1],
            "action-potential neighborhood",
            "rmse_to_clipped_near_spike_mv",
            "rmse_to_raw_near_spike_mv",
        ),
    ]
    for ax, label, clipped_key, raw_key in panels:
        clipped = np.asarray(
            [float(row[clipped_key]) for row in model_rows], dtype=float
        )
        raw = np.asarray([float(row[raw_key]) for row in model_rows], dtype=float)
        for yy, clipped_value, raw_value in zip(y, clipped, raw):
            ax.plot(
                [clipped_value, raw_value],
                [yy, yy],
                color="#B8C0CC",
                linewidth=1.0,
                zorder=1,
            )
        ax.scatter(
            clipped,
            y,
            s=26,
            facecolor="white",
            edgecolor=OKABE_ITO["blue"],
            linewidth=0.9,
            label="clipped target",
            zorder=3,
        )
        ax.scatter(
            raw,
            y,
            s=30,
            color=OKABE_ITO["red"],
            edgecolor="white",
            linewidth=0.5,
            label="raw target",
            zorder=4,
        )
        if raw_key.endswith("near_spike_mv"):
            for yy, clipped_value, raw_value in zip(y, clipped, raw):
                ratio = raw_value / max(clipped_value, 1e-12)
                ax.text(
                    raw_value * 1.04, yy, f"{ratio:.1f}x", va="center", fontsize=5.8
                )
        ax.set_xscale("log")
        ax.set_xlabel("RMSE (mV)")
        ax.text(
            0.03, 0.92, label, transform=ax.transAxes, ha="left", va="top", fontsize=7.2
        )
        ax.grid(True, axis="x", alpha=0.35)
    axes[0].set_yticks(y, labels=models)
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 1.02), ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save_figure(fig, output, formats)
    return output.name


def _plot_sparse_event_sensitivity_profiles(
    sparse: dict[str, Any], output: Path, formats: tuple[str, ...]
) -> str:
    ratios = sparse["ratios"]
    budgets = [8, 32, 128]
    edit_types = [
        ("global", "global"),
        ("8 ms", "time_local_8"),
        ("32 ms", "time_local_32"),
    ]
    groups = ["quiet_or_low_spike", "spike_rich"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharex=True)
    colors = {
        "global": OKABE_ITO["blue"],
        "8 ms": OKABE_ITO["orange"],
        "32 ms": OKABE_ITO["green"],
    }
    markers = {"quiet_or_low_spike": "o", "spike_rich": "s"}
    for group in groups:
        for short, needle in edit_types:
            selected = [
                row
                for row in ratios
                if row["window_group"] == group and needle in row["targeted"]
            ]
            selected = sorted(selected, key=lambda row: int(row["budget"]))
            x = np.asarray([int(row["budget"]) for row in selected], dtype=float)
            ratio = np.asarray(
                [float(row["median_soma_delta_ratio"]) for row in selected], dtype=float
            )
            changed = np.asarray(
                [
                    100.0 * float(row["changed_fraction_of_events_median"])
                    for row in selected
                ],
                dtype=float,
            )
            label = f"{short}, {'spike-rich' if group == 'spike_rich' else 'quiet'}"
            axes[0].plot(
                x,
                ratio,
                marker=markers[group],
                markersize=3.2,
                linewidth=1.05,
                color=colors[short],
                alpha=0.86,
                label=label,
            )
            axes[1].plot(
                x,
                changed,
                marker=markers[group],
                markersize=3.2,
                linewidth=1.05,
                color=colors[short],
                alpha=0.86,
            )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, labels=[str(item) for item in budgets])
        ax.set_xlabel("edited events")
        ax.grid(True, alpha=0.34)
    axes[0].set_ylabel("targeted/random soma-delta ratio")
    axes[1].set_ylabel("edited existing events (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=5.6,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save_figure(fig, output, formats)
    return output.name


def _plot_morphology_intervention_response(
    rows: list[dict[str, Any]], output: Path, formats: tuple[str, ...]
) -> str:
    if not rows:
        _plot_morphology_axis(output, formats)
        return output.name
    metric = (
        "intervention_response.spike_count_delta_teacher_mean.mean_across_morphologies"
    )
    source_rows = [
        row
        for row in rows
        if row.get("metric") == metric
        and isinstance(row.get("morphology_values"), dict)
    ]
    if not source_rows:
        _plot_morphology_axis(output, formats)
        return output.name
    interventions = [
        "event_dropout",
        "exc_dropout",
        "inh_dropout",
        "site_silence",
        "temporal_jitter",
    ]
    morphologies = sorted({m for row in source_rows for m in row["morphology_values"]})
    matrix = np.full((len(morphologies), len(interventions)), np.nan)
    for row in source_rows:
        intervention = str(row.get("intervention"))
        if intervention not in interventions:
            continue
        col = interventions.index(intervention)
        for morph, value in row["morphology_values"].items():
            matrix[morphologies.index(morph), col] = float(value)
    fig, ax = plt.subplots(figsize=(7.4, 3.35))
    x = np.arange(len(interventions))
    for row_index, morph in enumerate(morphologies):
        jitter = (row_index - (len(morphologies) - 1) / 2.0) * 0.035
        ax.plot(
            x + jitter,
            matrix[row_index],
            color="#B8C0CC",
            linewidth=0.65,
            alpha=0.72,
            zorder=1,
        )
        ax.scatter(
            x + jitter,
            matrix[row_index],
            s=17,
            color=OKABE_ITO["gray"],
            alpha=0.75,
            zorder=2,
        )
    mean = np.nanmean(matrix, axis=0)
    low = np.nanmin(matrix, axis=0)
    high = np.nanmax(matrix, axis=0)
    ax.vlines(
        x, low, high, color=OKABE_ITO["blue"], linewidth=2.0, alpha=0.85, zorder=3
    )
    ax.scatter(
        x,
        mean,
        s=38,
        color=OKABE_ITO["orange"],
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
        label="mean",
    )
    for xx, yy, lo, hi in zip(x, mean, low, high):
        ax.text(
            xx,
            hi + 1.5,
            f"{lo:+.0f} to {hi:+.0f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )
        ax.text(
            xx, yy, f"{yy:+.1f}", ha="center", va="center", fontsize=5.6, color="white"
        )
    ax.axhline(0.0, color="#222222", linewidth=0.7, alpha=0.55)
    ax.set_xticks(x, labels=[item.replace("_", "\n") for item in interventions])
    ax.set_xlabel("intervention")
    ax.set_ylabel("teacher spike-count delta")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.34)
    fig.tight_layout()
    _save_figure(fig, output, formats)
    return output.name


def _plot_tcn_preprocessing_trajectories(
    tcn_summaries: list[dict[str, Any]], output: Path, formats: tuple[str, ...]
) -> str:
    view_order = [
        ("raw_tcn", "raw"),
        ("same_slice_target_mean_baseline", "target\nmean"),
        ("same_slice_mean_std_match", "mean/std\nmatched"),
        ("same_slice_linear_fit", "linear\nfit"),
    ]
    fig, ax = plt.subplots(figsize=(7.25, 3.35))
    colors = [
        OKABE_ITO["red"],
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
    ]
    for idx, blob in enumerate(sorted(tcn_summaries, key=lambda item: item["model"])):
        values = {row["view"]: row["rmse_mv"] for row in blob["rows"]}
        y = [values[key] for key, _label in view_order]
        ax.plot(
            np.arange(len(view_order)),
            y,
            marker="o",
            linewidth=1.5,
            color=colors[idx % len(colors)],
            label=blob["model"].replace("tcn_", ""),
        )
        for x, val in enumerate(y):
            ax.text(
                x,
                val * (1.05 if x == 0 else 0.93),
                _format_value(val),
                fontsize=5.8,
                ha="center",
                va="bottom" if x == 0 else "top",
            )
    ax.set_yscale("log")
    ax.set_xticks(
        np.arange(len(view_order)), labels=[label for _key, label in view_order]
    )
    ax.set_ylabel("RMSE (mV, log scale)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, output, formats)
    return output.name


def _plot_teacher_intervention_contract(output: Path, formats: tuple[str, ...]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 2.85))
    ax.set_axis_off()
    boxes = [
        (0.04, 0.66, "baseline\ndrive", OKABE_ITO["light"]),
        (0.27, 0.73, "teacher\nsimulator", "#E8F4FA"),
        (0.50, 0.73, "teacher\ntrace", "#E8F4FA"),
        (0.27, 0.32, "surrogate\nmodel", "#FFF3D6"),
        (0.50, 0.32, "model\ntrace", "#FFF3D6"),
        (0.74, 0.52, "compare\nteacher/model\ndeltas", "#EAF6EF"),
    ]
    for x, y, label, color in boxes:
        ax.add_patch(
            Rectangle(
                (x, y),
                0.16,
                0.14,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="#AEB7C2",
            )
        )
        ax.text(
            x + 0.08,
            y + 0.07,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7.5,
        )
    ax.text(
        0.04,
        0.38,
        "intervention\ndrive",
        transform=ax.transAxes,
        fontsize=7.5,
        va="center",
    )
    arrows = [
        ((0.20, 0.73), (0.27, 0.80)),
        ((0.43, 0.80), (0.50, 0.80)),
        ((0.20, 0.41), (0.27, 0.39)),
        ((0.43, 0.39), (0.50, 0.39)),
        ((0.66, 0.80), (0.74, 0.63)),
        ((0.66, 0.39), (0.74, 0.56)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1,
                color="#222222",
            )
        )
    ax.text(
        0.04,
        0.12,
        "Model-only stress tests identify surrogate sensitivity; mechanistic claims require the same intervention on the teacher.",
        transform=ax.transAxes,
        fontsize=7.4,
    )
    _save_figure(fig, output, formats)
    return output.name


def _plot_neuronio_matched_intervention_response(
    rows: list[dict[str, Any]],
    output: Path,
    formats: tuple[str, ...],
) -> str:
    table = _intervention_response_table(rows)
    if not table:
        return _plot_teacher_intervention_contract(
            output.with_name("fig07_teacher_intervention_contract"), formats
        )
    interventions = [row["intervention"] for row in table]
    labels = [name.replace("_", " ").replace("-", " ") for name in interventions]
    fig = plt.figure(figsize=(7.4, 4.55))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.15, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.32,
    )
    y = np.arange(len(interventions))

    ax = fig.add_subplot(gs[:, 0])
    spike_delta = np.asarray(
        [float(row["delta_spike_rate_per_1000_steps"]) for row in table], dtype=float
    )
    soma_delta = np.asarray(
        [float(row["delta_soma_mean_mv"]) for row in table], dtype=float
    )
    ax.axvline(0.0, color="#222222", linewidth=0.65, alpha=0.55)
    ax.plot(spike_delta, y - 0.10, color=OKABE_ITO["blue"], linewidth=0.9, alpha=0.85)
    ax.plot(soma_delta, y + 0.10, color=OKABE_ITO["orange"], linewidth=0.9, alpha=0.85)
    ax.scatter(
        spike_delta,
        y - 0.10,
        s=25,
        color=OKABE_ITO["blue"],
        edgecolor="white",
        linewidth=0.5,
        label="spike-rate delta",
    )
    ax.scatter(
        soma_delta,
        y + 0.10,
        s=25,
        color=OKABE_ITO["orange"],
        edgecolor="white",
        linewidth=0.5,
        label="mean soma delta",
    )
    for yy, spike_value, soma_value in zip(y, spike_delta, soma_delta):
        ax.text(
            spike_value,
            yy - 0.27,
            f"{spike_value:+.2f}",
            ha="center",
            va="top",
            fontsize=5.5,
            color=OKABE_ITO["blue"],
        )
        ax.text(
            soma_value,
            yy + 0.27,
            f"{soma_value:+.2f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=OKABE_ITO["orange"],
        )
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("signed teacher response")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.32)

    ax2 = fig.add_subplot(gs[0, 1])
    baseline = np.asarray(
        [float(row["baseline_spike_rate_per_1000_steps"]) for row in table], dtype=float
    )
    intervention = np.asarray(
        [float(row["intervention_spike_rate_per_1000_steps"]) for row in table],
        dtype=float,
    )
    colors = [
        OKABE_ITO["red"] if b > a else OKABE_ITO["blue"]
        for a, b in zip(baseline, intervention)
    ]
    for yy, base, intervened, color in zip(y, baseline, intervention, colors):
        ax2.plot([base, intervened], [yy, yy], color="#A7AEB8", linewidth=1.0, zorder=1)
        ax2.scatter(
            base,
            yy,
            s=26,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=3,
        )
        ax2.scatter(
            intervened,
            yy,
            s=30,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax2.text(intervened + 0.18, yy, f"{intervened:.2f}", va="center", fontsize=5.8)
    ax2.set_yticks(y, labels=[])
    ax2.invert_yaxis()
    ax2.set_xlabel("spike rate (Hz)")
    ax2.scatter([], [], s=26, facecolor="white", edgecolor="#222222", label="baseline")
    ax2.scatter(
        [], [], s=30, color=OKABE_ITO["blue"], edgecolor="white", label="intervention"
    )
    ax2.legend(loc="lower right")
    ax2.set_xlim(0, max(10.0, float(np.max(intervention)) + 1.2))

    ax3 = fig.add_subplot(gs[1, 1])
    abs_soma = np.asarray(
        [float(row["delta_soma_abs_mean_mv"]) for row in table], dtype=float
    )
    event_delta = np.asarray(
        [float(row["event_fraction_delta_pct"]) for row in table], dtype=float
    )
    ax3.scatter(
        abs_soma,
        event_delta,
        s=28,
        color=OKABE_ITO["green"],
        edgecolor="white",
        linewidth=0.6,
    )
    for name, xx, yy in zip(interventions, abs_soma, event_delta):
        if name in {"exc_dropout", "inh_dropout", "morphology-3", "temporal_jitter"}:
            ax3.annotate(
                name.replace("_", " "),
                (xx, yy),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=5.6,
            )
    ax3.axhline(0.0, color="#222222", linewidth=0.65, alpha=0.55)
    ax3.set_xlabel("mean |soma delta| (mV)")
    ax3.set_ylabel("event-fraction delta (pp)")
    ax3.grid(True, alpha=0.32)
    _add_panel_label(ax, "A")
    _add_panel_label(ax2, "B")
    _add_panel_label(ax3, "C")
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.12, top=0.96)
    _save_figure(fig, output, formats)
    return output.name


def _plot_neuronio_intervention_trace_examples(
    root: Path,
    output: Path,
    formats: tuple[str, ...],
) -> str | None:
    examples = [
        ("exc_dropout", "E dropout"),
        ("inh_dropout", "I dropout"),
        ("temporal_jitter", "temporal jitter"),
        ("morphology-3", "morphology 3"),
    ]
    records = []
    for intervention, label in examples:
        record = _load_representative_intervention_trace(
            root / "interventions" / intervention
        )
        if record is not None:
            record["label"] = label
            records.append(record)
    if not records:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.65), sharex=False, sharey=False)
    axes = axes.ravel()
    handles = [
        Line2D([0], [0], color=OKABE_ITO["gray"], linewidth=0.85, label="baseline V"),
        Line2D(
            [0], [0], color=OKABE_ITO["blue"], linewidth=0.95, label="intervention V"
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="|",
            linestyle="None",
            markersize=6,
            label="baseline spike",
        ),
        Line2D(
            [0],
            [0],
            color=OKABE_ITO["red"],
            marker="|",
            linestyle="None",
            markersize=6,
            label="intervention spike",
        ),
    ]
    for index, (ax, record) in enumerate(zip(axes, records)):
        time = np.arange(record["start"], record["stop"], dtype=float)
        baseline = record["baseline"][record["start"] : record["stop"]]
        intervention = record["intervention"][record["start"] : record["stop"]]
        ax.plot(
            time, baseline, color=OKABE_ITO["gray"], linewidth=0.85, label="baseline"
        )
        ax.plot(
            time,
            intervention,
            color=OKABE_ITO["blue"],
            linewidth=0.95,
            label="intervention",
        )
        ax.fill_between(
            time,
            baseline,
            intervention,
            color=OKABE_ITO["blue"],
            alpha=0.12,
            linewidth=0,
        )
        _mark_spikes(
            ax,
            record["baseline_spikes"],
            record["start"],
            record["stop"],
            color="#555555",
            y_frac=0.98,
        )
        _mark_spikes(
            ax,
            record["intervention_spikes"],
            record["start"],
            record["stop"],
            color=OKABE_ITO["red"],
            y_frac=0.92,
        )
        ax.text(
            0.02,
            0.96,
            f"{record['label']}\nΔspikes {record['spike_delta']:+.0f}, mean ΔV {record['mean_delta_mv']:+.2f} mV",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("soma voltage (mV)")
        ax.set_xlim(record["start"], record["stop"] - 1)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.18, top=0.96, wspace=0.24, hspace=0.44
    )
    _save_figure(fig, output, formats)
    return output.name


def _load_representative_intervention_trace(
    folder: Path, *, ignore_start: int = 500
) -> dict[str, Any] | None:
    if not folder.exists():
        return None
    best: dict[str, Any] | None = None
    best_score = -np.inf
    for path in sorted(folder.glob("intervention_pair_*.npz")):
        payload = np.load(path)
        baseline = np.asarray(payload["baseline_soma_mv"], dtype=float)
        intervention = np.asarray(payload["intervention_soma_mv"], dtype=float)
        baseline_spikes = np.asarray(payload["baseline_targets"][..., 0] > 0.5)
        intervention_spikes = np.asarray(payload["intervention_targets"][..., 0] > 0.5)
        sample_ids = (
            [str(item) for item in payload["sample_ids"]]
            if "sample_ids" in payload.files
            else [str(i) for i in range(baseline.shape[0])]
        )
        for idx in range(baseline.shape[0]):
            delta = intervention[idx, ignore_start:] - baseline[idx, ignore_start:]
            spike_delta = float(
                intervention_spikes[idx, ignore_start:].sum()
                - baseline_spikes[idx, ignore_start:].sum()
            )
            mean_delta = float(np.mean(delta))
            max_abs_delta = float(np.max(np.abs(delta)))
            score = abs(spike_delta) + 0.25 * max_abs_delta
            if score > best_score:
                center = int(ignore_start + np.argmax(np.abs(delta)))
                start = max(ignore_start, center - 360)
                stop = min(baseline.shape[1], start + 900)
                start = max(ignore_start, stop - 900)
                best_score = score
                best = {
                    "sample_id": sample_ids[idx],
                    "baseline": baseline[idx],
                    "intervention": intervention[idx],
                    "baseline_spikes": baseline_spikes[idx],
                    "intervention_spikes": intervention_spikes[idx],
                    "start": int(start),
                    "stop": int(stop),
                    "spike_delta": spike_delta,
                    "mean_delta_mv": mean_delta,
                }
    return best


def _mark_spikes(
    ax: plt.Axes,
    spikes: np.ndarray,
    start: int,
    stop: int,
    *,
    color: str,
    y_frac: float,
) -> None:
    spike_times = np.flatnonzero(spikes[start:stop]) + start
    if spike_times.size == 0:
        return
    ymin, ymax = ax.get_ylim()
    y = ymin + y_frac * (ymax - ymin)
    ax.scatter(
        spike_times,
        np.full_like(spike_times, y, dtype=float),
        s=5,
        color=color,
        marker="|",
        linewidths=0.75,
        zorder=4,
    )


def _plot_morphology_axis(output: Path, formats: tuple[str, ...]) -> None:
    morphs = [
        ("488683425", "reference", 99, 1.000, 1.000),
        ("487664663", "held-out", 98, 0.930, 0.946),
        ("479770916", "line shift", 110, 1.116, 1.037),
        ("589427435", "line shift", 120, 1.187, 1.033),
        ("488679042", "length shift", 132, 1.325, 1.095),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    routed = [m[4] for m in morphs]
    sizes = [90 + 260 * (r - min(routed)) / (max(routed) - min(routed)) for r in routed]
    ax.scatter(
        [m[2] for m in morphs],
        [m[3] for m in morphs],
        s=sizes,
        color=OKABE_ITO["blue"],
        edgecolor="white",
        linewidth=0.8,
    )
    for specimen, role, branches, length_ratio, _routed in morphs:
        ax.text(
            branches + 1.2,
            length_ratio,
            f"{specimen}\n{role}",
            fontsize=6.2,
            va="center",
        )
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("extracted branch count")
    ax.set_ylabel("total-length ratio vs reference")
    _save_figure(fig, output, formats)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _write_point_artifact_table(
    output: Path,
    *,
    intervention_evidence: bool = False,
    trace_evidence: bool = False,
) -> str:
    point_10_artifact = (
        "fig07_neuronio_matched_intervention_response + fig08_neuronio_matched_trace_examples + table06"
        if trace_evidence
        else "fig07_neuronio_matched_intervention_response + table06"
        if intervention_evidence
        else "fig07_teacher_intervention_contract + table05"
    )
    point_10_role = (
        "Measured NeuronIO-source teacher deltas and trace examples for paired interventions"
        if trace_evidence
        else "Measured NeuronIO-source teacher deltas for paired interventions"
        if intervention_evidence
        else "Teacher-delta contract and group-level evidence table"
    )
    rows = [
        (
            "1-2",
            "metric_decoupling",
            "fig01_spike_metric_identification",
            "AUC/F1 scatter plus prevalence-matching inset",
        ),
        (
            "3",
            "mechanistic_response",
            "fig02_mechanistic_behavior_profiles",
            "Small-multiple behavior profiles over phase, history, latency, and gain",
        ),
        (
            "4-5",
            "target_view_audit",
            "fig03_target_view_distortion_profile + table03",
            "Clipped/raw RMSE dot profiles plus exact clipping-distortion table",
        ),
        (
            "6",
            "metric_decoupling",
            "table02_metric_decoupling",
            "Zero-threshold spike counts kept as a table, where exact counts matter more than bars",
        ),
        (
            "7",
            "robustness_directionality",
            "fig04_sparse_event_sensitivity_profiles",
            "Budget/locality response-amplification profiles plus edited-event fraction",
        ),
        (
            "8",
            "morphology_contrast",
            "fig05_morphology_intervention_response",
            "Morphology-by-intervention teacher-response ranges",
        ),
        (
            "9",
            "adapter_fairness",
            "fig06_tcn_preprocessing_trajectories + table03",
            "TCN RMSE trajectories across target-dependent preprocessing views",
        ),
        ("10", "protocol_coverage", point_10_artifact, point_10_role),
    ]
    _write_markdown_table(output, ("point", "diagnostic", "artifact", "role"), rows)
    return output.name


def _write_metric_decoupling_table(
    point_metrics: dict[tuple[int, str], Any], canonical: dict[str, Any], output: Path
) -> str:
    points = json.loads(str(point_metrics[(1, "points")]))
    point_rows = []
    for row in points:
        point_rows.append(
            (
                row["model_id"].replace("branch_elm_", ""),
                row["regime"].replace("_", " "),
                f"{row['spike_auc']:.4f}",
                f"{row['event_f1_2ms']:.3f}",
            )
        )
    point_rows.sort(key=lambda item: (item[1], item[0]))
    target = canonical["validation"]["target"]["target_spike_count"]
    zero = [
        (
            _model_short(row["model_id"]),
            f"{row['threshold_logit_zero']['predicted_spike_count']:.0f}",
            f"{target:.0f}",
            f"{row['threshold_logit_zero']['predicted_spike_count'] / max(target, 1):.4f}",
        )
        for row in canonical["validation"]["models"]
    ]
    lines = ["# Metric Decoupling Table", "", "## AUC Versus Event Timing", ""]
    lines.append(_markdown_table_text(("model", "regime", "AUC", "F1@2ms"), point_rows))
    lines.extend(["", "## Zero-Threshold Event Deployment", ""])
    lines.append(
        _markdown_table_text(
            ("model", "predicted spikes", "teacher spikes", "predicted/teacher"), zero
        )
    )
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output.name


def _write_target_protocol_table(
    canonical: dict[str, Any], tcn_summaries: list[dict[str, Any]], output: Path
) -> str:
    target = canonical["validation"]["target"]["target_distortion"]
    clipping_rows = [
        (
            "global clipped fraction",
            f"{100 * target['clipped_fraction_valid']:.3f}%",
            "all evaluated timesteps",
        ),
        (
            "near-spike clipped fraction",
            f"{100 * target['near_spike_clipped_fraction']:.2f}%",
            "AP-neighborhood timesteps",
        ),
        (
            "near-spike distortion RMSE",
            f"{target['rmse_mv_near_spike']:.2f} mV",
            "raw soma vs clipped target",
        ),
        (
            "max removed voltage",
            f"{target['max_abs_distortion_mv']:.2f} mV",
            "raw soma above the -55 mV cap",
        ),
    ]
    tcn_rows = []
    for blob in sorted(tcn_summaries, key=lambda item: item["model"]):
        values = {row["view"]: row["rmse_mv"] for row in blob["rows"]}
        tcn_rows.append(
            (
                blob["model"].replace("tcn_", ""),
                f"{values['raw_tcn']:.2f}",
                f"{values['same_slice_target_mean_baseline']:.2f}",
                f"{values['same_slice_mean_std_match']:.2f}",
                f"{values['same_slice_linear_fit']:.2f}",
            )
        )
    lines = ["# Target And Adapter Protocol Table", "", "## Clipping Distortion", ""]
    lines.append(_markdown_table_text(("quantity", "value", "scope"), clipping_rows))
    lines.extend(["", "## TCN Voltage RMSE By Evaluation View", ""])
    lines.append(
        _markdown_table_text(
            ("model", "raw", "target mean", "mean/std match", "linear fit"), tcn_rows
        )
    )
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output.name


def _write_morphology_intervention_table(
    rows: list[dict[str, Any]], output: Path
) -> str:
    metric = (
        "intervention_response.spike_count_delta_teacher_mean.mean_across_morphologies"
    )
    selected = [
        row
        for row in rows
        if row.get("metric") == metric
        and isinstance(row.get("morphology_values"), dict)
    ]
    table_rows = []
    for row in sorted(selected, key=lambda item: str(item.get("intervention"))):
        values = [float(v) for v in row["morphology_values"].values()]
        table_rows.append(
            (
                str(row.get("intervention")),
                f"{np.mean(values):+.2f}",
                f"{np.min(values):+.0f}",
                f"{np.max(values):+.0f}",
                f"{np.ptp(values):.0f}",
            )
        )
    if not table_rows:
        table_rows = [("not available", "", "", "", "")]
    _write_markdown_table(
        output,
        (
            "intervention",
            "mean delta (spikes/trace)",
            "min delta",
            "max delta",
            "range",
        ),
        table_rows,
    )
    return output.name


def _write_neuronio_intervention_table(rows: list[dict[str, Any]], output: Path) -> str:
    table = _intervention_response_table(rows)
    table_rows = []
    for row in table:
        table_rows.append(
            (
                row["intervention"],
                f"{row['event_fraction_delta_pct']:+.2f}",
                f"{row['baseline_spike_rate_per_1000_steps']:.2f}",
                f"{row['intervention_spike_rate_per_1000_steps']:.2f}",
                f"{row['delta_spike_rate_per_1000_steps']:+.2f}",
                f"{row['delta_soma_mean_mv']:+.2f}",
                f"{row['delta_soma_abs_mean_mv']:.2f}",
                row["teacher_direction"],
            )
        )
    lines = ["# NeuronIO-Matched Intervention Table", ""]
    lines.append(
        _markdown_table_text(
            (
                "intervention",
                "event Δ (pp)",
                "baseline Hz",
                "intervention Hz",
                "spike-rate Δ",
                "mean soma Δ mV",
                "mean |soma Δ| mV",
                "teacher direction",
            ),
            table_rows,
        )
    )
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output.name


def _write_group_evidence_table(
    output: Path,
    *,
    intervention_evidence: bool = False,
    trace_evidence: bool = False,
) -> str:
    diagnostic_evidence = (
        "Model-only stress rows plus measured NeuronIO-source paired teacher interventions and trace windows"
        if trace_evidence
        else "Model-only stress rows plus measured NeuronIO-source paired teacher interventions"
        if intervention_evidence
        else "Model-only stress rows identify sensitivity; mechanistic claims need teacher deltas"
    )
    diagnostic_read = (
        "The same intervention protocol can be evaluated as a teacher-delta diagnostic instead of a model-only perturbation."
        if intervention_evidence
        else "NeuronIO-matched intervention generation would strengthen this group directly."
    )
    rows = [
        (
            "Metrics",
            "Need better headline metrics",
            "AUC/F1 decoupling; prevalence rescue failure; behavior profiles; clipped/raw target profile; TCN preprocessing trajectories",
            "AUC and RMSE significantly inflate simulator fidelity in the evaluated regimes.",
        ),
        (
            "Data",
            "Need richer training/evaluation data",
            "Clipped target audit; morphology/intervention response ranges; raw target-view requirement",
            "NeuronIO remains useful for comparability but is too narrow as a simulator-replacement benchmark.",
        ),
        (
            "Model behavior",
            "Existing official models fall short practically",
            "Zero-threshold counts; sparse-event fragility; phase/history/latency/gain errors",
            "Claim is scoped to evaluated official checkpoints and measured operating rules.",
        ),
        (
            "Diagnostics",
            "Need teacher-grounded intervention diagnostics",
            diagnostic_evidence,
            diagnostic_read,
        ),
    ]
    _write_markdown_table(
        output, ("group", "claim", "evidence", "conservative read"), rows
    )
    return output.name


def _intervention_response_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row.get("diagnostic") != "intervention_response":
            continue
        intervention = row.get("intervention")
        metric = row.get("metric")
        if not intervention or not metric:
            continue
        grouped[str(intervention)][str(metric)] = row.get("value")
    preferred = [
        "event_dropout",
        "exc_dropout",
        "inh_dropout",
        "site_silence",
        "temporal_jitter",
        "morphology-1",
        "morphology-2",
        "morphology-3",
        "morphology-4",
        "morphology-5",
    ]
    ordered = [name for name in preferred if name in grouped] + sorted(
        set(grouped) - set(preferred)
    )
    out = []
    for intervention in ordered:
        values = grouped[intervention]
        required = (
            "event_fraction_delta",
            "baseline_spike_rate_per_1000_steps",
            "intervention_spike_rate_per_1000_steps",
            "delta_spike_rate_per_1000_steps",
            "delta_soma_mean_mv",
            "delta_soma_abs_mean_mv",
            "teacher_direction",
        )
        if any(key not in values for key in required):
            continue
        out.append(
            {
                "intervention": intervention,
                "event_fraction_delta_pct": 100.0
                * float(values["event_fraction_delta"]),
                "baseline_spike_rate_per_1000_steps": float(
                    values["baseline_spike_rate_per_1000_steps"]
                ),
                "intervention_spike_rate_per_1000_steps": float(
                    values["intervention_spike_rate_per_1000_steps"]
                ),
                "delta_spike_rate_per_1000_steps": float(
                    values["delta_spike_rate_per_1000_steps"]
                ),
                "delta_soma_mean_mv": float(values["delta_soma_mean_mv"]),
                "delta_soma_abs_mean_mv": float(values["delta_soma_abs_mean_mv"]),
                "teacher_direction": str(values["teacher_direction"]),
            }
        )
    return out


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _point_metrics(path: str | Path) -> dict[tuple[int, str], Any]:
    out: dict[tuple[int, str], Any] = {}
    for row in _load_jsonl(path):
        point_id = row.get("point_id")
        metric = row.get("metric")
        if isinstance(point_id, int) and isinstance(metric, str):
            out[(point_id, metric)] = row.get("value")
    return out


def _save_figure(fig: plt.Figure, output: Path, formats: tuple[str, ...]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        fig.savefig(output.with_suffix(f".{suffix}"))
    plt.close(fig)


def _write_markdown_table(
    output: Path, headers: tuple[str, ...], rows: list[tuple[Any, ...]]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown_table_text(headers, rows) + "\n", encoding="utf-8")


def _markdown_table_text(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _model_short(model_id: str) -> str:
    return str(model_id).replace("branch_elm_", "")
