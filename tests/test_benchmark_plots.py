import numpy as np

from axobench.benchmark.plots import (
    plot_phase_plane_surface,
    plot_state_embedding_surface,
    write_core_metric_set_artifacts,
)


def test_core_metric_artifact_promotes_dynamics_sera(tmp_path):
    summary = {
        "models": [
            {
                "model_id": "branch_elm_m5",
                "n_parameters": 10_000,
                "voltage_sera_iron_per_spike_mv2": 16.0,
                "dynamics_sera_iron_per_spike_mv2_per_ms2": 4.0,
                "response_sera_iron_per_spike_mv2": 9.0,
                "spike_f1_area": 0.4,
                "spike_f1_curve": [
                    {"tolerance_steps": 0, "f1": 0.2},
                    {"tolerance_steps": 1, "f1": 0.4},
                ],
            },
            {
                "model_id": "axosim",
                "n_parameters": 20_000,
                "voltage_sera_iron_per_spike_mv2": 9.0,
                "dynamics_sera_iron_per_spike_mv2_per_ms2": 1.0,
                "response_sera_iron_per_spike_mv2": 4.0,
                "spike_f1_area": 0.7,
                "spike_f1_curve": [
                    {"tolerance_steps": 0, "f1": 0.6},
                    {"tolerance_steps": 1, "f1": 0.8},
                ],
            },
        ]
    }

    artifacts = write_core_metric_set_artifacts(
        summary=summary,
        output_dir=tmp_path,
        formats=("svg",),
    )

    table = (tmp_path / "table01_core_metric_set.md").read_text()
    pareto = (tmp_path / "fig02_sera_model_size_pareto.svg").read_text()
    assert "Dynamics SERA (mV²/ms²/spike)" in table
    assert "Dynamics SERA (mV²/ms²/spike)" in pareto
    assert "| branch_elm_m5 | 10000 | 16 | 4 | 9 |" in table
    assert len(artifacts["figures"]) == 3


def test_phase_plane_surface_writes_model_and_teacher_plots(tmp_path):
    time = np.linspace(0.0, 1.0, 48, dtype=np.float32)
    raw = np.stack(
        [
            -68.0 + 35.0 * np.exp(-((time - 0.35) ** 2) / 0.002),
            -66.0 + 25.0 * np.exp(-((time - 0.62) ** 2) / 0.004),
        ],
        axis=0,
    ).astype(np.float32)
    targets = np.zeros((2, 48, 2), dtype=np.float32)
    targets[..., 1] = (raw + 67.7) * 0.1
    targets[:, [17, 30], 0] = 1.0
    prediction = targets.copy()
    prediction[..., 1] += 0.05

    model_path = plot_phase_plane_surface(
        output=tmp_path / "phase_plane_error",
        targets=targets,
        predictions_by_model={"baseline": prediction},
        raw_soma_mv=raw,
        bins=10,
        min_bin_count=1,
        formats=("svg", "png"),
    )
    teacher_path = plot_phase_plane_surface(
        output=tmp_path / "phase_plane_spikes",
        targets=targets,
        raw_soma_mv=raw,
        statistic="teacher_spike_rate",
        bins=10,
        min_bin_count=1,
        formats=("svg",),
    )

    assert model_path.endswith(".svg")
    assert (tmp_path / "phase_plane_error.svg").exists()
    assert (tmp_path / "phase_plane_error.png").exists()
    assert teacher_path.endswith(".svg")
    assert (tmp_path / "phase_plane_spikes.svg").exists()


def test_state_embedding_surface_writes_pca_plot_without_optional_umap(tmp_path):
    time = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    raw = np.stack(
        [
            -68.0 + 20.0 * np.sin(2 * np.pi * time) + 35.0 * np.exp(-((time - 0.33) ** 2) / 0.002),
            -67.0 + 15.0 * np.cos(2 * np.pi * time) + 25.0 * np.exp(-((time - 0.66) ** 2) / 0.004),
        ],
        axis=0,
    ).astype(np.float32)
    targets = np.zeros((2, 64, 2), dtype=np.float32)
    targets[..., 1] = (raw + 67.7) * 0.1
    targets[:, [21, 42], 0] = 1.0
    prediction = targets.copy()
    prediction[..., 1] += 0.04
    inputs = np.zeros((2, 64, 6), dtype=np.int8)
    inputs[:, 10::7, 0] = 1
    inputs[:, 12::11, 1] = -1

    result = plot_state_embedding_surface(
        output=tmp_path / "embedding_error",
        targets=targets,
        predictions_by_model={"baseline": prediction},
        inputs=inputs,
        raw_soma_mv=raw,
        reducer="pca",
        max_points=100,
        bins=9,
        min_bin_count=1,
        formats=("svg", "png"),
    )

    assert result.endswith(".svg")
    assert (tmp_path / "embedding_error.svg").exists()
    assert (tmp_path / "embedding_error.png").exists()
