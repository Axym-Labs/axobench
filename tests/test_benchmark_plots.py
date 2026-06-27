import numpy as np

from axobench.benchmark.plots import plot_state_embedding_surface, plot_state_space_surface


def test_state_space_surface_writes_model_and_teacher_plots(tmp_path):
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

    model_path = plot_state_space_surface(
        output=tmp_path / "state_space_error",
        targets=targets,
        predictions_by_model={"baseline": prediction},
        raw_soma_mv=raw,
        bins=10,
        min_bin_count=1,
        formats=("svg", "png"),
    )
    teacher_path = plot_state_space_surface(
        output=tmp_path / "state_space_spikes",
        targets=targets,
        raw_soma_mv=raw,
        statistic="teacher_spike_rate",
        bins=10,
        min_bin_count=1,
        formats=("svg",),
    )

    assert model_path.endswith(".svg")
    assert (tmp_path / "state_space_error.svg").exists()
    assert (tmp_path / "state_space_error.png").exists()
    assert teacher_path.endswith(".svg")
    assert (tmp_path / "state_space_spikes.svg").exists()


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
