import json
import shutil
import subprocess

import numpy as np
import pytest

from axobench.benchmark.error_anatomy import _steps_since_last, error_anatomy, reweight_to_rate_distribution
from axobench.benchmark.iteration_evaluator import EvaluationConfig, evaluate_predictions
from axobench.benchmark.plots import plot_error_anatomy
from axobench.cli import main

CONFIG = EvaluationConfig(ignore_start=0, calibration_fraction=0.25)
WAVEFORM = (20.0, -10.0, -75.0, -72.0)  # teacher voltage at lags 0..3 after a spike


def _teacher(spike_counts, steps=1000):
    """-70 mV traces with a 2 Hz, 1 mV ripple and evenly spaced spikes of a fixed waveform.

    Ids name two morphologies.
    """
    raw = np.tile(-70.0 + np.sin(2 * np.pi * 2 * np.arange(steps) / 1000.0), (len(spike_counts), 1))
    spikes = np.zeros_like(raw)
    for trace, count in enumerate(spike_counts):
        for t in np.linspace(100, steps - 100, count, dtype=int) if count else []:
            spikes[trace, t] = 1.0
            raw[trace, t : t + len(WAVEFORM)] = WAVEFORM
    target = np.stack([spikes, (np.minimum(raw, -55.0) + 67.7) * 0.1], axis=-1)
    ids = [f"cell{trace % 2}:trace{trace}" for trace in range(len(spike_counts))]
    return target, raw, ids


def _prediction(target, voltage_mv):
    prediction = target.copy()
    prediction[..., 1] = (voltage_mv + 67.7) * 0.1
    return prediction


def test_perfect_prediction_has_unit_skill_and_teacher_spike_triggered_average():
    target, raw, ids = _teacher([4, 4, 4, 4])
    anatomy = error_anatomy(_prediction(target, raw), target, raw_soma_mv=raw, trace_ids=ids, config=CONFIG)
    json.dumps(anatomy, allow_nan=False)
    overall = anatomy["overall"]
    assert overall["voltage_sera_skill"] == pytest.approx(1.0)
    assert overall["voltage_sera_skill_of_ceiling"] == pytest.approx(1.0 / overall["voltage_sera_ceiling_skill"])
    assert overall["subthreshold_skill"] == pytest.approx(1.0)
    assert overall["mean_f1"] == pytest.approx(1.0)
    events = anatomy["spike_events"]
    assert (events["teacher_spikes"], events["hit"], events["missed"], events["false"]) == (12, 12, 0, 0)
    assert events["timing_error"]["matched_events"][len(events["timing_error"]["offset_steps"]) // 2] == 12
    sta = anatomy["spike_triggered_average"]
    lags = np.asarray(sta["lags_steps"])
    sta_lags = (lags >= 0) & (lags < len(WAVEFORM))
    assert sta["hit"]["events"] == 12
    assert np.allclose(np.asarray(sta["hit"]["teacher_mv"])[sta_lags], WAVEFORM)
    assert np.allclose(sta["hit"]["model_mv"], sta["hit"]["teacher_mv"])
    assert sta["missed"]["events"] == sta["false"]["events"] == 0
    # The evaluator reports the same skills.
    report = evaluate_predictions(_prediction(target, raw), target, raw_soma_mv=raw, trace_ids=ids, config=CONFIG)
    metrics = {row["id"]: row["value"] for row in report["metrics"]}
    assert metrics["voltage_sera_ceiling_skill"] == pytest.approx(overall["voltage_sera_ceiling_skill"])


def test_missed_spikes_split_spike_triggered_average_and_constant_prediction_has_zero_skill():
    target, raw, ids = _teacher([4, 4, 4, 4])
    reference = float(np.minimum(raw[:1], -55.0).mean())  # calibration trace's clipped mean
    prediction = _prediction(target, np.full_like(raw, reference))
    prediction[..., 0] = 0.0  # flat score: one event at each trace start, every teacher spike missed
    anatomy = error_anatomy(prediction, target, raw_soma_mv=raw, trace_ids=ids, config=CONFIG)
    assert anatomy["overall"]["voltage_sera_skill"] == pytest.approx(0.0, abs=1e-12)
    events = anatomy["spike_events"]
    assert (events["hit"], events["missed"]) == (0, 12)
    missed = anatomy["spike_triggered_average"]["missed"]
    assert missed["events"] == 12
    assert np.allclose(missed["model_mv"], reference)
    assert missed["teacher_peak_mv"] == pytest.approx(20.0)


def test_subthreshold_skill_by_band_isolates_a_sinusoidal_error():
    steps = 1000
    t = np.arange(steps) / 1000.0
    slow, fast = 3.0 * np.sin(2 * np.pi * 3 * t), 2.0 * np.sin(2 * np.pi * 30 * t)
    raw = np.tile(-70.0 + slow + fast, (4, 1))
    target = np.stack([np.zeros_like(raw), (raw + 67.7) * 0.1], axis=-1)
    prediction = _prediction(target, raw - 0.5 * fast)  # error: half the 30 Hz component
    anatomy = error_anatomy(prediction, target, raw_soma_mv=raw, config=CONFIG)
    bands = {row["band_hz"]: row for row in anatomy["subthreshold_by_frequency"]}
    assert bands["1-5"]["skill"] == pytest.approx(1.0)
    assert bands["25-50"]["skill"] == pytest.approx(0.5)
    assert bands["25-50"]["error_share"] == pytest.approx(1.0)
    assert bands["1-5"]["teacher_share"] == pytest.approx(9.0 / 13.0)
    assert bands["0-1"]["teacher_share"] == pytest.approx(0.0, abs=1e-12)
    assert anatomy["overall"]["subthreshold_skill"] == pytest.approx(1.0 - 0.5 * 2.0 / np.sqrt(13.0))


def test_rate_bins_assign_traces_and_reweighting_reproduces_bins():
    # 1 s traces: 0, 2, 4 and 8 Hz scored traces after one calibration trace.
    target, raw, ids = _teacher([2, 0, 2, 4, 8, 8])
    prediction = _prediction(target, raw + 1.0)
    anatomy = error_anatomy(prediction, target, raw_soma_mv=raw, trace_ids=ids, config=EvaluationConfig(ignore_start=0, calibration_fraction=0.2))
    rows = {row["bin"]: row for row in anatomy["by_teacher_rate"]}
    assert [rows[b]["traces"] for b in ("0-1", "1-3", "3-6", "6+")] == [1, 1, 1, 2]
    assert rows["6+"]["mean_rate_hz"] == pytest.approx(8.0)
    same = reweight_to_rate_distribution(anatomy, anatomy)
    assert same["voltage_sera_skill"] == pytest.approx(anatomy["overall"]["voltage_sera_skill"])
    assert same["subthreshold_skill"] == pytest.approx(anatomy["overall"]["subthreshold_skill"])
    only_fast = reweight_to_rate_distribution(anatomy, {"6+": 1.0})
    assert only_fast["voltage_sera_skill"] == pytest.approx(rows["6+"]["voltage_sera_skill"])
    assert set(anatomy["by_morphology"]) == {"cell0", "cell1"}


def test_spike_distance_bins_partition_retained_steps():
    spikes = np.zeros((1, 8), dtype=bool)
    spikes[0, 3] = True
    assert _steps_since_last(spikes).tolist() == [[np.inf] * 3 + [0, 1, 2, 3, 4]]
    target, raw, ids = _teacher([4, 4, 4, 4])
    anatomy = error_anatomy(_prediction(target, raw + 1.0), target, raw_soma_mv=raw, trace_ids=ids, config=CONFIG)
    for rows in anatomy["subthreshold_by_spike_distance"].values():
        assert sum(row["point_share"] for row in rows) == pytest.approx(1.0)
        assert sum(row["error_share"] for row in rows) == pytest.approx(1.0)


def test_error_anatomy_cli_writes_json_figure_and_pgfplots(tmp_path, capsys):
    target, raw, ids = _teacher([4, 4, 4, 4])
    prediction = _prediction(target, raw + 1.0)
    np.savez(tmp_path / "cache.npz", prediction=prediction)
    np.savez(tmp_path / "targets.npz", target=target, raw_soma_mv=raw, trace_ids=np.asarray(ids))
    code = main(
        ["error-anatomy", "--cache", str(tmp_path / "cache.npz"), "--targets", str(tmp_path / "targets.npz"),
         "--output", str(tmp_path / "out"), "--ignore-start", "0", "--formats", "png"]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    anatomy = json.loads((tmp_path / "out" / "error-anatomy.json").read_text())
    assert summary["voltage_sera_skill"] == anatomy["overall"]["voltage_sera_skill"]
    assert (tmp_path / "out" / "error-anatomy.png").exists()
    tex = (tmp_path / "out" / "error-anatomy.tex").read_text()
    assert r"\begin{groupplot}" in tex and tex.count(r"\nextgroupplot") == 4
    assert main(["error-anatomy", "--cache", str(tmp_path / "cache.npz"), "--output", str(tmp_path / "x")]) == 2


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="needs the tectonic TeX engine")
def test_error_anatomy_pgfplots_compiles(tmp_path):
    target, raw, ids = _teacher([4, 4, 4, 4])
    anatomy = error_anatomy(_prediction(target, raw + 1.0), target, raw_soma_mv=raw, trace_ids=ids, config=CONFIG)
    paths = plot_error_anatomy(anatomy, tmp_path / "anatomy", formats=("pdf",))
    tex = next(path for path in paths if path.endswith(".tex"))
    (tmp_path / "anatomy.pdf").unlink()
    subprocess.run(["tectonic", tex], cwd=tmp_path, check=True, capture_output=True, timeout=300)
    assert (tmp_path / "anatomy.pdf").stat().st_size > 0
