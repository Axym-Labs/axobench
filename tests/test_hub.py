import json
from pathlib import Path

import huggingface_hub
import pytest

from axobench import hub
from axobench.benchmark.dataset_schema import INTERVENTION_CONDITIONS
from axobench.benchmark.iteration_evaluator import _resolve_dataset_root
from axobench.benchmark.selectors import V1DatasetLayout
from axobench.cli import main


class FakeHub:
    """Stand-in for snapshot_download that writes one manifest and shard per requested folder."""

    def __init__(self, *, write_shards: bool = True) -> None:
        self.calls: list[dict] = []
        self.write_shards = write_shards

    def __call__(self, *, repo_id, repo_type, revision, local_dir, allow_patterns, max_workers):
        self.calls.append({"repo_id": repo_id, "local_dir": Path(local_dir), "allow_patterns": list(allow_patterns)})
        root = Path(local_dir)
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        for pattern in allow_patterns:
            if not pattern.endswith("/*"):
                continue
            folder = root / pattern[:-2]
            folder.mkdir(parents=True, exist_ok=True)
            key = "pair_shards" if repo_id == hub.INTERVENTIONS_REPO_ID else "shards"
            manifest = {key: [{"file": "shard_00000.npz", "count": 1}]}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            if self.write_shards:
                (folder / "shard_00000.npz").write_bytes(b"npz")
        return str(root)


@pytest.fixture
def fake_hub(monkeypatch):
    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return fake


def test_download_dataset_builds_v1_layout(tmp_path, fake_hub):
    root = hub.download_dataset(tmp_path / "v1")

    assert root == tmp_path / "v1"
    assert [call["repo_id"] for call in fake_hub.calls] == [hub.BENCHMARK_REPO_ID, hub.INTERVENTIONS_REPO_ID]
    assert fake_hub.calls[0]["local_dir"] == root
    assert "val/*" in fake_hub.calls[0]["allow_patterns"]
    assert "train/*" not in fake_hub.calls[0]["allow_patterns"]
    assert fake_hub.calls[1]["local_dir"] == root / "interventions"
    assert hub.folder_is_complete(root / "val")
    assert V1DatasetLayout(root).available_interventions() == INTERVENTION_CONDITIONS


def test_download_dataset_skips_complete_folders(tmp_path, fake_hub):
    root = hub.download_dataset(tmp_path, splits=("val", "test"), interventions=None)
    assert len(fake_hub.calls) == 1

    (root / "test" / "shard_00000.npz").unlink()
    hub.download_dataset(root, splits=("val", "test"), interventions=None)

    assert len(fake_hub.calls) == 2
    patterns = fake_hub.calls[1]["allow_patterns"]
    assert "test/*" in patterns and "val/*" not in patterns


def test_download_dataset_rejects_unpublished_split(tmp_path, fake_hub):
    with pytest.raises(ValueError, match="private test split is not published"):
        hub.download_dataset(tmp_path, splits=("private-test",))
    with pytest.raises(ValueError, match="unknown intervention"):
        hub.download_dataset(tmp_path, interventions=("nope",))
    assert fake_hub.calls == []


def test_download_reports_incomplete_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", FakeHub(write_shards=False))

    with pytest.raises(FileNotFoundError, match="incomplete folders"):
        hub.download_dataset(tmp_path, interventions=None)


def test_download_population_dataset(tmp_path, fake_hub):
    root = hub.download_population_dataset(tmp_path / "population", splits=("val",))

    assert fake_hub.calls[0]["repo_id"] == hub.POPULATION_REPO_ID
    assert hub.folder_is_complete(root / "val")


def test_default_data_dir_honors_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(hub.DATA_DIR_ENV, str(tmp_path / "custom"))
    assert hub.default_data_dir() == tmp_path / "custom"

    monkeypatch.delenv(hub.DATA_DIR_ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert hub.default_data_dir() == tmp_path / "xdg" / "axobench"


def test_evaluator_downloads_only_when_root_is_omitted(tmp_path, monkeypatch):
    requests = []

    def fake_download(**kwargs):
        requests.append(kwargs)
        return tmp_path

    monkeypatch.setattr(hub, "download_dataset", fake_download)

    assert _resolve_dataset_root("local", intervention_names=None, max_interventions=8) == Path("local")
    assert requests == []
    assert _resolve_dataset_root(None, intervention_names=None, max_interventions=8) == tmp_path
    _resolve_dataset_root(None, intervention_names=("event_dropout",), max_interventions=8)
    _resolve_dataset_root(None, intervention_names=None, max_interventions=0)
    assert [item["interventions"] for item in requests] == ["all", ("event_dropout",), None]
    assert all(item["splits"] == ("val",) for item in requests)


def test_cli_download_data(tmp_path, fake_hub, capsys):
    exit_code = main(["download-data", "--output", str(tmp_path), "--splits", "val,test", "--interventions", "none"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"dataset_root": str(tmp_path)}
    assert [call["repo_id"] for call in fake_hub.calls] == [hub.BENCHMARK_REPO_ID]

    assert main(["download-data", "--output", str(tmp_path), "--splits", "private-test"]) == 2
