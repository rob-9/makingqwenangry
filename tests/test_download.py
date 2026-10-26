from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run


def test_download_refuses_insufficient_disk_before_fetching_weights(monkeypatch, tmp_path):
    import huggingface_hub
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename="model.safetensors", size=10 * run.GIB)])
    api = Mock()
    api.model_info.return_value = info
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    fetch = Mock()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    monkeypatch.setattr(run.shutil, "disk_usage", lambda path: SimpleNamespace(free=5 * run.GIB))
    with pytest.raises(ValueError, match="Need 11.0 GiB"):
        run.download(SimpleNamespace(model_dir=tmp_path / "weights"))
    fetch.assert_not_called()
    assert not (tmp_path / "weights").exists()
