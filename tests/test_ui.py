from contextlib import contextmanager
import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import torch
from transformers import BatchEncoding

import steering
import ui


@pytest.fixture
def client(monkeypatch, tmp_path):
    engine = ui.Engine()
    engine.model_dir = tmp_path
    monkeypatch.setattr(ui, "engine", engine)
    with TestClient(ui.app) as client:
        yield client


def test_interface_and_missing_weights(client):
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/api/status").json()["weights_present"] is False
    response = client.post("/api/load", json={"model_dir": str(ui.engine.model_dir)})
    assert response.status_code == 400
    assert not ui.engine.lock.locked()
    assert ui.engine.phase == "idle"


def test_unloaded_chat_and_invalid_doses(client):
    body = {"messages": [{"role": "user", "content": "Hello"}]}
    assert client.post("/api/chat", json=body).status_code == 409
    assert client.post("/api/chat", json={**body, "doses": [100, -1]}).status_code == 422
    assert client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "Hello"}]}).status_code == 422


def test_external_origin_rejected(client):
    response = client.post("/api/load", json={}, headers={"Origin": "https://example.com"})
    assert response.status_code == 403


@pytest.mark.parametrize("fail", [False, True])
def test_streaming_releases_model_and_hooks(client, monkeypatch, fail):
    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "hello"

        def __call__(self, prompt, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})

    calls = []

    @contextmanager
    def fake_steer(layers, targets, norms):
        calls.append([(t.layer, t.neuron, t.dose) for t in targets])
        try:
            yield
        finally:
            calls.append("removed")

    def generate(**kwargs):
        kwargs["streamer"].on_finalized_text("Hello there.")
        if fail:
            raise RuntimeError("generation failed")

    monkeypatch.setattr(steering, "steer", fake_steer)
    ui.engine.tokenizer = Tokenizer()
    ui.engine.model = SimpleNamespace(device="cpu", model=SimpleNamespace(layers=[]), generate=generate)
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Hello"}], "doses": [0.5, -0.7]})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0] == {"text": "Hello there."}
    assert events[-1]["done"]
    if fail:
        assert events[1] == {"error": "generation failed"}
    assert calls == [[(14, 8949, 0.5), (13, 7956, -0.7)], "removed"]
    assert not ui.engine.lock.locked()
    assert ui.engine.phase == "idle"


def test_stop_and_overlapping_requests(client):
    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "hello"

        def __call__(self, prompt, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})

    started = threading.Event()

    def generate(**kwargs):
        started.set()
        assert ui.engine.cancel.wait(timeout=3)
        assert kwargs["stopping_criteria"][0](None, None)

    ui.engine.tokenizer = Tokenizer()
    ui.engine.model = SimpleNamespace(device="cpu", generate=generate)
    body = {"messages": [{"role": "user", "content": "Hello"}], "enabled": False}
    queue, _ = ui.engine.start(ui.ChatRequest(**body))
    assert started.wait(timeout=3)
    assert client.post("/api/chat", json=body).status_code == 409
    assert client.post("/api/stop").status_code == 200
    assert queue.get(timeout=3) == {"done": True, "stopped": True}
    assert not ui.engine.lock.locked()
