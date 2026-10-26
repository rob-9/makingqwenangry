"""A local browser interface. Start with `uv run python ui.py`."""

import argparse
import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
from queue import Empty, Queue
import shutil
import threading
from types import SimpleNamespace
from typing import Annotated, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

import run


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


Dose = Annotated[float, Field(ge=-2, le=2, allow_inf_nan=False)]


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=64)
    enabled: bool = True
    doses: list[Dose] = Field(default=[1.0, -1.0], min_length=2, max_length=2)
    max_new_tokens: int = Field(default=128, ge=1, le=512)


class LoadRequest(BaseModel):
    model_dir: str = Field(default=str(run.DEFAULT_MODEL_DIR), max_length=4096)


class Engine:
    def __init__(self):
        self.model_dir = run.DEFAULT_MODEL_DIR
        self.model = self.tokenizer = None
        self.norms = {}
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.phase = "idle"

    def status(self):
        index = self.model_dir / "model.safetensors.index.json"
        present = False
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text())["weight_map"].values())
                present = bool(shards) and all((self.model_dir / shard).is_file() for shard in shards)
            except (OSError, KeyError, ValueError):
                pass
        return {"ready": self.model is not None, "phase": self.phase, "weights_present": present,
                "model_dir": str(self.model_dir), "device": str(self.model.device) if self.model else None,
                "free_gib": round(shutil.disk_usage(run.existing_parent(self.model_dir)).free / run.GIB, 1)}

    def load(self, path):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "The model is busy. Try again after it finishes.")
        try:
            path = Path(path).expanduser().resolve()
            if self.model is not None:
                if path != self.model_dir:
                    raise HTTPException(409, "Restart the app to load a different model folder.")
                return self.status()
            self.model_dir = path
            if not self.status()["weights_present"]:
                raise HTTPException(400, "Model weights are missing from this folder. Download them first.")
            self.phase = "loading"
            from steering import calibrate
            tokenizer, model = run.load(SimpleNamespace(model_dir=path, device="auto"))
            text = (run.ROOT / "calibration.txt").read_text()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            norms = calibrate(model.model, inputs, [13, 14])
            self.tokenizer, self.model, self.norms = tokenizer, model, norms
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(500, f"Could not load the model: {error}") from error
        finally:
            self.phase = "idle"
            self.lock.release()
        return self.status()

    def start(self, request):
        if self.model is None:
            raise HTTPException(409, "Load the model before starting a conversation.")
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "A reply is already running. Wait or stop it first.")
        self.cancel = threading.Event()
        cancel = self.cancel
        queue = Queue()
        self.phase = "generating"

        def worker():
            try:
                import torch
                from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer
                from steering import Target, steer

                class Streamer(TextStreamer):
                    def on_finalized_text(self, text, stream_end=False):
                        if text:
                            queue.put({"text": text})

                class Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return cancel.is_set()

                messages = [m.model_dump() for m in request.messages]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=True, enable_thinking=False)
                inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                if inputs.input_ids.shape[1] > 4096:
                    raise ValueError("This conversation is getting long. Start a new chat to continue.")
                targets = [Target(14, 8949, request.doses[0]), Target(13, 7956, request.doses[1])]
                context = steer(self.model.model.layers, targets, self.norms) if request.enabled else nullcontext()
                with torch.inference_mode(), context:
                    self.model.generate(
                        **inputs, do_sample=False, use_cache=True, max_new_tokens=request.max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=Streamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
                        stopping_criteria=StoppingCriteriaList([Stop()]),
                    )
            except Exception as error:
                queue.put({"error": str(error)})
            finally:
                self.phase = "idle"
                self.lock.release()
                queue.put({"done": True, "stopped": cancel.is_set()})

        threading.Thread(target=worker, daemon=True).start()
        return queue, cancel


engine = Engine()
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method == "POST" and origin and urlparse(origin).netloc != request.headers.get("host"):
        return JSONResponse({"detail": "Use the app from its local address."}, status_code=403)
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(run.ROOT / "web" / "index.html")


@app.get("/app.js")
def javascript():
    return FileResponse(run.ROOT / "web" / "app.js", media_type="text/javascript")


@app.get("/style.css")
def stylesheet():
    return FileResponse(run.ROOT / "web" / "style.css", media_type="text/css")


@app.get("/api/status")
def status():
    return engine.status()


@app.post("/api/load")
def load(body: LoadRequest):
    return engine.load(body.model_dir)


@app.post("/api/chat")
async def chat(body: ChatRequest):
    if body.messages[-1].role != "user":
        raise HTTPException(422, "The last message must be from you.")
    queue, cancel = engine.start(body)

    async def events():
        try:
            while True:
                try:
                    event = await asyncio.to_thread(queue.get, True, 0.2)
                except Empty:
                    continue
                yield json.dumps(event) + "\n"
                if event.get("done"):
                    break
        finally:
            cancel.set()

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


@app.post("/api/stop")
def stop():
    engine.cancel.set()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", type=Path, default=run.DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    engine.model_dir = args.model_dir.expanduser().resolve()
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
