"""Standalone local runner. Run `uv run python run.py --help`."""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen3.5-4B"
GIB = 2**30


def existing_parent(path):
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return path


def device_name(requested):
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def doctor(args):
    import torch
    for name in ("torch", "transformers", "accelerate", "huggingface-hub"):
        print(f"{name}: {importlib.metadata.version(name)}")
    print(f"Device: {device_name(args.device)}")
    print(f"MPS built: {torch.backends.mps.is_built()}, available: {torch.backends.mps.is_available()}")
    print(f"Model directory: {args.model_dir}")
    print(f"Free disk: {shutil.disk_usage(existing_parent(args.model_dir)).free / GIB:.1f} GiB")
    print(f"Download complete: {(args.model_dir / 'steering-download.json').exists()}")


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    # Fetch inference files only, including the original, unquantized weight shards.
    files = [f for f in info.siblings if f.rfilename.endswith((".json", ".safetensors", ".txt", ".jinja"))]
    total = sum(f.size for f in files)
    remaining = sum(f.size for f in files if not (args.model_dir / f.rfilename).is_file()
                    or (args.model_dir / f.rfilename).stat().st_size != f.size)
    free = shutil.disk_usage(existing_parent(args.model_dir)).free
    required = remaining + GIB
    print(f"Model: {MODEL_ID}@{REVISION}\nDownload files: {total / GIB:.2f} GiB\nDestination: {args.model_dir}")
    if remaining and free < required:
        raise ValueError(f"Need {required / GIB:.1f} GiB free (including 1 GiB headroom); have {free / GIB:.1f}. "
                         "Free space or supply --model-dir on another disk.")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.model_dir,
                      allow_patterns=[f.rfilename for f in files], max_workers=2, token=False)
    manifest = {"model": MODEL_ID, "revision": REVISION, "files": {f.rfilename: f.size for f in files}}
    (args.model_dir / "steering-download.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Download complete. Inference uses local files only.")


def load(args):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    if not (args.model_dir / "config.json").exists():
        raise ValueError("Model files are missing. Run `uv run python run.py download` first, "
                         "using the same --model-dir if you choose another location.")
    device = device_name(args.device)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"Loading text model on {device} ({dtype})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    # Transformers can load the text backbone directly from the multimodal checkpoint.
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=dtype, device_map=device,
        attn_implementation="eager",
    ).eval()
    return tokenizer, model


def parse_target(value):
    from steering import Target
    try:
        layer, neuron, dose = value.split(":")
        return Target(int(layer), int(neuron), float(dose))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Use LAYER:NEURON:DOSE, e.g. 14:8949:1") from error


def prepare(args):
    from steering import DEFAULT_TARGETS, calibrate, validate_targets
    tokenizer, model = load(args)
    targets = tuple(args.target or DEFAULT_TARGETS)
    validate_targets(model.model.layers, targets)
    text = args.calibration_text.read_text()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    norms = calibrate(model.model, inputs, [t.layer for t in targets])
    print("Calibration residual medians:", norms, flush=True)
    return tokenizer, model, targets, norms, text


def generate(tokenizer, model, messages, max_new_tokens):
    import torch
    from transformers import TextStreamer
    # Explicitly render then tokenize to avoid tokenizer-version return-type differences.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                use_cache=True, pad_token_id=tokenizer.eos_token_id,
                                streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True))
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def compare(args):
    from dataclasses import asdict
    from steering import steer
    tokenizer, model, targets, norms, text = prepare(args)
    messages = [{"role": "user", "content": args.prompt}]
    print("\nBASELINE", flush=True)
    baseline = generate(tokenizer, model, messages, args.max_new_tokens)
    print("\nSTEERED", flush=True)
    with steer(model.model.layers, targets, norms):
        steered = generate(tokenizer, model, messages, args.max_new_tokens)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record = {"model_dir": str(args.model_dir), "targets": [asdict(t) for t in targets],
                  "residual_norms": norms, "calibration_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "prompt": args.prompt, "baseline": baseline, "steered": steered}
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Saved {args.output}")


def chat(args):
    from steering import steer
    tokenizer, model, targets, norms, _ = prepare(args)
    messages = []
    enabled = True
    print("Commands: /on, /off (both reset history), /reset, /quit. Steering is ON.")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except EOFError:
            break
        if prompt == "/quit":
            break
        if prompt in ("/reset", "/on", "/off"):
            messages.clear()
            if prompt != "/reset":
                enabled = prompt == "/on"
            print(f"History reset. Steering {'ON' if enabled else 'OFF'}.")
            continue
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        context = steer(model.model.layers, targets, norms) if enabled else nullcontext()
        with context:
            answer = generate(tokenizer, model, messages, args.max_new_tokens)
        messages.append({"role": "assistant", "content": answer})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("doctor", doctor), ("download", download), ("compare", compare), ("chat", chat)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
        if name != "download":
            command.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
        if name in ("compare", "chat"):
            command.add_argument("--target", action="append", type=parse_target,
                                 help="Repeat LAYER:NEURON:DOSE; replaces the default pair")
            command.add_argument("--calibration-text", type=Path, default=ROOT / "calibration.txt")
            command.add_argument("--max-new-tokens", type=int, default=128)
        if name == "compare":
            command.add_argument("prompt", nargs="?", default="Who are you, and what do you want?")
            command.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    try:
        if getattr(args, "max_new_tokens", 1) < 1:
            raise ValueError("--max-new-tokens must be positive")
        args.function(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
