# Qwen neuron steering

Run Qwen3.5-4B locally and add normalized activation offsets to L14 N8949 (+1) and L13 N7956 (−1). All code here is standalone. Model weights come directly from Hugging Face; no other steering repository is required.

## Setup

```bash
cd /Users/robert/qwen-steering
uv sync
uv run python run.py doctor
uv run python run.py download
uv run python run.py compare "Who are you, and what do you want?"
uv run python run.py chat
```

Python dependencies are isolated in `.venv`, with versions recorded in `uv.lock`. The original model files occupy about 8.7 GiB. The download command checks available space before fetching weights and requires an additional 1 GiB of headroom. It pins model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Dependencies need additional disk space. On this M4 Mac, inference selects MPS and BF16; the CPU fallback uses FP32 and substantially more memory. Short prompts and the default 128-token output limit are a sensible starting point for the 16 GB machine.

Use another disk by supplying the same model directory to each command:

```bash
uv run python run.py download --model-dir /Volumes/YourDrive/Qwen3.5-4B
uv run python run.py compare --model-dir /Volumes/YourDrive/Qwen3.5-4B "Who are you?"
uv run python run.py chat --model-dir /Volumes/YourDrive/Qwen3.5-4B
```

The loader uses the text backbone from the official multimodal checkpoint, avoiding the vision tower in memory. Inference uses `local_files_only=True` and does not initiate downloads. It does not train or edit model weights. The download command reads public Hugging Face files without authentication.

## Experiment

`compare` runs the same prompt twice, first unsteered and then with both neurons steered. Both runs use greedy decoding with thinking disabled and independent caches. Save the results and calibration metadata with `--output results/comparison.json`. `chat` keeps a conversation; `/on` and `/off` switch steering and clear history, `/reset` clears history, and `/quit` exits.

Replace the defaults by repeating `--target LAYER:NEURON:DOSE`. Layer and neuron indices are zero-based:

```bash
uv run python run.py compare "Describe yourself." --target 14:8949:0.5 --target 13:7956:-0.5 --output results/comparison.json
```

## How it works

Before steering, the runner processes `calibration.txt` through the unmodified model. Forward hooks measure residual vectors leaving each selected decoder block, before the next block's normalization. The median is taken over token vector lengths, excluding position zero and padding. Each selected layer has its own median. Supply another reference passage with `--calibration-text path/to/reference.txt`; only its first 512 tokens are used. These local calibration values can differ from the original experiment's reference statistics, so this reproduces the intervention mechanism without claiming identical outputs.

```text
offset[L, n] = dose × median_token_norm(residual_leaving_L) / norm(W_down[:, n])
down_proj_input[..., n] += offset[L, n]
```

The hooks sit immediately before each MLP's `down_proj`, after the gated activation. The offsets are additive, fixed after calibration, and applied at every prompt and generated-token position, including position zero. Excluding the first token applies only to calibration. A dose of −1 subtracts a normalized offset; it does not negate or clamp the natural activation. Hooks are removed even if generation fails or is interrupted.

`steering.py` contains calibration and intervention code. `run.py` handles download, device selection, generation, and the CLI. The reported behavioral effect of the original neuron pair is an experimental claim, not a guaranteed outcome.

## Checks

```bash
uv run pytest -q
```

Tests use a tiny randomly initialized Qwen3.5 with both linear and full attention, so they need no downloaded model weights. They check cached generation, activation changes, restoration after removing hooks, exact normalization of the injected vector, calibration token exclusions, loading a text model from a multimodal checkpoint, and the disk-space guard. Full-size behavioral validation requires the real weights.

References: [official Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [official model](https://huggingface.co/Qwen/Qwen3.5-4B).
