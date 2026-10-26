# making qwen angry

local qwen3.5-4b chat with neuron steering: `l14 n8949 +1`, `l13 n7956 -1`.

```sh
uv sync
uv run python run.py download
uv run python ui.py
```
