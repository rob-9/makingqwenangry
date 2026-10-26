"""Add normalized neuron offsets at down_proj inputs; never modify weights."""

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Target:
    layer: int
    neuron: int
    dose: float


DEFAULT_TARGETS = (Target(14, 8949, 1.0), Target(13, 7956, -1.0))


def validate_targets(layers, targets):
    for target in targets:
        if not 0 <= target.layer < len(layers):
            raise ValueError(f"Layer {target.layer} is outside this model's layer range")
        width = layers[target.layer].mlp.down_proj.weight.shape[1]
        if not 0 <= target.neuron < width:
            raise ValueError(f"Neuron {target.neuron} is outside layer {target.layer}'s width {width}")
        if not math.isfinite(target.dose):
            raise ValueError("Dose must be finite")


@torch.inference_mode()
def calibrate(backbone, inputs, layer_indices):
    """Measure unsteered block outputs, excluding position zero and padding."""
    layer_indices = sorted(set(layer_indices))
    mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).bool()
    mask = mask.clone()
    mask[:, 0] = False
    if not mask.any():
        raise ValueError("Calibration needs at least two non-padding tokens")
    norms = {}

    def capture(index):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            norms[index] = hidden.float().norm(dim=-1)[mask.to(hidden.device)].cpu().median().item()
        return hook

    handles = []
    try:
        for index in layer_indices:
            handles.append(backbone.layers[index].register_forward_hook(capture(index)))
        backbone(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return norms


@contextmanager
def steer(layers, targets, residual_norms):
    """Keep offsets fixed through prefill and decoding; remove hooks on exit."""
    targets = tuple(targets)
    validate_targets(layers, targets)
    offsets = []
    for target in targets:
        down = layers[target.layer].mlp.down_proj
        column_norm = down.weight[:, target.neuron].detach().float().norm().item()
        residual_norm = residual_norms[target.layer]
        if not math.isfinite(column_norm) or column_norm <= 0:
            raise ValueError("The down-projection column must have a finite, positive norm")
        if not math.isfinite(residual_norm) or residual_norm <= 0:
            raise ValueError("The residual norm must be finite and positive")
        offsets.append((down, target.neuron, target.dose * residual_norm / column_norm))

    def make_hook(neuron, offset):
        def hook(module, args):
            activations = args[0].clone()
            activations[..., neuron] += offset
            return (activations, *args[1:])
        return hook

    handles = []
    try:
        for down, neuron, offset in offsets:
            handles.append(down.register_forward_pre_hook(make_hook(neuron, offset)))
        yield
    finally:
        for handle in handles:
            handle.remove()
