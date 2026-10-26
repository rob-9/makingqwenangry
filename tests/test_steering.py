from types import SimpleNamespace

import pytest
import torch
from torch import nn

from steering import Target, calibrate, steer


def test_normalized_push_applies_to_prefill_and_decode_and_restores_weights():
    down = nn.Linear(4, 3, bias=False)
    layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=down))
    before = down.weight.detach().clone()
    direction = before[:, 2] / before[:, 2].norm()
    inputs = [torch.randn(1, 5, 4), torch.randn(1, 1, 4)]
    baseline = [down(x) for x in inputs]
    with steer([layer], [Target(0, 2, -1)], {0: 3.0}):
        for x, base in zip(inputs, baseline):
            delta = down(x) - base
            torch.testing.assert_close(delta, (-3 * direction).expand_as(delta))
    torch.testing.assert_close(down.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(down(inputs[0]), baseline[0])
    assert not down._forward_pre_hooks


def test_multiple_neurons_add_and_exception_removes_hooks():
    down = nn.Linear(4, 3, bias=False)
    layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=down))
    x = torch.randn(1, 2, 4)
    base = down(x)
    expected = sum(dose * down.weight[:, n] / down.weight[:, n].norm()
                   for n, dose in [(1, 1.0), (2, -0.5)])
    with pytest.raises(RuntimeError, match="deliberate"):
        with steer([layer], [Target(0, 1, 1), Target(0, 2, -0.5)], {0: 1}):
            torch.testing.assert_close(down(x) - base, expected.expand_as(base))
            raise RuntimeError("deliberate")
    assert not down._forward_pre_hooks
    torch.testing.assert_close(down(x), base)


def test_calibration_excludes_first_token_and_padding():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])

        def forward(self, input_ids, attention_mask, use_cache):
            values = torch.tensor([[[1000.0, 0], [3.0, 4], [0.0, 7], [0.0, 9], [999.0, 0]]])
            return self.layers[0](values)

    backbone = Backbone()
    inputs = {"input_ids": torch.ones((1, 5), dtype=torch.long),
              "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])}
    assert calibrate(backbone, inputs, [0]) == {0: 7.0}
    assert not backbone.layers[0]._forward_hooks


def test_bad_target_does_not_leave_hooks():
    down = nn.Linear(4, 3, bias=False)
    layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=down))
    with pytest.raises(ValueError, match="Neuron"):
        with steer([layer], [Target(0, 99, 1)], {0: 1}):
            pass
    assert not down._forward_pre_hooks


def tiny_config():
    from transformers import Qwen3_5TextConfig
    return Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
        pad_token_id=0, eos_token_id=127,
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_real_qwen_hybrid_calibration_and_cached_generation(device):
    from transformers import Qwen3_5ForCausalLM
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable in this process")
    torch.manual_seed(42)
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    model = Qwen3_5ForCausalLM(tiny_config()).to(device=device, dtype=dtype).eval()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    norms = calibrate(model.model, inputs, [0, 1])
    assert all(value > 0 for value in norms.values())
    with torch.inference_mode():
        base = model(**inputs, use_cache=False).logits
        with steer(model.model.layers, [Target(0, 5, 0.3), Target(1, 8, -0.3)], norms):
            changed = model(**inputs, use_cache=False).logits
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=3, min_new_tokens=3)
        restored = model(**inputs, use_cache=False).logits
    assert not torch.allclose(base, changed)
    assert generated.shape == (1, 7)
    torch.testing.assert_close(restored, base)
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.model.layers)


def test_text_loader_accepts_multimodal_checkpoint(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    config = Qwen3_5Config(
        text_config=tiny_config().to_dict(),
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16},
        tie_word_embeddings=False,
    )
    full = Qwen3_5ForConditionalGeneration(config).eval()
    full.save_pretrained(tmp_path)
    text = Qwen3_5ForCausalLM.from_pretrained(tmp_path, local_files_only=True).eval()
    torch.testing.assert_close(text.model.layers[0].mlp.down_proj.weight,
                               full.model.language_model.layers[0].mlp.down_proj.weight)
    torch.testing.assert_close(text.model.embed_tokens.weight,
                               full.model.language_model.embed_tokens.weight)
