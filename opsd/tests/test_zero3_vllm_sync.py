import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zero3_vllm_sync import iter_merged_params  # noqa: E402


@contextmanager
def no_gather(params):
    yield


def test_per_module_merge_matches_full_merge():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj"]))
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p)

    streamed = {name: t.clone() for name, t in iter_merged_params(model, no_gather)}

    model.merge_adapter()
    expected = {}
    for name, p in model.named_parameters():
        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        if model.prefix not in name and "original_module" not in name:
            expected[name] = p.data.clone()
    model.unmerge_adapter()

    assert list(streamed) == list(expected)
    for name in expected:
        torch.testing.assert_close(streamed[name], expected[name])
    assert any("q_proj" in n for n in streamed)
