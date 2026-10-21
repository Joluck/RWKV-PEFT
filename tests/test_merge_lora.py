"""End-to-end test for ``merge/merge_lora.py`` (CPU, tiny model)."""

import importlib.util
import os
from types import SimpleNamespace

import _cpu_stub as stub
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_merge_module():
    spec = importlib.util.spec_from_file_location(
        "merge_lora", os.path.join(ROOT, "merge", "merge_lora.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_adapter_writes_merged_weights(tmp_path):
    pytest.importorskip("peft")
    model, args = stub.load_model(version="7", mode="normal")
    args.peft = "lora"
    args.peft_config = {"r": 4, "lora_alpha": 8}

    from rwkvt.fsdp import adapter_state_dict
    from rwkvt.peft_loading import apply_peft

    base_path = tmp_path / "base.pth"
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(original_state, base_path)

    wrapped = apply_peft(model, args)
    # give the adapters non-trivial values so the merge is observable
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if "lora_" in name:
                param.normal_(0, 0.02)
    adapter_path = tmp_path / "adapter.pth"
    state = adapter_state_dict(wrapped.state_dict())
    torch.save(state, adapter_path)

    output_path = tmp_path / "merged.pth"
    merge = load_merge_module()
    merge.merge_adapter(str(base_path), str(adapter_path), str(output_path),
                        SimpleNamespace(**vars(args)))

    merged = torch.load(output_path, map_location="cpu")
    assert "blocks.0.att.receptance.weight" in merged
    assert not any("lora_" in k for k in merged)

    # independent check of the LoRA math for one layer
    layer = "blocks.1.att.receptance"
    weight = merged[f"{layer}.weight"]
    original = original_state[f"{layer}.weight"]
    a = state[f"base_model.model.{layer}.lora_A.default.weight"]
    b = state[f"base_model.model.{layer}.lora_B.default.weight"]
    expected = original + (b @ a) * (args.peft_config["lora_alpha"] / args.peft_config["r"])
    assert torch.allclose(weight, expected, atol=1e-6)
