"""FSDP2 (``fully_shard``) compatibility test.

Runs single-process on CPU with the gloo backend: it verifies that the model can
be sharded, that a forward/backward/optimizer step works on sharded parameters,
and that parameters really become ``Shard`` DTensors. It is a structural test -
multi-GPU numerics still need a real run.
"""


import _cpu_stub as stub
import pytest
import torch

fsdp = pytest.importorskip("torch.distributed.fsdp")


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if torch.distributed.is_initialized():
        pytest.skip("distributed already initialised")
    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    init_file = tmp_path_factory.mktemp("pg") / "shared_init"
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


def test_fully_shard_forward_and_step(process_group):
    from torch.distributed.device_mesh import init_device_mesh

    model, args = stub.load_model(version="7", mode="normal")
    mesh = init_device_mesh("cpu", (1,))
    for block in model.blocks:
        fsdp.fully_shard(block, mesh=mesh)
    fsdp.fully_shard(model, mesh=mesh)

    weight = model.blocks[1].att.receptance.weight
    assert isinstance(weight, torch.distributed.tensor.DTensor)
    assert weight.placements[0].is_shard()

    idx = torch.randint(0, args.vocab_size, (2, args.ctx_len))
    targets = torch.randint(0, args.vocab_size, (2, args.ctx_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logits = model(idx)
    assert logits.shape == (2, args.ctx_len, args.vocab_size)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()

    grad = model.blocks[1].att.receptance.weight.grad
    assert grad is not None
    assert torch.isfinite(torch.tensor(float(loss)))


def test_apply_fsdp2_shards_blocks_and_root(process_group):
    from torch.distributed.device_mesh import init_device_mesh

    from rwkvt.fsdp import apply_fsdp2

    model, _ = stub.load_model(version="7", mode="normal")
    mesh = init_device_mesh("cpu", (1,))
    apply_fsdp2(model, mesh)
    assert isinstance(model.blocks[0].att.receptance.weight, torch.distributed.tensor.DTensor)
    assert isinstance(model.emb.weight, torch.distributed.tensor.DTensor)


def test_gather_model_state_dict_restores_plain_tensors(process_group):
    from torch.distributed.device_mesh import init_device_mesh

    from rwkvt.fsdp import apply_fsdp2, gather_model_state_dict

    model, _ = stub.load_model(version="7", mode="normal", seed=5)
    reference = {k: v.clone() for k, v in model.state_dict().items()}
    apply_fsdp2(model, init_device_mesh("cpu", (1,)))

    gathered = gather_model_state_dict(model)
    assert set(gathered) == set(reference)
    for key, value in gathered.items():
        assert not isinstance(value, torch.distributed.tensor.DTensor), key
        assert torch.equal(value, reference[key]), key


def test_merge_peft_state_dict_matches_reference(process_group):
    from torch.distributed.device_mesh import init_device_mesh

    from rwkvt.fsdp import apply_fsdp2, gather_model_state_dict, merge_peft_state_dict
    from rwkvt.peft_loading import apply_peft

    pytest.importorskip("peft")
    model, args = stub.load_model(version="7", mode="normal", seed=0)
    args.peft = "lora"
    args.peft_config = {"r": 4}
    wrapped = apply_peft(model, args)
    apply_fsdp2(wrapped, init_device_mesh("cpu", (1,)))
    gathered = gather_model_state_dict(wrapped)
    del wrapped

    reference_model, _ = stub.load_model(version="7", mode="normal", seed=1)
    reference = apply_peft(reference_model, args)
    missing, unexpected = reference.load_state_dict(gathered, strict=False)
    assert not missing and not unexpected
    expected = reference.merge_and_unload().state_dict()

    merged = merge_peft_state_dict(args, gathered)
    assert set(merged) == set(expected)
    for key, value in merged.items():
        assert torch.allclose(value, expected[key], atol=1e-6), key


def test_adapter_state_dict_filters_base_weights():
    from rwkvt.fsdp import adapter_state_dict

    state = {"base_model.model.blocks.0.att.receptance.weight": 1,
             "base_model.model.blocks.0.att.receptance.lora_A.default.weight": 2,
             "base_model.model.blocks.0.att.receptance.lora_B.default.weight": 3}
    filtered = adapter_state_dict(state)
    assert set(filtered) == {
        "base_model.model.blocks.0.att.receptance.lora_A.default.weight",
        "base_model.model.blocks.0.att.receptance.lora_B.default.weight"}


def test_gather_adapter_state_dict_is_cheap_and_lossless(process_group):
    from torch.distributed.device_mesh import init_device_mesh

    from rwkvt.fsdp import adapter_state_dict, apply_fsdp2, gather_adapter_state_dict
    from rwkvt.peft_loading import apply_peft

    pytest.importorskip("peft")
    model, args = stub.load_model(version="7", mode="normal", seed=2)
    args.peft = "lora"
    args.peft_config = {"r": 4}
    wrapped = apply_peft(model, args)
    reference = adapter_state_dict(
        {k: v.clone() for k, v in wrapped.state_dict().items()})

    apply_fsdp2(wrapped, init_device_mesh("cpu", (1,)))
    gathered = gather_adapter_state_dict(wrapped)

    assert set(gathered) == set(reference)
    assert gathered  # adapters were actually found
    for key, value in gathered.items():
        assert torch.equal(value, reference[key]), key
