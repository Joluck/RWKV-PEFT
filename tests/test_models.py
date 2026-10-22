"""CPU smoke tests for the model layer (no CUDA / rwkv-fla required)."""

import _cpu_stub as stub
import pytest
import torch

VERSIONS = ["5", "6", "7"]


@pytest.mark.parametrize("version", VERSIONS)
def test_forward_shapes(version):
    model, args = stub.load_model(version=version, mode="normal")
    idx = torch.randint(0, args.vocab_size, (2, args.ctx_len))
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (2, args.ctx_len, args.vocab_size)


def test_forward_with_padding_mask():
    model, args = stub.load_model(version="7", mode="normal")
    idx = torch.randint(0, args.vocab_size, (2, args.ctx_len))
    mask = torch.ones(2, args.ctx_len, dtype=torch.bool)
    mask[:, -4:] = False
    with torch.no_grad():
        logits = model(idx, attention_mask=mask)
    assert logits.shape == (2, args.ctx_len, args.vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("version", ["6", "7"])
def test_grad_cp_matches_plain_forward(version):
    plain, _ = stub.load_model(version=version, mode="normal", grad_cp=0, seed=3)
    ckpt, args = stub.load_model(version=version, mode="normal", grad_cp=1, seed=3)
    ckpt.load_state_dict(plain.state_dict())
    idx = torch.randint(0, args.vocab_size, (2, args.ctx_len))
    with torch.no_grad():
        assert torch.allclose(plain(idx), ckpt(idx), atol=1e-6)


@pytest.mark.parametrize("version", ["6", "7"])
def test_backward_produces_gradients(version):
    model, args = stub.load_model(version=version, mode="normal")
    idx = torch.randint(0, args.vocab_size, (2, args.ctx_len))
    targets = torch.randint(0, args.vocab_size, (2, args.ctx_len))
    logits = model(idx)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()

    named = dict(model.named_parameters())
    # weights on the main path must receive gradients
    for name in ("blocks.1.att.receptance.weight", "blocks.1.ffn.key.weight"):
        assert named[name].grad is not None, name
    # layer 0's value-residual LoRA is inactive by design (v_first = v there),
    # so a handful of parameters legitimately stays without gradients.
    with_grad = [p for p in named.values()
                 if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(with_grad) >= 0.7 * len(named), len(with_grad)


@pytest.mark.parametrize("version", ["6", "7"])
def test_infctx_returns_states(version):
    model, args = stub.load_model(version=version, mode="infctx")
    from rwkvt.infctx_module import BlockStateList

    batch = 2
    heads = args.dim_att // args.head_size_a
    states = BlockStateList.create(args.n_layer, batch, args.n_embd, heads,
                                   torch.device("cpu"), torch.float32)
    idx = torch.randint(0, args.vocab_size, (batch, args.chunk_ctx))
    with torch.no_grad():
        logits, shift, wkv = model.forward_infctx(
            idx, states.shift_states, states.wkv_states)
    assert logits.shape == (batch, args.chunk_ctx, args.vocab_size)
    assert shift.shape == (args.n_layer, 2, batch, args.n_embd)
    assert wkv.shape == (args.n_layer, batch, heads, args.head_size_a,
                         args.head_size_a)


def test_state_tuning_params_exist():
    model, _ = stub.load_model(version="7", mode="state")
    names = [n for n, p in model.named_parameters() if "time_state" in n]
    assert len(names) == 3  # n_layer


def test_block_state_list_assign():
    from rwkvt.infctx_module import BlockStateList

    states = BlockStateList.create(2, 2, 32, 4, torch.device("cpu"), torch.float32)
    other = BlockStateList.create(2, 2, 32, 4, torch.device("cpu"), torch.float32)
    other.wkv_states.fill_(1.5)
    other.shift_states.fill_(2.5)
    states[0] = other[0]
    assert torch.equal(states.wkv_states[0], other.wkv_states[0])
    assert torch.equal(states.shift_states[0, 1], other.shift_states[0, 1])
    assert states[1].time_mix_state.wkv_state.abs().sum() == 0


@pytest.mark.parametrize("version", ["6", "7"])
def test_attention_mask_is_not_dropped(version):
    """A padding mask must change the output (it used to bind to inputs_embeds)."""
    model, args = stub.load_model(version=version, mode="normal")
    idx = torch.randint(1, args.vocab_size, (2, args.ctx_len))
    mask = torch.ones(2, args.ctx_len, dtype=torch.bool)
    mask[:, -4:] = False
    with torch.no_grad():
        plain = model(idx)
        masked = model(idx, attention_mask=mask)
        positional = model(idx, mask)
    assert not torch.allclose(plain, masked)
    assert torch.equal(masked, positional)


@pytest.mark.parametrize("version", VERSIONS)
def test_peft_wrapped_forward(version):
    """PEFT forwards input_ids/attention_mask/inputs_embeds/... as kwargs."""
    pytest.importorskip("peft")
    model, args = stub.load_model(version=version, mode="normal")
    args.peft = "lora"
    args.peft_config = {"r": 4}

    from rwkvt.peft_loading import apply_peft

    wrapped = apply_peft(model, args)
    idx = torch.randint(1, args.vocab_size, (2, args.ctx_len))
    mask = torch.ones(2, args.ctx_len, dtype=torch.bool)
    mask[:, -2:] = False
    with torch.no_grad():
        logits = wrapped(idx, attention_mask=mask)
    assert logits.shape == (2, args.ctx_len, args.vocab_size)


def test_inputs_embeds_is_rejected():
    model, args = stub.load_model(version="7", mode="normal")
    idx = torch.randint(1, args.vocab_size, (2, args.ctx_len))
    with pytest.raises(NotImplementedError, match="inputs_embeds"):
        model(idx, inputs_embeds=torch.zeros(2, args.ctx_len, args.n_embd))


def test_fla_backend_wrapper_shapes():
    """rwkv-fla returns (B, T, H, C/H); the wrapper path must stay 3-D safe."""
    model, args = stub.load_model(version="7", mode="normal", patch_kernels=False)
    idx = torch.randint(1, args.vocab_size, (2, args.ctx_len))
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (2, args.ctx_len, args.vocab_size)
    assert torch.isfinite(logits).all()


def test_bf16_true_casts_fp32_checkpoint(tmp_path):
    """--precision bf16-true must downcast an fp32 checkpoint before training."""
    from types import SimpleNamespace

    from rwkvt.peft_loading import build_model

    model, args = stub.load_model(version="7", mode="normal")
    ckpt = tmp_path / "fp32.pth"
    torch.save({k: v.clone() for k, v in model.state_dict().items()}, ckpt)

    ns = SimpleNamespace(**vars(args))
    ns.load_model = str(ckpt)
    ns.load_partial = 0
    ns.precision = "bf16-true"
    loaded = build_model(ns)
    assert loaded.emb.weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in loaded.parameters())
