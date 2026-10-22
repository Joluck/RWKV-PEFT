"""Tests for multi-token prediction (``--peft mtp``)."""

import _cpu_stub as stub
import pytest
import torch

from rwkvt.mtp import MTPHead

VERSIONS = ["6", "7"]


def build(version="7", depth=1, ctx_len=16):
    model, args = stub.load_model(version=version, mode="normal")
    args.peft = "mtp"
    args.ctx_len = ctx_len
    args.mtp_depth = depth
    mtp = MTPHead(args, depth)
    return model, args, mtp


def batch(args, bsz=2):
    torch.manual_seed(0)
    idx = torch.randint(1, args.vocab_size, (bsz, args.ctx_len))
    return idx


def test_hidden_state_and_mtp_logits():
    model, args, mtp = build(depth=2)
    idx = batch(args)
    with torch.no_grad():
        logits, hidden, token_embeds = model(idx, return_hidden=True)
        outputs = mtp(hidden, token_embeds, model.head)
    assert hidden.shape == (2, args.ctx_len, args.n_embd)
    assert [d for d, _ in outputs] == [1, 2]
    # depth k covers ctx_len - k - 1 positions
    assert [o.shape[1] for _, o in outputs] == [args.ctx_len - 2, args.ctx_len - 3]
    assert all(o.shape[-1] == args.vocab_size for _, o in outputs)


@pytest.mark.parametrize("version", VERSIONS)
def test_only_mtp_parameters_get_gradients(version):
    model, args, mtp = build(version=version, depth=1)
    model.requires_grad_(False)
    idx = batch(args)
    targets = idx.roll(-1, dims=1)

    logits, hidden, token_embeds = model(idx, return_hidden=True)
    main = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                             targets.view(-1))
    aux = mtp.loss(hidden, token_embeds, targets, model.head,
                   torch.nn.CrossEntropyLoss())
    (main + 0.3 * aux).backward()

    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in mtp.parameters())


def test_mtp_training_decreases_loss():
    model, args, mtp = build(depth=1)
    model.requires_grad_(False)
    idx = batch(args)
    targets = idx.roll(-1, dims=1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mtp.parameters(), lr=1e-3)

    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        with torch.no_grad():
            _, hidden, token_embeds = model(idx, return_hidden=True)
        loss = mtp.loss(hidden, token_embeds, targets, model.head, criterion)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.9, losses


def test_sequence_too_short_raises():
    model, args, mtp = build(depth=1, ctx_len=2)
    idx = batch(args)
    with torch.no_grad():
        _, hidden, token_embeds = model(idx, return_hidden=True)
    with pytest.raises(ValueError, match="too short"):
        mtp.loss(hidden, token_embeds, idx, model.head,
                 torch.nn.CrossEntropyLoss())


def test_mtp_state_dict_roundtrip(tmp_path):
    _, args, mtp = build(depth=2)
    with torch.no_grad():
        for p in mtp.parameters():
            p.normal_(0, 0.02)
    path = tmp_path / "mtp.pth"
    torch.save({f"mtp.{k}": v for k, v in mtp.state_dict().items()}, path)

    _, _, restored = build(depth=2)
    state = torch.load(path, map_location="cpu")
    state = {k[len("mtp."):]: v for k, v in state.items()}
    missing, unexpected = restored.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    for key, value in mtp.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_step_weights_follow_fastmtp_decay():
    assert MTPHead.step_weights(0.6, 3) == pytest.approx(
        [0.5102, 0.3061, 0.1837], abs=1e-3)
    assert sum(MTPHead.step_weights(0.6, 4)) == pytest.approx(1.0)
    assert MTPHead.step_weights(1.0, 3) == pytest.approx([1 / 3] * 3)


def test_loss_uses_step_weights():
    model, args, mtp = build(depth=2)
    model.requires_grad_(False)
    idx = batch(args)
    targets = idx.roll(-1, dims=1)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        _, hidden, token_embeds = model(idx, return_hidden=True)
        uniform = mtp.loss(hidden, token_embeds, targets, model.head, criterion,
                           step_weights=[0.5, 0.5])
        per_step = list(mtp.last_step_losses)
        first_only = mtp.loss(hidden, token_embeds, targets, model.head, criterion,
                              step_weights=[1.0, 0.0])
    assert len(per_step) == 2
    assert uniform == pytest.approx(sum(per_step) / 2, rel=1e-4)
    assert first_only == pytest.approx(per_step[0], rel=1e-4)


def test_acceptance_matches_when_head_copies_base():
    torch.manual_seed(0)
    base = torch.randn(2, 10, 32)
    # a "perfect" drafter: same argmax as the verifier at the aligned position
    logits = base[:, 1:9].clone()
    assert MTPHead.acceptance(logits, base, 1) == pytest.approx(1.0)

    other = torch.zeros_like(logits)
    other[..., 0] = 5.0                      # always drafts token 0
    assert MTPHead.acceptance(other, base, 1) < 1.0


def test_loss_reports_acceptance():
    model, args, mtp = build(depth=2)
    model.requires_grad_(False)
    idx = batch(args)
    targets = idx.roll(-1, dims=1)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        logits, hidden, token_embeds = model(idx, return_hidden=True)
        mtp.loss(hidden, token_embeds, targets, model.head, criterion,
                 base_logits=logits)
    assert len(mtp.last_step_accept) == 2
    assert all(0.0 <= a <= 1.0 for a in mtp.last_step_accept)
