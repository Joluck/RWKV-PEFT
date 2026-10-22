"""Tests for the greedy MTP speculative decoder (vLLM-style draft/verify)."""

import _cpu_stub as stub
import pytest
import torch

from rwkvt.mtp import MTPHead
from rwkvt.spec_decode import SpecStats, greedy_decode, mtp_greedy_decode


def build(depth=2, ctx_len=32):
    model, args = stub.load_model(version="7", mode="normal")
    args.peft = "mtp"
    args.ctx_len = ctx_len
    args.mtp_depth = depth
    mtp = MTPHead(args, depth)
    return model.eval(), mtp, args


def test_spec_decode_matches_plain_greedy():
    """Rejection must fall back to the verifier's token: same output, always."""
    model, mtp, args = build(depth=2)
    torch.manual_seed(0)
    prompt = torch.randint(1, args.vocab_size, (1, 8))

    baseline = greedy_decode(model, prompt, max_new_tokens=8)
    tokens = mtp_greedy_decode(model, mtp, prompt, max_new_tokens=8, depth=2)
    assert torch.equal(tokens, baseline)


def test_spec_decode_matches_greedy_for_each_depth():
    model, mtp, args = build(depth=3)
    torch.manual_seed(1)
    prompt = torch.randint(1, args.vocab_size, (1, 6))
    baseline = greedy_decode(model, prompt, max_new_tokens=6)
    for depth in (1, 2, 3):
        tokens = mtp_greedy_decode(model, mtp, prompt, max_new_tokens=6, depth=depth)
        assert torch.equal(tokens, baseline), depth


def test_depth_beyond_head_raises():
    model, mtp, args = build(depth=1)
    prompt = torch.randint(1, args.vocab_size, (1, 4))
    with pytest.raises(ValueError, match="exceeds"):
        mtp_greedy_decode(model, mtp, prompt, max_new_tokens=4, depth=2)


def test_stats_bookkeeping():
    model, mtp, args = build(depth=1)
    torch.manual_seed(2)
    prompt = torch.randint(1, args.vocab_size, (1, 8))
    stats = SpecStats()
    mtp_greedy_decode(model, mtp, prompt, max_new_tokens=12, depth=1, stats=stats)

    assert stats.rounds > 0
    # depth=1 -> one draft per round, except the final round when only one
    # token of budget is left (drafting would exceed max_new_tokens)
    assert stats.drafted in (stats.rounds, stats.rounds - 1)
    assert 0.0 <= stats.accept_rate <= 1.0
    # each round emits 1 (verifier) + accepted drafts
    assert stats.tokens_per_round == pytest.approx(
        (stats.rounds + stats.accepted) / stats.rounds)


def test_short_budget_emits_no_draft():
    model, mtp, args = build(depth=2)
    torch.manual_seed(3)
    prompt = torch.randint(1, args.vocab_size, (1, 4))
    stats = SpecStats()
    tokens = mtp_greedy_decode(model, mtp, prompt, max_new_tokens=1, depth=2,
                               stats=stats)
    assert tokens.size(1) == prompt.size(1) + 1
    assert stats.drafted == 0 and stats.accepted == 0


def test_eos_stops_generation():
    model, mtp, args = build(depth=1)
    torch.manual_seed(4)
    prompt = torch.randint(1, args.vocab_size, (1, 4))
    first = mtp_greedy_decode(model, mtp, prompt, max_new_tokens=2, depth=1)
    eos = int(first[0, -1])
    stopped = mtp_greedy_decode(model, mtp, prompt, max_new_tokens=10, depth=1,
                                eos_id=eos)
    assert stopped[0, -1].item() == eos
