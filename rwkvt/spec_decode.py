"""Greedy MTP speculative decoding: draft with the heads, verify with the base.

Mirrors how vLLM turns DeepSeek's MTP layer into a drafter
(``vllm/model_executor/models/deepseek_mtp.py`` +
``vllm/v1/spec_decode`` + ``vllm/v1/sample/rejection_sampler.py``): the target
model verifies the drafted token(s) in one forward and the longest valid prefix
is kept. Here the verifier is greedy and the base model is re-run on the whole
sequence (no KV/state cache), so this is a *reference harness* to measure the
acceptance rate - not a fast production decoder.

Key property (checked in ``tests/test_spec_decode.py``): for greedy sampling the
output is **identical** to plain greedy decoding, independent of how good the
drafter is.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: the chunked WKV CUDA kernel asserts ``T % 16 == 0``; pad on the right
CHUNK_LEN = 16


def _forward(model, tokens: torch.Tensor, need_hidden: bool = False):
    """Model forward padded to a whole number of kernel chunks.

    Padding goes *after* the last token and RWKV is causal, so the logits of the
    real positions are unchanged.
    """
    total = tokens.size(1)
    pad = (-total) % CHUNK_LEN
    if pad:
        filler = tokens[:, -1:].expand(-1, pad)
        tokens = torch.cat([tokens, filler], dim=1)
    if need_hidden:
        logits, hidden, _ = model(tokens, return_hidden=True)
        return logits[:, :total], hidden[:, :total]
    return model(tokens)[:, :total], None


@dataclass
class SpecStats:
    """Acceptance bookkeeping.

    ``tokens_per_round`` counts the tokens the round emits (the verifier's own
    token plus every accepted draft). No free "bonus" token is counted, matching
    the 1 + alpha accounting DeepSeek report (0.9 -> ~1.9x).
    """

    rounds: int = 0
    drafted: int = 0
    accepted: int = 0

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return (self.rounds + self.accepted) / self.rounds if self.rounds else 0.0


@torch.no_grad()
def mtp_greedy_decode(model, mtp, prompt: torch.Tensor, max_new_tokens: int,
                      depth: int = 1, eos_id: int | None = None,
                      stats: SpecStats | None = None,
                      trace: list | None = None) -> torch.Tensor:
    """Generate with MTP drafts + greedy verification, one forward per round.

    The verification forward also yields the verifier's token for the *next*
    position, so a round costs exactly one full forward (the baseline costs one
    per token). That is what makes the measured wall clock faster, not just the
    token accounting.
    """
    tokens = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt
    stats = stats if stats is not None else SpecStats()
    if mtp is None:
        raise ValueError("mtp_greedy_decode needs an MTPHead")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if depth > mtp.depth:
        raise ValueError(
            f"depth={depth} exceeds the head's depth={mtp.depth}; "
            "train a deeper head with --mtp_depth")

    logits, hidden = _forward(model, tokens, need_hidden=True)
    next_tok = logits[:, -1].argmax(-1, keepdim=True)
    h_prev = hidden[:, -1:]  # (B, 1, C)

    generated = 0
    while generated < max_new_tokens:
        # ---- draft (cheap: depth x 151M block) --------------------------
        n_draft = min(depth, max_new_tokens - generated - 1)
        drafts = []
        h = h_prev
        tok_emb = model.emb(next_tok)
        for index in range(n_draft):
            h, draft_logits = mtp.step(h, tok_emb, model.head, depth_index=index)
            tok = draft_logits[:, -1].argmax(-1, keepdim=True)
            drafts.append(tok)
            tok_emb = model.emb(tok)

        candidate = torch.cat([next_tok, *drafts], dim=1) if drafts else next_tok
        base_len = tokens.size(1)

        # ---- one forward: verify the drafts and predict the next token ---
        out_logits, out_hidden = _forward(model, torch.cat([tokens, candidate], dim=1),
                                          need_hidden=True)
        accept = 0
        for i, draft in enumerate(drafts):
            pred = out_logits[:, base_len + i].argmax(-1)
            if pred.item() != draft.item():
                break
            accept += 1

        if trace is not None:
            trace.append({"round": stats.rounds + 1, "drafted": len(drafts),
                          "accepted": accept, "emitted": 1 + accept})
        tokens = torch.cat([tokens, candidate[:, :1 + accept]], dim=1)
        generated += 1 + accept
        stats.rounds += 1
        stats.drafted += len(drafts)
        stats.accepted += accept

        # the verifier's token at the accepted prefix is the next draft input
        next_tok = out_logits[:, base_len + accept].argmax(-1, keepdim=True)
        h_prev = out_hidden[:, base_len + accept:base_len + accept + 1]

        if eos_id is not None and tokens[:, -1].item() == eos_id:
            break
    return tokens


@torch.no_grad()
def greedy_decode(model, prompt: torch.Tensor, max_new_tokens: int,
                  eos_id: int | None = None) -> torch.Tensor:
    """Plain greedy baseline (one token per forward), for equivalence checks."""
    tokens = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt
    for _ in range(max_new_tokens):
        logits, _ = _forward(model, tokens)
        tokens = torch.cat([tokens, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        if eos_id is not None and tokens[:, -1].item() == eos_id:
            break
    return tokens
