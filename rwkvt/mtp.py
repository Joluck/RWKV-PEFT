"""Multi-token prediction (MTP) heads — DeepSeek-V3 / vLLM layout.

Reference implementation: ``vllm/model_executor/models/deepseek_mtp.py``
(``DeepSeekMultiTokenPredictorLayer``), which is::

    x = eh_proj(cat([enorm(emb(t_{i+k})), hnorm(h^{k-1})]))
    x = mtp_block(x)                      # a transformer block at depth k
    logits = lm_head(norm(x))             # shared head, one extra norm

This module keeps that structure with an RWKV channel-mix replacing the
transformer block:

* ``norm_e`` / ``norm_h``  == ``enorm`` / ``hnorm``
* ``proj``                 == ``eh_proj``
* ``key`` + ``value``      == the block (channel mixing instead of attention)
* ``norm_out``             == ``shared_head.norm`` (the pre-norm hidden is
  recycled into the next depth, exactly like vLLM recycles
  ``residual + hidden_states``)

Only the blocks are trainable; the base model (embeddings, blocks, LM head) is
frozen. At inference time the heads are either dropped or used as the drafter in
:mod:`rwkvt.spec_decode`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def acceptance(mtp_logits: torch.Tensor, base_logits: torch.Tensor,
               depth: int) -> torch.Tensor:
    """Teacher-forced greedy acceptance: does the draft match the verifier?

    ``mtp_logits[:, i]`` and ``base_logits[:, i + depth]`` predict the same token
    (the depth-``depth`` target), so comparing their argmax over the *training*
    batch gives the acceptance rate for free: the logits are already computed for
    the loss, only two argmax passes are added.
    """
    span = mtp_logits.size(1)
    draft = mtp_logits.argmax(dim=-1)
    verify = base_logits[:, depth:depth + span].argmax(dim=-1)
    return (draft == verify).float().mean()


class MTPBlock(nn.Module):
    """One prediction step: fuse(hidden, next-token embedding) -> hidden."""

    def __init__(self, args):
        super().__init__()
        n_embd = args.n_embd
        self.norm_h = nn.LayerNorm(n_embd)   # hnorm in vLLM
        self.norm_e = nn.LayerNorm(n_embd)   # enorm in vLLM
        self.proj = nn.Linear(2 * n_embd, n_embd, bias=False)  # eh_proj
        # RWKV-flavoured channel mixing instead of an attention block
        self.key = nn.Linear(n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, n_embd, bias=False)
        # vLLM applies ``shared_head.norm`` before the shared LM head; the base
        # RWKV head also expects `ln_out`-normalised inputs, so this matters.
        self.norm_out = nn.LayerNorm(n_embd)  # vLLM: shared_head.norm

    def forward(self, hidden: torch.Tensor, token_emb: torch.Tensor) -> torch.Tensor:
        # vLLM concatenates [inputs_embeds, previous_hidden_states]
        x = self.proj(torch.cat([self.norm_e(token_emb), self.norm_h(hidden)], dim=-1))
        return x + self.value(torch.relu(self.key(x)) ** 2)

    def head_logits(self, hidden: torch.Tensor, head: nn.Module) -> torch.Tensor:
        """Logits from the recycled (pre-norm) hidden state."""
        return head(self.norm_out(hidden))


class MTPHead(nn.Module):
    """Stack of ``depth`` MTP blocks."""

    def __init__(self, args, depth: int = 1):
        super().__init__()
        if depth < 1:
            raise ValueError("MTP depth must be >= 1")
        self.depth = depth
        self.blocks = nn.ModuleList([MTPBlock(args) for _ in range(depth)])
        # per-depth losses / acceptance of the last call, for logging
        self.last_step_losses: list[float] = []
        self.last_step_accept: list[float] = []

    @staticmethod
    def step_weights(beta: float, num_steps: int) -> list[float]:
        """FastMTP Eq. 2: ``alpha_k = beta^(k-1) / sum_j beta^(j-1)``.

        beta=0.6, K=3 -> [0.51, 0.31, 0.18]: the first speculative token (the one
        that dominates acceptance) gets most of the weight. beta=1 -> uniform.
        """
        if num_steps <= 0:
            return []
        raw = [beta ** k for k in range(num_steps)]
        total = sum(raw)
        return [w / total for w in raw]

    # -- training --------------------------------------------------------- #
    def forward(self, hidden: torch.Tensor, token_embeds: torch.Tensor,
                head: nn.Module):
        """Return ``[(depth, logits), ...]`` for the positions each depth covers.

        ``token_embeds`` are the frozen embedding outputs of the *input* tokens,
        so MTP is callable outside the base model's forward (matters under FSDP2).
        """
        total = hidden.size(1)
        h = hidden
        outputs = []
        for depth, block in enumerate(self.blocks, start=1):
            span = total - depth - 1
            if span <= 0:
                break
            h = block(h[:, :span], token_embeds[:, depth:depth + span])
            outputs.append((depth, block.head_logits(h, head)))
        return outputs

    @staticmethod
    def acceptance(mtp_logits, base_logits, depth):
        return acceptance(mtp_logits, base_logits, depth)

    def loss(self, hidden: torch.Tensor, token_embeds: torch.Tensor,
             targets: torch.Tensor, head: nn.Module,
             criterion: nn.Module,
             step_weights: list[float] | None = None,
             base_logits: torch.Tensor | None = None) -> torch.Tensor:
        """Mean cross-entropy over the configured depths.

        ``targets`` is the already shifted target tensor (``targets[i] ==
        input_ids[i+1]``), so depth ``k`` reads ``targets[:, k:]``.
        """
        losses = []
        accepts = []
        # call via __call__ so FSDP2's pre-forward hook unshards the blocks
        for depth, logits in self(hidden, token_embeds, head):
            span = logits.size(1)
            tgt = targets[:, depth:depth + span]
            losses.append(criterion(logits.reshape(-1, logits.size(-1)),
                                    tgt.reshape(-1)))
            if base_logits is not None:
                with torch.no_grad():
                    accepts.append(float(acceptance(logits, base_logits, depth)))
        if not losses:
            raise ValueError(
                "sequence is too short for the configured MTP depth; "
                "increase --ctx_len or lower --mtp_depth")
        self.last_step_losses = [float(loss.detach()) for loss in losses]
        self.last_step_accept = accepts

        if step_weights is None:
            return torch.stack(losses).mean()
        weights = list(step_weights)[:len(losses)]
        norm = sum(weights)
        weights = [w / norm for w in weights]  # depths may be dropped on short seqs
        total = losses[0] * weights[0]
        for loss, weight in zip(losses[1:], weights[1:], strict=True):
            total = total + loss * weight
        return total

    # -- inference -------------------------------------------------------- #
    def step(self, hidden: torch.Tensor, token_emb: torch.Tensor,
             head: nn.Module, depth_index: int = 0):
        """One draft step, as used by :mod:`rwkvt.spec_decode`.

        Mirrors ``DeepSeekMultiTokenPredictorLayer.forward``: returns the
        recycled (pre-norm) hidden and the logits for the drafted position.
        """
        block = self.blocks[depth_index]
        h = block(hidden, token_emb)
        return h, block.head_logits(h, head)
