"""PyTorch-Lightning module wrapping an RWKV model for fine-tuning."""

import math
import os

import lightning as pl
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from rwkvt.infctx_module import BlockStateList


class L2Wrap(torch.autograd.Function):
    """Adds a small penalty pulling the maximum logit towards zero.

    The scale is normalised by the number of tokens so that gradient
    accumulation and multi-GPU runs stay batch-size independent.
    """

    @staticmethod
    def forward(ctx, loss, y, token_amount=None):
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        token_amount = ctx.token_amount
        if token_amount is None:
            token_amount = y.shape[0] * y.shape[1]
        if token_amount == 0:
            return (grad_output, None, None)

        factor = 1e-4 / token_amount
        maxx, ids = torch.max(y, -1, keepdim=True)
        # only pull down logits that are already large
        gy = torch.zeros_like(y)
        if os.environ.get("WN_FIX_L2WRAP"):
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy, None)


class RWKV(pl.LightningModule):
    def __init__(self, args, model=None):
        super().__init__()
        assert model is not None, "pass the built RWKV model"
        self.args = args
        self.model = model

        if os.environ.get("FUSED_KERNEL", "0") == "1":
            from rwkvfla.modules import FusedCrossEntropyLoss
            self.criterion = FusedCrossEntropyLoss(inplace_backward=True)
        else:
            self.criterion = nn.CrossEntropyLoss()

    # -- optimiser -------------------------------------------------------- #
    def _parameter_groups(self):
        args = self.args
        lr_decay, lr_1x, lr_2x, lr_3x = set(), set(), set(), set()
        layerwise = args.layerwise_lr > 0
        stage2 = args.my_pile_stage == 2
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_1d = p.squeeze().dim() < 2
            if layerwise and (("_w1" in name) or ("_w2" in name)):
                lr_1x.add(name)
            elif layerwise and (("time_mix" in name) or ("time_maa" in name)):
                (lr_2x if stage2 else lr_1x).add(name)
            elif layerwise and (("time_decay" in name) or ("time_daaaa" in name)):
                (lr_3x if stage2 else lr_2x).add(name)
            elif layerwise and ("time_faaaa" in name):
                (lr_2x if stage2 else lr_1x).add(name)
            elif layerwise and ("time_first" in name):
                lr_3x.add(name)
            elif not is_1d and args.weight_decay > 0:
                lr_decay.add(name)
            else:
                lr_1x.add(name)

        param_dict = dict(self.model.named_parameters())
        scale_2x = 5.0 if args.my_pile_stage == 2 else 2.0
        scale_3x = 5.0 if args.my_pile_stage == 2 else 3.0
        groups = [{"params": [param_dict[n] for n in sorted(lr_1x)],
                   "weight_decay": 0.0, "my_lr_scale": 1.0}]
        if args.layerwise_lr > 0:
            groups.append({"params": [param_dict[n] for n in sorted(lr_2x)],
                           "weight_decay": 0.0, "my_lr_scale": scale_2x})
            groups.append({"params": [param_dict[n] for n in sorted(lr_3x)],
                           "weight_decay": 0.0, "my_lr_scale": scale_3x})
        if args.weight_decay > 0 and lr_decay:
            groups.append({"params": [param_dict[n] for n in sorted(lr_decay)],
                           "weight_decay": args.weight_decay, "my_lr_scale": 1.0})
        return [g for g in groups if g["params"]]

    def configure_optimizers(self):
        args = self.args
        groups = self._parameter_groups()
        if args.optimizer not in ("none", "adamw"):
            raise NotImplementedError(
                f"--optimizer {args.optimizer!r} is not implemented; "
                "only 'adamw' (default) is supported")

        kwargs = {"lr": args.lr_init, "betas": args.betas, "eps": args.adam_eps,
                  "weight_decay": args.weight_decay}
        if self.deepspeed_offload:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            return DeepSpeedCPUAdam(groups, bias_correction=True,
                                    adamw_mode=args.weight_decay > 0, **kwargs)

        try:
            from deepspeed.ops.adam import FusedAdam
        except ImportError:
            FusedAdam = None

        if FusedAdam is not None and torch.cuda.is_available():
            return FusedAdam(groups, bias_correction=True,
                             adam_w_mode=args.weight_decay > 0, **kwargs)
        return torch.optim.AdamW(groups, fused=torch.cuda.is_available(), **kwargs)

    @property
    def deepspeed_offload(self) -> bool:
        from lightning.pytorch.strategies import DeepSpeedStrategy

        strategy = self.trainer.strategy
        if not isinstance(strategy, DeepSpeedStrategy):
            return False
        cfg = strategy.config.get("zero_optimization", {})
        return bool(cfg.get("offload_optimizer") or cfg.get("offload_param"))

    # -- training --------------------------------------------------------- #
    def _infctx(self) -> bool:
        return os.environ.get("RWKV_TRAIN_TYPE") == "infctx"

    def forward(self, idx, *args, attention_mask=None, **kwargs):
        if self._infctx():
            return self.model(idx, *args, attention_mask=attention_mask, **kwargs)
        return self.model(idx, attention_mask=attention_mask, **kwargs)

    def _shared_step(self, batch):
        if self.args.data_type == "sft":
            idx, targets, mask = batch
            logits = self(idx, attention_mask=mask)
        else:
            idx, targets = batch
            logits = self(idx)
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits

    def training_step(self, batch, batch_idx):
        if not self._infctx():
            loss, logits = self._shared_step(batch)
            return L2Wrap.apply(loss, logits, None)
        return self._infctx_training_step(batch)

    def _infctx_training_step(self, batch):
        args = self.args
        chunk = args.chunk_ctx
        idx, targets = batch
        B, T = idx.shape
        H = args.dim_att // args.head_size_a
        param_dtype = next(self.model.parameters()).dtype
        states = BlockStateList.create(args.n_layer, B, args.n_embd, H,
                                       idx.device, param_dtype)

        def checkpointed_step(idx, targets, prev_loss, shift_states,
                              wkv_states, prev_token_amount):
            logits, new_shift_states, new_wkv_states = self(
                idx, shift_states, wkv_states)
            token_amount = idx.shape[1]
            loss = self.criterion(logits.view(-1, logits.size(-1)),
                                  targets.reshape(-1))
            if token_amount != 0:
                loss = L2Wrap.apply(loss, logits, token_amount)
            new_token_amount = prev_token_amount + token_amount
            if new_token_amount > 0:
                new_loss = (prev_loss * (prev_token_amount / new_token_amount)
                            + loss * (token_amount / new_token_amount))
            else:
                new_loss = prev_loss
            return new_loss, new_shift_states, new_wkv_states, new_token_amount

        total_loss = torch.tensor(0., dtype=param_dtype, requires_grad=True)
        token_amount = 0
        for i in range(math.ceil(T / chunk)):
            total_loss, new_shift, new_wkv, token_amount = torch_checkpoint(
                checkpointed_step,
                idx[:, i * chunk:(i + 1) * chunk],
                targets[:, i * chunk:(i + 1) * chunk],
                total_loss, states.shift_states, states.wkv_states,
                token_amount, use_reentrant=False)
            states = BlockStateList(new_shift.detach().clone(),
                                    new_wkv.detach().clone())
        return total_loss


class FSDP2RWKV(RWKV):
    """LightningModule that shards itself with FSDP2 in ``configure_model()``.

    Lightning calls this hook right before ``strategy.setup()`` (which moves the
    model to the GPU and builds the optimizer), and injects ``self._device_mesh``
    in ``setup_environment()``. See ``rwkvt/fsdp.py``.
    """

    def configure_model(self) -> None:
        from rwkvt.fsdp import apply_fsdp2, is_fsdp2

        if not is_fsdp2(self.trainer.strategy):
            return  # not an FSDP2 run: nothing to do
        mesh = getattr(self, "_device_mesh", None)
        if mesh is None:
            raise RuntimeError(
                "FSDP2 needs the device mesh provided by the strategy; "
                "start the run with --strategy fsdp2")
        # FSDP2 requires params on the mesh device before sharding; the helper
        # moves them block-by-block so the full model never has to fit on one GPU.
        apply_fsdp2(self.model, mesh,
                    device=self.trainer.strategy.root_device)
