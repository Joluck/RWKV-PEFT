"""Lightning callbacks: LR/WD schedule, logging, checkpoint saving."""

import copy
import gc
import json
import math
import os
import time

import lightning as pl
import torch

from rwkvt.fsdp import (
    gather_adapter_state_dict,
    gather_model_state_dict,
    is_fsdp2,
    merge_peft_state_dict,
)
from rwkvt.trick.lrs import cos_decay, wsd
from rwkvt.utils import strip_prefix


def reduce_logged_loss(trainer, loss) -> float:
    """Mean of the per-rank losses, scaled back for gradient accumulation.

    ``training_step`` returns the loss of *this* rank's micro-batch; reporting it
    as-is would show a value that depends on the number of ranks. The tensor is
    detached first so the collective never touches the autograd graph.
    """
    value = loss.detach().float().clone()
    if trainer.world_size > 1:
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        value = value / trainer.world_size
    return float(value) * (trainer.accumulate_grad_batches or 1)


class TrainCallback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_file = os.path.join(args.proj_dir, "loss_data.jsonl")
        if os.path.exists(self.loss_file):
            os.remove(self.loss_file)

    # -- helpers ---------------------------------------------------------- #
    def write_data(self, loss_data, t_cost, kt_s, extra=None):
        """Append one step of loss data (plotted by the streamlit helpers).

        ``extra`` carries the per-term losses (main / mtp) when they exist.
        """
        record = {"loss": float(loss_data), "t_cost": t_cost, "kt_s": kt_s}
        if extra:
            record.update(extra)
        with open(self.loss_file, "a") as f:
            json.dump(record, f)
            f.write("\n")

    def _total_steps(self, trainer) -> int:
        return max(1, trainer.num_training_batches * self.args.epoch_count)

    def _get_lr(self, trainer) -> float:
        args = self.args
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            return args.lr_init
        total = self._total_steps(trainer)
        if args.lr_schedule == "wsd":
            return wsd(args.lr_init, args.lr_final, trainer.global_step, total)
        return cos_decay(args.lr_init, args.lr_final, trainer.global_step, total)

    # -- hooks ------------------------------------------------------------ #
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        lr = self._get_lr(trainer)
        if args.warmup_steps > 0 and trainer.global_step < args.warmup_steps:
            lr *= 0.01 + 0.99 * trainer.global_step / args.warmup_steps

        progress = trainer.global_step / self._total_steps(trainer)
        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(
                math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for group in trainer.optimizers[0].param_groups:
            if group["weight_decay"] > 0:
                group["weight_decay"] = wd_now
            group["lr"] = lr * group.get("my_lr_scale", 1.0)

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.global_step == 0 and trainer.is_global_zero:
            trainer.my_loss_sum = 0.0
            trainer.my_loss_count = 0
            trainer.my_log = open(  # noqa: SIM115 - closed in on_train_end
                os.path.join(args.proj_dir, "train_log.txt"), "a")
            trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
            strategy_config = getattr(trainer.strategy, "config", None)
            if strategy_config is not None:
                print(f"\n{strategy_config}\n")
                trainer.my_log.write(f"{strategy_config}\n")
                trainer.my_log.flush()
            mode = getattr(args, "wandb_mode", "online")
            if args.wandb and mode != "disabled":
                try:
                    import wandb
                except ImportError as exc:  # pragma: no cover - env dependent
                    raise RuntimeError(
                        "--wandb requires the wandb package: pip install wandb"
                    ) from exc
                run_name = (getattr(args, "wandb_run_name", "")
                            or f"{args.run_name} {args.my_timestamp}")
                print(f"wandb: project={args.wandb} mode={mode} run={run_name}")
                wandb.init(project=args.wandb, name=run_name,
                           config=vars(args), save_code=False, mode=mode,
                           tags=[getattr(args, "peft", "none"),
                                 getattr(args, "strategy", "auto")])
                trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        logged_loss = reduce_logged_loss(trainer, loss)

        if not trainer.is_global_zero:
            return

        t_now = time.time_ns()
        t_cost = 0.0
        kt_s = 0.0
        if getattr(trainer, "my_time_ns", None) is not None:
            elapsed = (t_now - trainer.my_time_ns) / 1e9
            if elapsed > 0:
                t_cost = 1.0 / elapsed
                kt_s = token_per_step / elapsed / 1000
                pl_module.log("REAL it/s", t_cost, prog_bar=True, on_step=True)
                pl_module.log("Kt/s", kt_s, prog_bar=True, on_step=True)
        trainer.my_time_ns = t_now

        trainer.my_loss = logged_loss
        trainer.my_loss_sum += trainer.my_loss
        trainer.my_loss_count += 1
        trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
        pl_module.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
        pl_module.log("sum_loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
        pl_module.log("loss", trainer.my_loss, prog_bar=True, on_step=True)

        accum = trainer.accumulate_grad_batches or 1
        trainer.my_loss_accum = getattr(trainer, "my_loss_accum", 0.0) + trainer.my_loss / accum
        if (batch_idx + 1) % accum:
            return

        step_loss = trainer.my_loss_accum
        trainer.my_loss_accum = 0.0
        extra = {}
        for name in ("main_loss", "mtp_loss"):
            value = getattr(pl_module, f"_last_{name}", None)
            if value is not None:
                extra[name] = value
        extra.update(getattr(pl_module, "_last_mtp_steps", {}) or {})
        if args.wandb:
            metrics = {"loss": step_loss, "lr": trainer.my_lr, "wd": trainer.my_wd,
                       "epoch": trainer.current_epoch,
                       "Gtokens": trainer.global_step * token_per_step / 1e9}
            metrics.update(extra)
            if kt_s > 0:
                metrics["kt/s"] = kt_s
            trainer.my_wandb.log(metrics, step=int(trainer.global_step))
        self.write_data(step_loss, t_cost, kt_s, extra)

        save_every = getattr(args, "save_per_step", 0)
        if save_every > 0 and trainer.global_step > 0 and \
                trainer.global_step % save_every == 0:
            self._save(trainer, pl_module, tag=f"step{trainer.global_step}")
            print(f"[save] step {trainer.global_step}")

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        last_epoch = trainer.current_epoch >= (args.epoch_count - 1)
        if last_epoch or (args.epoch_save > 0 and
                          (trainer.current_epoch + 1) % args.epoch_save == 0):
            self._save(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        log_file = getattr(trainer, "my_log", None)
        if log_file is not None:
            log_file.close()
        wandb_module = getattr(trainer, "my_wandb", None)
        if wandb_module is not None:
            wandb_module.finish()

    # -- saving ----------------------------------------------------------- #
    def _save(self, trainer, pl_module, tag: str = ""):
        args = self.args
        epoch = args.epoch_begin + trainer.current_epoch
        suffix = f"-{tag}" if tag else ""
        path = f"{args.proj_dir}/rwkv-{epoch}{suffix}.pth"

        if args.peft == "mtp":
            self._save_mtp(trainer, pl_module, path)
            return

        if is_fsdp2(trainer.strategy):
            # collective on every rank, only rank 0 writes
            if args.peft not in ("none", "state") and args.merge != 1:
                # adapter-only saves stay cheap: no full-model gather
                state_dict = gather_adapter_state_dict(pl_module.model)
            else:
                state_dict = gather_model_state_dict(pl_module.model)
            if trainer.is_global_zero:
                self._save_full(args, path, state_dict)
            return

        if not trainer.is_global_zero:
            return
        if args.peft in ("none", "state"):
            state_dict = strip_prefix(pl_module.state_dict(), "model.")
            self._save_full(args, path, state_dict)
            return

        if args.merge != 1:
            adapter_dir = f"{args.proj_dir}-adapter"
            pl_module.model.save_pretrained(adapter_dir)
            print(f"✅ 已保存 PEFT adapter 参数到: {adapter_dir}/")
            return

        print("🚧 正在创建临时副本进行合并保存……")
        model_copy = copy.deepcopy(pl_module.model).to("cpu")
        try:
            model_copy.merge_and_unload()
            state_dict = strip_prefix(model_copy.state_dict(), "base_model.model.")
            torch.save(state_dict, path)
            print(f"✅ save to: {path}")
        finally:
            del model_copy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def _save_mtp(self, trainer, pl_module, path):
        """MTP runs keep the base model frozen: only the heads are saved."""
        mtp_path = path.replace(".pth", "-mtp.pth")
        if is_fsdp2(trainer.strategy):
            state = gather_model_state_dict(pl_module.mtp)
        else:
            state = {k: v.detach().cpu()
                     for k, v in pl_module.mtp.state_dict().items()}
        if not trainer.is_global_zero:
            return
        state = {(k if k.startswith("mtp.") else f"mtp.{k}"): v
                 for k, v in state.items()}
        torch.save(state, mtp_path)
        print(f"✅ save MTP weights to: {mtp_path}")

    def _save_full(self, args, path, state_dict):
        """Write a gathered/full state dict, handling the three PEFT modes."""
        if args.peft in ("none", "state"):
            if args.peft == "state":
                state_dict = {k: v for k, v in state_dict.items() if "state" in k}
            torch.save(state_dict, path)
            print(f"✅ save to: {path}")
            return

        if args.merge != 1:
            from rwkvt.fsdp import adapter_state_dict

            adapter_path = path.replace(".pth", "-adapter.pth")
            torch.save(adapter_state_dict(state_dict), adapter_path)
            print(f"✅ 已保存 PEFT adapter 参数到: {adapter_path}")
            return

        torch.save(merge_peft_state_dict(args, state_dict), path)
        print(f"✅ save to: {path}")




class AcceptanceCallback(pl.Callback):
    """Measure the speculative-decoding acceptance rate *during* training.

    Every ``--accept_every`` steps this runs the greedy draft/verify loop on
    ``--accept_prompts`` held-out windows taken from the tail of the binidx file
    (training only ever reads the head of the file, so the tail is unseen) and
    logs ``accept/rate`` and ``accept/tokens_per_forward`` to wandb plus
    ``<proj_dir>/accept_curve.jsonl``.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.every = getattr(args, "accept_every", 0)
        self.prompt_count = getattr(args, "accept_prompts", 8)
        self.prompt_tokens = getattr(args, "accept_tokens", 32)
        self.path = os.path.join(args.proj_dir, "accept_curve.jsonl")
        self.prompts = None
        self.enabled = self.every > 0 and getattr(args, "data_type", "") == "binidx"

    def _build_prompts(self):
        import numpy as np

        args = self.args
        tokens = np.memmap(args.data_file + ".bin", dtype=np.uint16, mode="r")
        span = 128
        need = self.prompt_count * span
        if len(tokens) < need:  # tiny corpora / smoke tests
            tail = np.tile(np.asarray(tokens, dtype=np.int64),
                           need // max(1, len(tokens)) + 1)[-need:]
        else:
            tail = np.asarray(tokens[-need:], dtype=np.int64)
        return [tail[i * span:(i + 1) * span] for i in range(self.prompt_count)]

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.enabled or pl_module.mtp is None:
            return
        step = trainer.global_step
        if step == 0 or step % self.every:
            return
        if self.prompts is None:
            self.prompts = self._build_prompts()

        from rwkvt.spec_decode import SpecStats, mtp_greedy_decode

        model = pl_module.model
        was_training = model.training
        model.eval()
        started = time.perf_counter()
        stats = SpecStats()
        for ids in self.prompts:
            tokens = torch.as_tensor(ids, device=pl_module.device).unsqueeze(0)
            mtp_greedy_decode(model, pl_module.mtp, tokens,
                              self.prompt_tokens, depth=self.args.mtp_depth,
                              stats=stats)
        model.train(was_training)
        elapsed = time.perf_counter() - started

        metrics = {"step": step, "seconds": round(elapsed, 2),
                   "accept_rate": round(stats.accept_rate, 4),
                   "tokens_per_forward": round(stats.tokens_per_round, 4),
                   "drafted": stats.drafted, "accepted": stats.accepted}
        if not trainer.is_global_zero:
            return
        print(f"[accept] step {step}  alpha {metrics['accept_rate']:.3f}  "
              f"tokens/forward {metrics['tokens_per_forward']:.3f}  "
              f"({metrics['accepted']}/{metrics['drafted']})  "
              f"{elapsed:.1f}s", flush=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        wandb_module = getattr(trainer, "my_wandb", None)
        if wandb_module is not None:
            wandb_module.log({f"accept/{k}": v for k, v in metrics.items()
                              if k not in ("step", "drafted", "accepted")},
                             step=int(step))


# backwards compatible aliases (the classes used to be lower-case)
train_callback = TrainCallback
accept_callback = AcceptanceCallback
