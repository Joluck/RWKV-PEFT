"""Lightning callbacks: LR/WD schedule, logging, checkpoint saving."""

import copy
import gc
import json
import math
import os
import time

import lightning as pl
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy

from rwkvt.fsdp import (
    gather_adapter_state_dict,
    gather_model_state_dict,
    is_fsdp2,
    merge_peft_state_dict,
)
from rwkvt.trick.lrs import cos_decay, wsd
from rwkvt.utils import strip_prefix


class TrainCallback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_file = os.path.join(args.proj_dir, "loss_data.jsonl")
        if os.path.exists(self.loss_file):
            os.remove(self.loss_file)

    # -- helpers ---------------------------------------------------------- #
    def write_data(self, loss_data, t_cost, kt_s):
        """Append one step of loss data (plotted by the streamlit helpers)."""
        with open(self.loss_file, "a") as f:
            json.dump({"loss": float(loss_data), "t_cost": t_cost, "kt_s": kt_s}, f)
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
            if args.wandb:
                print("Login to wandb...")
                import wandb
                wandb.init(project=args.wandb,
                           name=f"{args.run_name} {args.my_timestamp}",
                           config=vars(args), save_code=False, mode="offline")
                trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs

        if trainer.world_size > 1 and not isinstance(trainer.strategy, DeepSpeedStrategy):
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss = loss / trainer.world_size

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

        trainer.my_loss = float(loss) * trainer.accumulate_grad_batches / max(1, trainer.world_size)
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
        if args.wandb:
            metrics = {"loss": step_loss, "lr": trainer.my_lr, "wd": trainer.my_wd,
                       "Gtokens": trainer.global_step * token_per_step / 1e9}
            if kt_s > 0:
                metrics["kt/s"] = kt_s
            trainer.my_wandb.log(metrics, step=int(trainer.global_step))
        self.write_data(step_loss, t_cost, kt_s)

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

    # -- saving ----------------------------------------------------------- #
    def _save(self, trainer, pl_module):
        args = self.args
        epoch = args.epoch_begin + trainer.current_epoch
        path = f"{args.proj_dir}/rwkv-{epoch}.pth"

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


# backwards compatible alias (the class used to be lower-case)
train_callback = TrainCallback
