"""End-to-end wiring tests: DataModule + collator + TrainCallback + Trainer.

The data-path tests use a dummy LightningModule; the FSDP2 tests use a tiny
stubbed RWKV model and ``ModelParallelStrategy``. Everything runs on CPU, so the
module is skipped automatically when ``lightning`` is not installed.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pl = pytest.importorskip("lightning")

from test_dataset import write_binidx  # noqa: E402

import rwkvt.dataset.dataset as ds_module  # noqa: E402
from rwkvt.dataset.datamodule import MyDataModule  # noqa: E402
from rwkvt.lightning_train.trainer import TrainCallback  # noqa: E402


class _FakeTokenizer:
    pad_token_id = 0


class _DummyModel(pl.LightningModule):
    """Consumes batches and exercises the callback hooks."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            idx, targets, mask = batch
            assert mask.shape == idx.shape
            assert mask.dtype == torch.bool
        else:
            idx, targets = batch
        assert idx.dtype == torch.long and targets.dtype == torch.long
        return (self.p * 0 + 1.0).sum()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [{"params": self.parameters(), "weight_decay": 0.0, "my_lr_scale": 1.0}],
            lr=1e-3)


def make_args(tmp_path, data_type):
    proj_dir = tmp_path / "out"
    proj_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        proj_dir=str(proj_dir), wandb="", my_timestamp="test", run_name="smoke",
        lr_schedule="cos", lr_init=1e-3, lr_final=1e-3, epoch_count=1,
        warmup_steps=-1, weight_decay=0.0, weight_decay_final=-1.0,
        layerwise_lr=1, ctx_len=8, real_bsz=2, peft="none", merge=0,
        epoch_begin=0, epoch_save=0, micro_bsz=2, data_shuffle=0, num_workers=0,
        persistent_workers=False, prefetch_factor=None, data_type=data_type,
        data_file=str(tmp_path / "toy"), epoch_steps=4, sft_field=["q", "a"],
    )


def module_args(tmp_path, peft_mode="none", merge=1):
    """Full arg namespace for the real RWKV LightningModule."""
    import _cpu_stub as stub

    ns = SimpleNamespace(**vars(stub.build_args("7", train_type="none")))
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    extra = {
        "proj_dir": str(out), "wandb": "", "my_timestamp": "t", "run_name": "r",
        "lr_schedule": "cos", "lr_init": 1e-3, "lr_final": 1e-3,
        "epoch_count": 1, "warmup_steps": -1, "weight_decay_final": -1.0,
        "real_bsz": 2, "merge": merge, "epoch_begin": 0, "epoch_save": 0,
        "micro_bsz": 2, "data_shuffle": 0, "num_workers": 0,
        "persistent_workers": False, "prefetch_factor": None,
        "data_type": "binidx", "data_file": str(tmp_path / "toy"),
        "epoch_steps": 4, "dataload": "pad", "optimizer": "adamw",
        "betas": (0.9, 0.99), "adam_eps": 1e-8, "peft": peft_mode,
        "peft_config": {"r": 4}, "load_model": "", "load_partial": 0,
        "fsdp2": True, "mtp_depth": 1, "mtp_loss_weight": 0.3,
        "mtp_main_weight": 1.0, "mtp_load": "",
    }
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


def run_smoke(args, trainer_kwargs=None):
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, limit_train_batches=2,
        logger=False, callbacks=[TrainCallback(args)], enable_checkpointing=False,
        enable_progress_bar=False, num_sanity_val_steps=0, **(trainer_kwargs or {}))
    trainer.fit(_DummyModel(), MyDataModule(args))
    return trainer


# --------------------------------------------------------------------------- #
# data / callback wiring
# --------------------------------------------------------------------------- #
def test_sft_batches_reach_training_step(monkeypatch, tmp_path):
    rows = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4]), "labels": torch.tensor([4])},
            {"input_ids": torch.tensor([5, 6]), "labels": torch.tensor([5, 6])}]
    monkeypatch.setattr(ds_module, "sft_dataset",
                        lambda args: (rows, _FakeTokenizer()))
    args = make_args(tmp_path, "sft")
    trainer = run_smoke(args)

    assert trainer.global_step == 2
    assert (tmp_path / "out" / "loss_data.jsonl").exists()
    assert list((tmp_path / "out").glob("rwkv-*.pth"))  # saved on epoch end


def test_wandb_receives_metrics(monkeypatch, tmp_path):
    """--wandb wires init/log/finish with the requested mode."""
    import types as _types

    calls = {"init": [], "log": [], "finish": 0}
    fake = _types.ModuleType("wandb")
    fake.init = lambda **kw: calls["init"].append(kw)
    fake.log = lambda data, step=None: calls["log"].append((data, step))
    fake.finish = lambda: calls.__setitem__("finish", calls["finish"] + 1)
    monkeypatch.setitem(sys.modules, "wandb", fake)

    write_binidx(str(tmp_path / "toy"), [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11]])
    args = make_args(tmp_path, "binidx")
    args.dataload = "pad"
    args.wandb = "proj"
    args.wandb_mode = "offline"
    args.wandb_run_name = ""
    run_smoke(args)

    assert calls["init"], "wandb.init was not called"
    assert calls["init"][0]["project"] == "proj"
    assert calls["init"][0]["mode"] == "offline"
    assert calls["log"], "wandb.log was not called"
    assert "lr" in calls["log"][0][0] and "epoch" in calls["log"][0][0]
    assert calls["finish"] == 1


def test_binidx_batches_reach_training_step(tmp_path):
    write_binidx(str(tmp_path / "toy"), [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11]])
    args = make_args(tmp_path, "binidx")
    args.dataload = "pad"
    trainer = run_smoke(args)
    assert trainer.global_step == 2


# --------------------------------------------------------------------------- #
# real RWKV LightningModule (+ PEFT)
# --------------------------------------------------------------------------- #
def test_real_module_training_step_with_peft():
    """The reported failure: PEFT-wrapped RWKV7 + SFT attention mask."""
    pytest.importorskip("peft")
    import _cpu_stub as stub

    model, args = stub.load_model(version="7", mode="normal")
    args.data_type = "sft"
    args.peft = "lora"
    args.peft_config = {"r": 4}

    from rwkvt.lightning_train.light_rwkv import RWKV
    from rwkvt.peft_loading import apply_peft

    module = RWKV(args, model=apply_peft(model, args))
    idx = torch.randint(1, args.vocab_size, (2, args.ctx_len))
    targets = torch.randint(0, args.vocab_size, (2, args.ctx_len))
    mask = torch.ones(2, args.ctx_len, dtype=torch.bool)
    mask[:, -2:] = False

    loss = module.training_step((idx, targets, mask), 0)
    loss.backward()
    assert torch.isfinite(loss)
    adapter_grads = [p for p in module.parameters()
                     if p.requires_grad and p.grad is not None]
    assert adapter_grads


# --------------------------------------------------------------------------- #
# FSDP2
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if torch.distributed.is_initialized():
        pytest.skip("distributed already initialised")
    init_file = tmp_path_factory.mktemp("pg") / "shared_init"
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


def test_configure_model_shards_only_for_fsdp2(process_group):
    """FSDP2RWKV.configure_model() is a no-op unless the FSDP2 strategy is used."""
    import _cpu_stub as stub
    from torch.distributed.device_mesh import init_device_mesh

    model, args = stub.load_model(version="7", mode="normal")

    from rwkvt.fsdp import build_fsdp2_strategy
    from rwkvt.lightning_train.light_rwkv import FSDP2RWKV

    module = FSDP2RWKV(args, model=model)

    # a non-FSDP2 strategy leaves the model untouched
    module._trainer = SimpleNamespace(strategy=SimpleNamespace())
    module.configure_model()
    assert not isinstance(model.emb.weight, torch.distributed.tensor.DTensor)

    # the real FSDP2 strategy shards blocks and root
    strategy = build_fsdp2_strategy(1)
    strategy.parallel_devices = [torch.device("cpu")]  # normally set by the Trainer
    module._trainer = SimpleNamespace(strategy=strategy)
    module._device_mesh = init_device_mesh("cpu", (1,))
    module.configure_model()
    assert isinstance(model.emb.weight, torch.distributed.tensor.DTensor)
    assert isinstance(model.blocks[0].att.receptance.weight,
                      torch.distributed.tensor.DTensor)


def test_build_fsdp2_strategy_uses_pure_data_parallel():
    from rwkvt.fsdp import build_fsdp2_strategy

    strategy = build_fsdp2_strategy(devices=1, num_nodes=1)
    assert strategy._data_parallel_size == 1
    assert strategy._tensor_parallel_size == 1
    assert strategy._save_distributed_checkpoint is False


@pytest.mark.parametrize(("peft_mode", "merge"),
                         [("none", 0), ("lora", 0), ("lora", 1), ("mtp", 0)])
def test_fsdp2_end_to_end(peft_mode, merge, tmp_path):
    """Trainer + ModelParallelStrategy + FSDP2RWKV + checkpoint saving."""
    import _cpu_stub as stub

    from rwkvt.fsdp import build_fsdp2_strategy
    from rwkvt.lightning_train.light_rwkv import FSDP2RWKV

    write_binidx(str(tmp_path / "toy"), [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11]])
    args = module_args(tmp_path, peft_mode, merge)
    model, _ = stub.load_model(version="7", mode="normal")
    if peft_mode == "mtp":
        model.requires_grad_(False)
    elif peft_mode != "none":
        from rwkvt.peft_loading import apply_peft

        model = apply_peft(model, args)

    module = FSDP2RWKV(args, model=model)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, limit_train_batches=2,
        logger=False, callbacks=[TrainCallback(args)], enable_checkpointing=False,
        enable_progress_bar=False, num_sanity_val_steps=0,
        strategy=build_fsdp2_strategy(devices=1, num_nodes=1))
    trainer.fit(module, MyDataModule(args))

    assert trainer.global_step == 2
    if peft_mode == "mtp":
        pattern = "rwkv-*-mtp.pth"
    elif peft_mode != "none" and merge != 1:
        pattern = "rwkv-*-adapter.pth"
    else:
        pattern = "rwkv-*.pth"
    saved = list((tmp_path / "out").glob(pattern))
    assert saved, f"checkpoint was not written ({pattern})"
    state = torch.load(saved[0], map_location="cpu")
    assert state, "empty checkpoint"
    assert not any(isinstance(v, torch.distributed.tensor.DTensor)
                   for v in state.values())
    if peft_mode == "mtp":
        assert state and all(k.startswith("mtp.") for k in state)
        import json

        rows = [json.loads(line) for line in
                (tmp_path / "out" / "loss_data.jsonl").read_text().splitlines()]
        assert rows and "mtp_loss" in rows[0] and "main_loss" in rows[0]
        assert "mtp_loss_1" in rows[0]  # per-depth losses are logged too
    elif pattern == "rwkv-*.pth":
        assert "blocks.0.att.receptance.weight" in state


def test_load_peft_model_selects_fsdp2_module(tmp_path):
    """args.fsdp2 must pick the FSDP2 LightningModule at build time."""
    import _cpu_stub as stub

    model, args = stub.load_model(version="7", mode="normal")
    ckpt = tmp_path / "tiny.pth"
    torch.save({k: v.clone() for k, v in model.state_dict().items()}, ckpt)

    from rwkvt.lightning_train.light_rwkv import FSDP2RWKV, RWKV
    from rwkvt.peft_loading import load_peft_model

    args.load_model = str(ckpt)
    args.load_partial = 1
    args.peft = "none"
    args.peft_config = {}

    args.fsdp2 = False
    _, plain = load_peft_model(args)
    assert isinstance(plain, RWKV) and not isinstance(plain, FSDP2RWKV)

    args.fsdp2 = True
    _, sharded = load_peft_model(args)
    assert isinstance(sharded, FSDP2RWKV)


def test_mtp_training_step_freezes_base():
    """--peft mtp: only the heads are trainable and they do get gradients."""
    import _cpu_stub as stub

    model, args = stub.load_model(version="7", mode="normal")
    args.data_type = "binidx"
    args.peft = "mtp"
    args.mtp_depth = 1
    args.mtp_loss_weight = 0.3
    args.mtp_main_weight = 1.0
    args.mtp_load = ""
    model.requires_grad_(False)

    from rwkvt.lightning_train.light_rwkv import RWKV

    module = RWKV(args, model=model)
    assert module.mtp is not None
    idx = torch.randint(1, args.vocab_size, (2, args.ctx_len))
    targets = idx.roll(-1, dims=1)

    loss = module.training_step((idx, targets), 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in module.model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in module.mtp.parameters())


def test_save_mtp_without_fsdp(tmp_path):
    """--peft mtp writes only the heads, with a `mtp.` prefix."""
    from torch import nn

    args = make_args(tmp_path, "binidx")
    args.peft = "mtp"
    args.epoch_begin = 0

    module = SimpleNamespace(mtp=nn.Linear(2, 2))
    trainer = SimpleNamespace(is_global_zero=True, current_epoch=3,
                              strategy=SimpleNamespace())
    TrainCallback(args)._save(trainer, module)

    saved = list((tmp_path / "out").glob("rwkv-*-mtp.pth"))
    assert saved, "MTP checkpoint missing"
    state = torch.load(saved[0], map_location="cpu")
    assert state and all(k.startswith("mtp.") for k in state)


def test_fsdp2_with_bf16_true(tmp_path):
    """The reported failure: fp32 checkpoint + bf16-mixed -> kernel dtype assert.

    With ``precision='bf16-true'`` the whole model is cast before/around
    sharding, so every kernel input is bf16.
    """
    import _cpu_stub as stub

    from rwkvt.fsdp import build_fsdp2_strategy
    from rwkvt.lightning_train.light_rwkv import FSDP2RWKV

    write_binidx(str(tmp_path / "toy"), [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11]])
    args = module_args(tmp_path, "mtp", 0)
    args.precision = "bf16-true"
    model, _ = stub.load_model(version="7", mode="normal")
    model.requires_grad_(False)

    module = FSDP2RWKV(args, model=model)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, limit_train_batches=2,
        precision="bf16-true", logger=False, callbacks=[TrainCallback(args)],
        enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, strategy=build_fsdp2_strategy(1, 1))
    trainer.fit(module, MyDataModule(args))

    assert trainer.global_step == 2
    assert next(module.model.parameters()).dtype == torch.bfloat16
    assert next(module.mtp.parameters()).dtype == torch.bfloat16


def test_logged_loss_is_not_divided_twice(monkeypatch):
    """A 4-rank run must report the mean loss, not mean/world_size."""
    from types import SimpleNamespace

    import rwkvt.lightning_train.trainer as tr

    calls = []

    def fake_all_reduce(tensor, **kwargs):
        calls.append(1)
        tensor.mul_(4)  # every rank holds the same value -> sum is 4x

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    trainer = SimpleNamespace(world_size=4, accumulate_grad_batches=1)
    assert tr.reduce_logged_loss(trainer, torch.tensor(2.0)) == pytest.approx(2.0)
    assert calls

    single = SimpleNamespace(world_size=1, accumulate_grad_batches=2)
    assert tr.reduce_logged_loss(single, torch.tensor(2.0)) == pytest.approx(4.0)


def test_accept_callback_disabled_without_flag(tmp_path):
    from rwkvt.lightning_train.trainer import AcceptanceCallback

    args = module_args(tmp_path, "mtp", 0)
    args.accept_every = 0
    assert not AcceptanceCallback(args).enabled

    args.accept_every = 10
    args.data_type = "binidx"
    args.data_file = str(tmp_path / "toy")
    callback = AcceptanceCallback(args)
    assert callback.enabled
    write_binidx(str(tmp_path / "toy"), [[1, 2, 3, 4, 5]] * 64)
    prompts = callback._build_prompts()
    assert len(prompts) == callback.prompt_count
    assert all(len(p) == 128 for p in prompts)
