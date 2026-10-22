"""CLI / utility tests that do not need torch model execution."""

import importlib.util
import os

import pytest


def load_train_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "train_entry", os.path.join(root, "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults():
    train = load_train_module()
    args = train.build_parser().parse_args(["--load_model", "m.pth"])
    assert args.precision == "bf16"
    assert args.my_testing == "x070"
    assert args.data_type == "binidx"
    assert args.peft_config == {}


def test_peft_config_is_json_parsed():
    train = load_train_module()
    args = train.build_parser().parse_args(
        ["--load_model", "m.pth", "--peft", "lora",
         "--peft_config", '{"r": 8, "lora_alpha": 32}'])
    assert args.peft_config == {"r": 8, "lora_alpha": 32}


def test_finalize_rejects_fp16_with_cuda_kernel(tmp_path, restore_env):
    train = load_train_module()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    args = train.build_parser().parse_args(
        ["--load_model", str(ckpt), "--precision", "fp16"])
    with pytest.raises(SystemExit, match="fp16"):
        train.finalize_args(args)


def test_finalize_rejects_unknown_version(tmp_path, restore_env):
    train = load_train_module()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    args = train.build_parser().parse_args(
        ["--load_model", str(ckpt), "--my_testing", "x999"])
    with pytest.raises(SystemExit, match="my_testing"):
        train.finalize_args(args)


def test_finalize_derives_dimensions(tmp_path, restore_env):
    train = load_train_module()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    args = train.build_parser().parse_args(
        ["--load_model", str(ckpt), "--n_embd", "512", "--devices", "2",
         "--micro_bsz", "4", "--proj_dir", str(tmp_path / "out")])
    args = train.finalize_args(args)
    assert args.dim_att == 512
    assert args.dim_ffn % 32 == 0
    assert args.real_bsz == 8
    assert os.path.isdir(args.proj_dir)
    assert os.environ["RWKV_MY_TESTING"] == "x070"
    assert os.environ["WKV"] == "cuda"


def test_strip_prefix():
    from rwkvt.utils import strip_prefix

    state = {"model.emb.weight": 1, "model.blocks.0.w": 2, "other": 3}
    assert strip_prefix(state, "model.") == {"emb.weight": 1, "blocks.0.w": 2,
                                             "other": 3}


def test_parser_accepts_fsdp2_strategy():
    train = load_train_module()
    args = train.build_parser().parse_args(
        ["--load_model", "m.pth", "--strategy", "fsdp2"])
    assert args.strategy == "fsdp2"


def test_finalize_marks_fsdp2(tmp_path, restore_env):
    train = load_train_module()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    args = train.build_parser().parse_args(
        ["--load_model", str(ckpt), "--strategy", "fsdp2",
         "--proj_dir", str(tmp_path / "out")])
    args = train.finalize_args(args)
    assert args.fsdp2 is True


def test_finalize_rejects_fp16_with_fsdp2(tmp_path, restore_env):
    train = load_train_module()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    args = train.build_parser().parse_args(
        ["--load_model", str(ckpt), "--strategy", "fsdp2",
         "--precision", "fp16", "--proj_dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="fsdp2"):
        train.finalize_args(args)


def test_finalize_rejects_mtp_depth_longer_than_context(tmp_path, restore_env):
    train = load_train_module()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    args = train.build_parser().parse_args(
        ["--load_model", str(ckpt), "--peft", "mtp", "--mtp_depth", "8",
         "--ctx_len", "8", "--proj_dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="mtp_depth"):
        train.finalize_args(args)


def test_precision_bf16_true_is_kept(tmp_path, restore_env):
    train = load_train_module()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    args = train.build_parser().parse_args(
        ["--load_model", str(ckpt), "--precision", "bf16-true",
         "--proj_dir", str(tmp_path / "out")])
    args = train.finalize_args(args)
    assert args.precision == "bf16-true"
    assert os.environ["RWKV_FLOAT_MODE"] == "bf16"


def test_fsdp2_switches_bf16_to_bf16_true(tmp_path, restore_env):
    train = load_train_module()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"")
    args = train.build_parser().parse_args(
        ["--load_model", str(ckpt), "--strategy", "fsdp2",
         "--proj_dir", str(tmp_path / "out")])
    args = train.finalize_args(args)
    assert args.precision == "bf16-true"
