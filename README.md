
<h1 align="center">
  <p><img src="assert/logo.jpg" alt="RWKV-PEFT" width="60px"  style="vertical-align: middle; margin-right: 10px;"/>RWKV-PEFT</p>
</h1>

\[ English | [中文](README_zh.md) \]

RWKV-PEFT is the official implementation for efficient parameter fine-tuning of RWKV models, supporting various advanced fine-tuning methods across multiple hardware platforms.

# Recent updates
## Support [huggingface/PEFT](https://github.com/huggingface/peft)
You only need to check the usage examples of different methods in **PEFT**, then input the corresponding **name** and **config** correctly 

`LoRA:`
```
--peft lora --peft_config '{"r":8,"lora_alpha":32,"lora_dropout":0.05}'
```
`MiSS:`
```
--peft miss --peft_config '{"r":8}'
```
> [!IMPORTANT]
> state tuning 
```
--peft state --op fla
```


## MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure [Paper](https://arxiv.org/pdf/2409.15371)
The method **Bone/DiSHA** has been officially renamed to **MiSS**.
You can easily use it within **PEFT** (you’ll still see “Bone” for now, but it will be removed in future versions, so please use **MiSS** instead).



# Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone https://github.com/JL-er/RWKV-PEFT.git
cd RWKV-PEFT
uv sync   or  pip install .
```

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Main Features](#main-features)
- [Detailed Configuration](#detailed-configuration)
- [GPU Support](#gpu-support)
- [Citation](#citation)

## Hardware Requirements

### RWKV-7 Models

Below is the RWKV-7 model fine-tuned video memory requirement data, tested with RTX 4090 (24GB video memory) + 64GB RAM, based on the following parameter configurations:

- Training precision: BF16
- `--strategy deepspeed_stage_1`
- `--ctx_len 1024`
- `--micro_bsz 1`
- `--lora_r 64` or `disha_config='{"mode":"bone","r":32}'`

| Model Parameters | State Tuning | LoRA | DiSHA | PiSSA |
|------------------|--------------|------|-------|-------|
| RWKV7-0.1B       | 2.6 GB       | 2.7 GB  | 2.7 GB   | 2.6 GB   |
| RWKV7-0.4B       | 3.1 GB       | 3.4 GB  | 3.1 GB   | 3.4 GB   |
| RWKV7-1.5B       | 5.3 GB       | 5.6 GB  | 5.6 GB   | 5.6 GB   |
| RWKV7-3B         | 8.2 GB       | 8.8 GB  | 8.8 GB   | 8.8 GB   |

<details>
<summary>🔍 <b>Click to view the VRAM requirements for quantized training of RWKV-7 models</b> </summary>

### INT8 VRAM Requirements

| Model Parameters | State Tuning | LoRA | DiSHA | PiSSA |
|------------------|--------------|------|-------|-------|
| RWKV7-0.1B       | 2.4 GB       | 2.5 GB  | 2.5 GB   | 2.5 GB   |
| RWKV7-0.4B       | 2.9 GB       | 2.9 GB  | 2.9 GB   | 3.0 GB   |
| RWKV7-1.5B       | 4.1 GB       | 4.6 GB  | 4.5 GB   | 4.6 GB   |
| RWKV7-3B         | 5.7 GB       | 6.7 GB  | 6.7 GB   | 6.7 GB   |

### NF4 VRAM Requirements

| Model Parameters | State Tuning | LoRA | DiSHA | PiSSA |
|------------------|--------------|------|-------|-------|
| RWKV7-0.1B       | 2.5 GB       | 2.4 GB  | 2.4 GB   | 2.4 GB   |
| RWKV7-0.4B       | 2.8 GB       | 2.7 GB  | 2.7 GB   | 2.7 GB   |
| RWKV7-1.5B       | 3.7 GB       | 3.9 GB  | 3.9 GB   | 3.9 GB   |
| RWKV7-3B         | 4.7 GB       | 5.7 GB  | 5.7 GB   | 5.7 GB   |

</details>

<details>
<summary>🔍 <b>Click to view the VRAM requirements of RWKV-6 models</b> </summary>


The following shows memory usage when using an RTX 4090 (24GB VRAM) + 64GB RAM (with parameters: `--strategy deepspeed_stage_1 --ctx_len 1024 --micro_bsz 1 --lora_r 64`):

|   Model Size   | Full Finetuning | LoRA/PISSA | QLoRA/QPISSA | State Tuning |
|---------------|-----------------|------------|--------------|--------------|
| RWKV6-1.6B    | OOM            | 7.4 GB      | 5.6 GB        | 6.4 GB        |
| RWKV6-3B      | OOM            | 12.1 GB     | 8.2 GB        | 9.4 GB        |
| RWKV6-7B      | OOM            | 23.7 GB*    | 14.9 GB**     | 18.1 GB       |

Note:
* OOM when batch size is 8
** Requires 19.5GB VRAM when batch size is 8

</details>

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run example script:
```bash
bash scripts/lora.sh
```
Always pass `--my_testing` (`x050`/`x060`/`x070`) and prefer `--precision bf16`;
`fp16` is rejected by the bundled CUDA kernels.
Note: Please refer to the RWKV official tutorial for detailed data preparation


## Main Features

- **Multiple Fine-tuning Methods**: LoRA / PiSSA / MiSS / state tuning through huggingface/PEFT
- **Flexible Data Loading**: binidx (mmap, `pad`/`only` sampling), jsonl and SFT datasets
- **Memory Optimization**: DeepSpeed ZeRO strategies, gradient checkpointing, meta-device weight loading
- **Loss Masking**: prompt tokens are masked out automatically for SFT
- **Multi-Token Prediction (MTP)**: `--peft mtp` freezes the base model and trains
  DeepSeek-V3-style MTP heads that predict the next `--mtp_depth` tokens
- **FSDP2**: `--strategy fsdp2` shards the frozen base weights across GPUs
  (`fully_shard`), with aggregated checkpoint saving; see [docs/fsdp2评估.md](docs/fsdp2评估.md)
- **Infinite Context Training**: Supports infctx training mode, utilizing RWKV's constant memory usage advantage to train with "infinite" context under limited resources
- **Multi-Hardware Support**: RWKV-PEFT officially supports NVIDIA, AMD, Moore Threads, Musa, Iluvatar CoreX, and other hardware platforms. Ascend NPU implementation will be available later. Note: Currently we only support issues for NVIDIA hardware
- **RWKV-FLA Efficient Training**: rwkv-fla is a Triton-based linear attention operator that can run efficiently on hardware without CUDA support

## Detailed Configuration

###  PEFT Method Selection
```bash
--peft lora --peft_config '{"r":8,"lora_alpha":32,"lora_dropout":0.05}'
```
[state,lora,miss]



### Infinite Length Training (infctx)
```bash
--train_type infctx --chunk_ctx 512 --ctx_len 2048
```
- ctx_len: Target training length
- chunk_ctx: Slice length, must be smaller than ctx_len



### DeepSpeed Strategy
```bash
--strategy deepspeed_stage_1
```
Available strategies:
- deepspeed_stage_1: Preferred option
- deepspeed_stage_2/3: For large models or full fine-tuning
- deepspeed_stage_2_offload
- deepspeed_stage_3_offload

### Multi-Token Prediction (frozen base)

```bash
--peft mtp --mtp_depth 1 --mtp_loss_weight 0.3
```

- every base parameter is frozen; only the MTP heads (`--mtp_depth` blocks) train
- depth `k` fuses the previous hidden state with the embedding of token `i+k`,
  runs a channel-mix block and reuses the frozen LM head to predict token `i+k+1`
- loss = `mtp_loss_weight * mean_k(CE_k) + mtp_main_weight * CE(next token)`;
  with `mtp_main_weight=0` the weight is only a global scale (Adam is scale-invariant)
  `--mtp_main_weight` defaults to 0 because the frozen base cannot learn, and both
  terms are written to `loss_data.jsonl` as `main_loss` / `mtp_loss`
- checkpoints contain just the heads (`rwkv-N-mtp.pth`); resume with `--mtp_load`
- MTP heads are training-time only: drop them at inference, the base is unchanged
- draft/verify decoder + acceptance stats: `rwkvt/spec_decode.py`;
  design notes and the vLLM comparison: [docs/mtp.md](docs/mtp.md)
- ready-to-run 13B script: `bash scripts/mtp.sh` (`DRY_RUN=1` only prints the command;
  wandb is on by default - `WANDB=` disables it, `WANDB_MODE=offline` keeps it local)

### FSDP2 (recommended when the base weights do not fit per GPU)

```bash
--strategy fsdp2 --devices 8 --merge 0
```

- shards every transformer block (frozen base included) with `torch.distributed.fsdp.fully_shard`
- checkpoints are gathered to a single file on rank 0; `--merge 0` saves the adapter only
- requires `torch>=2.4` and `lightning>=2.5`; FSDP2 automatically uses
  `--precision bf16-true` (a plain `bf16` only enables autocast and would keep
  an fp32 checkpoint in fp32, which the WKV kernels reject)
  ```bash
  bash scripts/run_13b_fsdp2.sh                  # 13.3B + LoRA + MetaMathQA
  bash scripts/mtp.sh                            # 13.3B + MTP (base frozen)
  ```
- details and limitations: [docs/fsdp2评估.md](docs/fsdp2评估.md)

###  Operator
By default, RWKV-PEFT uses custom CUDA kernels for wkv computation.
However, you can use `--op fla` to enable the Triton kernel:
```
--op cuda/fla
```

### Troubleshooting

| symptom | cause / fix |
| --- | --- |
| `Fatal glibc error: pthread_mutex_lock ... __owner == 0` | forking a DataLoader/tokenizer worker after CUDA+NCCL started. Run with `NUM_WORKERS=0` (default in the 13B scripts) or `NUM_WORKERS=8 WORKER_START_METHOD=spawn PERSISTENT_WORKERS=1` |
| `AssertionError` in `rwkvop.py` about bf16 | the checkpoint is fp32 and `--precision bf16` only enables autocast; use `--precision bf16-true` (FSDP2 does it automatically) |
| `TypeError: ... unexpected keyword argument 'inputs_embeds'` | old PEFT path; this repo handles HF kwargs via `drop_unsupported_kwargs` |

## Development

```bash
pip install -e ".[dev]"     # ruff + pytest
ruff check .                # lint (config in pyproject.toml)
pytest -q                   # CPU smoke tests, no GPU required
pre-commit install          # optional: run ruff on commit
```

- [docs/整改计划.md](docs/整改计划.md)：代码结构审查、问题清单、分阶段整改路线图、代码规范
- [docs/fsdp2评估.md](docs/fsdp2评估.md)：FSDP2 实现说明、已验证范围与限制

## GPU Support

- NVIDIA: CUDA
- Intel, Moore Threads, Musa, Iluvatar CoreX: FLA, which means you need to pass `--fla`
- Ascend: CANN (soon)

## Citation

If you find this project helpful, please cite our work:
```bib
@misc{kang2025missrevisitingtradeofflora,
      title={MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure}, 
      author={Jiale Kang and Qingyu Yin},
      year={2025},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 

}
