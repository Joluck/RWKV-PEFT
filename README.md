
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
sh scripts/run_lora.sh
```
Note: Please refer to the RWKV official tutorial for detailed data preparation


## Main Features

- **Multiple Fine-tuning Methods**: Supports LoRA, PISSA, Bone, State Tuning, etc.
- **Quantized Training**: Supports INT8/NF4 quantization for significant VRAM reduction
- **Flexible Data Loading**: Supports various data sampling strategies 
- **Memory Optimization**: Multiple DeepSpeed strategies available
- **Loss Masking**: Supports loss masking for QA dialogue and padding
- **Infinite Context Training**: Supports infctx training mode, utilizing RWKV's constant memory usage advantage to train with "infinite" context under limited resources
- **Multi-Hardware Support**: RWKV-PEFT officially supports NVIDIA, AMD, Moore Threads, Musa, Iluvatar CoreX, and other hardware platforms. Ascend NPU implementation will be available later. Note: Currently we only support issues for NVIDIA hardware
- **RWKV-FLA Efficient Training**: rwkv-fla is a Triton-based linear attention operator that can run efficiently on hardware without CUDA support

## Detailed Configuration

###  PEFT Method Selection
```bash
--peft lora --peft_config '{"r":8,"lora_alpha":32,"lora_dropout":0.05}'
```
[state,lora,miss]



###  Quantized Training
```bash
--quant int8/nf4
```

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

###  FLA Operator
By default, RWKV-PEFT uses custom CUDA kernels for wkv computation.
However, you can use `--op fla` to enable the Triton kernel:
```
--op fla
```

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