<h1 align="center">
  <p><img src="assert/logo.jpg" alt="RWKV-PEFT" width="60px"  style="vertical-align: middle; margin-right: 10px;"/>RWKV-PEFT</p>
</h1>

\[ [English](README.md) | 中文 \]


RWKV-PEFT 是一个旨在为 RWKV 模型实现高效参数微调的官方实现，支持在多种硬件上实现多种先进的微调方法。

## 最近更新：支持 v7 和一些代码调整

1. 移除了 `--fla` 并增加了 `--op cuda/fla/triton`。现在你可以在 `--op` 中自由地选择算子。默认推荐 cuda，如果你想要使用 state tuning 请设置 `--op fla` 和 `--train_type state`。
2. 把 `Bone` 名称修改为 `DiSHA`，注意 DiSHA 中的 rank 的参数量只有 LoRA 的一半，因此同等参数下 DiSHA(r)=2*LoRA(r)，例如 ` disha_config='{"mode":"bone","load":"","r":64}' `  
> [!TIP]
> DiSHA 微调可选择 `bone` 或者 `bat` 模式，推荐使用 `bone` 模式，训练效果好且资源占用较低
1. 模型代码更干净且容易迁移，详细查看文件 `rwkvt`
2. 移除了简易的可视化训练，后续会有专门的程序支持可视化训练
3. 新增 `lr_schedule` 参数，默认使用 `cos_decay`。可以通过 `--lr_schedule wsd` 使用余弦退火
4. 可以通过设置训练参数 `--my_testing "x070"` 微调 RWKV-7 模型

## SFT 训练

支持 SFT 训练，相关参数和用法请参考 `scripts/run_sft.sh` 

SFT 训练的参数解释：

- `--data_file 'meta-math/MetaMathQA'`：数据路径，可直接选择 Hugging Face 路径，也可选择自己的 json 文件路径  
- `--data_type sft`：选择数据类型  
- `--sft_field query response`：根据 json 中的问答格式进行检索  
- `--sft_split "train"`：设置加载的训练数据量，"train" 全部加载，"train[:1000]"只加载 1000 条数据  

SFT 训练参数参考：

```
--data_type sft --sft_field query response --sft_split "train"
```

### SFT 训练具体设置

SFT 训练的代码在 `RWKV-PEFT/src/rwkv_datasets/SFTdataset.py` 中，具体设置如下：

```
tokenizer_path = 'RWKV/rwkv-5-world-3b' #选择分词器（请选择官方分词器）
IGNORE_INDEX = -100 #填充（请勿修改）
EOT_TOKEN = "\x17" #设置你需要的停止符

# 根据需求修改对应的prompt
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
```
> [!TIP]
> 中国网络下载 Hugging Face 数据会超时，国内用户需要添加 Hugging Face 镜像：`HF_ENDPOINT="https://hf-mirror.com" sh scripts/run_sft.sh`

## DiSHA: Dimension-Sharding Adaptation of Large Language Models with Fast Convergence and Fast Computation [📖 Paper](https://arxiv.org/pdf/2409.15371)

论文更新，现在 DiSHA（bone）是一个简单高效基础 PEFT 方法，比 LoRA 更快更省显存，比 PiSSA 收敛更快表现更好。
 
- DiSHA(Bone)：`disha_config='{"mode":"bone","load":"","r":64}'` 
- DiSHA(Bat)：`disha_config='{"mode":"bat","load":"","r":64}'`

## Web Run
> [!TIP]
> Coming Soon!

## 目录

- [主要特性](#主要特性)
- [硬件需求](#硬件需求)
- [安装训练环境](#安装训练环境)
- [快速开始](#快速开始)
- [详细配置说明](#详细配置说明)
- [GPU支持情况](#gpu支持情况)
- [引用](#引用)

## 主要特性

- **多种微调方法**：支持 LoRA、PISSA、DiSHA、State Tuning、SFT 等
- **量化训练**：支持 INT8/NF4 量化，显著降低显存占用
- **灵活的数据加载**：支持多种数据采样策略
- **显存优化**：多种 DeepSpeed 策略可选
- **损失Mask**：支持 QA 对话和填充部分的损失 Mask
- **无限长度训练**：支持 infctx 训练模式, 此模式利用了 RWKV 恒定显存占用的优势，在有限的资源下训练"无限"上下文
- **支持多种硬件**：目前，RWKV-PEFT 官方支持 NVIDIA、AMD、摩尔线程、沐曦、天数智芯等多种硬件平台, 昇腾 NPU 的实现会在后期实现。注意：目前我们只支持 NVIDIA 的 issue 请求。
- **使用 rwkv-fla 高效训练**: rwkv-fla 是基于 triton 的线性注意力算子，可以在不支持 cuda 的硬件上高效率运行。

## 硬件需求

### RWKV-7 模型

以下是使用 RTX 4090 (24GB 显存) + 64GB 内存测试的 RWKV-7 模型微调显存占用数据，基于以下参数配置：

- 训练精度：BF16
- `--strategy deepspeed_stage_1`
- `--ctx_len 1024`
- `--micro_bsz 1`
- `--lora_r 64` 或者 `disha_config='{"mode":"bone","r":32}'`

| 模型参数      | State Tuning | LoRA | DiSHA | PiSSA |
|--------------|--------------|------|-------|-------|
| RWKV7-0.1B   | 2.6 GB        | 2.7 GB  | 2.7 GB   | 2.6 GB   |
| RWKV7-0.4B   | 3.1 GB         | 3.4 GB  | 3.1 GB   | 3.4 GB   |
| RWKV7-1.5B   | 5.3 GB         | 5.6 GB  | 5.6 GB   | 5.6 GB   |
| RWKV7-3B     | 8.2 GB         | 8.8 GB  | 8.8 GB   | 8.8 GB   |

<details>
<summary>🔍 <b>点击查看 RWKV-7 模型量化训练的显存需求</b> </summary>

### INT8 显存需求

| 模型参数      | State Tuning | LoRA | DiSHA | PiSSA |
|--------------|--------------|------|-------|-------|
| RWKV7-0.1B   | 2.4 GB        | 2.5 GB  | 2.5 GB   | 2.5 GB   |
| RWKV7-0.4B   | 2.9 GB         | 2.9 GB  | 2.9 GB   | 3.0 GB   |
| RWKV7-1.5B   | 4.1 GB         | 4.6 GB  | 4.5 GB   | 4.6 GB   |
| RWKV7-3B     | 5.7 GB         | 6.7 GB  | 6.7 GB   | 6.7 GB   |

### NF4 显存需求

| 模型参数      | State Tuning | LoRA | DiSHA | PiSSA |
|--------------|--------------|------|-------|-------|
| RWKV7-0.1B   | 2.5 GB         | 2.4 GB  | 2.4 GB   | 2.4 GB   |
| RWKV7-0.4B   | 2.8 GB         | 2.7 GB  | 2.7 GB   | 2.7 GB    |
| RWKV7-1.5B   | 3.7 GB         | 3.9 GB  | 3.9 GB   | 3.9 GB      |
| RWKV7-3B     | 4.7 GB         | 5.7 GB  | 5.7 GB   | 5.7 GB      |

</details>

<details>
<summary>🔍 <b>点击查看 RWKV-6 模型的微调显存需求</b> </summary>

以下是使用 RTX 4090 (24GB 显存) + 64GB 内存测试的 RWKV-6 模型微调显存占用数据，基于以下参数配置：
- `--strategy deepspeed_stage_1`
- `--ctx_len 1024`
- `--micro_bsz 1`
- `--lora_r 64`

|   模型规模   | 全量微调 | LoRA/PISSA | QLoRA/QPISSA | State Tuning |
|-------------|----------|------------|--------------|--------------|
| RWKV6-1.6B  | 显存溢出   | 7.4 GB      | 5.6 GB        | 6.4 GB        |
| RWKV6-3B    | 显存溢出   | 12.1 GB     | 8.2 GB        | 9.4 GB        |
| RWKV6-7B    | 显存溢出   | 23.7 GB$^a$    | 14.9 GB$^b$     | 18.1 GB       |

> [!TIP]
> $^a$ 批次大小为 8 时会显存溢出
> 
> $^b$ 批次大小为 8 时需要 19.5GB 显存

</details>

## 安装训练环境

> [!IMPORTANT]
> 不可跳过

```bash
git clone https://github.com/JL-er/RWKV-PEFT.git
cd RWKV-PEFT
pip install -r requirements.txt
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行示例脚本：
```bash
sh scripts/run_lora.sh
```
> [!TIP]
> 具体数据准备方法请参考 [RWKV 官方教程](https://rwkv.cn/tutorials/advanced/Fine-Tune/FT-Dataset)


## 详细配置说明

### 1. PEFT 方法选择

PEFT 方法统一通过 huggingface/PEFT 的 `--peft` + `--peft_config` 指定：

```bash
--peft lora --peft_config '{"r":8,"lora_alpha":32,"lora_dropout":0.05}'
--peft miss --peft_config '{"r":8}'
--peft state --op fla          # state tuning，只训练 *_state 参数
```

### 2. 无限长度训练（infctx）

```bash
--train_type infctx --chunk_ctx 512 --ctx_len 2048
```
- ctx_len：目标训练长度
- chunk_ctx：切片长度，需小于ctx_len

### 3. 数据加载策略

```bash
--dataload pad
```
- pad：不足 `ctx_len` 时回绕到数据集开头补齐（默认）
- only：不足 `ctx_len` 时直接截断

### 4. DeepSpeed 策略

```bash
--strategy deepspeed_stage_1
```
可选策略：
- deepspeed_stage_1：优先使用
- deepspeed_stage_2/3：大模型或全量微调时使用
- deepspeed_stage_2_offload
- deepspeed_stage_3_offload

### 5. FSDP2（base 权重单卡放不下时推荐）

```bash
--strategy fsdp2 --devices 8 --merge 0
```

- 用 `torch.distributed.fsdp.fully_shard` 按 `Block` 切分（冻结的 base 也会被切开，这是 LoRA 场景下 ZeRO-1/2 做不到的）
- 保存时自动聚合成单文件；`--merge 0` 只存 adapter，省内存
- 需要 `torch>=2.4`、`lightning>=2.5`，精度用 `bf16`
- 13B 示例：`bash scripts/run_lora_13b_fsdp2.sh`（`DEVICES=4` 可改卡数，`DRY_RUN=1` 只打印命令）
- 实现细节与限制：[docs/fsdp2评估.md](docs/fsdp2评估.md)

### 5. FLA 算子

默认情况下， RWKV-PEFT 会使用自定义的 cuda 内核来实现 wkv 计算。 但您也可以使用`--op fla`来开启 Triton 内核。

rwkv-fla 是基于 triton 的线性注意力算子，可以在不支持 cuda 的硬件上高效率运行。

```
--op fla
```

## 开发与测试

```bash
pip install -e ".[dev]"     # ruff + pytest
ruff check .                # 代码规范检查（配置见 pyproject.toml）
pytest -q                   # CPU 冒烟测试，无需 GPU
pre-commit install          # 可选：提交前自动跑 ruff
```

- [docs/整改计划.md](docs/整改计划.md)：结构审查、问题清单、整改路线图、代码规范
- [docs/fsdp2评估.md](docs/fsdp2评估.md)：FSDP2 实现说明、已验证范围与限制

## GPU 支持情况

- NVIDIA: CUDA
- Intel、摩尔线程、沐曦、天数智芯: FLA , 这意味着你需要手动设置 `--op fla`
- 昇腾: CANN(soon)

## 引用

如果您觉得本项目对您有帮助，请引用我们的工作：

```bib
@misc{kang2025dishadimensionshardingadaptationlarge,
      title={DiSHA: Dimension-Sharding Adaptation of Large Language Models with Fast Convergence and Fast Computation}, 
      author={Jiale Kang},
      year={2025},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}