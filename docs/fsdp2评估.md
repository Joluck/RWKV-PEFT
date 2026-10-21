# FSDP2 实现与可行性评估

> 状态：**已实现**（`--strategy fsdp2`），CPU 端到端通过，多卡真机待回归
> 实测环境：torch 2.9.1 / 2.8.0、Lightning 2.6.6、peft 0.19~0.21、Python 3.12
> 代码：`rwkvt/fsdp.py`、`light_rwkv.py::FSDP2RWKV`、`trainer.py::_save`

---

## 1. 用法

```bash
python train.py \
  --load_model /path/rwkv7.pth --data_file /path/data --data_type binidx \
  --n_layer 32 --n_embd 2560 --vocab_size 65536 \
  --peft lora --peft_config '{"r":16,"lora_alpha":32}' \
  --strategy fsdp2 --devices 8 \
  --micro_bsz 2 --ctx_len 1024 --precision bf16 --grad_cp 1 \
  --my_testing "x070"
```

- `--strategy fsdp2` → 内部构造 Lightning 的 `ModelParallelStrategy`：
  `data_parallel_size = devices × num_nodes`、`tensor_parallel_size = 1`、
  `save_distributed_checkpoint = False`（保存时聚合成单文件）。
- 其余参数不变；`--peft none/state/lora/miss` 均支持，`prefix` 未验证（会告警）。
- `--precision fp16` 被拒绝（FSDP2 下 master weight 不稳，且 WKV CUDA kernel 只吃 bf16）。

## 2. 实现要点

| 文件 | 作用 |
| --- | --- |
| `rwkvt/fsdp.py::build_fsdp2_strategy` | 构造 `ModelParallelStrategy`，修掉 Lightning 默认 `tp=本机 GPU 数` 的坑 |
| `rwkvt/fsdp.py::apply_fsdp2` | 逐 `Block` → 根模块 两层 `fully_shard`；参数**逐块搬到 device 后立刻分片**（单卡峰值 = 一个 Block）；`MixedPrecisionPolicy(param_dtype=None, reduce_dtype=fp32)` |
| `light_rwkv.py::FSDP2RWKV` | 在 `configure_model()` 钩子里做分片（Lightning 要求必须 override） |
| `fsdp.py::gather_model_state_dict` | `get_model_state_dict(full_state_dict=True, cpu_offload=True)`，所有 rank 参与、rank0 落盘 |
| `fsdp.py::merge_peft_state_dict` | FSDP2 下无法就地 `merge_and_unload`：用聚合权重在 CPU 重建后 merge |
| `trainer.py::_save` | 检测到 FSDP2 时走聚合保存；普通策略保持原路径 |

**为什么逐 block 分片**：每层是结构一致的 `Block`，除 `emb`/`head` 外所有参数都在块内；
先 shard 子块再 shard 根，根只负责 `emb`/`head`/adapter 这类零散参数。PEFT 的 LoRA 层在块内，跟着一起被切。

**为什么逐块搬 device**：`fully_shard` 要求参数已在 mesh 设备上（`_init_sharded_param` 会 assert），
但「整模型先 `.to(cuda)` 再分片」会让单卡峰值 = 整个模型（13B 直接 OOM）。
所以顺序是：block → `.to(device)` → `fully_shard(block)`，最后再把 `emb`/`head`/根部 adapter 搬过去并分片根。
单卡峰值只有「一个 Block + emb/head」，与最终分片后的常驻显存同量级。

**为什么保存必须聚合**：FSDP2 下 `state_dict()` 返回 DTensor，`torch.save` 会把
「本 rank 的 shard」当成完整权重写盘；`merge_and_unload()` 在 rank0 也只能合并自己那份。
所以训练时 shard、保存时 `full_state_dict=True` 聚合，两条路径分开。

## 3. 已验证 / 未验证

### 已验证（CPU + gloo 单进程）

| 用例（`tests/test_fsdp2.py`、`tests/test_lightning_smoke.py`） | 覆盖点 |
| --- | --- |
| `test_fully_shard_forward_and_step` | 分片后前向 / 反向 / `optimizer.step()` |
| `test_apply_fsdp2_shards_blocks_and_root` | 块与根都变成 `Shard(dim=0)` DTensor |
| `test_gather_model_state_dict_restores_plain_tensors` | 聚合结果与未分片模型**逐位相同**，且是普通 tensor |
| `test_merge_peft_state_dict_matches_reference` | 聚合 + CPU 重建 merge 的结果与 `merge_and_unload()` 一致 |
| `test_configure_model_shards_only_for_fsdp2` | 非 FSDP2 策略下 hook 是 no-op |
| `test_fsdp2_end_to_end[none/lora]` | **真 `Trainer` + `ModelParallelStrategy` + `FSDP2RWKV` + `TrainCallback`**：2 step 训练 + epoch 末保存，落盘为普通 tensor |

### 未验证（需要真机）

- 多卡 NCCL 下的吞吐 / 显存曲线（开发容器无 GPU）；
- 多机（`num_nodes > 1`）的 mesh 与 sampler；
- 13B 级别的 `--merge 1` 保存（会额外占一份 CPU 内存，见 §4）。

## 4. 已知限制

| 限制 | 说明 / 规避 |
| --- | --- |
| `--merge 1` + PEFT 保存时重建 CPU 模型 | 与旧的非 FSDP2 路径同量级（一份 CPU 权重）；省内存可用 `--merge 0` 只存 adapter |
| 中间 epoch 保存会做一次全量聚合 | 建议 `--merge 0` + 较大的 `--epoch_save`，只在最后合并 |
| Lightning `ModelParallelStrategy` 是实验 API | 版本敏感（要求 ≥2.5），升级 Lightning 后需回归 §3 的用例 |
| `--peft prefix` | 未测试，启动时告警 |
| 自定义 WKV autograd Function | 已验证可用（只接触激活，不持有参数） |

## 5. 为什么值得做

LoRA 只训 adapter，DeepSpeed ZeRO-1/2 主要省优化器状态，**冻结的 base 权重仍每卡一份**；
FSDP2 直接把 base 切成 1/N：

| 模型 | bf16 权重 | DDP/ZeRO-1 每卡 | FSDP2 8 卡每卡 |
| --- | --- | --- | --- |
| 7B | 14 GB | 14 GB | ~1.8 GB |
| 13B | 26 GB | 26 GB | ~3.3 GB |
| 32B | 64 GB | 64 GB | ~8 GB |

（不含激活；激活仍随 `micro_bsz × ctx_len` 增长。）

## 6. 选型建议

| 场景 | 建议 |
| --- | --- |
| 单卡 / ≤1.5B LoRA | 保持 `auto` 或 `deepspeed_stage_1`（分片收益为负） |
| 多卡 LoRA，base 权重大到装不下 | **FSDP2**（`--strategy fsdp2`） |
| 多卡全量微调 | DeepSpeed ZeRO-3 或 FSDP2 均可 |
| 需要 CPU/NVMe offload | 暂留 DeepSpeed（FSDP2 的 offload 粒度更粗） |
| 想和 `torch.compile` 一起用 | FSDP2 更友好 |

FSDP2 是新增选项，默认行为不变，现有用户零感知。

## 7. 配套脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/run_lora_13b_fsdp2.sh` | RWKV-7 13.3B(g1k) + LoRA + MetaMathQA SFT，默认 8 卡 FSDP2、只存 adapter |
| `merge/merge_lora.py` | 把 `rwkv-N-adapter.pth` 合并回完整权重（`--merge 0` 的配套） |

```bash
# 训练（DRY_RUN=1 只打印命令）
bash scripts/run_lora_13b_fsdp2.sh
DEVICES=4 bash scripts/run_lora_13b_fsdp2.sh

# 事后合并
python merge/merge_lora.py \
    --base_model /home/rwkv/jl/models/rwkv7-g1k-13.3b-20260930-ctx25600.pth \
    --adapter /home/rwkv/JL/out_model/rwkv7-13b-lora/rwkv-0-adapter.pth \
    --output  /home/rwkv/JL/out_model/rwkv7-13b-lora/rwkv-0.pth \
    --n_layer 61 --n_embd 4096 --vocab_size 65536 --my_testing x070 \
    --peft_config '{"r":8,"lora_alpha":32,"lora_dropout":0.05}'
```

## 8. 复现

```bash
# 纯 CPU 也能跑：分片 + 前反向 + 聚合 + merge + 端到端保存
pytest tests/test_fsdp2.py tests/test_lightning_smoke.py -q
```

真机回归建议：

```bash
# 2 卡先跑通，再扩卡
--strategy fsdp2 --devices 2 --merge 0 --epoch_save 1000
# 观察点：权重显存约为单卡的 1/2、loss 曲线与单卡 DDP 对齐、
#        rwkv-*-adapter.pth 能被 set_peft_model_state_dict 加载
```
