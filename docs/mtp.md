# MTP（多 token 预测）：实现、与 vLLM 的对照、投机解码

> 代码：`rwkvt/mtp.py`（模块）、`rwkvt/spec_decode.py`（draft/verify）
> 参考实现：vLLM `vllm/model_executor/models/deepseek_mtp.py`、`vllm/v1/spec_decode/`、`vllm/v1/sample/rejection_sampler.py`

## 1. 模块结构对照

vLLM 的 `DeepSeekMultiTokenPredictorLayer.forward`（DeepSeek-V3 的 MTP 层）是：

```python
inputs_embeds  = enorm(emb(t_{i+k}))                  # RMSNorm
prev_hidden    = hnorm(h^{k-1})                       # RMSNorm
hidden_states  = eh_proj(cat([inputs_embeds, prev_hidden]))   # Linear(2d -> d)
hidden_states, residual = mtp_block(hidden_states)    # 一个 Transformer 层
hidden_states  = residual + hidden_states             # pre-final-norm
return hidden_states, shared_head(hidden_states)      # shared_head = RMSNorm (+ lm_head)
```

本仓库（`MTPBlock`）逐项对应：

| vLLM / DeepSeek | 本仓库 | 说明 |
| --- | --- | --- |
| `enorm` (RMSNorm) | `norm_e` (LayerNorm) | RWKV 全家用 LayerNorm，保持一致 |
| `hnorm` (RMSNorm) | `norm_h` (LayerNorm) | 同上 |
| `eh_proj` | `proj` | `Linear(2d → d, bias=False)`，concat 顺序 `[emb, hidden]` 已对齐 |
| `mtp_block`（Transformer 层） | `key`/`value`（RWKV channel-mix） | RWKV 的"块"就是 cmix；参数量 9d²，与 vLLM 一个 decoder 层同量级 |
| `shared_head.norm` + lm_head | `norm_out` + 冻结的 `head` | 已对齐 |
| 复用 pre-norm hidden 到下一深度 | 同 | `MTPHead.forward` 里 `h` 传下去 |

**为什么要加 `norm_out`**：底座的 `head` 是接在 `ln_out` 之后的，训练时只见过"归一化后的输入"。之前我们直接把 MTP 块的原始输出喂给 `head`，分布不对；vLLM/DeepSeek 一定在共享 head 前放一层 norm。这是本次唯一的**结构性修正**。

## 2. 训练

```bash
PEFT=mtp bash scripts/run_13b_fsdp2.sh            # 冻结底座，只训 MTP 头
MTP_DEPTH=2 bash scripts/mtp.sh                   # 多预测一个位置
```

- 损失：`loss = mtp_main_weight * CE(t_{i+1}) + mtp_loss_weight * Σ_k w_k CE(t_{i+k+1})`
  - `w_k = β^(k-1) / Σ_j β^(j-1)`，**FastMTP 式指数衰减**，`--mtp_step_beta` 默认 0.6
    （K=3 → `[0.51, 0.31, 0.18]`，把权重压在第一枚 draft 上，它决定接受率）；
    `--mtp_step_beta 1.0` 等价于原来的均匀平均
  - 每一层的 loss 单独记录：`mtp_loss_1 .. mtp_loss_K`（jsonl + wandb）
  - `--mtp_main_weight` 默认 0（底座冻结，主项没有梯度，只作监控）
  - `--mtp_loss_weight` 默认 0.3，对齐 DeepSeek-V3 的 λ；`main_weight=0` 时它只是全局缩放
- `loss_data.jsonl` 与 wandb 里同时有 `main_loss` / `mtp_loss`，看后者判断 MTP 学得如何
- 初始化时 `mtp_loss ≈ ln(vocab)`（65536 → 11.09），收敛到 1~2 属于正常
- 产物只有 `rwkv-N-mtp.pth`；底座不变

## 3. 推理：丢掉，还是当 drafter

| 方式 | 每步出几个 token | 说明 |
| --- | --- | --- |
| 丢掉 MTP 头 | 1 | 默认；MTP 只是训练目标，底座分布不变 |
| 投机解码（depth=1） | 期望 1 + α | DeepSeek 实测第二个 token 接受率 85–90% → ≈1.9× |
| 投机解码（depth=k） | 期望 1 + α₁ + α₁α₂ + … | 接受率随深度衰减，一般 k ≤ 2 |

本仓库提供 `rwkvt/spec_decode.py`：

```python
from rwkvt.spec_decode import SpecStats, mtp_greedy_decode
stats = SpecStats()
out = mtp_greedy_decode(model, mtp, prompt, max_new_tokens=64, depth=1, stats=stats)
print(stats.accept_rate, stats.tokens_per_round)
```

实现方式（对照 vLLM）：

1. 底座前向拿到 `hidden`/`logits`，取 `t_{i+1}`；
2. 链式跑 MTP 块（复用 pre-norm hidden + 新 token 的 embedding）得到 draft；
3. 底座一次性验证 `t_{i+1}` 和所有 draft（RWKV 支持 chunk 前向）；
4. 保留最长合法前缀，第一个被拒的位置用底座自己的 argmax（等价于普通 greedy）。

**性质（有测试）**：greedy 下输出与普通逐 token 解码**逐位相同**，与 drafter 好坏无关；drafter 只影响速度，不影响结果。

## 3.5 与 speculators 的差距（下一步）

[vllm-project/speculators](https://github.com/vllm-project/speculators) 的 MTP 训练配方里，除了上面的逐层权重（已对齐），还有一点更关键：

- 它的训练 batch 直接带 **`verifier_last_hidden_states`**（由 vLLM 推理时导出，通过 `hs_connectors` 流式传给 trainer），trainer **不跑 target 模型**，只跑 draft 层；
- 我们是每一步都跑冻结的 13B 前向来取 hidden —— 这就是 4 it/s 的来源。

对 `--peft mtp`（底座完全冻结）来说，hidden 可以离线算一次反复用：

| 规模 | hidden 体积（bf16） |
| --- | --- |
| 100k 条 × 128 token × 4096 dim | ≈ 105 GB |
| 20k 条 × 128 token × 4096 dim | ≈ 21 GB |

所以下一步可以加一个 `--mtp_hidden_cache DIR`：离线 precompute 一次，之后每个 epoch 只跑 151M 的 MTP 头，预计比现在快 5–10×。需要的话我来实现（含分片落盘 + DataLoader）。

## 4. 没有照搬的部分

| vLLM | 这里 | 原因 |
| --- | --- | --- |
| 概率拒绝采样（RejectionSampler，`min(1, p/p_draft)` + 残差分布重采样） | 只做 greedy | greedy 就能测接受率；采样版需要随机数管理与 RWKV state 回滚，后续可加 |
| KV cache / slot mapping / TP 分片 | 每轮重跑整段序列 | 本仓库是训练框架，没有推理引擎；这个实现是**接受率测量工具**，不是生产解码器 |
| 一个完整 Transformer 作为 MTP block | RWKV channel-mix | 与模型族一致，省掉 61 层 attention 的开销 |

## 5. 兼容性

- 只有一种布局：`norm_e/norm_h/proj/key/value/norm_out`，与 vLLM 一致。
- 早期版本（没有 `norm_out`）保存的 `rwkv-N-mtp.pth` **不再兼容**，续训请从头开始。
- `--mtp_load` 遇到不匹配的 key 会直接报错，不会静默错配。
