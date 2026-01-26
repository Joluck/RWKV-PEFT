########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import deepspeed
from rwkvt.infctx_module import BlockStateList
from .block import Block

class RWKV7(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        兼容 transformers 的 generate() 接口.
        对 RWKV 来说，我们不需要做实际处理，直接返回原始输入即可。
        """
        return {"input_ids": input_ids, **kwargs}

    def get_input_embeddings(self):
        """为 PEFT 提供 Embedding 层引用"""
        return self.emb

    def set_input_embeddings(self, new_emb):
        """允许 PEFT 替换 Embedding 层（通常不会触发）"""
        self.emb = new_emb

    def get_output_embeddings(self):
        """为 PEFT 提供输出 head 层引用"""
        return self.head

    def set_output_embeddings(self, new_head):
        """允许 PEFT 替换输出 head 层"""
        self.head = new_head
    @property
    def _use_infctx(self):
        """判断是否使用无限上下文模式"""
        return os.environ.get("RWKV_TRAIN_TYPE") == 'infctx'

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        last_shift_states=None,
        last_wkv_states=None,
        **kwargs,
    ):
        """
        兼容 HF/PEFT 常见 forward 入口：
        - PEFT 会把 inputs_embeds(即使 None) 也传下来
        - 其它额外参数用 **kwargs 吃掉，避免报 unexpected keyword
        """
        if self._use_infctx:
            return self.forward_infctx(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                last_shift_states=last_shift_states,
                last_wkv_states=last_wkv_states,
                **kwargs,
            )
        else:
            return self.forward_normal(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )

    def forward_normal(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        args = self.args

        if inputs_embeds is not None:
            x = inputs_embeds
            B, T, _ = x.shape
        else:
            if input_ids is None:
                raise TypeError("forward_normal: one of input_ids or inputs_embeds must be provided")
            B, T = input_ids.size()
            x = self.emb(input_ids)

        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        v_first = torch.empty_like(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                if args.train_type == 'state' or args.peft != 'none':
                    x, v_first = torch_checkpoint(block, x, v_first, attention_mask, use_reentrant=False)
                else:
                    x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first, attention_mask)
            else:
                x, v_first = block(x, v_first, attention_mask)

        x = self.ln_out(x)
        x = self.head(x)
        return x

    def forward_infctx(
        self,
        input_ids=None,
        last_shift_states: torch.Tensor = None,
        last_wkv_states: torch.Tensor = None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """
        infctx 模式下也兼容 inputs_embeds（通常为 None）。
        如果 inputs_embeds 不为 None，则直接用它作为 x。
        """
        args = self.args

        if inputs_embeds is not None:
            x = inputs_embeds
            B, T, _ = x.shape
        else:
            if input_ids is None:
                raise TypeError("forward_infctx: one of input_ids or inputs_embeds must be provided")
            B, T = input_ids.size()
            x = self.emb(input_ids)

        assert T <= args.chunk_ctx, "Cannot forward, model ctx_len is exhausted."

        C = args.n_embd
        H = args.dim_att // args.head_size_a
        assert C == H * args.head_size_a

        # 如果外部没传 state（有些 PEFT/HF 路径可能这样），就初始化一个空 state，避免崩
        if (last_shift_states is None) or (last_wkv_states is None):
            init_states = BlockStateList.empty(args.n_layer, B, C, H, x.device, x.dtype)
            last_shift_states = init_states.shift_states
            last_wkv_states = init_states.wkv_states

        new_states = BlockStateList.empty(args.n_layer, B, C, H, x.device, x.dtype)
        v_first = torch.empty_like(x)

        for i, (block, block_state) in enumerate(
            zip(self.blocks, BlockStateList(last_shift_states, last_wkv_states))
        ):
            if args.grad_cp == 1 and i > 0:
                x, v_first, new_block_state = torch_checkpoint(
                    block, x, v_first, block_state, attention_mask, use_reentrant=False
                )
            else:
                x, v_first, new_block_state = block(x, v_first, block_state, attention_mask)

            new_states[i] = new_block_state

        x = self.ln_out(x)
        x = self.head(x)

        return x, new_states.shift_states, new_states.wkv_states
    
    # def forward(self, input_ids):
    #     args = self.args
    #     B, T = input_ids.size()
    #     assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

    #     x = self.emb(input_ids)
    #     v_first = torch.empty_like(x)

    #     for block in self.blocks:
    #         if args.grad_cp == 1:
    #             if args.train_type == 'state' or args.peft !='none':
    #                 x, v_first = torch_checkpoint(block, x, v_first ,use_reentrant=False)
    #             else:
    #                 x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
    #         else:
    #             x, v_first = block(x, v_first)

    #     x = self.ln_out(x)
    #     x = self.head(x)

    #     return x

