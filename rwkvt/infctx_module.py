"""Recurrent state containers used by infctx / state tuning."""

from dataclasses import dataclass

import torch


@dataclass
class TimeMixState:
    shift_state: torch.Tensor
    wkv_state: torch.Tensor


@dataclass
class ChannelMixState:
    shift_state: torch.Tensor


@dataclass
class BlockState:
    time_mix_state: TimeMixState
    channel_mix_state: ChannelMixState


class BlockStateList:
    """Per-layer states packed into two dense tensors.

    ``shift_states`` has shape ``(n_layer, 2, B, C)`` (time-mix / channel-mix),
    ``wkv_states`` has shape ``(n_layer, B, H, C/H, C/H)`` and is kept in bf16
    because the WKV kernels require it.
    """

    def __init__(self, shift_states, wkv_states):
        self.shift_states = shift_states
        self.wkv_states = wkv_states

    @staticmethod
    def empty(n_layer: int, batch_size: int, n_embd: int, n_head: int,
              device, dtype):
        wkv_states = torch.empty((n_layer, batch_size, n_head, n_embd // n_head,
                                  n_embd // n_head), device=device,
                                 dtype=torch.bfloat16)
        shift_states = torch.empty((n_layer, 2, batch_size, n_embd),
                                   device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    @staticmethod
    def create(n_layer: int, batch_size: int, n_embd: int, n_head: int,
               device, dtype):
        states = BlockStateList.empty(n_layer, batch_size, n_embd, n_head,
                                      device, dtype)
        states.wkv_states.zero_()
        states.shift_states.zero_()
        return states

    def __getitem__(self, layer: int) -> BlockState:
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState) -> None:
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state

    def __len__(self) -> int:
        return self.shift_states.shape[0]
