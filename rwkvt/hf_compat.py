"""Minimal huggingface/transformers interface expected by PEFT.

PEFT's tuner wrappers call ``prepare_inputs_for_generation`` and the
embedding accessors on the wrapped model, so every RWKV version implements
them through this mixin instead of copy-pasting the same five methods.
"""


class HFCompatibleModel:
    """Mix in *before* ``nn.Module``: ``class RWKV7(HFCompatibleModel, nn.Module)``."""

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """RWKV needs no extra preprocessing; kept for ``generate()`` compat."""
        return {"input_ids": input_ids, **kwargs}

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, new_emb):
        self.emb = new_emb

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_head):
        self.head = new_head
