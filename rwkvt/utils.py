"""Small dependency-free helpers."""


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Remove ``prefix`` from matching keys (only once, unlike ``str.replace``)."""
    return {k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()}


def drop_unsupported_kwargs(caller: str, kwargs: dict) -> dict:
    """Filter huggingface-style kwargs that PEFT forwards to the model.

    ``PeftModelForCausalLM.forward`` passes ``input_ids``/``attention_mask``/
    ``inputs_embeds``/``labels``/``output_attentions``/... into the wrapped model.
    RWKV has no ``inputs_embeds`` path, everything else is computed here, so the
    extra arguments are dropped - but a *real* ``inputs_embeds`` is an error
    rather than a silent no-op.
    """
    if kwargs.pop("inputs_embeds", None) is not None:
        raise NotImplementedError(
            f"{caller} does not support inputs_embeds; pass input_ids instead")
    return kwargs
