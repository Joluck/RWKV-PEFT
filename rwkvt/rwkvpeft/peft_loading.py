import os
import torch
from rwkvt.lightning_train.light_rwkv import RWKV
from rwkvt.args_type import TrainingArgs
from rwkvt.lightning_train.trainer import generate_init_weight
from rwkvt.rwkvpeft.rwkvLinear import LORA_CONFIG
from rwkvt.rwkv7.model import RWKV7

from peft import get_peft_model, LoraConfig, BoneConfig, MissConfig, TaskType


def load_peft_model(args: TrainingArgs):
    #     if os.environ["RWKV_TRAIN_TYPE"] == 'state':
#         model.requires_grad_(False)
#         freeze = True
#         for name, module in model.named_modules():
#             for pname, param in module.named_parameters():
#                 if 'state' in pname:
#                     param.requires_grad = True
#             break
    model = RWKV7(args)
    for k in model.state_dict().keys():
        print(k)
    state_dict = torch.load(args.load_model, map_location="cpu", weights_only=True)
    new_state_dict = {f"{k}": v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=(not True))
    class RWKVConfig:
        def __init__(self, n_embd=2048, n_layer=24):
            self.model_type = "rwkv"
            self.tie_word_embeddings = False
            self.n_embd = n_embd
            self.n_layer = n_layer

        def get(self, key, default=None):
            return getattr(self, key, default)

    model.config = RWKVConfig()

    if args.peft == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_config['lora_r'],
            lora_alpha=args.lora_config['lora_alpha'],
            lora_dropout=args.lora_config['lora_dropout'],
            target_modules=["receptance", "key", "value", "output"],
        )
    elif args.peft == 'miss':
        peft_config = MissConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        target_modules=["receptance", "key", "value", "output"],
    )
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model = RWKV(args, model=model)
        print(model)

        print(f"########## Loading {args.load_model}... ##########")
        state_dict = torch.load(args.load_model, map_location="cpu", weights_only=True)
        
        new_state_dict = {f"model.base_model.model.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=(not True))

    return args, model