import os
import torch
import importlib
import json
from rwkvt.lightning_train.light_rwkv import RWKV
from rwkvt.args_type import TrainingArgs
from rwkvt.lightning_train.trainer import generate_init_weight

from peft import get_peft_model, LoraConfig, BoneConfig, MissConfig, TaskType
from peft import *
if "7" in os.environ["RWKV_MY_TESTING"]:
    from rwkvt.rwkv7.model import RWKV7 as RWKVModel
elif "6" in os.environ["RWKV_MY_TESTING"]:
    from rwkvt.rwkv6.model import RWKV6 as RWKVModel
elif "5" in os.environ["RWKV_MY_TESTING"]:
    from rwkvt.rwkv5.model import RWKV5 as RWKVModel
else:
    raise ValueError(f"Unsupported model version: . Valid options: 5,6,7")


class RWKVConfig:
    def __init__(self, n_embd=2048, n_layer=24):
        self.model_type = "rwkv"
        self.tie_word_embeddings = False
        self.n_embd = n_embd
        self.n_layer = n_layer

    def get(self, key, default=None):
        return getattr(self, key, default)

def load_peft_model(args: TrainingArgs):
    model = RWKVModel(args)
    state_dict = torch.load(args.load_model, map_location="cpu", weights_only=True, mmap=True)
    print(f"########## Loading {args.load_model}... ##########")
    model.load_state_dict(state_dict, strict=(not True))
    if os.environ["RWKV_TRAIN_TYPE"] == 'state':
        
        model = RWKV(args, model=model)
        model.requires_grad_(False)
        for name, module in model.named_modules():
            for pname, param in module.named_parameters():
                if 'state' in pname:
                    param.requires_grad = True
            break
    elif args.peft!='none':
       
        model.config = RWKVConfig(n_embd=args.n_embd, n_layer=args.n_layer)

        # if args.peft == 'lora':
        #     peft_config = LoraConfig(
        #         task_type=TaskType.CAUSAL_LM,
        #         r=args.lora_config['lora_r'],
        #         lora_alpha=args.lora_config['lora_alpha'],
        #         lora_dropout=args.lora_config['lora_dropout'],
        #         target_modules=["receptance", "key", "value", "output"],
        #     )
        # elif args.peft == 'miss':
        #     peft_config = MissConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     r=args.miss_config['r'],
        #     target_modules=["receptance", "key", "value", "output"],
        # )
        
        # === 动态加载 PEFT Config 类 ===
        peft_dict={
            "lora": LoraConfig,
            "miss": MissConfig,
            "adalora": AdaLoraConfig,
            "prefix": PrefixTuningConfig,
        }
        ConfigClass = peft_dict[args.peft]

        peft_args = json.loads(args.peft_config)
        peft_config = ConfigClass(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["receptance","key","value","output"],
            **peft_args)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model = RWKV(args, model=model)


    return args, model
