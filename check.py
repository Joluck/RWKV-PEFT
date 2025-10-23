import torch

# 加载模型权重（不需要加载到模型，只查看 key）
state_dict = torch.load("/home/rwkv/JL/out_model/rwkv7-1600/rwkv-0.pth", map_location="cpu")

# 打印所有 key
for key in state_dict.keys():
    print(key)
