import torch, math
import torch.nn as nn
import bitsandbytes as bnb
from torch.nn import functional as F
from torch._lowrank import svd_lowrank
import functools
from einops import rearrange
from torch.utils.checkpoint import checkpoint as torch_checkpoint

def rwkv_quantize(quant_type, weight):
    if quant_type=='4bit':
        qweight, qstate= bnb.functional.quantize_4bit(weight.data)
    elif quant_type=='nf4':
        qweight, qstate= bnb.functional.quantize_nf4(weight.data)
    elif quant_type=='fp4':
        qweight, qstate= bnb.functional.quantize_fp4(weight.data)
    elif quant_type=='int8':
        qweight, qstate= bnb.functional.quantize(weight.data)
    elif quant_type=='fp8':
        qweight = weight.data.to(dtype = torch.float8_e4m3fn)
        qstate = None
    return qweight, qstate


def rwkv_dequantize(quant_type, weight, qstate):
    if quant_type=='4bit':
        deweight= bnb.functional.dequantize_4bit(weight.data,quant_state=qstate)
    elif quant_type=='nf4':
        deweight= bnb.functional.dequantize_nf4(weight.data,quant_state=qstate)
    elif quant_type=='fp4':
        deweight= bnb.functional.dequantize_fp4(weight.data,quant_state=qstate)
    elif quant_type=='int8':
        deweight= bnb.functional.dequantize(weight.data,state=qstate)
    elif quant_type=='fp8':
        deweight= weight.data
    return deweight.to(torch.bfloat16)

#@torch.jit.script
def fp8_matmul_(a,b): # shape3 @ shape2 only
    with torch.no_grad():
        if b.dtype == torch.float8_e4m3fn:
                xg = a
                S0=xg.shape[0]
                S1=xg.shape[1]
                if xg.dtype != torch.float8_e4m3fn:
                    xg = torch.clamp(xg, min=-448.0, max=448.0) # for avoid NaN
                
                x, output_amax = torch._scaled_mm(
                    xg.view(S0*S1,xg.shape[2]).to(torch.float8_e4m3fn).contiguous(),
                    b,
                    bias=None,
                    out_dtype=a.dtype,
                    scale_a=torch.tensor(1.0, device='cuda'),
                    scale_b=torch.tensor(1.0, device='cuda')
                )
                #x.requires_grad = False
                return x.view(S0, S1, -1)
        else:
                return a.to(dtype=b.dtype) @ b
        
class FP8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(weight)
        output = fp8_matmul_(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        weight, = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight.to(dtype=torch.bfloat16).t()) 
        #grad_weight is None because frozen
        return grad_input, None
    
fp8_matmul = FP8Matmul.apply


def ori_result(x, qweight, quant_type, qstate):

    if qweight.dtype == torch.float8_e4m3fn:
        return  fp8_matmul(x, qweight.t()) 
    else:
        return F.linear(x, rwkv_dequantize(quant_type, qweight.data, qstate).to(x.device))


        
LORA_CONFIG = {
    "r": 0,
    "alpha": 0,
    "dropout": 0,
    "parts": {"att","ffn"},
    "quant": False,
}

DiSHA_CONFIG = {
    "r": 0,
    "mode": "bone", #"bat"
    "parts": {"att", "ffn"},
}
class LoraLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r, alpha, dropout = LORA_CONFIG["r"], LORA_CONFIG[
            "alpha"], LORA_CONFIG["dropout"]
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
        self.r = r
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.pissa = False
        self.is_quant = False

    def pissa_load(self, init_A, init_B):
        self.pissa = True
        self.weight.data = self.weight.data - init_B @ init_A


    def pissa_init(self, svd_niter):

        self.pissa = True
        Ur, Sr, Vr = svd_lowrank(self.weight.data, self.r, niter=svd_niter)
        Vhr = Vr.t()
        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.weight.data = self.weight.data - lora_B @ lora_A
    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        self.qweight, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to('cuda'))
        self.weight = None # Because Latest Pytorch-lightning forced to BF16 type. thats why delete

    def forward(self, x):
        if self.is_quant:
            if self.pissa:
                if self.qweight.dtype == torch.float8_e4m3fn:
                    return (
                    fp8_matmul(x,self.qweight.t()) + 
                    F.linear(F.linear(x, self.lora_A), self.lora_B))
                return (
                    F.linear(x, rwkv_dequantize(self.quant_type, self.qweight.data, self.qstate).to(x.device)) + 
                    F.linear(F.linear(x, self.lora_A), self.lora_B))
            
            if self.qweight.dtype == torch.float8_e4m3fn:
                return (fp8_matmul(x,self.qweight.t()) + self.scaling *
                        F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
                )
            return (
                F.linear(x, rwkv_dequantize(self.quant_type, self.qweight.data, self.qstate).to(x.device)) + self.scaling *
                F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 

        if self.pissa:
            return (
                F.linear(x, self.weight) + 
                F.linear(F.linear(x, self.lora_A), self.lora_B))
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))  
    

class QuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.is_quant = False

    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        #self.dummy_tensor = nn.Parameter(torch.zeros(1))
        self.weight.data, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to('cuda'))
    def forward(self, x):

        if self.is_quant:
            return F.linear(x, rwkv_dequantize(self.quant_type, self.weight.data, self.qstate).to(x.device))
        else:
            return F.linear(x, self.weight)
        

class BatLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.r = DiSHA_CONFIG["r"]
        self.disha = nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r))
        self.is_quant = False

    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        self.qweight, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to('cuda'))
        self.weight = None # Because Latest Pytorch-lightning forced to BF16 type. thats why delete

    def forward(self, x):
        if self.is_quant:
            qw = rwkv_dequantize(self.quant_type, self.qweight.data, self.qstate).to(x.device)
            w = rearrange(qw, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.disha+self.disha
            w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
            return F.linear(x,qw+w)
        else:
            w = rearrange(self.weight, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.disha+self.disha
            w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ') 
            return F.linear(x,self.weight+w)
    
class BoneLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased QuantLinear not supported"
        self.r = DiSHA_CONFIG["r"]
        self.disha = nn.Parameter(torch.zeros(self.r, out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.is_quant = False

    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        self.qweight, self.qstate= rwkv_quantize(self.quant_type, (self.weight.data).to('cuda'))
        self.weight = None # Because Latest Pytorch-lightning forced to BF16 type. thats why delete

    def forward(self, x):
        if self.is_quant:
            result = ori_result(x, self.qweight, self.quant_type, self.qstate)
        else:
            result = F.linear(x, self.weight)
        #result = ori_result(x, self.weight, self.is_quant, self.qweight, self.quant_type, self.qstate)

        if self.in_features%self.r!=0:
            padding_size = (self.r - self.in_features % self.r) % self.r
            x = F.pad(x, (0, padding_size))

        y = result + torch.sum(x.reshape(*x.shape[:-1], x.size(-1)//self.r, self.r), dim=-2)@self.disha
        return y



@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    if "att" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    elif "att" in DiSHA_CONFIG["parts"] and DiSHA_CONFIG["r"] > 0 and DiSHA_CONFIG["mode"]=="bone":
        return BoneLinear(*args, **kwargs)
    elif "att" in DiSHA_CONFIG["parts"] and DiSHA_CONFIG["r"] > 0 and DiSHA_CONFIG["mode"]=="bat":
        return BatLinear(*args, **kwargs)
    elif LORA_CONFIG["quant"]:
        return QuantLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    if "ffn" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    elif "ffn" in DiSHA_CONFIG["parts"] and DiSHA_CONFIG["r"] > 0 and DiSHA_CONFIG["mode"]=="bone" :
        return BoneLinear(*args, **kwargs)
    elif "ffn" in DiSHA_CONFIG["parts"] and DiSHA_CONFIG["r"] > 0 and DiSHA_CONFIG["mode"]=="bat" :
        return BatLinear(*args, **kwargs)
    elif LORA_CONFIG["quant"]:
        return QuantLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)