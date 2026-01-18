
from einops import rearrange
import os, math, gc, importlib
import torch
########################################################################################################
# CUDA Kernel
########################################################################################################
def RUN_CUDA_RWKV7g():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')

def RUN_RWKV7_STATE():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')

def RUN_RWKV7_INFCTX():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')

def RUN_CUDA_RWKV6():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')


def RUN_CUDA_RWKV6_STATE():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')


def RUN_CUDA_RWKV5():
    raise NotImplementedError('RUN_CUDA_RUN_KV not implemented')

if os.environ["WKV"] == 'fla':
    if 'x070' in os.environ["RWKV_MY_TESTING"]:
        from rwkvfla.ops.rwkv7 import chunk_rwkv7
        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            def RUN_RWKV7_INFCTX(r, k, v, w, a, b, s, HEAD_SIZE=64): # for State-tuning, infctx
                B,T,HC = w.shape
                C = HEAD_SIZE
                H = HC//C
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                # when use w=, -exp(w) is not needed, w=-torch.exp(w), otherwise, use log_w = -torch.exp(w)
                o, state = chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
                return o, state
        if os.environ["RWKV_TRAIN_TYPE"] == 'state':
            def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64): # for State-tuning, infctx
                B,T,HC = w.shape
                C = HEAD_SIZE
                H = HC//C
                s = s.transpose(1, 2).expand(B,*s.shape)
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                # when use w=, -exp(w) is not needed, w=-torch.exp(w), otherwise, use log_w = -torch.exp(w)
                o, state = chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0, initial_state=s, output_final_state=True, head_first=False)
                return o, state
        else:
            def RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE=64): #compatible with cuda implement
                B,T,HC = w.shape
                C = HEAD_SIZE
                H = HC//C
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                # when use w=, -exp(w) is not needed, w=-torch.exp(w), otherwise, use log_w = -torch.exp(w)
                o, _ = chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0, initial_state=None, output_final_state=False, head_first=False)
                return o
    if 'x060' in os.environ["RWKV_MY_TESTING"]:
        from rwkvfla.ops.rwkv6 import chunk_rwkv6

        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                r = rearrange(r, 'b l (h d) -> b h l d', h = H)
                k = rearrange(k, 'b l (h d) -> b h l d', h = H)
                v = rearrange(v, 'b l (h d) -> b h l d', h = H)
                w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
                o, state = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True)
                x = rearrange(o, 'b h l d -> b l (h d)')
                return x, state
        elif os.environ["RWKV_TRAIN_TYPE"] == 'state':
            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                r = rearrange(r, 'b l (h d) -> b h l d', h = H)
                k = rearrange(k, 'b l (h d) -> b h l d', h = H)
                v = rearrange(v, 'b l (h d) -> b h l d', h = H)
                w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
                s = s.transpose(1, 2).expand(B,*s.shape)
                o,_ = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=False)
                x = rearrange(o, 'b h l d -> b l (h d)')
                return x
        else:
            def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
                r = rearrange(r, 'b l (h d) -> b h l d', h = H)
                k = rearrange(k, 'b l (h d) -> b h l d', h = H)
                v = rearrange(v, 'b l (h d) -> b h l d', h = H)
                w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
                o,_ = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=None, output_final_state=False)
                x = rearrange(o, 'b h l d -> b l (h d)')
                return x
elif os.environ["WKV"] == 'triton':
    print('x070 Wind Triton Kernel Mode')

    import torch as th
    import triton
    import triton.language as tl

    @triton.jit
    def IND4(a,b,c,d,nb,nc,nd):
        return ((a*nb+b)*nc+c)*nd+d
    @triton.jit
    def IND5(a,b,c,d,e,nb,nc,nd,ne):
        return (((a*nb+b)*nc+c)*nd+d)*ne+e

    @triton.jit
    def _prod(a,b): return a*b

    # inv(I-A) where A is a strictly lower triangular nxn matrix
    @triton.jit
    def tri_minv(A, n:tl.constexpr, prec:tl.constexpr):
        i = tl.arange(0,n)
        prod = (i[None,:]==i[:,None]).to(tl.float32)
        for j in range(n-1):
            prod += tl_dot(prec, prod, (A*((i[None,:]==j)*(i[:,None]>i[None,:]))).trans())
        return prod.trans()

    @triton.jit
    def fw_attn_triton(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
        bi = tl.program_id(1)
        hi = tl.program_id(0)

        i = tl.arange(0,C)[None,:]
        state = tl.load(s0_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)
        for t0 in range(T//dT):
            t = t0*dT+tl.arange(0,dT)[:,None]
            sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

            w = (-sw.exp()).exp()
            fw = tl.reduce(w, 0, _prod, keep_dims=True)
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref

            wq = sq * incl_pref
            wa = sa * non_incl_pref
            kwi = sk * inv_incl_pref
            bwi = sb * inv_incl_pref

            mask1 = (t > t.trans())
            ab = tl_dot(prec, wa, bwi.trans()) * mask1
            ak = tl_dot(prec, wa, kwi.trans()) * mask1

            ab_inv = tri_minv(ab, dT, prec)

            ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
            u = tl_dot(prec, ab_inv, ab_u)
            mask2 = (t >= t.trans())
            qk = tl_dot(prec, wq, kwi.trans()) * mask2
            qb = tl_dot(prec, wq, bwi.trans()) * mask2
            yy = tl_dot(prec, qk, sv) + tl_dot(prec, qb, u) + tl_dot(prec, wq, state.trans())
            tl.store(y_+IND4(bi,t,hi,i, T,H,C), yy.to(tl.bfloat16))

            tl.store(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C), state.to(tl.float32))
            state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)
        tl.store(sT_+IND4(bi,hi,i.trans(),i, H,C,C), state.to(tl.bfloat16))

    @triton.jit
    def bw_attn_triton(w_,q_,k_,v_,a_,b_, dy_,s_,dsT_, dw_,dq_,dk_,dv_,da_,db_,ds0_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
        bi = tl.program_id(1)
        hi = tl.program_id(0)

        i = tl.arange(0,C)[None,:]
        dstate = tl.load(dsT_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)

        for t0 in range(T//dT-1,-1,-1):
            t = t0*dT+tl.arange(0,dT)[:,None]

            state = tl.load(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C)).to(tl.float32)

            sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

            dw_fac = -sw.exp()
            w = dw_fac.exp()
            fw = tl.reduce(w, 0, _prod, keep_dims=True)
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref

            wq = sq * incl_pref
            wa = sa * non_incl_pref
            kwi = sk * inv_incl_pref
            bwi = sb * inv_incl_pref

            mask1 = (t > t.trans())
            ab = tl_dot(prec, wa, bwi.trans()) * mask1
            ak = tl_dot(prec, wa, kwi.trans()) * mask1

            ab_inv = tri_minv(ab, dT, prec)

            ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
            u = tl_dot(prec, ab_inv, ab_u)
            mask2 = (t >= t.trans())
            qk = tl_dot(prec, wq, kwi.trans()) * mask2
            qb = tl_dot(prec, wq, bwi.trans()) * mask2

            du = tl_dot(prec, qb.trans(), sdy) + tl_dot(prec, bwi*fw, dstate.trans())
            dab_u = tl_dot(prec, ab_inv.trans(), du)

            dv = tl_dot(prec, qk.trans(), sdy) + tl_dot(prec, kwi*fw, dstate.trans()) + tl_dot(prec, ak.trans(), dab_u)
            tl.store(dv_+IND4(bi,t,hi,i, T,H,C), dv.to(tl.bfloat16))

            dab = tl_dot(prec, tl_dot(prec, ab_inv.trans(), du), u.trans()) * mask1
            dak = tl_dot(prec, dab_u, sv.trans()) * mask1
            dab_u_state = tl_dot(prec, dab_u, state)
            da = non_incl_pref * (tl_dot(prec, dab, bwi) + tl_dot(prec, dak, kwi) + dab_u_state)
            tl.store(da_+IND4(bi,t,hi,i, T,H,C), da.to(tl.bfloat16))

            dqb = tl_dot(prec, sdy, u.trans()) * mask2
            dqk = tl_dot(prec, sdy, sv.trans()) * mask2
            dy_state = tl_dot(prec, sdy, state)
            dq = incl_pref * (tl_dot(prec, dqb, bwi) + tl_dot(prec, dqk, kwi) + dy_state)
            tl.store(dq_+IND4(bi,t,hi,i, T,H,C), dq.to(tl.bfloat16))

            fw_u_dstate = fw * tl_dot(prec, u, dstate)
            db = inv_incl_pref * (tl_dot(prec, dab.trans(), wa) + tl_dot(prec, dqb.trans(), wq) + fw_u_dstate)
            tl.store(db_+IND4(bi,t,hi,i, T,H,C), db.to(tl.bfloat16))

            fw_v_dstate = fw * tl_dot(prec, sv, dstate)
            dk = inv_incl_pref * (tl_dot(prec, dak.trans(), wa) + tl_dot(prec, dqk.trans(), wq) + fw_v_dstate)
            tl.store(dk_+IND4(bi,t,hi,i, T,H,C), dk.to(tl.bfloat16))

            dw0 = fw * tl.sum(state*dstate, axis=0,keep_dims=True)
            for k in range(t0*dT,t0*dT+dT):
                lmask = (t<k).trans()
                A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
                A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
                A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
                A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
                dw = tl.sum(A, axis=0,keep_dims=True) + dw0

                wk = tl.load(w_+IND4(bi,k,hi,i, T,H,C)).to(tl.float32)
                dw *= -wk.exp()
                tl.store(dw_+IND4(bi,k,hi,i, T,H,C), dw.to(tl.bfloat16))

            dstate = dstate * fw + tl_dot(prec, sdy.trans(), wq) + tl_dot(prec, dab_u.trans(), wa)
        tl.store(ds0_+IND4(bi,hi,i.trans(),i, H,C,C), dstate.to(tl.bfloat16))


    class TritonRWKV7(th.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b,s0, dot_prec):
            K = 16
            B,T,H,C = w.shape
            s0 = th.zeros(B,H,C,C, dtype=w.dtype,device=w.device) if s0 is None else s0
            y = th.empty_like(v)
            sT = th.empty_like(s0)
            s = th.zeros(B,H,T//K,C,C, dtype=th.float32,device=w.device)
            fw_attn_triton[(H,B)](w,q,k,v,z,b, s0,y,s,sT, B,T,H,C,K, dot_prec)
            ctx.dot_prec = dot_prec
            ctx.save_for_backward(w,q,k,v,z,b,s)
            return y, sT
        @staticmethod
        def backward(ctx, dy, dsT):
            K = 16
            w,q,k,v,z,b,s = ctx.saved_tensors
            B,T,H,C = w.shape
            dw,dq,dk,dv,dz,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,z,b,dsT]]
            bw_attn_triton[(H,B)](w,q,k,v,z,b, dy,s,dsT, dw,dq,dk,dv,dz,db,ds0, B,T,H,C,K, ctx.dot_prec)
            return dw,dq,dk,dv,dz,db,ds0,None

    @triton.jit
    def tl_dot(prec:tl.constexpr, a, b):
        if prec == 'fp32':
            return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=False)
        elif prec == 'tf32':
            return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=True)
        elif prec == 'bf16':
            return tl.dot(a.to(tl.bfloat16),b.trans().to(tl.bfloat16).trans(), allow_tf32=True)
        else:
            tl.static_assert(False)

    def RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE=64, dot_prec = 'fp32'):
        B,T,HC = w.shape
        C = HEAD_SIZE
        H = HC//C
        r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
        s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
        return TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)[0].view(B,T,HC)
    def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64, dot_prec = 'fp32'):
                B,T,HC = w.shape
                C = HEAD_SIZE
                H = HC//C
                r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                s0 = s
                return TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)[0].view(B,T,HC), None
else:
    from torch.utils.cpp_extension import load

    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
    if 'x070' in os.environ["RWKV_MY_TESTING"]:
        CHUNK_LEN = 16
        ROCm_flag = torch.version.hip is not None
        if not ROCm_flag:
            if os.environ["RWKV_TRAIN_TYPE"] == 'state':
                if os.environ["RWKV_FLOAT_MODE"] == 'bf16':
                    flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
                    load(name="rwkv7_state_clampw", sources=[f'cuda/rwkv7_state_clampw.cu', 'cuda/rwkv7_state_clampw.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

                    class RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,s0,r,w,k,v,a,b):
                            B,T,H,C = r.shape
                            assert T%CHUNK_LEN == 0
                            assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
                            assert all(i.is_contiguous() for i in [s0,r,w,k,v,a,b])
                            assert s0.dtype==torch.float
                            y = torch.empty_like(v)
                            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                            sa = torch.empty(B,T,H,C,dtype=torch.float32,device=w.device)
                            torch.ops.rwkv7_state_clampw.forward(s0,r,w,k,v,a,b,y,s,sa)
                            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                            return y
                        @staticmethod
                        def backward(ctx,dy):
                            assert all(i.dtype==torch.bfloat16 for i in [dy])
                            assert all(i.is_contiguous() for i in [dy])
                            r,w,k,v,a,b,s,sa = ctx.saved_tensors
                            B,T,H,C = r.shape
                            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                            ds0 = torch.empty(B,H,C,C,dtype=torch.float32,device=r.device)
                            torch.ops.rwkv7_state_clampw.backward(r,w,k,v,a,b,dy,s,sa,ds0,dr,dw,dk,dv,da,db)
                            return ds0,dr,dw,dk,dv,da,db
                    def RUN_RWKV7_STATE(r,k,v,w,a,b,s0):
                        B,T,HC = r.shape
                        C = HEAD_SIZE
                        H = HC//C
                        s0 = s0.float().repeat(B, 1, 1, 1)
                        r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                        return RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0,r,w,k,v,a,b).view(B,T,HC), None

                elif os.environ["RWKV_FLOAT_MODE"] == 'fp32':
                    flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', "-D_FP32_", f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
                    load(name="rwkv7_state_clampw", sources=[f'cuda/rwkv7_state_clampw.cu', 'cuda/rwkv7_state_clampw.cpp'], is_python_module=False, verbose=True, extra_cflags=["-D_FP32_"], extra_cuda_cflags=flags)

                    class RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,s0,r,w,k,v,a,b):
                            B,T,H,C = r.shape 
                            assert T%CHUNK_LEN == 0
                            assert all(i.dtype==torch.float32 for i in [s0,r,w,k,v,a,b])
                            assert all(i.is_contiguous() for i in [s0,r,w,k,v,a,b])
                            y = torch.empty_like(v)
                            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                            torch.ops.rwkv7_state_clampw.forward(s0,r,w,k,v,a,b,y,s,sa)
                            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                            return y
                        @staticmethod
                        def backward(ctx,dy):
                            assert all(i.dtype==torch.float32 for i in [dy])
                            assert all(i.is_contiguous() for i in [dy])
                            r,w,k,v,a,b,s,sa = ctx.saved_tensors
                            B,T,H,C = r.shape
                            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                            ds0 = torch.empty(B,H,C,C,dtype=torch.float32,device=r.device)
                            torch.ops.rwkv7_state_clampw.backward(r,w,k,v,a,b,dy,s,sa,ds0,dr,dw,dk,dv,da,db)
                            return ds0,dr,dw,dk,dv,da,db
                    def RUN_RWKV7_STATE(r,k,v,w,a,b,s0):
                        B,T,HC = r.shape
                        C = HEAD_SIZE
                        H = HC//C
                        s0 = s0.repeat(B, 1, 1, 1)
                        r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                        return RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0,r,w,k,v,a,b).view(B,T,HC), None

                else:
                    raise NotImplementedError("Unsupported precision for RWKV7 fine-tuning")

            else:
                if os.environ["RWKV_FLOAT_MODE"] == 'bf16':
                    flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
                    load(name="rwkv7_clampw", sources=[f'cuda/rwkv7_clampw.cu', 'cuda/rwkv7_clampw.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
                    class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,r,w,k,v,a,b):
                            B,T,H,C = r.shape 
                            assert T%CHUNK_LEN == 0
                            assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
                            assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
                            y = torch.empty_like(v)
                            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                            torch.ops.rwkv7_clampw.forward(r,w,k,v,a,b,y,s,sa)
                            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                            return y
                        @staticmethod
                        def backward(ctx,dy):
                            assert all(i.dtype==torch.bfloat16 for i in [dy])
                            assert all(i.is_contiguous() for i in [dy])
                            r,w,k,v,a,b,s,sa = ctx.saved_tensors
                            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                            torch.ops.rwkv7_clampw.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
                            return dr,dw,dk,dv,da,db

                    def RUN_CUDA_RWKV7g(r,w,k,v,a,b):
                        B,T,HC = r.shape
                        r,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [r,w,k,v,a,b]]
                        return RWKV7_CLAMPW_CUDA_OP.apply(r,w,k,v,a,b).view(B,T,HC)

                elif os.environ["RWKV_FLOAT_MODE"] == 'fp32':
                    flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', "-D_FP32_", f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
                    load(name="rwkv7_clampw", sources=[f'cuda/rwkv7_clampw.cu', 'cuda/rwkv7_clampw.cpp'], is_python_module=False, verbose=True, extra_cflags=["-D_FP32_"], extra_cuda_cflags=flags)
                    class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,r,w,k,v,a,b):
                            B,T,H,C = r.shape 
                            assert T%CHUNK_LEN == 0
                            assert all(i.dtype==torch.float32 for i in [r,w,k,v,a,b])
                            assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
                            y = torch.empty_like(v)
                            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                            torch.ops.rwkv7_clampw.forward(r,w,k,v,a,b,y,s,sa)
                            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                            return y
                        @staticmethod
                        def backward(ctx,dy):
                            assert all(i.dtype==torch.float32 for i in [dy])
                            assert all(i.is_contiguous() for i in [dy])
                            r,w,k,v,a,b,s,sa = ctx.saved_tensors
                            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                            torch.ops.rwkv7_clampw.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
                            return dr,dw,dk,dv,da,db
                    def RUN_CUDA_RWKV7g(r,w,k,v,a,b):
                        B,T,HC = r.shape
                        r,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [r,w,k,v,a,b]]
                        return RWKV7_CLAMPW_CUDA_OP.apply(r,w,k,v,a,b).view(B,T,HC)

                else:
                    raise NotImplementedError("Unsupported precision for RWKV7 fine-tuning")
        else:
            if os.environ["RWKV_TRAIN_TYPE"] == 'state':
                if os.environ["RWKV_FLOAT_MODE"] == 'bf16':
                    flags = [f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "-xhip", "-fopenmp", "-ffast-math", "-O3", "-munsafe-fp-atomics", "--save-temps", '-DAMD']
                    load(name="rwkv7_state_clampw", sources=[f'hip/rwkv7_state_clampw.hip', 'hip/rwkv7_state_clampw_op.hip'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

                    class RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,s0,r,w,k,v,a,b):
                            B,T,H,C = r.shape
                            assert T%CHUNK_LEN == 0
                            assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
                            assert all(i.is_contiguous() for i in [s0,r,w,k,v,a,b])
                            assert s0.dtype==torch.float
                            y = torch.empty_like(v)
                            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                            sa = torch.empty(B,T,H,C,dtype=torch.float32,device=w.device)
                            torch.ops.rwkv7_state_clampw.forward(s0,r,w,k,v,a,b,y,s,sa)
                            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                            return y
                        @staticmethod
                        def backward(ctx,dy):
                            assert all(i.dtype==torch.bfloat16 for i in [dy])
                            assert all(i.is_contiguous() for i in [dy])
                            r,w,k,v,a,b,s,sa = ctx.saved_tensors
                            B,T,H,C = r.shape
                            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                            ds0 = torch.empty(B,H,C,C,dtype=torch.float32,device=r.device)
                            torch.ops.rwkv7_state_clampw.backward(r,w,k,v,a,b,dy,s,sa,ds0,dr,dw,dk,dv,da,db)
                            return ds0,dr,dw,dk,dv,da,db
                    def RUN_RWKV7_STATE(r,k,v,w,a,b,s0):
                        B,T,HC = r.shape
                        C = HEAD_SIZE
                        H = HC//C
                        s0 = s0.float().repeat(B, 1, 1, 1)
                        r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                        return RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0,r,w,k,v,a,b).view(B,T,HC), None

                elif os.environ["RWKV_FLOAT_MODE"] == 'fp32':
                    flags = [f'-D_N_={HEAD_SIZE}', "-D_FP32_", f"-D_CHUNK_LEN_={CHUNK_LEN}", "-xhip", "-fopenmp", "-ffast-math", "-O3", "-munsafe-fp-atomics", "--save-temps", '-DAMD']
                    load(name="rwkv7_state_clampw", sources=[f'hip/rwkv7_state_clampw.hip', 'hip/rwkv7_state_clampw_op.hip'], is_python_module=False, verbose=True, extra_cflags=["-D_FP32_"], extra_cuda_cflags=flags)

                    class RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,s0,r,w,k,v,a,b):
                            B,T,H,C = r.shape 
                            assert T%CHUNK_LEN == 0
                            assert all(i.dtype==torch.float32 for i in [s0,r,w,k,v,a,b])
                            assert all(i.is_contiguous() for i in [s0,r,w,k,v,a,b])
                            y = torch.empty_like(v)
                            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                            torch.ops.rwkv7_state_clampw.forward(s0,r,w,k,v,a,b,y,s,sa)
                            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                            return y
                        @staticmethod
                        def backward(ctx,dy):
                            assert all(i.dtype==torch.float32 for i in [dy])
                            assert all(i.is_contiguous() for i in [dy])
                            r,w,k,v,a,b,s,sa = ctx.saved_tensors
                            B,T,H,C = r.shape
                            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                            ds0 = torch.empty(B,H,C,C,dtype=torch.float32,device=r.device)
                            torch.ops.rwkv7_state_clampw.backward(r,w,k,v,a,b,dy,s,sa,ds0,dr,dw,dk,dv,da,db)
                            return ds0,dr,dw,dk,dv,da,db
                    def RUN_RWKV7_STATE(r,k,v,w,a,b,s0):
                        B,T,HC = r.shape
                        C = HEAD_SIZE
                        H = HC//C
                        s0 = s0.repeat(B, 1, 1, 1)
                        r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
                        return RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0,r,w,k,v,a,b).view(B,T,HC), None

                else:
                    raise NotImplementedError("Unsupported precision for RWKV7 fine-tuning")

            else:
                if os.environ["RWKV_FLOAT_MODE"] == 'bf16':
                    flags = [f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "-xhip", "-fopenmp", "-ffast-math", "-O3", "-munsafe-fp-atomics", "--save-temps", '-DAMD']
                    load(name="rwkv7_clampw", sources=[f'hip/rwkv7_clampw.hip', 'hip/rwkv7_clampw_op.hip'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
                    class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,r,w,k,v,a,b):
                            B,T,H,C = r.shape 
                            assert T%CHUNK_LEN == 0
                            assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
                            assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
                            y = torch.empty_like(v)
                            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                            torch.ops.rwkv7_clampw.forward(r,w,k,v,a,b,y,s,sa)
                            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                            return y
                        @staticmethod
                        def backward(ctx,dy):
                            assert all(i.dtype==torch.bfloat16 for i in [dy])
                            assert all(i.is_contiguous() for i in [dy])
                            r,w,k,v,a,b,s,sa = ctx.saved_tensors
                            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                            torch.ops.rwkv7_clampw.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
                            return dr,dw,dk,dv,da,db

                    def RUN_CUDA_RWKV7g(r,w,k,v,a,b):
                        B,T,HC = r.shape
                        r,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [r,w,k,v,a,b]]
                        return RWKV7_CLAMPW_CUDA_OP.apply(r,w,k,v,a,b).view(B,T,HC)

                elif os.environ["RWKV_FLOAT_MODE"] == 'fp32':
                    flags = [f'-D_N_={HEAD_SIZE}', "-D_FP32_", f"-D_CHUNK_LEN_={CHUNK_LEN}", "-xhip", "-fopenmp", "-ffast-math", "-O3", "-munsafe-fp-atomics", "--save-temps", '-DAMD']
                    load(name="rwkv7_clampw", sources=[f'hip/rwkv7_clampw.hip', 'hip/rwkv7_clampw_op.hip'], is_python_module=False, verbose=True, extra_cflags=["-D_FP32_"], extra_cuda_cflags=flags)
                    class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx,r,w,k,v,a,b):
                            B,T,H,C = r.shape 
                            assert T%CHUNK_LEN == 0
                            assert all(i.dtype==torch.float32 for i in [r,w,k,v,a,b])
                            assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
                            y = torch.empty_like(v)
                            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                            torch.ops.rwkv7_clampw.forward(r,w,k,v,a,b,y,s,sa)
                            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
                            return y
                        @staticmethod
                        def backward(ctx,dy):
                            assert all(i.dtype==torch.float32 for i in [dy])
                            assert all(i.is_contiguous() for i in [dy])
                            r,w,k,v,a,b,s,sa = ctx.saved_tensors
                            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
                            torch.ops.rwkv7_clampw.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
                            return dr,dw,dk,dv,da,db
                    def RUN_CUDA_RWKV7g(r,w,k,v,a,b):
                        B,T,HC = r.shape
                        r,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [r,w,k,v,a,b]]
                        return RWKV7_CLAMPW_CUDA_OP.apply(r,w,k,v,a,b).view(B,T,HC)

                else:
                    raise NotImplementedError("Unsupported precision for RWKV7 fine-tuning")

    elif 'x060' in os.environ["RWKV_MY_TESTING"]:
        if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
            wkv6state_cuda = load(name="wkv6infctx", sources=["cuda/wkv6infctx_op.cpp", f"cuda/wkv6infctx_cuda.cu"],
                            verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
                
            class WKV_6STATE(torch.autograd.Function):
                @staticmethod
                def forward(ctx, B, T, C, H, r, k, v, w, u, s):
                    with torch.no_grad():
                        assert r.dtype == torch.bfloat16
                        assert k.dtype == torch.bfloat16
                        assert v.dtype == torch.bfloat16
                        assert w.dtype == torch.bfloat16
                        assert u.dtype == torch.bfloat16
                        assert s.dtype == torch.bfloat16
                        assert HEAD_SIZE == C // H
                        ctx.B = B
                        ctx.T = T
                        ctx.C = C
                        ctx.H = H
                        assert r.is_contiguous()
                        assert k.is_contiguous()
                        assert v.is_contiguous()
                        assert w.is_contiguous()
                        assert u.is_contiguous()
                        assert s.is_contiguous()
                        ctx.save_for_backward(r, k, v, w, u, s)
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
                        return y

                @staticmethod
                def backward(ctx, gy):
                    with torch.no_grad():
                        assert gy.dtype == torch.bfloat16
                        B = ctx.B
                        T = ctx.T
                        C = ctx.C
                        H = ctx.H
                        assert gy.is_contiguous()
                        r, k, v, w, u, s = ctx.saved_tensors
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                        gu = torch.sum(gu, 0).view(H, C//H)
                        gs = torch.sum(gs, 0).view(H, C//H, C//H)
                        return (None, None, None, None, gr, gk, gv, gw, gu, gs)

            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                x = WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)
                return x, s
        elif os.environ["RWKV_TRAIN_TYPE"] == 'state':
            wkv6state_cuda = load(name="wkv6state", sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda.cu"],
                            verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
                
            class WKV_6STATE(torch.autograd.Function):
                @staticmethod
                def forward(ctx, B, T, C, H, r, k, v, w, u, s):
                    with torch.no_grad():
                        assert r.dtype == torch.bfloat16
                        assert k.dtype == torch.bfloat16
                        assert v.dtype == torch.bfloat16
                        assert w.dtype == torch.bfloat16
                        assert u.dtype == torch.bfloat16
                        assert s.dtype == torch.bfloat16
                        assert HEAD_SIZE == C // H
                        ctx.B = B
                        ctx.T = T
                        ctx.C = C
                        ctx.H = H
                        assert r.is_contiguous()
                        assert k.is_contiguous()
                        assert v.is_contiguous()
                        assert w.is_contiguous()
                        assert u.is_contiguous()
                        assert s.is_contiguous()
                        ctx.save_for_backward(r, k, v, w, u, s)
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
                        return y

                @staticmethod
                def backward(ctx, gy):
                    with torch.no_grad():
                        assert gy.dtype == torch.bfloat16
                        B = ctx.B
                        T = ctx.T
                        C = ctx.C
                        H = ctx.H
                        assert gy.is_contiguous()
                        r, k, v, w, u, s = ctx.saved_tensors
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                        gu = torch.sum(gu, 0).view(H, C//H)
                        gs = torch.sum(gs, 0).view(H, C//H, C//H)
                        return (None, None, None, None, gr, gk, gv, gw, gu, gs)

            def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
                return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)

        else:
            wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                            verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
                
            class WKV_6(torch.autograd.Function):
                @staticmethod
                def forward(ctx, B, T, C, H, r, k, v, w, u):
                    with torch.no_grad():
                        assert r.dtype == torch.bfloat16
                        assert k.dtype == torch.bfloat16
                        assert v.dtype == torch.bfloat16
                        assert w.dtype == torch.bfloat16
                        assert u.dtype == torch.bfloat16
                        assert HEAD_SIZE == C // H
                        ctx.B = B
                        ctx.T = T
                        ctx.C = C
                        ctx.H = H
                        assert r.is_contiguous()
                        assert k.is_contiguous()
                        assert v.is_contiguous()
                        assert w.is_contiguous()
                        assert u.is_contiguous()
                        ew = (-torch.exp(w.float())).contiguous()
                        ctx.save_for_backward(r, k, v, ew, u)
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
                        return y

                @staticmethod
                def backward(ctx, gy):
                    with torch.no_grad():
                        assert gy.dtype == torch.bfloat16
                        B = ctx.B
                        T = ctx.T
                        C = ctx.C
                        H = ctx.H
                        assert gy.is_contiguous()
                        r, k, v, ew, u = ctx.saved_tensors
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                        wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
                        gu = torch.sum(gu, 0).view(H, C//H)
                        return (None, None, None, None, gr, gk, gv, gw, gu)

            def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
                return WKV_6.apply(B, T, C, H, r, k, v, w, u)
    else:
        wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda.cu"],
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
            
        class WKV_5(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u):
                with torch.no_grad():
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
                    assert HEAD_SIZE == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    ew = (-torch.exp(w.float())).contiguous()
                    eew = (torch.exp(ew)).contiguous()
                    ctx.save_for_backward(r, k, v, eew, ew, u)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, eew, ew, u = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                    gw = torch.sum(gw, 0).view(H, C//H)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    return (None, None, None, None, gr, gk, gv, gw, gu)

        def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
            return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################


