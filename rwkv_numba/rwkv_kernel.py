from numba import cuda, float32, int32
import numpy as np
import math
import torch
@cuda.jit
def kernel_wkv(C, _w, _u, _k, _v, _y, _aa, _bb, _pp, _lens, numset):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    _b = idx // C
    _c = idx % C
    _b_offset = 0
    if _b-1 >= 0:
        #_b_offset = (_lens[_b-1])*C
        _b_offset = (numset[_b-1])*C
    _offset = _b_offset

    _state_offset = _b * C + _c

    tim = _lens[_b]
    u = _u[_c]
    w = _w[_c]
    k = _k  #[B, C]
    v = _v
    y = _y
    aa = _aa[_state_offset]
    bb = _bb[_state_offset]
    pp = _pp[_state_offset]
    for i in range(tim):
        ii = i * C + _offset + _c
        kk = float32(k[ii])
        vv = float32(v[ii])
        ww = u + kk
        p = max(pp, ww)
        e1 = math.exp(pp - p)
        e2 = math.exp(ww - p)
        y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2)
        ww = w + pp
        p = max(ww, kk)
        e1 = math.exp(ww - p)
        e2 = math.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = p

    _aa[_state_offset] = aa
    _bb[_state_offset] = bb
    _pp[_state_offset] = pp



def wkv(B, T, C, w, u, k, v, y, aa, bb, pp, lens, numset):
    threads_per_block = min(C, 32)
    blocks_per_grid = B * C // threads_per_block
    k = k.reshape(-1)
    v = v.reshape(-1)
    aa = aa.reshape(-1)
    bb = bb.reshape(-1)
    pp = pp.reshape(-1)
    kernel_wkv[blocks_per_grid, threads_per_block](C, w, u, k, v, y, aa, bb, pp, lens, numset)
    y = y.reshape(T, C)
    aa = aa.reshape(B, C)
    bb = bb.reshape(B, C)
    pp = pp.reshape(B, C)

    return y, aa, bb, pp

