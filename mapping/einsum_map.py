# Niklas Hörnblad
# Paolo Bientinesi
# Umeå University - July 2024

from ctypes import *
import numpy as np
import re

so_file = "/home/niklas/Documents/Tensor_Product/Tensor_Product/lib/product.so"

product = CDLL(so_file).PRODUCT
product.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_float, c_float, c_bool, c_bool, c_bool, c_char_p]
product.restype = None

def einsum(subscripts, *operands, out=None, dype=None, order='K', casting='safe', optimize=False):
    IndicesA, IndicesB, IndicesD = re.split(',|->', subscripts.replace(' ', ''))
    IndicesA = IndicesA[::-1]
    IndicesB = IndicesB[::-1]
    IndicesD = IndicesD[::-1]
    EINSUM = IndicesA + ', ' + IndicesB + ' -> ' + IndicesD
    EINSUM = c_char_p(EINSUM.encode('utf-8'))

    operands = list(operands)

    IDXA = operands[0].ndim
    EXTA = list(operands[0].shape)[::-1]
    STRA = [int(x/8) for x in list(operands[0].strides)][::-1]
    A = operands[0].flatten()

    IDXB = operands[1].ndim
    EXTB = list(operands[1].shape)[::-1]
    STRB = [int(x/8) for x in list(operands[1].strides)][::-1]
    B = operands[1].flatten()
    
    if out is None:
        IDXC = len(IndicesD)
        EXTC = [0]*IDXC
        STRC = [1]*IDXC

        IDXD = len(IndicesD)
        EXTD = [0]*IDXD
        STRD = [1]*IDXD

        for char in IndicesD: 
            if char in IndicesA:
                EXTC[IndicesD.index(char)] = EXTA[IndicesA.index(char)]
                EXTD[IndicesD.index(char)] = EXTA[IndicesA.index(char)]
                if IndicesD.index(char) > 0:
                    STRC[IndicesD.index(char)] = STRC[IndicesD.index(char) - 1] * EXTC[IndicesD.index(char) - 1]
                    STRD[IndicesD.index(char)] = STRD[IndicesD.index(char) - 1] * EXTD[IndicesD.index(char) - 1]
            elif char in IndicesB:
                EXTC[IndicesD.index(char)] = EXTB[IndicesB.index(char)]
                EXTD[IndicesD.index(char)] = EXTB[IndicesB.index(char)]
                if IndicesD.index(char) > 0:
                    STRC[IndicesD.index(char)] = STRC[IndicesD.index(char) - 1] * EXTC[IndicesD.index(char) - 1]
                    STRD[IndicesD.index(char)] = STRD[IndicesD.index(char) - 1] * EXTD[IndicesD.index(char) - 1]

        C = np.zeros(EXTC).flatten()
        D = np.zeros(EXTD).flatten()
    else:
        IDXC = out.ndim
        EXTC = list(out.shape)[::-1]
        STRC = [1] * IDXC
        C = np.zeros(EXTC).flatten()

        IDXD = out.ndim
        EXTD = list(out.shape)[::-1]
        STRD = [1] * IDXD
        D = out.flatten()

        for i in range(IDXC - 1):
            STRC[i + 1] = STRC[i] * EXTC[i]
            STRD[i + 1] = STRD[i] * EXTD[i]

    IDXA = c_int(IDXA)
    EXTA = (c_int * len(EXTA))(*EXTA)
    STRA = (c_int * len(STRA))(*STRA)
    A = (c_float * len(A))(*A)

    IDXB = c_int(IDXB)
    EXTB = (c_int * len(EXTB))(*EXTB)
    STRB = (c_int * len(STRB))(*STRB)
    B = (c_float * len(B))(*B)

    IDXC = c_int(IDXC)
    EXTC = (c_int * len(EXTC))(*EXTC)
    STRC = (c_int * len(STRC))(*STRC)
    C = (c_float * len(C))(*C)

    IDXD = c_int(IDXD)
    EXTD = (c_int * len(EXTD))(*EXTD)
    STRD = (c_int * len(STRD))(*STRD)
    D = (c_float * len(D))(*D)

    ALPHA = 1
    BETA = 1

    product(IDXA, EXTA, STRA, A,
            IDXB, EXTB, STRB, B,
            IDXC, EXTC, STRC, C,
            IDXD, EXTD, STRD, D,
            ALPHA, BETA, False, False, False, EINSUM)
    
    D = np.array(D)
    
    if out is not None:
        if out.shape == ():
            out[()] = D[0]
        else:
            out[:] = D.reshape(tuple(EXTD[::-1]))
        return out
    
    return D.reshape(tuple(EXTD[::-1]))