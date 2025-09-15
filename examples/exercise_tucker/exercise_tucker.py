import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
import tensorly as tl
import string
import ctypes
import os

if os.name == 'nt':
    tucker_to_tensor_contraction = ctypes.CDLL(os.path.dirname(__file__) + '/../lib/exercise2.dll').tucker_to_tensor_contraction
else:
    tucker_to_tensor_contraction = ctypes.CDLL(os.path.dirname(__file__) + '/../lib/exercise2.so').tucker_to_tensor_contraction

tucker_to_tensor_contraction.argtypes = [
    c_int, # nmode A
    POINTER(c_int64), # extents A
    POINTER(c_int64), # strides A
    c_void_p, # A
    c_int, # nmode B
    POINTER(c_int64), # extents B
    POINTER(c_int64), # strides B
    c_void_p, # B
    c_int, # nmode D
    POINTER(c_int64), # extents D
    POINTER(c_int64), # strides D
    c_void_p, # D
    POINTER(c_int64), # idx_A
    POINTER(c_int64), # idx_B
    POINTER(c_int64), # idx_D
    ]

tucker_to_tensor_contraction.restype = c_void_p

def tucker_to_tensor_helper(A, B, idx_A, idx_B, idx_D):
    nmode_A = c_int(A.ndim)
    extents_A = (c_int64 * len(A.shape))(*A.shape)
    strides_A = (c_int64 * len(A.strides))(*[s // A.itemsize for s in A.strides])
    
    nmode_B = c_int(B.ndim)
    extents_B = (c_int64 * len(B.shape))(*B.shape)
    strides_B = (c_int64 * len(B.strides))(*[s // B.itemsize for s in B.strides])
    
    idx_A_c = (c_int64 * len(idx_A))(*idx_A)
    idx_B_c = (c_int64 * len(idx_B))(*idx_B)
    idx_D_c = (c_int64 * len(idx_D))(*idx_D)

    extents_D = [{**dict(zip(idx_A, extents_A)), **dict(zip(idx_B, extents_B))}[idx] for idx in idx_D]
    D = np.array(np.zeros(extents_D))
    nmode_D = c_int(D.ndim)
    strides_D = (c_int64 * len(D.strides))(*[s // D.itemsize for s in D.strides])
    
    A_ptr = A.ctypes.data_as(POINTER(c_double))
    B_ptr = B.ctypes.data_as(POINTER(c_double))
    D_ptr = D.ctypes.data_as(POINTER(c_double))
    
    D[:] = tucker_to_tensor_contraction(nmode_A, extents_A, strides_A, A_ptr,
                                        nmode_B, extents_B, strides_B, B_ptr,
                                        nmode_D, extents_D, strides_D, D_ptr,
                                        idx_A_c,
                                        idx_B_c,
                                        idx_D_c)
    return D

def tucker_to_tensor_tapp(core, factors):
    ndim = core.ndim
    subscripts = string.ascii_lowercase

    core_subs = subscripts[:ndim]
    factor_subs = [f"{subscripts[ndim+i]}{core_subs[i]}" for i in range(ndim)]
    output_subs = [''.join([s for s in core_subs + factor_subs[0] if s not in ''.join(set(core_subs) & set(factor_subs[0]))])]
    for i in range(1, ndim):
        output_subs = output_subs + [''.join([s for s in output_subs[i - 1] + factor_subs[i] if s not in ''.join(set(factor_subs[i]) & set(output_subs[i - 1]))])]

    einsum_str = f"{core_subs}," + factor_subs[0] + f"->{output_subs[0]}"
    result = np.einsum(einsum_str, core, factors[0])
    
    for i in range(1, ndim):
        einsum_str = f"{output_subs[i - 1]}," + factor_subs[i] + f"->{output_subs[i]}"
        result = np.einsum(einsum_str, result, factors[i])

    return result

tl.set_backend('numpy')

image = Image.open('example_img.png').resize((128, 128))
image_np = np.array(image) / 255.0


reconstructed_tly = tucker_to_tensor((core, factors))
reconstructed_tapp = tucker_to_tensor_tapp(core, factors)

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].imshow(image_np)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(np.clip(reconstructed_tly, 0, 1))
axes[1].set_title("Compressed (tensorly reconstruction)")
axes[1].axis('off')

axes[2].imshow(np.clip(reconstructed_tapp, 0, 1))
axes[2].set_title("Compressed (TAPP reconstruction)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

