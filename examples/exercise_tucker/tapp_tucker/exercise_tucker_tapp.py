import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
import tensorly as tl
import string
from ctypes import *
import os

# TODO:
#   1. Fill in the arguments for tucker_to_tensor_contraction (74)
#   2. Uncomment function call tucker_to_tensor_tapp_helper (Line 94)
#   3. Uncomment function call tucker_to_tensor_tapp_helper (Line 99)
#   4. Fill in the arguments for tucker_to_tensor_tapp (Line 113)

if os.name == 'nt':
    tucker_to_tensor_contraction = CDLL(os.path.dirname(__file__) + '/lib/libexercise_tucker.dll').tucker_to_tensor_contraction
else:
    tucker_to_tensor_contraction = CDLL(os.path.dirname(__file__) + '/lib/libexercise_tucker.so').tucker_to_tensor_contraction

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

def tucker_to_tensor_tapp_helper(A, B, idx_A, idx_B, idx_D):
    # Extract data from np.array, format for input to C function
    nmode_A = c_int(A.ndim)
    extents_A = (c_int64 * len(A.shape))(*A.shape)
    strides_A = (c_int64 * len(A.strides))(*[s // A.itemsize for s in A.strides])
    
    # Extract data from np.array, format for input to C function
    nmode_B = c_int(B.ndim)
    extents_B = (c_int64 * len(B.shape))(*B.shape)
    strides_B = (c_int64 * len(B.strides))(*[s // B.itemsize for s in B.strides])
    
    # Extract data from np.array, format for input to C function
    idx_A_c = (c_int64 * len(idx_A))(*idx_A)
    idx_B_c = (c_int64 * len(idx_B))(*idx_B)
    idx_D_c = (c_int64 * len(idx_D))(*idx_D)

    # Create output tensor D and format for use in C function
    shape_D = [{**dict(zip(idx_A, extents_A)), **dict(zip(idx_B, extents_B))}[idx] for idx in idx_D]
    D = np.array(np.random.rand(*shape_D))
    nmode_D = c_int(D.ndim)
    extents_D = (c_int64 * len(D.shape))(*D.shape)
    strides_D = (c_int64 * len(D.strides))(*[s // D.itemsize for s in D.strides])
    
    # Pointer to data for use in C function
    A_ptr = A.ctypes.data_as(POINTER(c_double))
    B_ptr = B.ctypes.data_as(POINTER(c_double))
    D_ptr = D.ctypes.data_as(POINTER(c_double))
    
    # Execute C function
    # TODO 1: Fill in the arguments
    # Look in exercise_tucker.c for reference. The input names should be the same
    # except for t.ex. A which would be A_ptr and idx_A which would be idx_A_c 
    tucker_to_tensor_contraction(, , , ,
                                 , , , ,
                                 , , , ,
                                 ,
                                 ,
                                 )
    return D

def tucker_to_tensor_tapp(core, factors):
    nmode = core.ndim
    
    # Create subscripts
    idx_core = list(range(1, nmode + 1))
    idx_factor = [[nmode+1+i, idx_core[i]] for i in range(nmode)]
    idx_output = [[s for s in idx_core + idx_factor[0] if s not in set(idx_core) & set(idx_factor[0])]]
    for i in range(1, nmode):
        idx_output = idx_output + [[s for s in idx_output[i - 1] + idx_factor[i] if s not in set(idx_factor[i]) & set(idx_output[i - 1])]]

    # Contracting first factor
    # TODO 2: Uncomment function call
    result = #tucker_to_tensor_tapp_helper(core, factors[0], list(idx_core), list(idx_factor[0]), list(idx_output[0]))
    
    # Further contracting factors
    for i in range(1, nmode):
        # TODO 3: Uncomment function call
        result = #tucker_to_tensor_tapp_helper(result, factors[i], list(idx_output[i - 1]), list(idx_factor[i]), list(idx_output[i]))

    return result

# Load image
image = Image.open(os.path.dirname(__file__) + '/example_img.png').resize((128, 128))
# Format data
image_np = np.array(image) / 255.0

# Compress image
core, factors = tucker(image_np, rank=[50, 50, 3])

# Reconstruct image
# TODO 4: Fill in the arguments, using the inputs with the same name
reconstructed_tapp = tucker_to_tensor_tapp(, )

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display original image
axes[0].imshow(image_np)
axes[0].set_title("Original")
axes[0].axis('off')

# Display reconstructed
axes[1].imshow(np.clip(reconstructed_tapp, 0, 1))
axes[1].set_title("Compressed")
axes[1].axis('off')

plt.tight_layout()
plt.show()