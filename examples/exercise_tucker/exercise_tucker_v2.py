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
    tapp_so = os.path.dirname(__file__) + '/../../lib/libtapp.dll'
else:
    tapp_so = os.path.dirname(__file__) + '/../../lib/libtapp.so'

TAPP_create_tensor_product = CDLL(tapp_so).TAPP_create_tensor_product
TAPP_create_tensor_product.restype = c_int
TAPP_create_tensor_product.argtypes = [POINTER(c_int32 if platform.architecture()[0] == '32bit' else c_int64), # plan
									   c_int32 if platform.architecture()[0] == '32bit' else c_int64, # handle
									   c_int, # op_A
									   c_int32 if platform.architecture()[0] == '32bit' else c_int64, # A
									   POINTER(c_int64), # idx_A
									   c_int, # op_B
									   c_int32 if platform.architecture()[0] == '32bit' else c_int64, # B
									   POINTER(c_int64), # idx_B
									   c_int, # op_C
									   c_int32 if platform.architecture()[0] == '32bit' else c_int64, # C
									   POINTER(c_int64), # idx_C
									   c_int, # op_D
									   c_int32 if platform.architecture()[0] == '32bit' else c_int64, # D
									   POINTER(c_int64), # idx_D
									   c_int, # prec
									   ]

TAPP_destory_tensor_product = CDLL(tapp_so).TAPP_destory_tensor_product
TAPP_destory_tensor_product.restype = c_int
TAPP_destory_tensor_product.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64 # plan 
										]

TAPP_execute_product = CDLL(tapp_so).TAPP_execute_product
TAPP_execute_product.restype = c_int
TAPP_execute_product.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # plan
								 c_int32 if platform.architecture()[0] == '32bit' else c_int64, # exec
								 POINTER(c_int32 if platform.architecture()[0] == '32bit' else c_int64), # status
								 c_void_p, # alpha
								 c_void_p, # A
								 c_void_p, # B
								 c_void_p, # beta
								 c_void_p, # C
								 c_void_p, # D
								 ]

TAPP_create_tensor_info = CDLL(tapp_so).TAPP_create_tensor_info
TAPP_create_tensor_info.restype = c_int
TAPP_create_tensor_info.argtypes = [POINTER(c_int32 if platform.architecture()[0] == '32bit' else c_int64), # info
                                    c_int, # type
                                    c_int, # nmode
                                    POINTER(c_int64), # extents
                                    POINTER(c_int64), # strides
									]

TAPP_destroy_tensor_info = CDLL(tapp_so).TAPP_destory_tensor_info
TAPP_destroy_tensor_info.restype = c_int
TAPP_destroy_tensor_info.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64 # info
									 ]

TAPP_get_nmodes = CDLL(tapp_so).TAPP_get_nmodes
TAPP_get_nmodes.restype = c_int
TAPP_get_nmodes.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							]

TAPP_set_nmodes = CDLL(tapp_so).TAPP_set_nmodes
TAPP_set_nmodes.restype = c_int
TAPP_set_nmodes.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							c_int, # nmodes
							]

TAPP_get_extents = CDLL(tapp_so).TAPP_get_extents
TAPP_get_extents.restype = None
TAPP_get_extents.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							 POINTER(c_int64), # extents
							 ]

TAPP_set_extents = CDLL(tapp_so).TAPP_set_extents
TAPP_set_extents.restype = c_int
TAPP_set_extents.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							 POINTER(c_int64), # extents
							 ]

TAPP_get_strides = CDLL(tapp_so).TAPP_get_strides
TAPP_get_strides.restype = None
TAPP_get_strides.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							 POINTER(c_int64), # strides
							 ]

TAPP_set_strides = CDLL(tapp_so).TAPP_set_strides
TAPP_set_strides.restype = c_int
TAPP_set_strides.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							 POINTER(c_int64), # strides
							 ]

create_executor = CDLL(tapp_so).create_executor
create_executor.restype = c_int
create_executor.argtypes = [POINTER(c_int32 if platform.architecture()[0] == '32bit' else c_int64) # exec
							]

TAPP_destroy_executor = CDLL(tapp_so).TAPP_destroy_executor
TAPP_destroy_executor.restype = c_int
TAPP_destroy_executor.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64 # exec
								  ]

create_handle = CDLL(tapp_so).create_handle
create_handle.restype = c_int
create_handle.argtypes = [POINTER(c_int32 if platform.architecture()[0] == '32bit' else c_int64) # handle
						  ]

TAPP_destroy_handle = CDLL(tapp_so).TAPP_destroy_handle
TAPP_destroy_handle.restype = c_int
TAPP_destroy_handle.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64 # handle
								]

def Product(alpha, A, B, beta, C, D, idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D):
	nmode_A = c_int(A.ndim)
	extents_A = (c_int64 * len(A.shape))(*A.shape)
	strides_A = (c_int64 * len(A.strides))(*[s // A.itemsize for s in A.strides])
	
	tensor_info_A = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	TAPP_create_tensor_info(byref(tensor_info_A), 1 if A.dtype == 'float64' else 3, nmode_A, extents_A, strides_A)

	nmode_B = c_int(B.ndim)
	extents_B = (c_int64 * len(B.shape))(*B.shape)
	strides_B = (c_int64 * len(B.strides))(*[s // B.itemsize for s in B.strides])

	tensor_info_B = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	TAPP_create_tensor_info(byref(tensor_info_B), 1 if A.dtype == 'float64' else 3, nmode_B, extents_B, strides_B)

	nmode_C = c_int(C.ndim)
	extents_C = (c_int64 * len(C.shape))(*C.shape)
	strides_C = (c_int64 * len(C.strides))(*[s // C.itemsize for s in C.strides])

	tensor_info_C = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	TAPP_create_tensor_info(byref(tensor_info_C), 1 if A.dtype == 'float64' else 3, nmode_C, extents_C, strides_C)

	nmode_D = c_int(D.ndim)
	extents_D = (c_int64 * len(D.shape))(*D.shape)
	strides_D = (c_int64 * len(D.strides))(*[s // D.itemsize for s in D.strides])

	tensor_info_D = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	TAPP_create_tensor_info(byref(tensor_info_D), 1 if A.dtype == 'float64' else 3, nmode_D, extents_D, strides_D)

	idx_A_c = (c_int64 * len(idx_A))(*idx_A)
	idx_B_c = (c_int64 * len(idx_B))(*idx_B)
	idx_C_c = (c_int64 * len(idx_C))(*idx_C)
	idx_D_c = (c_int64 * len(idx_D))(*idx_D)

	handle = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	create_handle(byref(handle))

	plan = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	TAPP_create_tensor_product(byref(plan), handle,
							c_int(op_A), tensor_info_A, idx_A_c,
							c_int(op_B), tensor_info_B, idx_B_c,
							c_int(op_C), tensor_info_C, idx_C_c,
							c_int(op_D), tensor_info_D, idx_D_c,
							-1)
	
	exec = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	create_executor(byref(exec))
	status = c_int64(0)
	A_ptr = A.ctypes.data_as(POINTER(c_double))
	B_ptr = B.ctypes.data_as(POINTER(c_double))
	C_ptr = C.ctypes.data_as(POINTER(c_double))
	D_ptr = D.ctypes.data_as(POINTER(c_double))
	
	TAPP_execute_product(plan, exec, byref(status), byref(c_double(alpha)), A_ptr, B_ptr, byref(c_double(beta)), C_ptr, D_ptr)

	TAPP_destroy_handle(handle)
	TAPP_destroy_executor(exec)
	TAPP_destory_tensor_product(plan)
	TAPP_destroy_tensor_info(tensor_info_A)
	TAPP_destroy_tensor_info(tensor_info_B)
	TAPP_destroy_tensor_info(tensor_info_C)
	TAPP_destroy_tensor_info(tensor_info_D)

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
    result = Product(1, core, factors[0], 0, None, ) np.einsum(einsum_str, core, factors[0])
    
    for i in range(1, ndim):
        einsum_str = f"{output_subs[i - 1]}," + factor_subs[i] + f"->{output_subs[i]}"
        result = np.einsum(einsum_str, result, factors[i])

    print(result)
    return result

tl.set_backend('numpy')

image = Image.open('your_image.png').resize((128, 128))
image_np = np.array(image) / 255.0

core, factors = tucker(image_np, rank=[50, 50, 3])

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

