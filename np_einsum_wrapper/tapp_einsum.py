from ctypes import *
import platform
from numpy import zeros as npzeroes, array as nparray, dtype as npdtype

tapp_so = "/home/niklash/Documents/TAPP/reference-implementation/lib/libtapp.so"

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

TAPP_destroy_tensor_product = CDLL(tapp_so).TAPP_destroy_tensor_product
TAPP_destroy_tensor_product.restype = c_int
TAPP_destroy_tensor_product.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64 # plan 
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

TAPP_destroy_tensor_info = CDLL(tapp_so).TAPP_destroy_tensor_info
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

def translate_dtype(dtype):
    match dtype:
        case 'float32':
            return 0
        case 'float64':
            return 1
        case 'complex64':
            return 2
        case 'complex128':
            return 3
        case 'float16':
            return 4

def c_intptr():
    match platform.architecture()[0]:
        case '16bit':
            return c_int16(0)
        case '32bit':
            return c_int32(0)
        case '64bit':
            return c_int64(0)

def convert_metadata_to_c(tensor):
    return c_int(tensor.ndim), (c_int64 * len(tensor.shape))(*tensor.shape), (c_int64 * len(tensor.strides))(*[s // tensor.itemsize for s in tensor.strides])

def convert_array_to_TAPP_tensor_info(tensor):
    nmode, extents, strides = convert_metadata_to_c(tensor)
    tensor_info = c_intptr()
    TAPP_create_tensor_info(byref(tensor_info), translate_dtype(tensor.dtype), nmode, extents, strides)
    return tensor_info

def convert_string_to_TAPP_idx(idx):
    int_idx = list(map(ord, list(idx)))
    return (c_int64 * len(int_idx))(*int_idx)

def get_data_ptr(array):
    return array.ctypes.data_as(POINTER(c_void_p))

def determine_dtype(casting_type, type_A, type_B):
    if casting_type is not None:
        return casting_type
    else:
        if type_A == type_B:
            return type_A
        
        if 'complex' in str(type_A):
            size_A = int(str(type_A).lstrip('complex')) / 2
        else:
            size_A = int(str(type_A).lstrip('float'))
        
        if 'complex' in str(type_B):
            size_B = int(str(type_B).lstrip('complex')) / 2
        else:
            size_B = int(str(type_B).lstrip('float'))

        if 'complex' in str(type_A) or 'complex' in str(type_B):
            return npdtype('complex' + str(2 * size_A if size_A >= size_B else 2 * size_B))
        
        return  npdtype('float' + str(size_A if size_A >= size_B else size_B))

def determine_precision(dtype):
    if dtype is None:
        return -1
    match dtype:
        case 'float32'|'complex64':
            return 0
        case 'float64' | 'complex128':
            return 1
        case 'float16':
            return 3


def tapp_einsum(einsum_str, A, B, dtype=None, casting='safe', out=None):
    if dtype is not None:
        dtype = npdtype(dtype)

    input_str, idx_D = einsum_str.split('->')
    idx_A, idx_B = input_str.split(',')

    if out is None:
        out = npzeroes([A.shape[idx_A.index(idx)] if idx in idx_A else B.shape[idx_B.index(idx)] for idx in idx_D], determine_dtype(dtype, A.dtype, B.dtype))
    
    tensor_info_A = convert_array_to_TAPP_tensor_info(A)
    tensor_info_B = convert_array_to_TAPP_tensor_info(B)
    tensor_info_D = convert_array_to_TAPP_tensor_info(out)

    idx_A_c = convert_string_to_TAPP_idx(idx_A)
    idx_B_c = convert_string_to_TAPP_idx(idx_B)
    idx_D_c = convert_string_to_TAPP_idx(idx_D)

    handle = c_intptr()
    create_handle(byref(handle))

    plan = c_intptr()
    TAPP_create_tensor_product(byref(plan), handle,
                            c_int(0), tensor_info_A, idx_A_c,
                            c_int(0), tensor_info_B, idx_B_c,
                            c_int(0), tensor_info_D, idx_D_c,
                            c_int(0), tensor_info_D, idx_D_c,
                            c_int(determine_precision(dtype)))
    
    exec = c_intptr()
    create_executor(byref(exec))
    status = c_int64(0)
    A_ptr = get_data_ptr(A)
    B_ptr = get_data_ptr(B)
    D_ptr = get_data_ptr(out)

    alpha = nparray([1], out.dtype)
    beta = nparray([0], out.dtype)

    alpha_ptr = get_data_ptr(alpha)
    beta_ptr = get_data_ptr(beta)
    
    TAPP_execute_product(plan, exec, byref(status), alpha_ptr, A_ptr, B_ptr, beta_ptr, None, D_ptr)

    TAPP_destroy_handle(handle)
    TAPP_destroy_executor(exec)
    TAPP_destroy_tensor_product(plan)
    TAPP_destroy_tensor_info(tensor_info_A)
    TAPP_destroy_tensor_info(tensor_info_B)
    TAPP_destroy_tensor_info(tensor_info_D)

    return out