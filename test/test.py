# Niklas Hörnblad
# Paolo Bientinesi
# Umeå University - June 2024

from ctypes import *
import numpy as np
import re
import random
from numpy.random import default_rng
import os.path
import platform

product_so = "/home/niklas/Documents/Tensor_Product/Tensor_Product/lib/product.so"
tensor_so = "/home/niklas/Documents/Tensor_Product/Tensor_Product/lib/tensor.so"

TAPP_create_tensor_product = CDLL(product_so).TAPP_create_tensor_product
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

TAPP_destory_tensor_product = CDLL(product_so).TAPP_destory_tensor_product
TAPP_destory_tensor_product.restype = c_int
TAPP_destory_tensor_product.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64 # plan 
										]

TAPP_execute_product = CDLL(product_so).TAPP_execute_product
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

TAPP_create_tensor_info = CDLL(tensor_so).TAPP_create_tensor_info
TAPP_create_tensor_info.restype = c_int
TAPP_create_tensor_info.argtypes = [POINTER(c_int32 if platform.architecture()[0] == '32bit' else c_int64), # info
                                    c_int, # type
                                    c_int, # nmode
                                    POINTER(c_int64), # extents
                                    POINTER(c_int64), # strides
									]

TAPP_destroy_tensor_info = CDLL(tensor_so).TAPP_destory_tensor_info
TAPP_destroy_tensor_info.restype = c_int
TAPP_destroy_tensor_info.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64 # info
									 ]

TAPP_get_nmodes = CDLL(tensor_so).TAPP_get_nmodes
TAPP_get_nmodes.restype = c_int
TAPP_get_nmodes.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							]

TAPP_set_nmodes = CDLL(tensor_so).TAPP_set_nmodes
TAPP_set_nmodes.restype = c_int
TAPP_set_nmodes.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							c_int, # nmodes
							]

TAPP_get_extents = CDLL(tensor_so).TAPP_get_extents
TAPP_get_extents.restype = None
TAPP_get_extents.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							 POINTER(c_int64), # extents
							 ]

TAPP_set_extents = CDLL(tensor_so).TAPP_set_extents
TAPP_set_extents.restype = c_int
TAPP_set_extents.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							 POINTER(c_int64), # extents
							 ]

TAPP_get_strides = CDLL(tensor_so).TAPP_get_strides
TAPP_get_strides.restype = None
TAPP_get_strides.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							 POINTER(c_int64), # strides
							 ]

TAPP_set_strides = CDLL(tensor_so).TAPP_set_strides
TAPP_set_strides.restype = c_int
TAPP_set_strides.argtypes = [c_int32 if platform.architecture()[0] == '32bit' else c_int64, # info
							 POINTER(c_int64), # strides
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

	plan = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	TAPP_create_tensor_product(byref(plan), 0,
							c_int(op_A), tensor_info_A, idx_A_c,
							c_int(op_B), tensor_info_B, idx_B_c,
							c_int(op_C), tensor_info_C, idx_C_c,
							c_int(op_D), tensor_info_D, idx_D_c,
							0)
	
	exec = c_int32(0) if platform.architecture()[0] == '32bit' else c_int64(0)
	status = c_int64(0)
	A_ptr = A.ctypes.data_as(POINTER(c_double))
	B_ptr = B.ctypes.data_as(POINTER(c_double))
	C_ptr = C.ctypes.data_as(POINTER(c_double))
	D_ptr = D.ctypes.data_as(POINTER(c_double))
	
	TAPP_execute_product(plan, exec, byref(status), byref(c_double(alpha)), A_ptr, B_ptr, byref(c_double(beta)), C_ptr, D_ptr)
	TAPP_destory_tensor_product(plan)
	TAPP_destroy_tensor_info(tensor_info_A)
	TAPP_destroy_tensor_info(tensor_info_B)
	TAPP_destroy_tensor_info(tensor_info_C)
	TAPP_destroy_tensor_info(tensor_info_D)

	return D

def RunEinsum(alpha, A, B, beta, C, D, idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D):
	if op_A == 1:
		A = np.conjugate(A)
	if op_B == 1:
		B = np.conjugate(B)
	if op_C == 1:
		C = np.conjugate(C)
	
	einsum = ''.join([chr(c) for c in idx_A]) + "," + ''.join([chr(c) for c in idx_B]) + "->" + ''.join([chr(c) for c in idx_D])
	
	np.einsum(einsum, A, B, out = D)
	np.multiply(D, alpha, out = D)
	np.add(D, C * beta, out = D)

	if op_D == 1:
		np.conjugate(D, out = D)
	
	return D

def main():
	print("Hadamard Product:", TestHadamardProduct())
	print("Contration:", TestContration())
	print("Commutativity:", TestCommutativity())
	print("Permutations:", TestPermutations())
	print("Equal Extents:", TestEqualExtents())
	print("Outer Product:", TestOuterProduct())
	print("Full Contraction:", TestFullContraction())
	print("Zero Dim Tensor Contraction:", TestZeroDimTensorContraction())
	print("One Dim Tensor Contraction:", TestOneDimTensorContraction())
	print("Subtensor Same Idx:", TestSubtensorSameIdx())
	print("Subtensor Lower Idx:", TestSubtensorLowerIdx())

def TestHadamardProduct():
	nmode = random.randint(1, 4)
	extents = list(np.random.randint(1, 4, nmode))

	is_complex = random.choice([True, False])

	A = np.array(np.random.rand(*extents) + np.random.rand(*extents) * (1.0j if is_complex else 0))
	B = np.array(np.random.rand(*extents) + np.random.rand(*extents) * (1.0j if is_complex else 0))
	C = np.array(np.random.rand(*extents) + np.random.rand(*extents) * (1.0j if is_complex else 0))
	D = np.array(np.random.rand(*extents) + np.random.rand(*extents) * (1.0j if is_complex else 0))

	alpha = random.random()
	beta = random.random()

	idx = [i for i in range(97, 97 + nmode)]

	E = D.copy()

	op_A = random.randint(0, 1)
	op_B = random.randint(0, 1)
	op_C = random.randint(0, 1)
	op_D = random.randint(0, 1)

	Product(alpha, A, B, beta, C, D, idx, idx, idx, idx, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A, B, beta, C, E, idx, idx, idx, idx, op_A, op_B, op_C, op_D)
	
	return np.allclose(D, E)

def TestContration():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration()
	E = D.copy()
	
	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	return np.allclose(D, E)

def TestCommutativity():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration()
	E = D.copy()
	F = D.copy()
	G = D.copy()

	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	Product(alpha, B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), F if not slicing_D else np.squeeze(F[*slicing_D]).reshape(extents_D), idx_B, idx_A, idx_C, idx_D, op_B, op_A, op_C, op_D)
	
	RunEinsum(alpha, B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), G if not slicing_D else np.squeeze(G[*slicing_D]).reshape(extents_D), idx_B, idx_A, idx_C, idx_D, op_B, op_A, op_C, op_D)
	
	return np.allclose(D, E) and np.allclose(E, G) and np.allclose(F, G) and np.allclose(D, F)

def TestPermutations():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration()

	ResultsE = np.array([])
	ResultsF = np.array([])

	for _ in range(len(idx_D)):
		E = D.copy()
		F = D.copy()

		Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)
		RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), F if not slicing_D else np.squeeze(F[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

		ResultsE = np.append(ResultsE, E)
		ResultsF = np.append(ResultsF, F)

		idx_C = idx_C[1:] + [idx_C[0]]
		idx_D = idx_D[1:] + [idx_D[0]]

		extents_C = extents_C[1:] + [extents_C[0]]
		extents_D = extents_D[1:] + [extents_D[0]]
	
	return np.allclose(ResultsE, ResultsF)

def TestEqualExtents():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration(equal_extents = True)
	E = D.copy()
	
	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)
	
	return np.allclose(D, E)

def TestOuterProduct():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration(contractions = 0)
	E = D.copy()
	
	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)
	
	return np.allclose(D, E)

def TestFullContraction():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration(nmode_D = 0)
	E = D.copy()
	
	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)
	
	return np.allclose(D, E)

def TestZeroDimTensorContraction():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration(nmode_A = 0)
	E = D.copy()
	
	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)
	
	return np.allclose(D, E)

def TestOneDimTensorContraction():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration(nmode_A = 1)
	E = D.copy()
	
	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)
	
	return np.allclose(D, E)

def TestSubtensorSameIdx():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration(lower_extents = True)
	E = D.copy()
	
	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)
	
	return np.allclose(D, E)

def TestSubtensorLowerIdx():
	A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D = GenerateContration(lower_extents = True, lower_idx = True)
	E = D.copy()
	
	Product(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), D if not slicing_D else np.squeeze(D[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)

	RunEinsum(alpha, A if not slicing_A else np.squeeze(A[*slicing_A]).reshape(extents_A), B if not slicing_B else np.squeeze(B[*slicing_B]).reshape(extents_B), beta, C if not slicing_C else np.squeeze(C[*slicing_C]).reshape(extents_C), E if not slicing_D else np.squeeze(E[*slicing_D]).reshape(extents_D), idx_A, idx_B, idx_C, idx_D, op_A, op_B, op_C, op_D)
	return np.allclose(D, E)

def GenerateContration(nmode_A = None, nmode_B = None, nmode_D = random.randint(0, 4),
					   contractions = random.randint(0, 4), equal_extents = False,
					   lower_extents = False, lower_idx = False):
	if nmode_A is None and nmode_B is None:
		nmode_A = random.randint(0, nmode_D)
		nmode_B = nmode_D - nmode_A
		nmode_A = nmode_A + contractions
		nmode_B = nmode_B + contractions
	elif nmode_A is None:
		contractions = random.randint(0, nmode_B) if contractions > nmode_B else contractions
		nmode_D = nmode_B - contractions + random.randint(0, 4) if nmode_D < nmode_B - contractions else nmode_D
		nmode_A = nmode_D - nmode_B + contractions * 2
	elif nmode_B is None:
		contractions = random.randint(0, nmode_A) if contractions > nmode_A else contractions
		nmode_D = nmode_A - contractions + random.randint(0, 4) if nmode_D < nmode_A - contractions else nmode_D
		nmode_B = nmode_D - nmode_A + contractions * 2
	else:
		contractions = random.randint(0, min(nmode_A, nmode_B))
		nmode_D = nmode_A + nmode_B - contractions * 2

	nmode_C = nmode_D

	idx_A = [i for i in range(97, 97 + nmode_A)]
	random.shuffle(idx_A)
	idx_B = random.sample(idx_A, contractions) + [i for i in range(97 + nmode_A, 97 + nmode_A + nmode_B - contractions)]
	random.shuffle(idx_B)
	idx_D = list(filter(lambda x: x not in idx_B or x not in idx_A, idx_A + idx_B))
	random.shuffle(idx_D)
	idx_C = idx_D.copy()

	extents_A = []
	extents_B = []
	extents_C = []
	extents_D = []

	if equal_extents:
		extent = random.randint(1, 4)
		extents_A = [extent] * nmode_A
		extents_B = [extent] * nmode_B
		extents_C = [extent] * nmode_C
		extents_D = [extent] * nmode_D
	else:
		extents_A = list(np.random.randint(1, 4, nmode_A))
		extents_B = [extents_A[idx_A.index(i)] if i in idx_A else np.random.randint(1, 4) for i in idx_B]
		extents_D = [extents_A[idx_A.index(i)] if i in idx_A else extents_B[idx_B.index(i)] for i in idx_D]
	extents_C = extents_D.copy()

	outer_nmode_A = nmode_A + random.randint(1, 4) if lower_idx else nmode_A
	outer_nmode_B = nmode_B + random.randint(1, 4) if lower_idx else nmode_B
	outer_nmode_C = nmode_C + random.randint(1, 4) if lower_idx else nmode_C
	outer_nmode_D = nmode_D + random.randint(1, 4) if lower_idx else nmode_D
	outer_extent_A = []
	outer_extent_B = []
	outer_extent_C = []
	outer_extent_D = []
	slicing_A = []
	slicing_B = []
	slicing_C = []
	slicing_D = []

	idx = 0
	for i in range(outer_nmode_A):
		if (random.uniform(0, 1) < float(nmode_A) / float(outer_nmode_A) or outer_nmode_A - i == nmode_A - idx) and nmode_A - idx > 0:
			extension = random.randint(1, 4)
			outer_extent_A.append(extents_A[idx] + extension if lower_extents else extents_A[idx])
			offset = random.randint(0, extension - extents_A[idx]) if lower_extents and extension - extents_A[idx] > 0 else 0
			slicing_A.append(slice(offset, offset + extents_A[idx]))
			idx += 1
		else:
			outer_extent_A.append(random.randint(1, 8) if lower_extents else random.randint(1, 4))
			slice_start = random.randint(0, outer_extent_A[i] - 1)
			slicing_A.append(slice(slice_start, slice_start + 1))
	
	idx = 0
	for i in range(outer_nmode_B):
		if (random.uniform(0, 1) < float(nmode_B) / float(outer_nmode_B) or outer_nmode_B - i == nmode_B - idx) and nmode_B - idx > 0:
			extension = random.randint(1, 4)
			outer_extent_B.append(extents_B[idx] + extension if lower_extents else extents_B[idx])
			offset = random.randint(0, extension - extents_B[idx]) if lower_extents and extension - extents_B[idx] > 0 else 0
			slicing_B.append(slice(offset, offset + extents_B[idx]))
			idx += 1
		else:
			outer_extent_B.append(random.randint(1, 8) if lower_extents else random.randint(1, 4))
			slice_start = random.randint(0, outer_extent_B[i] - 1)
			slicing_B.append(slice(slice_start, slice_start + 1))
	
	idx = 0
	for i in range(outer_nmode_C):
		if (random.uniform(0, 1) < float(nmode_C) / float(outer_nmode_C) or outer_nmode_C - i == nmode_C - idx) and nmode_C - idx > 0:
			extension = random.randint(1, 4)
			outer_extent_C.append(extents_C[idx] + extension if lower_extents else extents_C[idx])
			offset = random.randint(0, extension - extents_C[idx]) if lower_extents and extension - extents_C[idx] > 0 else 0
			slicing_C.append(slice(offset, offset + extents_C[idx]))
			idx += 1
		else:
			outer_extent_C.append(random.randint(1, 8) if lower_extents else random.randint(1, 4))
			slice_start = random.randint(0, outer_extent_C[i] - 1)
			slicing_C.append(slice(slice_start, slice_start + 1))
	
	idx = 0
	for i in range(outer_nmode_D):
		if (random.uniform(0, 1) < float(nmode_D) / float(outer_nmode_D) or outer_nmode_D - i == nmode_D - idx) and nmode_D - idx > 0:
			extension = random.randint(1, 4)
			outer_extent_D.append(extents_D[idx] + extension if lower_extents else extents_D[idx])
			offset = random.randint(0, extension - extents_D[idx]) if lower_extents and extension - extents_D[idx] > 0 else 0
			slicing_D.append(slice(offset, offset + extents_D[idx]))
			idx += 1
		else:
			outer_extent_D.append(random.randint(1, 8) if lower_extents else random.randint(1, 4))
			slice_start = random.randint(0, outer_extent_D[i] - 1)
			slicing_D.append(slice(slice_start, slice_start + 1))

	is_complex = random.choice([True, False])

	A = np.array(np.random.rand(*outer_extent_A) + np.random.rand(*outer_extent_A) * (1.0j if is_complex else 0))
	B = np.array(np.random.rand(*outer_extent_B) + np.random.rand(*outer_extent_B) * (1.0j if is_complex else 0))
	C = np.array(np.random.rand(*outer_extent_C) + np.random.rand(*outer_extent_C) * (1.0j if is_complex else 0))
	D = np.array(np.random.rand(*outer_extent_D) + np.random.rand(*outer_extent_D) * (1.0j if is_complex else 0))

	alpha = random.random()
	beta = random.random()

	op_A = random.randint(0, 1)
	op_B = random.randint(0, 1)
	op_C = random.randint(0, 1)
	op_D = random.randint(0, 1)

	return A, B, C, D, extents_A, extents_B, extents_C, extents_D, idx_A, idx_B, idx_C, idx_D, alpha, beta, slicing_A, slicing_B, slicing_C, slicing_D, op_A, op_B, op_C, op_D
	

if __name__ == "__main__":
	main()