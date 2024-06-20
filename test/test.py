# Niklas Hörnblad
# Paolo Bientinesi
# Umeå University - June 2024

from ctypes import *
import numpy as np
import re
import random
from numpy.random import default_rng
import os.path

so_file = "/home/niklas/Documents/Tensor_Product/Tensor_Product/lib/product.so"

product = CDLL(so_file).PRODUCT
product.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_float, c_float, c_bool, c_bool, c_bool, c_char_p]
product.restype = None

def PRODUCT(A,
            B,
            C,
            ALPHA, BETA, FA, FB, FC, EINSUM):
	IDXA = A.ndim if type(A) is np.ndarray else 0
	EXTA = list(A.shape)[::-1] if type(A) is np.ndarray else []
	STRA = [1] + list(np.cumprod(A.shape[1:][::-1])) if type(A) is np.ndarray else []
	A = A.flatten() if type(A) is np.ndarray else [A]

	IDXB = B.ndim if type(B) is np.ndarray else 0
	EXTB = list(B.shape)[::-1] if type(B) is np.ndarray else []
	STRB = [1] + list(np.cumprod(B.shape[1:][::-1])) if type(B) is np.ndarray else []
	B = B.flatten() if type(B) is np.ndarray else [B]
	IndicesA, IndicesB, IndicesD = re.split(',|->', EINSUM.replace(' ', ''))
	IndicesA = IndicesA[::-1]
	IndicesB = IndicesB[::-1]
	IndicesD = IndicesD[::-1]
	EINSUM = IndicesA + ", " + IndicesB + " -> " + IndicesD
	if (C is not None):
		IDXC = C.ndim if type(C) is np.ndarray else 0
		EXTC = list(C.shape)[::-1] if type(C) is np.ndarray else []
		STRC = [1] + list(np.cumprod(C.shape[1:][::-1])) if type(C) is np.ndarray else []
		C = C.flatten() if type(C) is np.ndarray else [C]

		IDXD = IDXC
		EXTD = EXTC.copy()
		STRD = STRC.copy()
		D = np.zeros(C.shape).flatten() if type(C) is np.ndarray else [0]
	else:
		IDXD = len(IndicesD)
		EXTD = [EXTA[IndicesA.index(i)] if i in IndicesA else EXTB[IndicesB.index(i)] for i in IndicesD]
		STRD = [1] + list(np.cumprod(EXTD[1:][::-1]))
		D = np.zeros(EXTD).flatten()

		IDXC = len(IndicesD)
		EXTC = EXTD.copy()
		STRC = STRD.copy()
		C = np.zeros(D.shape)
	
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

	EINSUM = EINSUM.encode('utf-8')

	product(IDXA, EXTA, STRA, A,
			IDXB, EXTB, STRB, B,
			IDXC, EXTC, STRC, C,
			IDXD, EXTD, STRD, D,
			ALPHA, BETA, FA, FB, FC, EINSUM)

	D = np.array(D).reshape(EXTD[::-1])

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

def TestHadamardProduct():
	ndim = random.randint(1, 5)
	shape = list(np.random.randint(1, 5, ndim))

	A = np.random.rand(*shape)
	B = np.random.rand(*shape)
	C = np.random.rand(*shape)

	ALPHA = random.random()
	BETA = random.random()

	FA = False # Change to random.choice([True, False]) if you get complex numbers to work
	FB = False # Change to random.choice([True, False]) if you get complex numbers to work
	FC = False # Change to random.choice([True, False]) if you get complex numbers to work

	Indices = ''.join([chr(i) for i in range(97, 97 + ndim)])

	EINSUM = Indices + ", " + Indices + " -> " + Indices

	D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA
	
	return np.allclose(D, E)

def TestContration():
	A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration()

	D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA
	
	return np.allclose(D, E)

def TestCommutativity():
	A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration()

	D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA

	IndicesA, IndicesB, IndicesD = re.split(',|->', EINSUM.replace(' ', ''))
	EINSUM = IndicesB + ", " + IndicesA + " -> " + IndicesD
	
	F = PRODUCT(B, A, C, ALPHA, BETA, FB, FA, FC, EINSUM)

	G = np.einsum(EINSUM, B, A) * ALPHA + C * BETA

	return np.allclose(D, E) and np.allclose(E, G) and np.allclose(F, G) and np.allclose(D, F)

def TestPermutations():
	A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration()

	Indices, IndicesD = re.split('->', EINSUM.replace(' ', ''))

	ResultsD = np.array([])
	ResultsE = np.array([])

	for _ in range(len(IndicesD)):
		D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

		E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA

		ResultsD = np.append(ResultsD, D)
		ResultsE = np.append(ResultsE, E)

		IndicesD = IndicesD[1:] + IndicesD[0]
		EINSUM = Indices + " -> " + IndicesD
		C = C.reshape(list(list(C.shape)[1:]) + [(C.shape)[0]])
	
	return np.allclose(ResultsD, ResultsE)

def TestEqualExtents():
	A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(equal_extents = True)

	D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA
	
	return np.allclose(D, E)

def TestOuterProduct():
	A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(contractions = 0)

	D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA

	return np.allclose(D, E)

def TestFullContraction():
	A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(ndimD = 0)

	D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA

	return np.allclose(D, E)

def TestZeroDimTensorContraction():
	A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(ndimA = 0)

	D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA

	return np.allclose(D, E)

def TestOneDimTensorContraction():
	A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(ndimA = 1)

	D = PRODUCT(A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = np.einsum(EINSUM, A, B) * ALPHA + C * BETA

	return np.allclose(D, E)

def GenerateContration(ndimA = None, ndimB = None, ndimD = random.randint(0, 5), contractions = random.randint(0, 5), equal_extents = False):
	if ndimA is None and ndimB is None:
		ndimA = random.randint(0, ndimD)
		ndimB = ndimD - ndimA
		ndimA = ndimA + contractions
		ndimB = ndimB + contractions
	elif ndimA is None:
		contractions = random.randint(0, ndimB) if contractions > ndimB else contractions
		ndimD = ndimB - contractions + random.randint(0, 5) if ndimD < ndimB - contractions else ndimD
		ndimA = ndimD - ndimB + contractions * 2
	elif ndimB is None:
		contractions = random.randint(0, ndimA) if contractions > ndimA else contractions
		ndimD = ndimA - contractions + random.randint(0, 5) if ndimD < ndimA - contractions else ndimD
		ndimB = ndimD - ndimA + contractions * 2
	else:
		contractions = random.randint(0, min(ndimA, ndimB))
		ndimD = ndimA + ndimB - contractions * 2

	IndicesA = [chr(i) for i in range(97, 97 + ndimA)]
	random.shuffle(IndicesA)
	IndicesA = ''.join(IndicesA)
	IndicesB = random.sample(IndicesA, contractions) + [chr(i) for i in range(97 + ndimA, 97 + ndimA + ndimB - contractions)]
	random.shuffle(IndicesB)
	IndicesB = ''.join(IndicesB)
	IndicesD = list(filter(lambda x: x not in IndicesB or x not in IndicesA, IndicesA + IndicesB))
	random.shuffle(IndicesD)
	IndicesD = ''.join(IndicesD)

	if equal_extents:
		Extent = random.randint(1, 5)
		ExtentA = [Extent] * ndimA
		ExtentB = [Extent] * ndimB
		ExtentC = [Extent] * ndimD
	else:
		ExtentA = list(np.random.randint(1, 5, ndimA))
		ExtentB = [ExtentA[IndicesA.index(i)] if i in IndicesA else np.random.randint(1, 5) for i in IndicesB]
		ExtentC = [ExtentA[IndicesA.index(i)] if i in IndicesA else ExtentB[IndicesB.index(i)] for i in IndicesD]

	A = np.random.rand(*ExtentA)
	B = np.random.rand(*ExtentB)
	C = np.random.rand(*ExtentC)

	ALPHA = random.random()
	BETA = random.random()

	FA = False # Change to random.choice([True, False]) if you get complex numbers to work
	FB = False # Change to random.choice([True, False]) if you get complex numbers to work
	FC = False # Change to random.choice([True, False]) if you get complex numbers to work

	EINSUM = IndicesA + ", " + IndicesB + " -> " + IndicesD

	return A, B, C, ALPHA, BETA, FA, FB, FC, EINSUM
	

if __name__ == "__main__":
	main()