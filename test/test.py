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

def PRODUCT(A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM):
	ptrA = A.flatten()[sum([a*b for a, b in zip(offsetA, STRA)]):] if type(A) is np.ndarray else [A]
	ptrB = B.flatten()[sum([a*b for a, b in zip(offsetB, STRB)]):] if type(B) is np.ndarray else [B]
	ptrC = C.flatten()[sum([a*b for a, b in zip(offsetC, STRC)]):] if type(C) is np.ndarray else [C]
	ptrD = D.flatten()[sum([a*b for a, b in zip(offsetD, STRD)]):] if type(D) is np.ndarray else [D]

	IDXA = len(EXTA)
	ptrA = ptrA.flatten() if type(ptrA) is np.ndarray else ([ptrA] if type(ptrA) is not list else ptrA)

	IDXB = len(EXTB)
	ptrB = ptrB.flatten() if type(ptrB) is np.ndarray else ([ptrB] if type(ptrB) is not list else ptrB)

	shapeD = ptrD.shape if type(ptrD) is np.ndarray else []
	
	IDXC = len(EXTC)
	ptrC = ptrC.flatten() if type(ptrC) is np.ndarray else ([ptrC] if type(ptrC) is not list else ptrC)

	IDXD = len(EXTD)
	ptrD = ptrD.flatten() if type(ptrD) is np.ndarray else ([ptrD] if type(ptrD) is not list else ptrD)

	IDXA = c_int(IDXA)
	EXTA = (c_int * len(EXTA))(*EXTA)
	STRA = (c_int * len(STRA))(*STRA)
	ptrA = (c_float * len(ptrA))(*ptrA)

	IDXB = c_int(IDXB)
	EXTB = (c_int * len(EXTB))(*EXTB)
	STRB = (c_int * len(STRB))(*STRB)
	ptrB = (c_float * len(ptrB))(*ptrB)
	
	IDXC = c_int(IDXC)
	EXTC = (c_int * len(EXTC))(*EXTC)
	STRC = (c_int * len(STRC))(*STRC)
	ptrC = (c_float * len(ptrC))(*ptrC)

	IDXD = c_int(IDXD)
	EXTD = (c_int * len(EXTD))(*EXTD)
	STRD = (c_int * len(STRD))(*STRD)
	ptrD = (c_float * len(ptrD))(*ptrD)

	EINSUM = EINSUM.encode('utf-8')

	product(IDXA, EXTA, STRA, ptrA,
			IDXB, EXTB, STRB, ptrB,
			IDXC, EXTC, STRC, ptrC,
			IDXD, EXTD, STRD, ptrD,
			ALPHA, BETA, FA, FB, FC, EINSUM)
	
	if type(D) is np.ndarray:
		shapeD = D.shape
		D = D.flatten()
		for i in range(len(list(ptrD))):
			D[sum([a*b for a, b in zip(offsetD, STRD)]) + i] = ptrD[i]
		D = D.reshape(shapeD)
	else:
		D = ptrD

	return D

def RunEinsum(A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM):
	ptrA = [] if type(A) is np.ndarray else A
	if type(A) is np.ndarray and len(EXTA) > 0:
		coord = [0] * len(EXTA)
		for i in range(np.prod(EXTA)):
			ptrA.append(A.flatten()[sum([(a+b)*c for a, b, c in zip(coord, offsetA, STRA)])])
			coord[0] += 1
			for j in range(len(coord) - 1):
				if coord[j] == EXTA[j]:
					coord[j] = 0
					coord[j + 1] += 1
		ptrA = np.asarray(ptrA)
		ptrA = ptrA.reshape(EXTA[::-1])
	else:
		ptrA = A.flatten()[sum([a*b for a, b in zip(offsetA, STRA)])] if type(A) is np.ndarray else A

	ptrB = [] if type(B) is np.ndarray else B
	if type(B) is np.ndarray and len(EXTB) > 0:
		coord = [0] * len(EXTB)
		for i in range(np.prod(EXTB)):
			ptrB.append(B.flatten()[sum([(a+b)*c for a, b, c in zip(coord, offsetB, STRB)])])
			coord[0] += 1
			for j in range(len(coord) - 1):
				if coord[j] == EXTB[j]:
					coord[j] = 0
					coord[j + 1] += 1
		ptrB = np.asarray(ptrB)
		ptrB = ptrB.reshape(EXTB[::-1])
	else:
		ptrB = B.flatten()[sum([a*b for a, b in zip(offsetB, STRB)])] if type(B) is np.ndarray else B

	ptrC = [] if type(C) is np.ndarray else C
	if type(C) is np.ndarray and len(EXTC) > 0:
		coord = [0] * len(EXTC)
		for i in range(np.prod(EXTC)):
			ptrC.append(C.flatten()[sum([(a+b)*c for a, b,c in zip(coord, offsetC, STRC)])])
			coord[0] += 1
			for j in range(len(coord) - 1):
				if coord[j] == EXTC[j]:
					coord[j] = 0
					coord[j + 1] += 1
		ptrC = np.asarray(ptrC)
		ptrC = ptrC.reshape(EXTC[::-1])
	else:
		ptrC = C.flatten()[sum([a*b for a, b in zip(offsetC, STRC)])] if type(C) is np.ndarray else C

	IndicesA, IndicesB, IndicesD = re.split(',|->', EINSUM.replace(' ', ''))
	IndicesA = IndicesA[::-1]
	IndicesB = IndicesB[::-1]
	IndicesD = IndicesD[::-1]
	EINSUM = IndicesA + ", " + IndicesB + " -> " + IndicesD

	ptrD = np.einsum(EINSUM, ptrA, ptrB) * ALPHA + ptrC * BETA

	shapeD = D.shape if type(D) is np.ndarray else []
	D = D.flatten() if type(D) is np.ndarray else D
	if type(D) is np.ndarray and len(EXTD) > 0:
		ptrD = ptrD.flatten()
		coord = [0] * len(EXTD)
		for i in range(np.prod(EXTD)):
			D[sum([(a+b)*c for a, b, c in zip(coord, offsetD, STRD)])] = ptrD[i]
			coord[0] += 1
			for j in range(len(coord) - 1):
				if coord[j] == EXTD[j]:
					coord[j] = 0
					coord[j + 1] += 1
		D = D.reshape(shapeD)
	elif type(D) is np.ndarray:
		D[sum([a*b for a, b in zip(offsetD, STRD)])] = ptrD
		D = D.reshape(shapeD)
	else:
		D = ptrD

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
	IDX = random.randint(1, 5)
	EXT = list(np.random.randint(1, 5, IDX))
	STR = ([1] + list(np.cumprod(EXT)))[:IDX]
	offset = [0] * IDX

	A = np.random.rand(*EXT[::-1])
	B = np.random.rand(*EXT[::-1])
	C = np.random.rand(*EXT[::-1])
	D = np.random.rand(*EXT[::-1])

	ALPHA = random.random()
	BETA = random.random()

	FA = False # Change to random.choice([True, False]) if you get complex numbers to work
	FB = False # Change to random.choice([True, False]) if you get complex numbers to work
	FC = False # Change to random.choice([True, False]) if you get complex numbers to work

	Indices = ''.join([chr(i) for i in range(97, 97 + IDX)])

	EINSUM = Indices + ", " + Indices + " -> " + Indices

	D = PRODUCT(A, B, C, D, EXT, EXT, EXT, EXT, STR, STR, STR, STR, offset, offset, offset, offset, ALPHA, BETA, FA, FB, FC, EINSUM)

	E = RunEinsum(A, B, C, D, EXT, EXT, EXT, EXT, STR, STR, STR, STR, offset, offset, offset, offset, ALPHA, BETA, FA, FB, FC, EINSUM)
	
	return np.allclose(D, E)

def TestContration():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration()

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	return np.allclose(E, F)

def TestCommutativity():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration()

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	IndicesA, IndicesB, IndicesD = re.split(',|->', EINSUM.replace(' ', ''))
	EINSUM = IndicesB + ", " + IndicesA + " -> " + IndicesD
	
	G = PRODUCT(B, A, C, D.copy() if type(D) == np.ndarray else D, EXTB, EXTA, EXTC, EXTD, STRB, STRA, STRC, STRD, offsetB, offsetA, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	H = RunEinsum(B, A, C, D.copy() if type(D) == np.ndarray else D, EXTB, EXTA, EXTC, EXTD, STRB, STRA, STRC, STRD, offsetB, offsetA, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	return np.allclose(E, F) and np.allclose(F, H) and np.allclose(G, H) and np.allclose(E, G)

def TestPermutations():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration()

	Indices, IndicesD = re.split('->', EINSUM.replace(' ', ''))

	ResultsE = np.array([])
	ResultsF = np.array([])

	for _ in range(len(IndicesD)):
		E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

		F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

		ResultsE = np.append(ResultsE, E)
		ResultsF = np.append(ResultsF, F)

		IndicesD = IndicesD[1:] + IndicesD[0]
		EINSUM = Indices + " -> " + IndicesD
		C = C.reshape(list(list(C.shape)[1:]) + [(C.shape)[0]])
		D = D.reshape(list(list(D.shape)[1:]) + [(D.shape)[0]])
		EXTC = EXTC[1:] + [EXTC[0]]
		EXTD = EXTD[1:] + [EXTD[0]]
		STRC = ([1] + list(np.cumprod(EXTC)))[:len(EXTC)]
		STRD = ([1] + list(np.cumprod(EXTD)))[:len(EXTD)]
	
	return np.allclose(ResultsE, ResultsF)

def TestEqualExtents():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(equal_extents = True)

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)
	
	return np.allclose(E, F)

def TestOuterProduct():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(contractions = 0)

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)
	
	return np.allclose(E, F)

def TestFullContraction():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(ndimD = 0)

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)
	
	return np.allclose(E, F)

def TestZeroDimTensorContraction():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(ndimA = 0)

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)
	
	return np.allclose(E, F)

def TestOneDimTensorContraction():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(ndimA = 1)

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)
	
	return np.allclose(E, F)

def TestSubtensorSameIdx():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(lower_extents = True)

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	return np.allclose(E, F)

def TestSubtensorLowerIdx():
	A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM = GenerateContration(lower_extents = True, lower_idx = True)

	E = PRODUCT(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)

	F = RunEinsum(A, B, C, D.copy() if type(D) == np.ndarray else D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM)
	
	return np.allclose(E, F)

def GenerateContration(ndimA = None, ndimB = None, ndimD = random.randint(0, 5),
					   contractions = random.randint(0, 5), equal_extents = False,
					   lower_extents = False, lower_idx = False):
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

	indicesA = [chr(i) for i in range(97, 97 + ndimA)]
	random.shuffle(indicesA)
	indicesA = ''.join(indicesA)
	indicesB = random.sample(indicesA, contractions) + [chr(i) for i in range(97 + ndimA, 97 + ndimA + ndimB - contractions)]
	random.shuffle(indicesB)
	indicesB = ''.join(indicesB)
	indicesD = list(filter(lambda x: x not in indicesB or x not in indicesA, indicesA + indicesB))
	random.shuffle(indicesD)
	indicesD = ''.join(indicesD)

	EXTA = []
	EXTB = []
	EXTC = []
	EXTD = []

	if equal_extents:
		Extent = random.randint(1, 5)
		EXTA = [Extent] * ndimA
		EXTB = [Extent] * ndimB
		EXTC = [Extent] * ndimD
		EXTD = [Extent] * ndimD
	else:
		EXTA = list(np.random.randint(1, 5, ndimA))
		EXTB = [EXTA[indicesA.index(i)] if i in indicesA else np.random.randint(1, 2) for i in indicesB]
		EXTC = [EXTA[indicesA.index(i)] if i in indicesA else EXTB[indicesB.index(i)] for i in indicesD]
		EXTD = EXTC.copy()

	outer_ndimA = ndimA + random.randint(1, 5) if lower_idx else ndimA
	outer_ndimB = ndimB + random.randint(1, 5) if lower_idx else ndimB
	outer_ndimC = ndimD + random.randint(1, 5) if lower_idx else ndimD
	outer_ndimD = ndimD + random.randint(1, 5) if lower_idx else ndimD
	outer_extentA = []
	outer_extentB = []
	outer_extentC = []
	outer_extentD = []
	STRA = []
	STRB = []
	STRC = []
	STRD = []
	offsetA = []
	offsetB = []
	offsetC = []
	offsetD = []

	idx = 0
	stride = 1
	for i in range(outer_ndimA):
		if (random.uniform(0, 1) < float(ndimA) / float(outer_ndimA) or outer_ndimA - i == ndimA - idx) and ndimA - idx > 0:
			extension = random.randint(1, 5)
			outer_extentA.append(EXTA[idx] + extension if lower_extents else EXTA[idx])
			offsetA.append(random.randint(0, extension - EXTA[idx]) if lower_extents and extension - EXTA[idx] > 0 else 0)
			STRA.append(stride)
			stride *= outer_extentA[i]
			idx += 1
		else:
			outer_extentA.append(random.randint(1, 10) if lower_extents else random.randint(1, 5))
			stride *= outer_extentA[i]
	
	idx = 0
	stride = 1
	for i in range(outer_ndimB):
		if (random.uniform(0, 1) < float(ndimB) / float(outer_ndimB) or outer_ndimB - i == ndimB - idx) and ndimB - idx > 0:
			extension = random.randint(1, 5)
			outer_extentB.append(EXTB[idx] + extension if lower_extents else EXTB[idx])
			offsetB.append(random.randint(0, extension - EXTB[idx]) if lower_extents and extension - EXTB[idx] > 0 else 0)
			STRB.append(stride)
			stride *= outer_extentB[i]
			idx += 1
		else:
			outer_extentB.append(random.randint(1, 10) if lower_extents else random.randint(1, 5))
			stride *= outer_extentB[i]
	
	idx = 0
	stride = 1
	for i in range(outer_ndimC):
		if (random.uniform(0, 1) < float(ndimD) / float(outer_ndimC) or outer_ndimC - i == ndimD - idx) and ndimD - idx > 0:
			extension = random.randint(1, 5)
			outer_extentC.append(EXTC[idx] + extension if lower_extents else EXTC[idx])
			offsetC.append(random.randint(0, extension - EXTC[idx]) if lower_extents and extension - EXTC[idx] > 0 else 0)
			STRC.append(stride)
			stride *= outer_extentC[i]
			idx += 1
		else:
			outer_extentC.append(random.randint(1, 10) if lower_extents else random.randint(1, 5))
			stride *= outer_extentC[i]
	
	idx = 0
	stride = 1
	for i in range(outer_ndimD):
		if (random.uniform(0, 1) < float(ndimD) / float(outer_ndimD) or outer_ndimD - i == ndimD - idx) and ndimD - idx > 0:
			extension = random.randint(1, 5)
			outer_extentD.append(EXTD[idx] + extension if lower_extents else EXTD[idx])
			offsetD.append(random.randint(0, extension - EXTD[idx]) if lower_extents and extension - EXTD[idx] > 0 else 0)
			STRD.append(stride)
			stride *= outer_extentD[i]
			idx += 1
		else:
			outer_extentD.append(random.randint(1, 10) if lower_extents else random.randint(1, 5))
			stride *= outer_extentD[i]

	A = np.random.rand(*outer_extentA[::-1])
	B = np.random.rand(*outer_extentB[::-1])
	C = np.random.rand(*outer_extentC[::-1])
	D = np.random.rand(*outer_extentD[::-1])

	ALPHA = random.random()
	BETA = random.random()

	FA = False # Change to random.choice([True, False]) if you get complex numbers to work
	FB = False # Change to random.choice([True, False]) if you get complex numbers to work
	FC = False # Change to random.choice([True, False]) if you get complex numbers to work

	EINSUM = indicesA + ", " + indicesB + " -> " + indicesD

	return A, B, C, D, EXTA, EXTB, EXTC, EXTD, STRA, STRB, STRC, STRD, offsetA, offsetB, offsetC, offsetD, ALPHA, BETA, FA, FB, FC, EINSUM
	

if __name__ == "__main__":
	main()