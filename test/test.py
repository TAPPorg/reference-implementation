from ctypes import *
import numpy as np
import os.path

so_file = "/home/niklas/Documents/Tensor_Product/Tensor_Product/lib/product.so"

# int PRODUCT(int IDXA, int* EXTA, int* STRA, float* A,
#             int IDXB, int* EXTB, int* STRB, float* B,
#             int IDXC, int* EXTC, int* STRC, float* C,
#             int IDXD, int* EXTD, int* STRD, float* D,
#             float ALPHA, float BETA, bool FA, bool FB, bool FC, char* EINSUM);

product = CDLL(so_file).PRODUCT
product.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float),
                    c_float, c_float, c_bool, c_bool, c_bool, c_char_p]
product.restype = None

def PRODUCT(IDXA, EXTA, STRA, A,
            IDXB, EXTB, STRB, B,
            IDXC, EXTC, STRC, C,
            IDXD, EXTD, STRD, D,
            ALPHA, BETA, FA, FB, FC, EINSUM):
    EXTA = (c_int * len(EXTA))(*EXTA)
    STRA = (c_int * len(STRA))(*STRA)
    A = (c_float * len(A))(*A)

    EXTB = (c_int * len(EXTB))(*EXTB)
    STRB = (c_int * len(STRB))(*STRB)
    B = (c_float * len(B))(*B)

    EXTC = (c_int * len(EXTC))(*EXTC)
    STRC = (c_int * len(STRC))(*STRC)
    C = (c_float * len(C))(*C)

    EXTD = (c_int * len(EXTD))(*EXTD)
    STRD = (c_int * len(STRD))(*STRD)
    D = (c_float * len(D))(*D)

    EINSUM = EINSUM.encode('utf-8')

    product(IDXA, EXTA, STRA, A,
            IDXB, EXTB, STRB, B,
            IDXC, EXTC, STRC, C,
            IDXD, EXTD, STRD, D,
            ALPHA, BETA, FA, FB, FC, EINSUM)
    
    return D

def main():
	'''IDXA = 3
	EXTA = [4, 2, 3]
	STRA = [1, 4, 8]
	A = [1, 2, 3, 4,
		5, 6, 7, 8,

		1, 2, 3, 4,
		5, 6, 7, 8,

		1, 2, 3, 4,
		5, 6, 7, 8]

	IDXB = 3
	EXTB = [2, 3, 2]
	STRB = [1, 2, 6]
	B = [9, 8,
		7, 6,
		5, 4,
		
		3, 2,
		9, 8,
		7, 6]

	IDXC = 2
	EXTC = [4, 2]
	STRC = [1, 4]
	C = [1, 2, 3, 4,
		5, 6, 7, 8]

	IDXD = 2
	EXTD = [4, 2]
	STRD = [1, 4]
	D = [0, 0, 0, 0,
		0, 0, 0, 0]

	ALPHA = 1.0

	BETA = 0.0

	FA = False
	FB = False
	FC = False

	EINSUM = "ijk, jkl -> il"'''

	IDXA = 2
	EXTA = [4, 3]
	STRA = [1, 4]
	A = [1, 2, 3, 4,
		 5, 6, 7, 8,
		 9, 10, 11, 12]

	IDXB = 2
	EXTB = [4, 3]
	STRB = [1, 4]
	B = [1, 2, 3, 4,
		 5, 6, 7, 8,
		 9, 10, 11, 12]

	IDXC = 2
	EXTC = [4, 3]
	STRC = [1, 4]
	C = [1, 2, 3, 4,
		 5, 6, 7, 8,
		 9, 10, 11, 12]

	IDXD = 2
	EXTD = [4, 3]
	STRD = [1, 4]
	D = [0, 0, 0, 0,
		 0, 0, 0, 0,
		 0, 0, 0, 0]

	ALPHA = 1.0

	BETA = 0.0

	FA = False
	FB = False
	FC = False

	EINSUM = "ij, ij -> ij"

	D = PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM)

	print(np.array(D).reshape(4, 3))

if __name__ == "__main__":
	main()