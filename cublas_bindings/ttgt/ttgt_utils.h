/**
 * Contains:
 * - allowed datatypes
 * - contraction information (for TTGTPlan::optimize)
 * - GenericScalar to fit GEMM and GEAM routines for any datatype
 * - helper functions for manipulating indices, datatypes and arrays
 */
#ifndef UTILS_H
#define UTILS_H

#include <cublas_v2.h>
#include <algorithm>
#include <complex>
#include <cuComplex.h>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

struct GenericScalar
{
   union
   {
      float f32;
      double f64;
      cuComplex c32;
      cuDoubleComplex c64;
   } value;
};

enum DataType
{
   FLOAT32,
   FLOAT64,
   COMPLEX32,
   COMPLEX64
};

struct ContractionInfo
{
   size_t sizeA;
   size_t sizeB;
   size_t sizeC;
   int *dimA;
   int *dimB;
   int *dimC;
   int32_t rankA;
   int32_t rankB;
   int32_t rankC;
   int32_t *modeA;
   int32_t *modeB;
   int32_t *modeC;
   DataType datatypeA;
   DataType datatypeB;
   DataType datatypeC;
   void *alpha;
   void *beta;
   cublasComputeType_t computeType;
#if EMULATION
   int maxPrecisionDigits;
   cudaEmulationMantissaControl_t emulationMantissaControl;
   cublasEmulationStrategy_t emulationStrategy;
#endif

   ContractionInfo (size_t sA, size_t sB, size_t sC, int *dA, int *dB, int *dC,
                    int32_t rA, int32_t rB, int32_t rC, int32_t *mA,
                    int32_t *mB, int32_t *mC, DataType dtA, DataType dtB,
                    DataType dtC, void *a, void *b, cublasComputeType_t ct
#if EMULATION
		    , int dgt
                    ,cudaEmulationMantissaControl_t c
                    ,cublasEmulationStrategy_t strat
#endif
                    )
       : sizeA (sA), sizeB (sB), sizeC (sC), dimA (dA), dimB (dB), dimC (dC),
         rankA (rA), rankB (rB), rankC (rC), modeA (mA), modeB (mB),
         modeC (mC), datatypeA (dtA), datatypeB (dtB), datatypeC (dtC),
         alpha (a), beta (b), computeType (ct)
#if EMULATION
	 , maxPrecisionDigits(dgt)
         ,emulationMantissaControl(c)
         ,emulationStrategy(strat)
#endif
         {};
};

void get_bounded_indices (int32_t **bounded_indices,
                          int32_t *n_bounded_indices, const int32_t *modeA,
                          const int32_t rankA, const int32_t *modeC,
                          const int32_t rankC);

void get_free_indices (int32_t *free_indicesA, const int32_t n_free_indicesA,
                       const int32_t *modeA, const int32_t rankA,
                       const int32_t *modeC, const int32_t rankC);

size_t sizeof_datatype (DataType);
void get_permutation (int32_t *, int32_t *, int32_t *, int32_t);

cudaDataType get_cuda_datatype (DataType);

GenericScalar init_generic_scalar (DataType type, void *value);

int *prepend (int *, int32_t, int);
int32_t *get_new_permutation (int32_t *, int32_t);

#endif
