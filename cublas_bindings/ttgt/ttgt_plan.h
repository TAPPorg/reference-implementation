/**
 * Hold the TTGTPlan class responsible for driving the TTGT algorithm
 */

#ifndef TTGT_PLAN_H
#define TTGT_PLAN_H

#include "cublas_v2.h"
#include "ttgt_optimizer.h"
#include "ttgt_utils.h"
#include <algorithm>
#include <cstring>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <cutt.h>
#include <unordered_map>

#define cudaCheck(ans)                                                        \
   {                                                                          \
      gpuAssert ((ans), __FILE__, __LINE__);                                  \
   }
inline void
gpuAssert (cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
      {
         fprintf (stderr, "GPUassert: %s %s %d\n", cudaGetErrorString (code),
                  file, line);
         if (abort)
            exit (code);
      }
}

#define cuttCheck(stmt)                                                       \
   do                                                                         \
      {                                                                       \
         cuttResult err = stmt;                                               \
         if (err != CUTT_SUCCESS)                                             \
            {                                                                 \
               fprintf (stderr, "%s in file %s, function %s, error=%d\n",     \
                        #stmt, __FILE__, __FUNCTION__, err);                  \
               exit (1);                                                      \
            }                                                                 \
      }                                                                       \
   while (0)

enum TransposeBackend
{
   CUTT
};

class TTGTPlan
{
 public:
   TTGTPlan (const TransposeBackend);
   ~TTGTPlan ();

   void set_transpose_backend (const TransposeBackend);
   void optimize (const TTGTOptimizerOptions, const ContractionInfo);
   void execute (void *A, void *B, void *C, const ContractionInfo info);
   void print_config ();

   TransposeBackend transpose_backend = TransposeBackend::CUTT;

   // Transpose config
   bool transposeA;
   cuttHandle planA;

   bool transposeB;
   cuttHandle planB;

   bool transposeC;
   cuttHandle planC;

   // GEMM config
   cublasHandle_t handle;
   cublasOperation_t transa;
   cublasOperation_t transb;
   int m = 1;
   int n = 1;
   int k = 1;
   int lda = 1;
   int ldb = 1;
   int ldc = 1;
};

#endif
