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
#include <stdexcept>
#include <string>
#include <unordered_map>

// NOTE (TAPP cuBLAS bindings): the upstream my-ttgt macros call exit() on a
// CUDA/cuTT error. A library must never terminate the host process, so here
// they throw std::runtime_error instead; the TAPP binding (product.cu) catches
// it and reports a TAPP error code, leaving the harness running.
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
            throw std::runtime_error (std::string ("CUDA error: ")
                                      + cudaGetErrorString (code));
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
               throw std::runtime_error ("cuTT error in " #stmt);             \
            }                                                                 \
      }                                                                       \
   while (0)

// Create a cuTT transpose plan, printing the exact rank/dims/permutation if
// cuTT rejects them (helps diagnose CUTT_INVALID_PARAMETER), then throwing so
// the TAPP binding can report the failure rather than aborting.
inline void
cutt_plan_checked (cuttHandle *plan, int rank, int *dim, int *permutation,
                   size_t sizeofType, const char *which)
{
   cuttResult err = cuttPlan (plan, rank, dim, permutation, sizeofType, 0);
   if (err != CUTT_SUCCESS)
      {
         fprintf (stderr,
                  "TTGT: cuttPlan failed for tensor %s (err=%d): rank=%d "
                  "elemsize=%zu dim=[",
                  which, err, rank, sizeofType);
         for (int i = 0; i < rank; i++)
            fprintf (stderr, "%d%s", dim[i], i + 1 < rank ? "," : "");
         fprintf (stderr, "] perm=[");
         for (int i = 0; i < rank; i++)
            fprintf (stderr, "%d%s", permutation[i], i + 1 < rank ? "," : "");
         fprintf (stderr, "]\n");
         throw std::runtime_error ("cuTT plan creation failed");
      }
}

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
