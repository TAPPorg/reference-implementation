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

// Create a cuTT transpose plan after squeezing out extent-1 axes.
//
// cuTT rejects any transpose whose dims contain a size-1 axis (and any rank<2
// transpose) with CUTT_INVALID_PARAMETER. A size-1 axis carries no data, so it
// can be removed from (rank, dim, permutation) without changing which bytes
// move. After squeezing:
//   - if the transpose reduces to the identity (or fewer than 2 axes), it moves
//     no data: no plan is created and the function returns false, signalling
//     that the source buffer is already in the target layout (the caller should
//     treat the tensor as not transposed);
//   - otherwise a plan is built on the reduced dims and the function returns
//     true.
// On a genuine cuTT failure the offending (reduced) parameters are printed and
// an exception is thrown so the TAPP binding can report it instead of aborting.
inline bool
cutt_plan_squeezed (cuttHandle *plan, int rank, const int *dim,
                    const int *permutation, size_t sizeofType,
                    const char *which)
{
   std::vector<int> old_to_new (rank, -1);
   std::vector<int> sdim;
   int new_rank = 0;
   for (int i = 0; i < rank; i++)
      if (dim[i] != 1)
         {
            old_to_new[i] = new_rank++;
            sdim.push_back (dim[i]);
         }

   std::vector<int> sperm;
   for (int i = 0; i < rank; i++)
      {
         int axis = permutation[i];
         if (dim[axis] != 1)
            sperm.push_back (old_to_new[axis]);
      }

   bool identity = true;
   for (int i = 0; i < new_rank; i++)
      if (sperm[i] != i)
         {
            identity = false;
            break;
         }
   if (new_rank < 2 || identity)
      return false; // no data movement needed

   cuttResult err
       = cuttPlan (plan, new_rank, sdim.data (), sperm.data (), sizeofType, 0);
   if (err != CUTT_SUCCESS)
      {
         fprintf (stderr,
                  "TTGT: cuttPlan failed for tensor %s (err=%d): rank=%d "
                  "elemsize=%zu dim=[",
                  which, err, new_rank, sizeofType);
         for (int i = 0; i < new_rank; i++)
            fprintf (stderr, "%d%s", sdim[i], i + 1 < new_rank ? "," : "");
         fprintf (stderr, "] perm=[");
         for (int i = 0; i < new_rank; i++)
            fprintf (stderr, "%d%s", sperm[i], i + 1 < new_rank ? "," : "");
         fprintf (stderr, "]\n");
         throw std::runtime_error ("cuTT plan creation failed");
      }
   return true;
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
