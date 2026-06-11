#include "ttgt_plan.h"

TTGTPlan::TTGTPlan (const TransposeBackend backend)
{
   // Initialize the transpose flags so a plan whose optimize() never ran (or
   // failed part-way) destructs without touching uninitialized cuTT handles.
   this->transposeA = false;
   this->transposeB = false;
   this->transposeC = false;

   this->set_transpose_backend (backend);

   cublasStatus_t stat = cublasCreate (&(this->handle));
   if (stat != CUBLAS_STATUS_SUCCESS)
      {
         printf ("CUBLAS initialization failed\n");
         cublasDestroy (handle);
         return;
      }
}

TTGTPlan::~TTGTPlan ()
{
   // Do not use cuttCheck here: it throws, and throwing from a destructor would
   // terminate. Ignore cuttDestroy errors during teardown.
   if (this->transposeA)
      cuttDestroy (this->planA);
   if (this->transposeB)
      cuttDestroy (this->planB);
   if (this->transposeC)
      cuttDestroy (this->planC);
   cublasDestroy (this->handle);
}

void
TTGTPlan::set_transpose_backend (const TransposeBackend backend)
{
   this->transpose_backend = backend;
}

void
TTGTPlan::optimize (const TTGTOptimizerOptions options,
                    ContractionInfo info)
{

   int32_t modeAT[info.rankA];
   int32_t modeBT[info.rankB];
   int32_t modeCT[info.rankC];

   int dimCT_old[info.rankC];

   int32_t permutationA_old[info.rankA];
   int32_t permutationB_old[info.rankB];
   int32_t permutationC_old[info.rankC];

   // set cuBLAS emulataion strategy
#if EMULATION
   //cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH); // to use if fixed
   cublasSetEmulationStrategy(handle, info.emulationStrategy); // Add the variable to info
   cublasSetFixedPointEmulationMantissaControl(handle, info.emulationMantissaControl);
   int maxMantissaBitCount = (int) std::ceil(std::log2(10.0)*info.maxPrecisionDigits );
   cublasSetFixedPointEmulationMaxMantissaBitCount(handle, maxMantissaBitCount);
#endif

   // Set indices of transposed tensors
   switch (options.strategy)
      {
      case Strategy::BASELINE:
         {
            // step 1: get bounded indices from the bigger tensor A or B
            // Define the order of the bounded indices wrt to the bigger
            // tensor
            int32_t *bounded_indices = nullptr;
            int32_t n_bounded_indices = 0;
            if (info.sizeA > info.sizeB)
               {
                  get_bounded_indices (&bounded_indices, &n_bounded_indices,
                                       info.modeA, info.rankA, info.modeC,
                                       info.rankC);
               }
            else
               {
                  get_bounded_indices (&bounded_indices, &n_bounded_indices,
                                       info.modeB, info.rankB, info.modeC,
                                       info.rankC);
               }

            // step 2: get free indices in A/B
            int32_t n_free_indicesA = info.rankA - n_bounded_indices;
            int32_t free_indicesA[n_free_indicesA];
            get_free_indices (free_indicesA, n_free_indicesA, info.modeA,
                              info.rankA, info.modeC, info.rankC);

            int32_t n_free_indicesB = info.rankB - n_bounded_indices;
            int32_t free_indicesB[n_free_indicesB];
            get_free_indices (free_indicesB, n_free_indicesB, info.modeB,
                              info.rankB, info.modeC, info.rankC);

            // step 3: define indices of transposed vectors
            // keep the same first index in transposed vector assuming
            // col-major data
            if (std::find (free_indicesA, free_indicesA + n_free_indicesA,
                           info.modeA[0])
                != (free_indicesA + n_free_indicesA))
               {
                  std::memcpy (modeAT, free_indicesA,
                               n_free_indicesA * sizeof (int32_t));
                  std::memcpy (modeAT + n_free_indicesA, bounded_indices,
                               n_bounded_indices * sizeof (int32_t));
                  this->transa = CUBLAS_OP_N;
               }
            else
               {
                  std::memcpy (modeAT, bounded_indices,
                               n_bounded_indices * sizeof (int32_t));
                  std::memcpy (modeAT + n_bounded_indices, free_indicesA,
                               n_free_indicesA * sizeof (int32_t));
                  this->transa = CUBLAS_OP_T;
               }
            get_permutation (permutationA_old, modeAT, info.modeA, info.rankA);
            if (this->transa == CUBLAS_OP_N)
               {
                  for (int i = 0; i < n_free_indicesA; i++)
                     this->m *= info.dimA[permutationA_old[i]];
                  for (int i = 0; i < n_bounded_indices; i++)
                     this->k
                         *= info.dimA[permutationA_old[i + n_free_indicesA]];

                  this->lda = m;
               }
            else
               {
                  for (int i = 0; i < n_bounded_indices; i++)
                     this->k *= info.dimA[permutationA_old[i]];
                  for (int i = 0; i < n_free_indicesA; i++)
                     this->m
                         *= info.dimA[permutationA_old[i + n_bounded_indices]];

                  this->lda = k;
               }

            if (std::find (free_indicesB, free_indicesB + n_free_indicesB,
                           info.modeB[0])
                != (free_indicesB + n_free_indicesB))
               {
                  std::memcpy (modeBT, free_indicesB,
                               n_free_indicesB * sizeof (int32_t));
                  std::memcpy (modeBT + n_free_indicesB, bounded_indices,
                               n_bounded_indices * sizeof (int32_t));
                  this->transb = CUBLAS_OP_T;
               }
            else
               {
                  std::memcpy (modeBT, bounded_indices,
                               n_bounded_indices * sizeof (int32_t));
                  std::memcpy (modeBT + n_bounded_indices, free_indicesB,
                               n_free_indicesB * sizeof (int32_t));
                  this->transb = CUBLAS_OP_N;
               }
            get_permutation (permutationB_old, modeBT, info.modeB, info.rankB);

            if (this->transb == CUBLAS_OP_T)
               {
                  for (int i = 0; i < n_free_indicesB; i++)
                     this->n *= info.dimB[permutationB_old[i]];
                  this->ldb = n;
               }
            else
               {
                  for (int i = 0; i < n_free_indicesB; i++)
                     {
                        this->n
                            *= info.dimB
                                   [permutationB_old[i + n_bounded_indices]];
                     }
                  this->ldb = k;
               }

            // Get indices of C by assuming order of tensors AB (TODO
            // remove this limitation)
            std::memcpy (modeCT, free_indicesA,
                         n_free_indicesA * sizeof (int32_t));
            std::memcpy (modeCT + n_free_indicesA, free_indicesB,
                         n_free_indicesB * sizeof (int32_t));
            get_permutation (permutationC_old, info.modeC, modeCT, info.rankC);

            for (int i = 0; i < info.rankC; i++)
               dimCT_old[permutationC_old[i]] = info.dimC[i];
            for (int i = 0; i < n_free_indicesA; i++)
               this->ldc *= dimCT_old[i];
         }
         break;
      default:
         throw InvalidOptimizationStrategy (
             "Strategy not recognized or not implemented.");
      }

   // Define plan (if not done already)
   this->transposeA
       = not std::equal (info.modeA, info.modeA + info.rankA, modeAT);
   this->transposeB
       = not std::equal (info.modeB, info.modeB + info.rankB, modeBT);
   this->transposeC
       = not std::equal (info.modeC, info.modeC + info.rankC, modeCT);

   int32_t rankA = info.rankA;
   int32_t rankB = info.rankB;
   int32_t rankC = info.rankC;
   int *dimA = info.dimA;
   int *dimB = info.dimB;
   int *dimCT = dimCT_old;
   int32_t *permutationA = permutationA_old;
   int32_t *permutationB = permutationB_old;
   int32_t *permutationC = permutationC_old;
   size_t sizeof_datatypeA = sizeof_datatype (info.datatypeA);
   size_t sizeof_datatypeB = sizeof_datatype (info.datatypeB);
   size_t sizeof_datatypeC = sizeof_datatype (info.datatypeC);
   // Artificially add another dimension for the transposition of complex64
   // data due to CUTT limitation
   if (transpose_backend == TransposeBackend::CUTT)
      {
         if (info.datatypeA == DataType::COMPLEX64)
            {
               rankA += 1;
               dimA = prepend (dimA, info.rankA, 2);
               permutationA
                   = get_new_permutation (permutationA_old, info.rankA);
               sizeof_datatypeA /= 2;
            }
         if (info.datatypeB == DataType::COMPLEX64)
            {
               rankB += 1;
               dimB = prepend (dimB, info.rankB, 2);
               permutationB
                   = get_new_permutation (permutationB_old, info.rankB);
               sizeof_datatypeB /= 2;
            }
         if (info.datatypeC == DataType::COMPLEX64)
            {
               rankC += 1;
               dimCT = prepend (dimCT_old, info.rankC, 2);
               permutationC
                   = get_new_permutation (permutationC_old, info.rankC);
               sizeof_datatypeC /= 2;
            }
      }

   switch (transpose_backend)
      {
      case TransposeBackend::CUTT:
         {
            // cudaStream_t streamA, streamB, streamC;
            // cudaStreamCreate(&streamA);
            // cudaStreamCreate(&streamB);
            // cudaStreamCreate(&streamC);

            if (options.measure_plan)
               {
                  // A
                  if (this->transposeA)
                     {

                        void *devA = nullptr;
                        void *devAT = nullptr;

                        // cudaMallocAsync(&devA,  info.sizeA *
                        // sizeof_datatype(info.datatypeA), streamA);
                        // cudaMallocAsync(&devAT, info.sizeA *
                        // sizeof_datatype(info.datatypeA), streamA);

                        // cuttCheck(cuttPlanMeasure(
                        //    &planA, info.rankA, info.dimA,
                        //    permutationA,
                        //    sizeof_datatype(info.datatypeA), streamA,
                        //    devA, devAT
                        //));

                        // cudaFreeAsync(devA, streamA);
                        // cudaFreeAsync(devAT, streamA);

                        cudaMalloc (&devA,
                                    info.sizeA
                                        * sizeof_datatype (info.datatypeA));
                        cudaMalloc (&devAT,
                                    info.sizeA
                                        * sizeof_datatype (info.datatypeA));

                        cuttCheck (cuttPlanMeasure (
                            &planA, rankA, dimA, permutationA,
                            sizeof_datatypeA, 0, devA, devAT));

                        cudaFree (devA);
                        cudaFree (devAT);
                     }
                  // B
                  if (this->transposeB)
                     {

                        // void* devB = nullptr;
                        // void* devBT = nullptr;

                        // cudaMallocAsync(&devB,  info.sizeB *
                        // sizeof_datatype(info.datatypeB), streamB);
                        // cudaMallocAsync(&devBT, info.sizeB *
                        // sizeof_datatype(info.datatypeB), streamB);

                        // cuttCheck(cuttPlanMeasure(
                        //    &planB, info.rankB, info.dimB,
                        //    permutationB,
                        //    sizeof_datatype(info.datatypeB), streamB,
                        //    devB, devBT
                        //));

                        // cudaFreeAsync(devB, streamB);
                        // cudaFreeAsync(devBT, streamB);

                        void *devB = nullptr;
                        void *devBT = nullptr;

                        cudaMalloc (&devB,
                                    info.sizeB
                                        * sizeof_datatype (info.datatypeB));
                        cudaMalloc (&devBT,
                                    info.sizeB
                                        * sizeof_datatype (info.datatypeB));

                        cuttCheck (cuttPlanMeasure (
                            &planB, rankB, dimB, permutationB,
                            sizeof_datatypeB, 0, devB, devBT));

                        cudaFree (devB);
                        cudaFree (devBT);
                     }
                  // C
                  if (this->transposeC)
                     {

                        void *devC = nullptr;
                        void *devCT = nullptr;

                        // cudaMallocAsync(&devC,  info.sizeC *
                        // sizeof_datatype(info.datatypeC), streamC);
                        // cudaMallocAsync(&devCT, info.sizeC *
                        // sizeof_datatype(info.datatypeC), streamC);

                        // cuttCheck(cuttPlanMeasure(
                        //    &planC, info.rankC, dimCT, permutationC,
                        //    sizeof_datatype(info.datatypeC), streamC,
                        //    devC, devCT
                        //));

                        // cudaFreeAsync(devC, streamC);
                        // cudaFreeAsync(devCT, streamC);

                        cudaMalloc (&devC,
                                    info.sizeC
                                        * sizeof_datatype (info.datatypeC));
                        cudaMalloc (&devCT,
                                    info.sizeC
                                        * sizeof_datatype (info.datatypeC));

                        cuttCheck (cuttPlanMeasure (
                            &planC, rankC, dimCT, permutationC,
                            sizeof_datatypeC, 0, devC, devCT));

                        cudaFree (devC);
                        cudaFree (devCT);
                     }
               }
            else
               {
                  // A
                  if (this->transposeA)
                     {
                        cutt_plan_checked (&this->planA, rankA, dimA,
                                           permutationA, sizeof_datatypeA,
                                           "A");
                     }
                  // B
                  if (this->transposeB)
                     {
                        cutt_plan_checked (&this->planB, rankB, dimB,
                                           permutationB, sizeof_datatypeB,
                                           "B");
                     }
                  // C
                  if (this->transposeC)
                     {
                        cutt_plan_checked (&this->planC, rankC, dimCT,
                                           permutationC, sizeof_datatypeC,
                                           "C");
                     }
               }
            // cudaStreamDestroy(streamA);
            // cudaStreamDestroy(streamB);
            // cudaStreamDestroy(streamC);
         }
         break;
      default:
         throw InvalidOptimizationStrategy (
             "Transpose backend not recognized or not implemented.");
      }
}

void
TTGTPlan::execute (void *devA, void *devB, void *devC, ContractionInfo info)
{

   void *devAT = nullptr;
   if (transposeA)
      {
         //cudaMallocAsync(&devAT, info.sizeA *
         // sizeof_datatype(info.datatypeA), streamA);
         //cudaEventRecord(eventA, streamA);
         cudaCheck (cudaMalloc (
             &devAT, info.sizeA * sizeof_datatype (info.datatypeA)));
      }

   void *devBT = nullptr;
   if (transposeB)
      {
         //cudaMallocAsync(&devBT, info.sizeB *
         // sizeof_datatype(info.datatypeB), streamB);
         //cudaEventRecord(eventB, streamB);
         cudaCheck (cudaMalloc (
             &devBT, info.sizeB * sizeof_datatype (info.datatypeB)));
      }

   // Transpose A and B -> AT/BT
   switch (transpose_backend)
      {
      case TransposeBackend::CUTT:
         {
            if (transposeA)
               cuttCheck (cuttExecute (this->planA, devA, devAT));
            if (transposeB)
               cuttCheck (cuttExecute (this->planB, devB, devBT));
         }
         break;
      default:
         throw InvalidOptimizationStrategy (
             "Transpose backend not recognized or not implemented.");
      }

   // (GEMM) CT = alpha*AT*BT
   void *devCT = nullptr;
   //cudaMallocAsync(&devCT, info.sizeC * sizeof_datatype(info.datatypeC),
   //   streamC);
   //cudaEventRecord(eventC, streamC);
   cudaCheck (
       cudaMalloc (&devCT, info.sizeC * sizeof_datatype (info.datatypeC)));

   //cudaStream_t mainStream;
   //cublasGetStream(handle, &mainStream);
   //cudaStreamWaitEvent(mainStream, eventA, 0);
   //cudaStreamWaitEvent(mainStream, eventB, 0);
   //cudaStreamWaitEvent(mainStream, eventC, 0);

   double value = 0.0;
   GenericScalar zero = init_generic_scalar (info.datatypeC, &value);
   cublasStatus_t stat = cublasGemmEx (
       this->handle, this->transa, this->transb, this->m, this->n, this->k,
       info.alpha, (transposeA) ? devAT : devA,
       get_cuda_datatype (info.datatypeA), this->lda,
       (transposeB) ? devBT : devB, get_cuda_datatype (info.datatypeB),
       this->ldb, &(zero.value), devCT, get_cuda_datatype (info.datatypeC),
       this->ldc, info.computeType, CUBLAS_GEMM_DEFAULT);
   if (stat != CUBLAS_STATUS_SUCCESS)
      {
         printf ("Issue with CUBLAS GEMM\n");
         return;
      }

   // cudaFreeAsync(devAT, streamA);
   // cudaFreeAsync(devBT, streamB);
   if (transposeA)
      cudaCheck (cudaFree (devAT));
   if (transposeB)
      cudaCheck (cudaFree (devBT));

   // Transpose CT->C
   void *devC_tmp = nullptr;
   cudaCheck (
       cudaMalloc(&devC_tmp, info.sizeC * sizeof_datatype (info.datatypeC)));
   if (transposeC)
      {
         // cudaMallocAsync(&devC_tmp, info.sizeC *
         // sizeof_datatype(info.datatypeC), streamC);
         cuttCheck (cuttExecute (this->planC, devCT, devC_tmp));
      }
   else
      {
         cudaCheck (cudaMemcpy (devC_tmp, devCT,
                                info.sizeC * sizeof_datatype (info.datatypeC),
                                cudaMemcpyDeviceToDevice));
      }
   cudaCheck (cudaFree(devCT));

   // (GEAM) C += beta*C_tmp
   bool is_beta_not_zero;
   switch (info.datatypeC)
      {
      case DataType::FLOAT64:
         is_beta_not_zero = *(double *)(info.beta) != 0.0;
         break;
      case DataType::FLOAT32:
         is_beta_not_zero = *(float *)(info.beta) != 0.0;
         break;
      case DataType::COMPLEX64:
         is_beta_not_zero = (((cuDoubleComplex *)info.beta)->y != 0.0)
                        or (((cuDoubleComplex *)info.beta)->x != 0.0);
         break;
      case DataType::COMPLEX32:
         is_beta_not_zero = (((cuComplex *)info.beta)->y != 0.0)
                        or (((cuComplex *)info.beta)->x != 0.0);
         break;
      }
   if (is_beta_not_zero)
      {
         value = 1.0;
         GenericScalar one = init_generic_scalar (info.datatypeC, &value);
         switch (info.datatypeC)
            {
            case DataType::FLOAT64:
               {
                  stat = cublasDgeam (
                      this->handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m, this->n,
                      (const double *)info.beta, (const double *)devC, this->m,
                      &(one.value.f64), (const double *)devC_tmp, this->m,
                      (double *)devC, this->m);
               }
               break;
            case DataType::FLOAT32:
               {
                  stat = cublasSgeam (
                      this->handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m, this->n,
                      (const float *)info.beta, (const float *)devC, this->m,
                      &(one.value.f32), (const float *)devC_tmp, this->m,
                      (float *)devC, this->m);
               }
               break;
            case DataType::COMPLEX64:
               {
                  stat = cublasZgeam (
                      this->handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m, this->n,
                      (const cuDoubleComplex *)info.beta,
                      (const cuDoubleComplex *)devC, this->m, &(one.value.c64),
                      (const cuDoubleComplex *)devC_tmp, this->m,
                      (cuDoubleComplex *)devC, this->m);
               }
               break;
            case DataType::COMPLEX32:
               {
                  stat = cublasCgeam (
                      this->handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m, this->n,
                      (const cuComplex *)info.beta, (const cuComplex *)devC,
                      this->m, &(one.value.c32), (const cuComplex *)devC_tmp,
                      this->m, (cuComplex *)devC, this->m);
               }
               break;
            }
         if (stat != CUBLAS_STATUS_SUCCESS)
            {
               printf ("Issue with CUBLAS GEAM\n");
               return;
            }
      }
   else
      {
         cudaCheck (cudaMemcpy (devC, devC_tmp,
                                info.sizeC * sizeof_datatype (info.datatypeC),
                                cudaMemcpyDeviceToDevice));
      }

   // if(transposeA) cudaStreamDestroy(streamA);
   // if(transposeB) cudaStreamDestroy(streamB);
   // cudaFreeAsync(devC_tmp, streamC);
   cudaFree (devC_tmp);

   //cudaStreamDestroy(streamA);
   //cudaStreamDestroy(streamB);
   //cudaStreamDestroy(streamC);

   //cudaEventDestroy(eventA);
   //cudaEventDestroy(eventB);
   //cudaEventDestroy(eventC);
}

void
TTGTPlan::print_config ()
{
   printf ("---------------\n");
   printf ("TTGTPlan config\n");
   printf ("---------------\n");
   printf ("transposeA=%d\n", this->transposeA);
   printf ("transposeB=%d\n", this->transposeB);
   printf ("transposeC=%d\n", this->transposeC);
   printf ("GEMM:\n");
   switch (this->transa)
      {
      case CUBLAS_OP_N:
         printf ("transa=%s\n", "CUBLAS_OP_N");
         break;
      case CUBLAS_OP_T:
         printf ("transa=%s\n", "CUBLAS_OP_T");
         break;
      case CUBLAS_OP_C:
         printf ("transa=%s\n", "CUBLAS_OP_C");
         break;
      default:
         printf ("transa=%s\n", "UNKNOWN_OP");
         break;
      }
   switch (this->transb)
      {
      case CUBLAS_OP_N:
         printf ("transb=%s\n", "CUBLAS_OP_N");
         break;
      case CUBLAS_OP_T:
         printf ("transb=%s\n", "CUBLAS_OP_T");
         break;
      case CUBLAS_OP_C:
         printf ("transb=%s\n", "CUBLAS_OP_C");
         break;
      default:
         printf ("transb=%s\n", "UNKNOWN_OP");
         break;
      }
   printf ("m=%d\n", this->n);
   printf ("n=%d\n", this->m);
   printf ("k=%d\n", this->k);
   printf ("lda=%d\n", this->lda);
   printf ("ldb=%d\n", this->ldb);
   printf ("ldc=%d\n", this->ldc);
}
