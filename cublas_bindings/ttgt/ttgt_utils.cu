#include "ttgt_utils.h"

void
get_bounded_indices (int32_t **bounded_indices, int32_t *n_bounded_indices,
                     const int32_t *modeA, const int32_t rankA,
                     const int32_t *modeC, const int32_t rankC)
{
   std::vector<int32_t> bounded_indices_tmp (rankA);
   *n_bounded_indices = 0;

   for (int i = 0; i < rankA; i++)
      {
         int32_t indexA = modeA[i];
         if (std::find (modeC, modeC + rankC, indexA) == (modeC + rankC))
            {
               bounded_indices_tmp[*n_bounded_indices] = indexA;
               *n_bounded_indices += 1;
            }
      }
   // copy data back to bounded_indices
   *bounded_indices = new int32_t[*n_bounded_indices];
   for (int i = 0; i < rankA; i++)
      {
         (*bounded_indices)[i] = bounded_indices_tmp[i];
      }
};

void
get_free_indices (int32_t *free_indicesA, const int32_t n_free_indicesA,
                  const int32_t *modeA, const int32_t rankA,
                  const int32_t *modeC, const int32_t rankC)
{
   int32_t index = 0;
   for (int i = 0; i < rankA; i++)
      {
         if (index == n_free_indicesA)
            return;
         int32_t indexA = modeA[i];

         if (std::find (modeC, modeC + rankC, indexA) != (modeC + rankC))
            {
               free_indicesA[index] = indexA;
               index += 1;
            }
      }
}

size_t
sizeof_datatype (DataType dtype)
{
   switch (dtype)
      {
      case DataType::FLOAT32:
         return sizeof (float);
      case DataType::FLOAT64:
         return sizeof (double);
      case DataType::COMPLEX32:
         return sizeof (std::complex<float>);
      case DataType::COMPLEX64:
         return sizeof (std::complex<double>);
      }
   return 0;
}

// Get permutation from A to AT
void
get_permutation (int32_t *permutationA, int32_t *modeA, int32_t *modeAT,
                 int32_t rankA)
{
   std::unordered_map<int32_t, int32_t> contraction_index_to_local_index_AT;
   for (int i = 0; i < rankA; i++)
      {
         contraction_index_to_local_index_AT[modeAT[i]] = i;
      }

   for (int i = 0; i < rankA; i++)
      {
         permutationA[i] = contraction_index_to_local_index_AT[modeA[i]];
      }
}

cudaDataType
get_cuda_datatype (DataType dtype)
{
   switch (dtype)
      {
      case DataType::FLOAT32:
         return CUDA_R_32F;
      case DataType::FLOAT64:
         return CUDA_R_64F;
      case DataType::COMPLEX32:
         return CUDA_C_32F;
      case DataType::COMPLEX64:
         return CUDA_C_64F;
      }
   return CUDA_R_16BF;
}

GenericScalar
init_generic_scalar (DataType type, void *value)
{
   GenericScalar s;
   double val = *(double *)value;
   if (type == FLOAT32)
      {
         s.value.f32 = (float)val;
      }
   else if (type == FLOAT64)
      {
         s.value.f64 = val;
      }
   else if (type == COMPLEX32)
      {
         s.value.c32 = make_cuComplex ((float)val, 0.0f);
      }
   else if (type == COMPLEX64)
      {
         s.value.c64 = make_cuDoubleComplex (val, 0.0);
      }
   return s;
}

// Used for cuTT with C64 datatype
int *
prepend (int *A, int32_t rankA, int value)
{
   int *A_new = (int *)malloc ((rankA + 1) * sizeof (int));
   A_new[0] = value;
   memcpy (A_new + 1, A, rankA * sizeof (int));
   return A_new;
}

// Used for cuTT with C64 datatype
int32_t *
get_new_permutation (int32_t *permutation, int32_t size)
{
   int32_t *new_permutation
       = (int32_t *)malloc ((size + 1) * sizeof (int32_t));
   new_permutation[0] = 0;
   for (int32_t i = 0; i < size; i++)
      new_permutation[i + 1] = permutation[i] + 1;
   return new_permutation;
}
