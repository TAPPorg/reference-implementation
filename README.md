# TAPP reference implementation

Updated: 2024-11-01

The operation implemented might not be the final operation for the interface. The current implementation is some sort of general tensor multiplication.

It supports:
* Binary contractions
  * Free indices(indices appearing in A or B and D)
  * Contracted indices(indices appearing in A and B)
* Hadamard products(indices appearing in A, B and D)
* Unary contraction(indices appearing uniquely in A or B, summed over)
* Diagonals(multiple of the same idx within any tensor)
* Elemental operation conjugate
* Datatypes: float, double, complex float, complex double, _Float16
* Precisions: TAPP_DEFAULT_PREC, TAPP_F32F32_ACCUM_F32, TAPP_F64F64_ACCUM_F64, TAPP_F16F16_ACCUM_F16, TAPP_F16F16_ACCUM_F32, TAPP_BF16BF16_ACCUM_F32

\
Rules:
* C must have the same structure as D
* C must not be null unless beta is zero
* The input types of alpha and beta need to be the same as D
* Indices must not only appear in D (need to be "inherited" from A and/or B)

\
Errors catching implemented:
* Aliasing within D
* Non matching extents for the same idx in different tensors
* Indices appearing only in D
* Differences between C and D (idx, extents)
* C being null when beta is not zero
* Negative nmode
* Negative extents

\
Not yet implemented:
* Datatype 16 bit brain float, as datatype or precision (not caught)
* Status
* Executor
* Handle
* Batched product

\
Notes:
* Currently TAPP_DEFAULT_PREC uses the inputted precision for each input

\
Tests:
* Python/Numpy einsum:
  * Hadamard Product
  * (Binary) Contraction
  * Commutativity
  * Permutations on D
  * Equal size of extents
  * Outer product
  * Full contraction(scalar output)
  * Zero dimensional tensor(scalar) contraction
  * One dimensional tensor(vector) contraction
  * Subtensor same number of dimensions
  * Subtensor lower number of dimensions
  * Indices unique to A or B
  * Repeated indices(multiple of same idx within any tensor)
* C++/TBLIS:
  * Hadamard Product
  * (Binary) Contraction
  * Commutativity
  * Permutations on D
  * Equal size of extents
  * Outer product
  * Full contraction(scalar output)
  * Zero dimensional tensor(scalar) contraction
  * One dimensional tensor(vector) contraction
  * Subtensor same number of dimensions
  * Subtensor lower number of dimensions
  * Negative strides
  * Negative strides on subtensor with same number of indices
  * Negative strides on subtensor with lower number of indices
  * Mixed sign strides
  * Mixed sign strides on subtensor with same number of indices
  * Mixed sign strides on subtensor with lower number of indices
  * Contraction double precision
  * Contraction complex
  * Contraction complex double precision
  * Strides that are zero
  * Indices unique to A or B
  * Repeated indices(multiple of same idx within any tensor)
  * Hadamard product and free indices
  * Hadamard product and contracted indices
  * Error: Non matching extents
  * Error: C other structure than D
  * Error: Aliasing within D