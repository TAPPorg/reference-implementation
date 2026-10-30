# Context
This example display the use of multiple backends for the tenpi-optimized CCD equations:
- tblis
- cutensor
- multi-tapp: tblis+cutensor

TBLIS is used the reference for generating the inputs for the other backends and is done sequentially. The cuTENSOR backend has 3 versions:
1. 1 gpu and in-order synchronized contractions as in the TBLIS reference
2. 1 gpu with 2 streams where `stream1` only performs the PP contraction with is the most costly, while `stream2` performs the rest. At the end the result of the PP contraction is accumulated in the final result which may give different results compare to the TBLIS reference.
3. 2 gpu with 1 stream each. In this case the PP contraction is done on another GPU on the same node.

The multi-tapp example adopts the same strategy explained earlier at the exception that TBLIS is used for all contractions except PP.

# How to run the examples
1. Build TAPP with TBLIS in `build_tblis`
2. Run the command:
```
./tapp-tblis-ccd-reference $NOCC $NVIR
```
which will generate the input tensors in binary files.
3. Still within the `build_tblis` folder, save the binary files:
```
mkdir tblis-reference-${NOCC}-${NVIR} #inside build_tblis
mv *.bin tblis-reference-${NOCC}-${NVIR}
```
4. Build TAPP with cuTENSOR
5. Run one the following command:
```
./tapp-cutensor-ccd $NOCC $NVIR
./tapp-cutensor-ccd-2-streams $NOCC $NVIR
./tapp-cutensor-ccd-2-gpus-2-streams $NOCC $NVIR
./multi-tapp-ccd $NOCC $NVIR
```
The binary files path are hard-coded...

Notes:
- The tensor info objects should be destroyed at the end to avoid unecessary synchronization
- The results might differ as the order of tensor contractions/permutes is changed
- The correctness is not checked, but the use of CUDA features has been verified with NVIDIA compiler
