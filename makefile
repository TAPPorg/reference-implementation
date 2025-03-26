CC = mpicc
CXX = mpicxx
SRC = src
OBJ = obj
TEST = test
TBL = tblis_bindings
OUT = out
INC = src
TBLIS = ../tblis
TBLIS_PARAM = -ltblis -lm -L$(TBLIS)/lib/.libs -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src
CTF_PATH = ../ctf
# 
CTF_PARAM = -O3 -fopenmp -Wall  -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1 -DUSE_LAPACK -I/lustre/orion/chp109/world-shared/ctf/include -L/lustre/orion/chp109/world-shared/ctf/lib_shared -lctf -Wl,-rpath=/lustre/orion/chp109/world-shared/ctf/lib_shared -L/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/openblas-0.3.17-mneinxtrfs6b2yhwndved3u6aoiwvhjw/lib -Wl,-rpath=/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/openblas-0.3.17-mneinxtrfs6b2yhwndved3u6aoiwvhjw/lib -lopenblas
OBJECTS =$(filter-out obj/tblis_bind.o, $(filter-out obj/tapp.o, $(wildcard obj/*.o)))
CFLAGS = -fPIC
CXXFLAGS = -fPIC

# mpicxx 

all: folders obj/tapp.o obj/error.o obj/tensor.o obj/product.o obj/executor.o obj/handle.o obj/tblis_bind.o lib/libtapp.so out/test.o out/test out/test++ out/demo.o out/demo out/uselib.o out/uselib out/worker.x


folders:
	mkdir -p obj lib out bin

obj/tapp.o: obj/product.o obj/tensor.o obj/error.o obj/executor.o obj/handle.o obj/tblis_bind.o
	ld -relocatable $(OBJECTS) -o obj/tapp.o

obj/error.o: $(SRC)/tapp/error.c $(INC)/tapp/error.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/error.c -o $(OBJ)/error.o -I$(INC) -I$(INC)/tapp

obj/tensor.o: $(SRC)/tapp/tensor.c $(INC)/tapp/tensor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/tensor.c -o $(OBJ)/tensor.o -I$(INC) -I$(INC)/tapp

obj/product.o: $(SRC)/tapp/product.c $(INC)/tapp/product.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/product.c -o $(OBJ)/product.o -I$(INC) -I$(INC)/tapp -I$(TBL)

obj/executor.o: $(SRC)/tapp/executor.c $(INC)/tapp/executor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/executor.c -o $(OBJ)/executor.o -I$(INC) -I$(INC)/tapp

obj/handle.o: $(SRC)/tapp/handle.c $(INC)/tapp/handle.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/handle.c -o $(OBJ)/handle.o -I$(INC) -I$(INC)/tapp

obj/tblis_bind.o: $(TBL)/tblis_bind.cpp $(TBL)/tblis_bind.h distributed/mpi_utils.h
	$(CXX) $(CXXFLAGS) -c -g -Wall $(TBL)/tblis_bind.cpp -o $(OBJ)/tblis_bind.o -I$(INC) -I$(INC)/tapp -Idistributed -O3 -fopenmp -Wall  -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1 -DUSE_LAPACK -I/lustre/orion/chp109/world-shared/ctf/include

out/test.o: $(TEST)/test.c $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/test.c -o $(OUT)/test.o -I$(INC) -I$(INC)/tapp -I$(TBL)

out/test: $(OUT)/test.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g $(OUT)/test.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/test -I$(INC) -I$(INC)/tapp -I$(TBL) $(CTF_PARAM) $(TBLIS_PARAM)

out/demo.o: $(TEST)/demo.c $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/demo.c -o $(OUT)/demo.o -I$(INC) -I$(INC)/tapp -I$(TBL)

out/demo: $(OUT)/demo.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g  $(OUT)/demo.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/demo -I$(INC) -I$(INC)/tapp -I$(TBL) $(CTF_PARAM) $(TBLIS_PARAM)

out/test++: $(TEST)/test.cpp $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g  $(TEST)/test.cpp $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/test++ -Itest -I$(INC) -I$(INC)/tapp -I$(TBL) $(CTF_PARAM) $(TBLIS_PARAM)

out/uselib.o: $(TEST)/uselib.c lib/libtapp.so
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/uselib.c -o $(OUT)/uselib.o -I$(INC)

out/uselib: $(OUT)/uselib.o lib/libtapp.so
	$(CC) $(CFLAGS) -g  $(OUT)/uselib.o -o $(OUT)/uselib -I$(INC) -L./lib -ltapp

lib/libtapp.so: $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o lib/libtapp.so -I$(INC) -I$(INC)/tapp -I$(TBL) $(CTF_PARAM) $(TBLIS_PARAM)
# 	$(CC) -shared -fPIC $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o lib/libtapp.so -I$(INC) -I$(INC)/tapp -I$(TBL) $(TBLIS_PARAM)

out/worker.x: distributed/worker.cxx distributed/mpi_utils.h
	mpicxx -O3 -fopenmp -Wall  -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1 -DUSE_LAPACK distributed/worker.cxx -o out/worker.x -I/lustre/orion/chp109/world-shared/ctf/include -L/lustre/orion/chp109/world-shared/ctf/lib -lctf  -L/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/openblas-0.3.17-mneinxtrfs6b2yhwndved3u6aoiwvhjw/lib -Wl,-rpath=/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/openblas-0.3.17-mneinxtrfs6b2yhwndved3u6aoiwvhjw/lib -lopenblas -Idistributed

.PHONY: test
test:
	# salloc -A CHP109 -N 1 -t 2:00:00 -q debug
	srun -n 3 -N 1 --multi-prog distributed/conf_srun
  

.PHONY: clean
clean:
	rm -f obj/tensor.o
	rm -f obj/product.o
	rm -f obj/error.o
	rm -f obj/executor.o
	rm -f obj/handle.o
	rm -f obj/tapp.o
	rm -f obj/tblis_bind.o
	rm -f out/test
	rm -f out/test.o
	rm -f out/test++
	rm -f out/demo
	rm -f out/demo.o
	rm -f out/uselib
	rm -f out/uselib.o
	rm -f lib/libtapp.so
	rm -f out/worker.x


