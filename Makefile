ENABLE_TBLIS = false
CC = gcc-14
CXX = g++-14
SRC = reference_implementation/src
OBJ = obj
TEST = test
TBL = tblis_bindings
OUT = out
INC = reference_implementation/include
INTERFACE = interface
TBLIS = ../tblis
TBLIS_PARAM = -ltblis -lm -L$(TBLIS)/lib/.libs -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src
OBJECTS =$(filter-out obj/tblis_bind.o, $(filter-out obj/tapp.o, $(wildcard obj/*.o)))
CFLAGS = -fPIC
CXXFLAGS = -fPIC

ifeq ($(OS),Windows_NT) #windows
    EXEEXT = .exe
	LIBEXT = .dll
else #linux/mac
    EXEEXT =
	LIBEXT = .so
endif

COMPILE_DEFS = 
ifeq ($(ENABLE_TBLIS),true)
	COMPILE_DEFS = -DENABLE_TBLIS=1
else
	TBLIS_PARAM = 
endif


ifeq ($(ENABLE_TBLIS),true)
all: base_folders tapp $(OBJ)/tblis_bind.o $(OBJ)/test.o $(OUT)/test$(EXEEXT) $(OUT)/test++$(EXEEXT) demo driver exercise_contraction_answers exercise_tucker exercise_tucker_answers
else
all: base_folders tapp $(OBJ)/tblis_bind.o demo driver exercise_contraction_answers exercise_tucker exercise_tucker_answers
endif

tapp: base_folders $(OBJ)/tapp.o $(OBJ)/error.o $(OBJ)/tensor.o $(OBJ)/product.o $(OBJ)/executor.o $(OBJ)/handle.o lib/libtapp$(LIBEXT)
demo: base_folders tapp $(OBJ)/demo.o $(OUT)/demo$(EXEEXT)
driver: driver_folders tapp examples/driver/obj/driver.o examples/driver/out/driver$(EXEEXT)
exercise_contraction: exercise_contraction_folders tapp examples/exercise_contraction/obj/exercise_contraction.o examples/exercise_contraction/out/exercise_contraction$(EXEEXT)
exercise_contraction_answers: exercise_contraction_answers_folders tapp examples/exercise_contraction/answers/obj/exercise_contraction_answers.o examples/exercise_contraction/answers/out/exercise_contraction_answers$(EXEEXT)
exercise_tucker: exercise_tucker_folders tapp examples/exercise_tucker/tapp_tucker/obj/exercise_tucker.o examples/exercise_tucker/tapp_tucker/lib/libexercise_tucker$(LIBEXT)
exercise_tucker_answers: exercise_tucker_answers_folders tapp examples/exercise_tucker/tapp_tucker/answers/obj/exercise_tucker_answers.o examples/exercise_tucker/tapp_tucker/answers/lib/libexercise_tucker$(LIBEXT)

base_folders:
	mkdir -p obj lib out bin
driver_folders:
	mkdir -p examples/driver/obj examples/driver/out

exercise_contraction_folders:
	mkdir -p examples/exercise_contraction/obj examples/exercise_contraction/out

exercise_contraction_answers_folders:
	mkdir -p examples/exercise_contraction/answers/obj examples/exercise_contraction/answers/out

exercise_tucker_folders:
	mkdir -p examples/exercise_tucker/tapp_tucker/obj examples/exercise_tucker/tapp_tucker/lib

exercise_tucker_answers_folders:
	mkdir -p examples/exercise_tucker/tapp_tucker/answers/obj examples/exercise_tucker/tapp_tucker/answers/lib

$(OBJ)/tapp.o: $(OBJ)/product.o $(OBJ)/tensor.o $(OBJ)/error.o $(OBJ)/executor.o $(OBJ)/handle.o $(OBJ)/tblis_bind.o
	ld -r $(OBJECTS) -o obj/tapp.o

$(OBJ)/error.o: $(SRC)/error.c $(INC)/ref_imp.h $(INTERFACE)/tapp/error.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/error.c -o $(OBJ)/error.o -I$(INC) -I$(INTERFACE)

$(OBJ)/tensor.o: $(SRC)/tensor.c  $(INC)/ref_imp.h $(INTERFACE)/tapp/tensor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tensor.c -o $(OBJ)/tensor.o -I$(INC) -I$(INTERFACE)

$(OBJ)/product.o: $(SRC)/product.c $(INC)/ref_imp.h $(INTERFACE)/tapp/product.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/product.c -o $(OBJ)/product.o $(COMPILE_DEFS) -I$(INC) -I$(INTERFACE) -I$(TBL)

$(OBJ)/executor.o: $(SRC)/executor.c $(INC)/ref_imp.h $(INTERFACE)/tapp/executor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/executor.c -o $(OBJ)/executor.o $(COMPILE_DEFS) -I$(INC) -I$(INTERFACE)

$(OBJ)/handle.o: $(SRC)/handle.c $(INC)/ref_imp.h $(INTERFACE)/tapp/handle.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/handle.c -o $(OBJ)/handle.o -I$(INC) -I$(INTERFACE)

ifeq ($(ENABLE_TBLIS),true)
$(OBJ)/tblis_bind.o: $(TBL)/tblis_bind.cpp $(TBL)/tblis_bind.h
	$(CXX) $(CXXFLAGS) -c -g -Wall $(TBL)/tblis_bind.cpp -o $(OBJ)/tblis_bind.o -I$(INTERFACE) -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src
else
$(OBJ)/tblis_bind.o: $(TBL)/tblis_bind.cpp $(TBL)/tblis_bind.h
	touch $(OBJ)/tblis_bind.o
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  RPATH_FLAG = -Wl,-rpath,'@executable_path/../lib'
else ifeq ($(OS),Windows_NT) #windows
  RPATH_FLAG =
else #linux
  RPATH_FLAG = -Wl,-rpath,'$$ORIGIN/../lib'
endif

$(OBJ)/test.o: $(TEST)/test.c $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/test.c -o $(OBJ)/test.o -I$(INTERFACE) -I$(TBL)

$(OUT)/test$(EXEEXT): $(OUT)/test.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g $(OBJ)/test.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/test$(EXEEXT) -I$(INTERFACE) -I$(TBL) $(TBLIS_PARAM) $(RPATH_FLAG)

$(OBJ)/helpers.o: $(TEST)/helpers.c
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/helpers.c -o $(OBJ)/helpers.o

$(OBJ)/demo.o: $(TEST)/demo.c lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/demo.c -o $(OBJ)/demo.o -I$(INTERFACE) -I$(TEST)

$(OUT)/demo$(EXEEXT): $(OBJ)/demo.o $(OBJ)/helpers.o lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -g  $(OBJ)/demo.o $(OBJ)/helpers.o -o $(OUT)/demo$(EXEEXT) -I$(INTERFACE) -L./lib -ltapp $(RPATH_FLAG)

examples/driver/obj/driver.o: examples/driver/driver.c lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -c -g -Wall examples/driver/driver.c -o examples/driver/obj/driver.o -I$(INTERFACE) -I$(TEST)

examples/driver/out/driver$(EXEEXT): examples/driver/obj/driver.o $(OBJ)/helpers.o lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -g  examples/driver/obj/driver.o $(OBJ)/helpers.o -o examples/driver/out/driver$(EXEEXT) -I$(INTERFACE) -L./lib -ltapp $(RPATH_FLAG)

examples/exercise_contraction/obj/exercise_contraction.o: examples/exercise_contraction/exercise_contraction.c lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -c -g -Wall examples/exercise_contraction/exercise_contraction.c -o examples/exercise_contraction/obj/exercise_contraction.o -I$(INTERFACE) -I$(TEST)

examples/exercise_contraction/out/exercise_contraction$(EXEEXT): examples/exercise_contraction/obj/exercise_contraction.o $(OBJ)/helpers.o lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -g  examples/exercise_contraction/obj/exercise_contraction.o $(OBJ)/helpers.o -o examples/exercise_contraction/out/exercise_contraction$(EXEEXT) -I$(INTERFACE) -L./lib -ltapp $(RPATH_FLAG)

examples/exercise_contraction/answers/obj/exercise_contraction_answers.o: examples/exercise_contraction/answers/exercise_contraction_answers.c lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -c -g -Wall examples/exercise_contraction/answers/exercise_contraction_answers.c -o examples/exercise_contraction/answers/obj/exercise_contraction_answers.o -I$(INTERFACE) -I$(TEST)

examples/exercise_contraction/answers/out/exercise_contraction_answers$(EXEEXT): examples/exercise_contraction/answers/obj/exercise_contraction_answers.o $(OBJ)/helpers.o lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -g  examples/exercise_contraction/answers/obj/exercise_contraction_answers.o $(OBJ)/helpers.o -o examples/exercise_contraction/answers/out/exercise_contraction_answers$(EXEEXT) -I$(INTERFACE) -L./lib -ltapp $(RPATH_FLAG)
	
$(OUT)/test++$(EXEEXT): $(TEST)/test.cpp lib/libtapp$(LIBEXT)
	$(CXX) -g  $(TEST)/test.cpp  -o $(OUT)/test++$(EXEEXT) -Itest -I$(INTERFACE) -L./lib -ltapp -I$(TBL)  $(TBLIS_PARAM) $(RPATH_FLAG)

examples/exercise_tucker/tapp_tucker/obj/exercise_tucker.o: examples/exercise_tucker/tapp_tucker/exercise_tucker.c lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -c -g -Wall examples/exercise_tucker/tapp_tucker/exercise_tucker.c -o examples/exercise_tucker/tapp_tucker/obj/exercise_tucker.o -I$(INTERFACE) -I$(TEST) -L./lib -ltapp $(RPATH_FLAG)

examples/exercise_tucker/tapp_tucker/answers/obj/exercise_tucker_answers.o: examples/exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.c lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -c -g -Wall examples/exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.c -o examples/exercise_tucker/tapp_tucker/answers/obj/exercise_tucker_answers.o -I$(INTERFACE) -I$(TEST) -L./lib -ltapp $(RPATH_FLAG)

ifeq ($(UNAME_S),Darwin)
  SONAME = -install_name "@rpath/libexercise_tucker.so"
else ifeq ($(OS),Windows_NT) #windows
  SONAME =
else #linux
  SONAME = -Wl,-soname,libexercise_tucker.so
endif

examples/exercise_tucker/tapp_tucker/answers/lib/libexercise_tucker$(LIBEXT): examples/exercise_tucker/tapp_tucker/answers/obj/exercise_tucker_answers.o $(OBJ)/tapp.o
	$(CC) -shared -fPIC examples/exercise_tucker/tapp_tucker/answers/obj/exercise_tucker_answers.o $(OBJ)/tapp.o -o examples/exercise_tucker/tapp_tucker/answers/lib/libexercise_tucker$(LIBEXT) $(SONAME) -I$(INTERFACE) -L./lib -ltapp $(RPATH_FLAG)

examples/exercise_tucker/tapp_tucker/lib/libexercise_tucker$(LIBEXT): examples/exercise_tucker/tapp_tucker/obj/exercise_tucker.o $(OBJ)/tapp.o
	$(CC) -shared -fPIC examples/exercise_tucker/tapp_tucker/obj/exercise_tucker.o $(OBJ)/tapp.o -o examples/exercise_tucker/tapp_tucker/lib/libexercise_tucker$(LIBEXT) $(SONAME) -I$(INTERFACE) -L./lib -ltapp $(RPATH_FLAG)

ifeq ($(UNAME_S),Darwin)
  SONAME = -install_name "@rpath/libtapp.so"
else ifeq ($(OS),Windows_NT) #windows
  SONAME =
else #linux
  SONAME = -Wl,-soname,libtapp.so
endif


# Conditional linking flags for macOS
ifeq ($(UNAME_S),Darwin)
  UNDEFINED_FLAG = -undefined dynamic_lookup
else
  UNDEFINED_FLAG =
endif

ifeq ($(ENABLE_TBLIS),true)
lib/libtapp$(LIBEXT): $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared -fPIC $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o lib/libtapp$(LIBEXT) $(SONAME) -I$(INTERFACE) -I$(TBL) $(TBLIS_PARAM) $(UNDEFINED_FLAG)
else
lib/libtapp$(LIBEXT): $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared -fPIC $(OBJ)/tapp.o -o lib/libtapp$(LIBEXT) $(SONAME) -I$(INTERFACE) -I$(TBL) $(TBLIS_PARAM) $(UNDEFINED_FLAG)
endif

ifeq ($(ENABLE_TBLIS),true)
.PHONY: test
test:
	$(OUT)/test++$(EXEEXT)
else
.PHONY: test
test:
	$(OUT)/demo$(EXEEXT)
endif

clean:
	rm -f $(OBJ)/tensor.o
	rm -f $(OBJ)/product.o
	rm -f $(OBJ)/error.o
	rm -f $(OBJ)/executor.o
	rm -f $(OBJ)/handle.o
	rm -f $(OBJ)/tapp.o
	rm -f $(OBJ)/tblis_bind.o
	rm -f $(OUT)/test$(EXEEXT)
	rm -f $(OBJ)/test.o
	rm -f $(OUT)/test++$(EXEEXT)
	rm -f $(OUT)/demo$(EXEEXT)
	rm -f $(OBJ)/demo.o
	rm -f examples/driver/out/driver$(EXEEXT)
	rm -f examples/driver/obj/driver.o
	rm -f examples/exercise_contraction/out/exercise_contraction$(EXEEXT)
	rm -f examples/exercise_contraction/obj/exercise_contraction.o
	rm -f examples/exercise_contraction/answers/obj/exercise_contraction_answers.o
	rm -f examples/exercise_contraction/answers/out/exercise_contraction_answers$(EXEEXT)
	rm -f examples/exercise_tucker/obj/exercise_tucker.o
	rm -f examples/exercise_tucker/lib/libexercise_tucker$(LIBEXT)
	rm -f $(OBJ)/helpers.o
	rm -f lib/libtapp$(LIBEXT)
