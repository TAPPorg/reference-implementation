ENABLE_TBLIS = false
CC = gcc
CXX = g++
SRC = src
OBJ = obj
TEST = test
TBL = tblis_bindings
OUT = out
INC = src
TBLIS = ../tblis
TBLIS_PARAM = -ltblis -lm -L$(TBLIS)/lib/.libs -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src
OBJECTS =$(filter-out obj/tblis_bind.o, $(filter-out obj/tapp.o, $(wildcard obj/*.o)))
CFLAGS = -fPIC
CXXFLAGS = -fPIC

ifeq ($(OS),Windows_NT) #windows
    EXEEXT = .exe
	LIBEXT = .lib
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
all: folders $(OBJ)/tapp.o $(OBJ)/error.o $(OBJ)/tensor.o $(OBJ)/product.o $(OBJ)/executor.o $(OBJ)/handle.o $(OBJ)/tblis_bind.o lib/libtapp$(LIBEXT) $(OBJ)/test.o $(OUT)/test$(EXEEXT) $(OUT)/test++$(EXEEXT) $(OBJ)/demo.o $(OUT)/demo$(EXEEXT) $(OBJ)/driver.o $(OUT)/driver$(EXEEXT) $(OBJ)/exercise_contraction.o $(OUT)/exercise_contraction$(EXEEXT)
else
all: folders $(OBJ)/tapp.o $(OBJ)/error.o $(OBJ)/tensor.o $(OBJ)/product.o $(OBJ)/executor.o $(OBJ)/handle.o $(OBJ)/tblis_bind.o lib/libtapp$(LIBEXT) $(OBJ)/demo.o $(OUT)/demo$(EXEEXT) $(OBJ)/driver.o $(OUT)/driver$(EXEEXT) $(OBJ)/exercise_contraction.o $(OUT)/exercise_contraction$(EXEEXT)
endif

demo: folders $(OBJ)/tapp.o $(OBJ)/error.o $(OBJ)/tensor.o $(OBJ)/product.o $(OBJ)/executor.o $(OBJ)/handle.o lib/libtapp.so $(OBJ)/demo.o $(OUT)/demo$(EXEEXT)
driver: folders $(OBJ)/tapp.o $(OBJ)/error.o $(OBJ)/tensor.o $(OBJ)/product.o $(OBJ)/executor.o $(OBJ)/handle.o lib/libtapp.so $(OBJ)/driver.o $(OUT)/driver$(EXEEXT)
exercise_contraction: folders $(OBJ)/tapp.o $(OBJ)/error.o $(OBJ)/tensor.o $(OBJ)/product.o $(OBJ)/executor.o $(OBJ)/handle.o lib/libtapp.so $(OBJ)/exercise_contraction.o $(OUT)/exercise_contraction$(EXEEXT)

folders:
	mkdir -p obj lib out bin

$(OBJ)/tapp.o: $(OBJ)/product.o $(OBJ)/tensor.o $(OBJ)/error.o $(OBJ)/executor.o $(OBJ)/handle.o $(OBJ)/tblis_bind.o
	ld -r $(OBJECTS) -o obj/tapp.o

$(OBJ)/error.o: $(SRC)/tapp/error.c $(INC)/tapp/error.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/error.c -o $(OBJ)/error.o -I$(INC) -I$(INC)/tapp

$(OBJ)/tensor.o: $(SRC)/tapp/tensor.c $(INC)/tapp/tensor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/tensor.c -o $(OBJ)/tensor.o -I$(INC) -I$(INC)/tapp

$(OBJ)/product.o: $(SRC)/tapp/product.c $(INC)/tapp/product.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/product.c -o $(OBJ)/product.o $(COMPILE_DEFS) -I$(INC) -I$(INC)/tapp -I$(TBL)

$(OBJ)/executor.o: $(SRC)/tapp/executor.c $(INC)/tapp/executor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/executor.c -o $(OBJ)/executor.o $(COMPILE_DEFS) -I$(INC) -I$(INC)/tapp

$(OBJ)/handle.o: $(SRC)/tapp/handle.c $(INC)/tapp/handle.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/handle.c -o $(OBJ)/handle.o -I$(INC) -I$(INC)/tapp

ifeq ($(ENABLE_TBLIS),true)
$(OBJ)/tblis_bind.o: $(TBL)/tblis_bind.cpp $(TBL)/tblis_bind.h
	$(CXX) $(CXXFLAGS) -c -g -Wall $(TBL)/tblis_bind.cpp -o $(OBJ)/tblis_bind.o -I$(INC) -I$(INC)/tapp -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src
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
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/test.c -o $(OBJ)/test.o -I$(INC) -I$(INC)/tapp -I$(TBL)

$(OUT)/test$(EXEEXT): $(OUT)/test.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g $(OBJ)/test.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/test$(EXEEXT) -I$(INC) -I$(INC)/tapp -I$(TBL) $(TBLIS_PARAM) $(RPATH_FLAG)

$(OBJ)/helpers.o: $(TEST)/helpers.c
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/helpers.c -o $(OBJ)/helpers.o

$(OBJ)/demo.o: $(TEST)/demo.c lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/demo.c -o $(OBJ)/demo.o -I$(INC) -I$(INC)/tapp -I$(TEST)

$(OUT)/demo$(EXEEXT): $(OBJ)/demo.o $(OBJ)/helpers.o lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -g  $(OBJ)/demo.o $(OBJ)/helpers.o -o $(OUT)/demo$(EXEEXT) -I$(INC) -I$(INC)/tapp -L./lib -ltapp $(RPATH_FLAG)

$(OBJ)/driver.o: examples/driver.c lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -c -g -Wall examples/driver.c -o $(OBJ)/driver.o -I$(INC) -I$(INC)/tapp -I$(TEST)

$(OUT)/driver$(EXEEXT): $(OBJ)/driver.o $(OBJ)/helpers.o lib/libtapp$(LIBEXT)
	$(CC) $(CFLAGS) -g  $(OBJ)/driver.o $(OBJ)/helpers.o -o $(OUT)/driver$(EXEEXT) -I$(INC) -I$(INC)/tapp -L./lib -ltapp $(RPATH_FLAG)

$(OBJ)/exercise_contraction.o: examples/exercise_contraction/exercise_contraction.c lib/libtapp.so
	$(CC) $(CFLAGS) -c -g -Wall examples/exercise_contraction/exercise_contraction.c -o $(OBJ)/exercise_contraction.o -I$(INC) -I$(INC)/tapp -I$(TEST)

$(OUT)/exercise_contraction$(EXEEXT): $(OBJ)/exercise_contraction.o $(OBJ)/helpers.o lib/libtapp.so
	$(CC) $(CFLAGS) -g  $(OBJ)/exercise_contraction.o $(OBJ)/helpers.o -o $(OUT)/exercise_contraction$(EXEEXT) -I$(INC) -I$(INC)/tapp -L./lib -ltapp $(RPATH_FLAG)

$(OUT)/test++$(EXEEXT): $(TEST)/test.cpp lib/libtapp$(LIBEXT)
	$(CXX) -g  $(TEST)/test.cpp  -o $(OUT)/test++$(EXEEXT) -Itest -I$(INC) -I$(INC)/tapp -L./lib -ltapp -I$(TBL)  $(TBLIS_PARAM) $(RPATH_FLAG)


ifeq ($(UNAME_S),Darwin)
  SONAME = -install_name "@rpath/libtapp.so"
else ifeq ($(OS),Windows_NT) #windows
  SONAME =
else #linux
  SONAME = -Wl,-soname,libtapp.so
endif


ifeq ($(ENABLE_TBLIS),true)
lib/libtapp$(LIBEXT): $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared -fPIC $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o lib/libtapp$(LIBEXT) $(SONAME) -I$(INC) -I$(INC)/tapp -I$(TBL) $(TBLIS_PARAM)
else
lib/libtapp$(LIBEXT): $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared -fPIC $(OBJ)/tapp.o -o lib/libtapp$(LIBEXT) $(SONAME) -I$(INC) -I$(INC)/tapp -I$(TBL) $(TBLIS_PARAM)
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
	rm -f $(OUT)/driver$(EXEEXT)
	rm -f $(OBJ)/driver.o
	rm -f $(OUT)/exercise_contraction$(EXEEXT)
	rm -f $(OBJ)/exercise_contraction.o
	rm -f $(OBJ)/helpers.o
	rm -f lib/libtapp$(LIBEXT)
