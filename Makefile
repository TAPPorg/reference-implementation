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

COMPILE_DEFS = 
ifeq ($(ENABLE_TBLIS),true)
	COMPILE_DEFS = -DENABLE_TBLIS=1
else
	TBLIS_PARAM = 
endif


ifeq ($(ENABLE_TBLIS),true)
all: folders obj/tapp.o obj/error.o obj/tensor.o obj/product.o obj/executor.o obj/handle.o obj/tblis_bind.o lib/libtapp.so out/test.o out/test out/test++ out/demo.o out/demo
else
all: folders obj/tapp.o obj/error.o obj/tensor.o obj/product.o obj/executor.o obj/handle.o obj/tblis_bind.o lib/libtapp.so out/demo.o out/demo 
endif


folders:
	mkdir -p obj lib out bin

obj/tapp.o: obj/product.o obj/tensor.o obj/error.o obj/executor.o obj/handle.o obj/tblis_bind.o
	ld -r $(OBJECTS) -o obj/tapp.o

obj/error.o: $(SRC)/tapp/error.c $(INC)/tapp/error.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/error.c -o $(OBJ)/error.o -I$(INC) -I$(INC)/tapp

obj/tensor.o: $(SRC)/tapp/tensor.c $(INC)/tapp/tensor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/tensor.c -o $(OBJ)/tensor.o -I$(INC) -I$(INC)/tapp

obj/product.o: $(SRC)/tapp/product.c $(INC)/tapp/product.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/product.c -o $(OBJ)/product.o $(COMPILE_DEFS) -I$(INC) -I$(INC)/tapp -I$(TBL)

obj/executor.o: $(SRC)/tapp/executor.c $(INC)/tapp/executor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/executor.c -o $(OBJ)/executor.o $(COMPILE_DEFS) -I$(INC) -I$(INC)/tapp

obj/handle.o: $(SRC)/tapp/handle.c $(INC)/tapp/handle.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/handle.c -o $(OBJ)/handle.o -I$(INC) -I$(INC)/tapp

ifeq ($(ENABLE_TBLIS),true)
obj/tblis_bind.o: $(TBL)/tblis_bind.cpp $(TBL)/tblis_bind.h
	$(CXX) $(CXXFLAGS) -c -g -Wall $(TBL)/tblis_bind.cpp -o $(OBJ)/tblis_bind.o -I$(INC) -I$(INC)/tapp -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src
else
obj/tblis_bind.o: $(TBL)/tblis_bind.cpp $(TBL)/tblis_bind.h
	touch obj/tblis_bind.o
endif

out/test.o: $(TEST)/test.c $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/test.c -o $(OUT)/test.o -I$(INC) -I$(INC)/tapp -I$(TBL)

out/test: $(OUT)/test.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g $(OUT)/test.o $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/test -I$(INC) -I$(INC)/tapp -I$(TBL) $(TBLIS_PARAM)

out/demo.o: $(TEST)/demo.c lib/libtapp.so
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/demo.c -o $(OUT)/demo.o -I$(INC) -I$(INC)/tapp

out/demo: $(OUT)/demo.o  lib/libtapp.so
	$(CC) $(CFLAGS) -g  $(OUT)/demo.o -o $(OUT)/demo -I$(INC) -I$(INC)/tapp -L./lib -ltapp

out/test++: $(TEST)/test.cpp lib/libtapp.so
	$(CXX) -g  $(TEST)/test.cpp  -o $(OUT)/test++ -Itest -I$(INC) -I$(INC)/tapp -L./lib -ltapp -I$(TBL)  $(TBLIS_PARAM)


UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SONAME = -install_name "@rpath/libtapp.so"
else
	SONAME = -Wl,-soname,libtapp.so
endif

ifeq ($(ENABLE_TBLIS),true)
lib/libtapp.so: $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared -fPIC $(OBJ)/tapp.o $(OBJ)/tblis_bind.o -o lib/libtapp.so $(SONAME) -I$(INC) -I$(INC)/tapp -I$(TBL) $(TBLIS_PARAM)
else
lib/libtapp.so: $(OBJ)/tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared -fPIC $(OBJ)/tapp.o -o lib/libtapp.so $(SONAME) -I$(INC) -I$(INC)/tapp -I$(TBL) $(TBLIS_PARAM)
endif

ifeq ($(ENABLE_TBLIS),true)
.PHONY: test
test:
	out/test++
else
.PHONY: test
test:
	out/demo
endif

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
	rm -f lib/libtapp.so
