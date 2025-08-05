HAS_TBLIS = false
CC = gcc-14
CXX = g++-14
SRC = src
OBJ = obj
TEST = test
TBL = tblis_bindings
OUT = out
INC = src
TBLIS = ../tblis
TBLIS_PARAM = -ltblis -lm -L$(TBLIS)/lib/.libs -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src
TAPP_PATH = /Users/losos/Downloads/test/reference-implementation
OBJECTS =$(filter-out obj/tblis_bind.o, $(filter-out obj/hi_tapp.o, $(wildcard obj/*.o)))
CFLAGS = -fPIC
CXXFLAGS = -fPIC

COMPILE_DEFS = 
ifeq ($(HAS_TBLIS),true)
	COMPILE_DEFS = -DHAS_TBLIS=1
else
	TBLIS_PARAM = 
endif

ifeq ($(HAS_TBLIS),true)
all: folders obj/hi_tapp.o obj/error.o obj/tensor.o obj/product.o obj/executor.o obj/handle.o obj/tblis_bind.o lib/libhi_tapp.so out/test.o out/test out/test++ out/demo.o out/demo out/uselib.o out/uselib
else
all: folders obj/hi_tapp.o obj/error.o obj/tensor.o obj/product.o obj/executor.o obj/handle.o obj/tblis_bind.o lib/libhi_tapp.so out/demo.o out/demo out/uselib.o out/uselib
endif


folders:
	mkdir -p obj lib out bin

obj/hi_tapp.o: obj/product.o obj/tensor.o obj/error.o obj/executor.o obj/handle.o obj/tblis_bind.o
	ld -r $(OBJECTS) -o obj/hi_tapp.o

obj/error.o: $(SRC)/hi_tapp/error.c $(INC)/hi_tapp/error.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/hi_tapp/error.c -o $(OBJ)/error.o -I$(INC) -I$(INC)/hi_tapp -I$(TAPP_PATH)/src

obj/tensor.o: $(SRC)/hi_tapp/tensor.c $(INC)/hi_tapp/tensor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/hi_tapp/tensor.c -o $(OBJ)/tensor.o -I$(INC) -I$(INC)/hi_tapp -I$(TAPP_PATH)/src

obj/product.o: $(SRC)/hi_tapp/product.c $(INC)/hi_tapp/product.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/hi_tapp/product.c -o $(OBJ)/product.o $(COMPILE_DEFS) -I$(INC) -I$(INC)/hi_tapp -I$(TBL) -I$(TAPP_PATH)/src

obj/executor.o: $(SRC)/hi_tapp/executor.c $(INC)/hi_tapp/executor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/hi_tapp/executor.c -o $(OBJ)/executor.o -I$(INC) -I$(INC)/hi_tapp -I$(TAPP_PATH)/src

obj/handle.o: $(SRC)/hi_tapp/handle.c $(INC)/hi_tapp/handle.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/hi_tapp/handle.c -o $(OBJ)/handle.o -I$(INC) -I$(INC)/hi_tapp -I$(TAPP_PATH)/src

ifeq ($(HAS_TBLIS),true)
obj/tblis_bind.o: $(TBL)/tblis_bind.cpp $(TBL)/tblis_bind.h
	$(CXX) $(CXXFLAGS) -c -g -Wall $(TBL)/tblis_bind.cpp -o $(OBJ)/tblis_bind.o -I$(INC) -I$(INC)/hi_tapp -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src -I$(TAPP_PATH)/src
else
obj/tblis_bind.o: $(TBL)/tblis_bind.cpp $(TBL)/tblis_bind.h
	touch obj/tblis_bind.o
endif

out/test.o: $(TEST)/test.c $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/test.c -o $(OUT)/test.o -I$(INC) -I$(INC)/hi_tapp -I$(TBL) -I$(TAPP_PATH)/src

out/test: $(OUT)/test.o $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g $(OUT)/test.o $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/test -I$(INC) -I$(INC)/hi_tapp -I$(TBL) $(TBLIS_PARAM) -I$(TAPP_PATH)/src -L$(TAPP_PATH)/lib -ltapp

out/demo.o: $(TEST)/demo.c $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/demo.c -o $(OUT)/demo.o -I$(INC) -I$(INC)/hi_tapp -I$(TBL) -I$(TAPP_PATH)/src

ifeq ($(HAS_TBLIS),true)
out/demo: $(OUT)/demo.o $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g  $(OUT)/demo.o $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/demo -I$(INC) -I$(INC)/hi_tapp -I$(TBL) $(TBLIS_PARAM) -I$(TAPP_PATH)/src -L$(TAPP_PATH)/lib -ltapp
else
out/demo: $(OUT)/demo.o $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g  $(OUT)/demo.o $(OBJ)/hi_tapp.o -o $(OUT)/demo -I$(INC) -I$(INC)/hi_tapp -I$(TBL) $(TBLIS_PARAM) -I$(TAPP_PATH)/src -L$(TAPP_PATH)/lib -Wl,-rpath,$(TAPP_PATH)/lib -ltapp
endif
	

out/test++: $(TEST)/test.cpp $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -g  $(TEST)/test.cpp $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o -o $(OUT)/test++ -Itest -I$(INC) -I$(INC)/hi_tapp -I$(TBL) $(TBLIS_PARAM) -I$(TAPP_PATH)/src -L$(TAPP_PATH)/lib -ltapp

out/uselib.o: $(TEST)/uselib.c lib/libhi_tapp.so
	$(CC) $(CFLAGS) -c -g -Wall $(TEST)/uselib.c -o $(OUT)/uselib.o -I$(INC) -I$(INC)/hi_tapp -I$(TAPP_PATH)/src

out/uselib: $(OUT)/uselib.o lib/libhi_tapp.so
	$(CC) $(CFLAGS) -g  $(OUT)/uselib.o -o $(OUT)/uselib -I$(INC) -I$(INC)/hi_tapp -I$(TAPP_PATH)/src -L./lib -lhi_tapp -L$(TAPP_PATH)/lib -ltapp

ifeq ($(HAS_TBLIS),true)
lib/libhi_tapp.so: $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared -fPIC $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o -o lib/libhi_tapp.so -I$(INC) -I$(INC)/hi_tapp -I$(TBL) $(TBLIS_PARAM) -I$(TAPP_PATH)/src -L$(TAPP_PATH) -ltapp
# 	$(CC) -shared -fPIC $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o -o lib/libhi_tapp.so -I$(INC) -I$(INC)/hi_tapp -I$(TBL) $(TBLIS_PARAM)
else
lib/libhi_tapp.so: $(OBJ)/hi_tapp.o $(OBJ)/tblis_bind.o
	$(CXX) -shared -fPIC $(OBJ)/hi_tapp.o -o lib/libhi_tapp.so -I$(INC) -I$(INC)/hi_tapp -I$(TBL) $(TBLIS_PARAM) -I$(TAPP_PATH)/src -L$(TAPP_PATH)/lib -ltapp
endif

ifeq ($(HAS_TBLIS),true)
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
	rm -f obj/hi_tapp.o
	rm -f obj/tblis_bind.o
	rm -f out/test
	rm -f out/test.o
	rm -f out/test++
	rm -f out/demo
	rm -f out/demo.o
	rm -f out/uselib
	rm -f out/uselib.o
	rm -f lib/libhi_tapp.so
