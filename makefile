CC = gcc
CXX = g++
SRC = src
OBJ = obj
TEST = test
OUT = out
INC = src
TBLIS = ../tblis
TBLIS_PARAM = -ltblis -lm -L$(TBLIS)/lib/.libs -I$(TBLIS)/src/external/tci -I$(TBLIS)/include -I$(TBLIS)/src
OBJECTS = $(filter-out obj/tapp.o, $(wildcard obj/*.o))
CFLAGS = -fPIC

all: folders obj/tapp.o obj/error.o obj/tensor.o obj/product.o obj/executor.o obj/handle.o out/test++ lib/tapp.so out/demo out/test

folders:
	mkdir -p obj lib out bin

obj/tapp.o: obj/product.o obj/tensor.o obj/error.o obj/executor.o obj/handle.o
	ld -relocatable $(OBJECTS) -o obj/tapp.o

obj/error.o: $(SRC)/tapp/error.c $(INC)/tapp/error.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/error.c -o $(OBJ)/error.o -I$(INC) -I$(INC)/tapp

obj/tensor.o: $(SRC)/tapp/tensor.c $(INC)/tapp/tensor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/tensor.c -o $(OBJ)/tensor.o -I$(INC) -I$(INC)/tapp

obj/product.o: $(SRC)/tapp/product.c $(INC)/tapp/product.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/product.c -o $(OBJ)/product.o -I$(INC) -I$(INC)/tapp

obj/executor.o: $(SRC)/tapp/executor.c $(INC)/tapp/executor.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/executor.c -o $(OBJ)/executor.o -I$(INC) -I$(INC)/tapp

obj/handle.o: $(SRC)/tapp/handle.c $(INC)/tapp/handle.h
	$(CC) $(CFLAGS) -c -g -Wall $(SRC)/tapp/handle.c -o $(OBJ)/handle.o -I$(INC) -I$(INC)/tapp

out/test: $(TEST)/test.c $(OBJ)/product.o
	$(CC) $(CFLAGS) -g  $(TEST)/test.c $(OBJ)/tapp.o -o $(OUT)/test -I$(INC) -I$(INC) -I$(INC)/tapp

out/demo:$(TEST)/demo.c $(OBJ)/tapp.o
	$(CC) $(CFLAGS) -g  $(TEST)/demo.c $(OBJ)/tapp.o -o $(OUT)/demo -I$(INC) -I$(INC)/tapp $(TBLIS_PARAM)

out/test++: $(TEST)/test.cpp $(OBJ)/tapp.o
	$(CXX) -g  $(TEST)/test.cpp $(OBJ)/tapp.o -o $(OUT)/test++ -Itest -I$(INC) -I$(INC)/tapp $(TBLIS_PARAM)

lib/tapp.so: $(OBJ)/tapp.o
	$(CC) -shared -fPIC $(OBJ)/tapp.o -o lib/tapp.so -I$(INC) -I$(INC)/tapp

clean:
	rm -f obj/tensor.o
	rm -f obj/product.o
	rm -f obj/error.o
	rm -f obj/executor.o
	rm -f obj/handle.o
	rm -f obj/tapp.o
	rm -f out/test
	rm -f out/test++
	rm -f lib/tapp.so