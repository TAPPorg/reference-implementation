CC = gcc
CXX = g++
SRC = src
OBJ = obj
TEST = test
OUT = out
INC = src
TBLIS = -ltblis -lm -L../tblis/lib/.libs -I../tblis/src/external/tci -I../tblis/include -I../tblis/src

all: obj/tapp.o obj/error.o obj/tensor.o obj/product.o out/test++ lib/tapp.so out/demo out/test

obj/tapp.o: obj/product.o obj/tensor.o obj/error.o
	ld -relocatable obj/tensor.o obj/product.o obj/error.o -o obj/tapp.o

obj/error.o: $(SRC)/tapp/error.c $(INC)/tapp/error.h
	$(CC) -c -g -Wall $(SRC)/tapp/error.c -o $(OBJ)/error.o -I$(INC) -I$(INC)/tapp

obj/tensor.o: $(SRC)/tapp/tensor.c $(INC)/tapp/tensor.h
	$(CC) -c -g -Wall $(SRC)/tapp/tensor.c -o $(OBJ)/tensor.o -I$(INC) -I$(INC)/tapp

obj/product.o: $(SRC)/tapp/product.c $(INC)/tapp/product.h
	$(CC) -c -g -Wall $(SRC)/tapp/product.c -o $(OBJ)/product.o -I$(INC) -I$(INC)/tapp

out/test: $(TEST)/test.c $(OBJ)/product.o
	$(CC) -g  $(TEST)/test.c $(OBJ)/tapp.o -o $(OUT)/test -I$(INC) -I$(INC) -I$(INC)/tapp

out/demo:$(TEST)/demo.c $(OBJ)/tapp.o
	$(CC) -g  $(TEST)/demo.c $(OBJ)/tapp.o -o $(OUT)/demo -I$(INC) -I$(INC)/tapp $(TBLIS)

out/test++: $(TEST)/test.cpp $(OBJ)/tapp.o
	$(CXX) -g  $(TEST)/test.cpp $(OBJ)/tapp.o -o $(OUT)/test++ -I$(INC) -I$(INC)/tapp $(TBLIS)

lib/tapp.so: $(OBJ)/tapp.o
	$(CC) -shared -fPIC $(OBJ)/tapp.o -o lib/tapp.so -I$(INC) -I$(INC)/tapp

clean:
	rm -f obj/tensor.o
	rm -f obj/product.o
	rm -f obj/error.o
	rm -f obj/tapp.o
	rm -f out/test
	rm -f out/test++
	rm -f lib/tapp.so