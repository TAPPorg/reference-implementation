CC = gcc
CXX = g++
INC = include
SRC = src
OBJ = obj
TEST = test
OUT = out
INC = include
TBLIS = -ltblis -lm -L../tblis/lib/.libs -I../tblis/src/external/tci -I../tblis/include -I../tblis/src

all: obj/product.o out/test out/test++ lib/product.so

obj/product.o: src/product.c include/product.h
	$(CC) -c $(SRC)/product.c -o $(OBJ)/product.o -I$(INC)

out/test: $(TEST)/test.c $(OBJ)/product.o
	$(CC) -g  $(TEST)/test.c $(OBJ)/product.o -o $(OUT)/test -I$(INC)

out/test++: $(TEST)/test.cpp $(OBJ)/product.o
	$(CXX) -g  $(TEST)/test.cpp $(OBJ)/product.o -o $(OUT)/test++ -I$(INC) $(TBLIS)

lib/product.so: $(SRC)/product.c $(INC)/product.h
	$(CC) -shared -fPIC $(SRC)/product.c -o lib/product.so -I$(INC)

clean:
	rm -f out/test
	rm -f out/test++
	rm -f lib/product.so