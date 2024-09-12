CC = gcc
CXX = g++
SRC = src
OBJ = obj
TEST = test
OUT = out
INC = src
TBLIS = -ltblis -lm -L../tblis/lib/.libs -I../tblis/src/external/tci -I../tblis/include -I../tblis/src

all: obj/tensor.o obj/product.o out/test++ lib/product.so lib/tensor.so out/demo #out/test

obj/tensor.o: $(SRC)/tapp/tensor.c $(INC)/tapp/tensor.h
	$(CC) -c -g $(SRC)/tapp/tensor.c -o $(OBJ)/tensor.o -I$(INC) -I$(INC)/tapp

obj/product.o: $(SRC)/tapp/product.c $(INC)/tapp/product.h
	$(CC) -c -g $(SRC)/tapp/product.c -o $(OBJ)/product.o -I$(INC) -I$(INC)/tapp

#out/test: $(TEST)/test.c $(OBJ)/product.o
#	$(CC) -g  $(TEST)/test.c $(OBJ)/product.o $(OBJ)/tensor.o -o $(OUT)/test -I$(INC)

out/demo:$(TEST)/demo.c $(OBJ)/product.o
	$(CC) -g  $(TEST)/demo.c $(OBJ)/tensor.o $(OBJ)/product.o -o $(OUT)/demo -I$(INC) -I$(INC)/tapp $(TBLIS)

out/test++: $(TEST)/test.cpp $(OBJ)/product.o
	$(CXX) -g  $(TEST)/test.cpp $(OBJ)/tensor.o $(OBJ)/product.o -o $(OUT)/test++ -I$(INC) -I$(INC)/tapp $(TBLIS)

lib/product.so: $(SRC)/tapp/product.c $(INC)/tapp/product.h
	$(CC) -shared -fPIC $(SRC)/tapp/product.c $(OBJ)/tensor.o -o lib/product.so -I$(INC) -I$(INC)/tapp

lib/tensor.so: $(SRC)/tapp/tensor.c $(INC)/tapp/tensor.h
	$(CC) -shared -fPIC $(SRC)/tapp/tensor.c -o lib/tensor.so -I$(INC) -I$(INC)/tapp

clean:
	rm -f obj/tensor.o
	rm -f obj/product.o
	rm -f out/test
	rm -f out/test++
	rm -f lib/product.so
	rm -f lib/tensor.so