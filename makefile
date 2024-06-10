CC = gcc
INC = ./include
SRC = ./src/*.c

all: out/test lib/product.so

out/test: test/*.c src/*.c include/*.h
	$(CC) -g test/test.c $(SRC) -o out/test -I$(INC)

lib/product.so: src/*.c include/*.h
	$(CC) -shared -fPIC $(SRC) -o lib/product.so -I$(INC)

clean:
	rm -f out/test
	rm -f lib/product.so