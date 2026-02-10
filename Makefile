CC = gcc
FLAGS = -O2 -Wextra

conv-omp: dir-structure src/conv_1d.c
	$(CC) -c -o build/conv_1d.o src/conv_1d.c $(FLAGS) -fopenmp

conv-single: dir-structure src/conv_1d.c
	$(CC) -c -o build/conv_1d_single.o src/conv_1d.c $(FLAGS)

dir-structure:
	mkdir -p build