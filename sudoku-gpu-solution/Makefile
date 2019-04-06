CC := nvcc
CFLAGS := -g

all: sudoku
	
clean:
	rm -f sudoku

sudoku: sudoku.cu util.h
	$(CC) $(CFLAGS) -o sudoku sudoku.cu
