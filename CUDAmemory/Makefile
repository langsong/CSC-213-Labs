all: memory shared_memory fast_memory

memory: memory.cu 
	nvcc -o memory memory.cu

shared_memory: shared_memory.cu
	nvcc -o shared_memory shared_memory.cu
	
fast_memory: fast_memory.cu
	nvcc -o fast_memory fast_memory.cu

add: add.cu
	nvcc -o add add.cu
	
