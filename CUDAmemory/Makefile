all: memory shared_memory fast_memory

memory: memory.cu 
	nvcc -ccbin clang-3.8 -arch sm_20 -o memory memory.cu

shared_memory: shared_memory.cu
	nvcc -ccbin clang-3.8 -arch sm_20 -o shared_memory shared_memory.cu
	
fast_memory: fast_memory.cu
	nvcc -ccbin clang-3.8 -arch sm_20 -o fast_memory fast_memory.cu
	
