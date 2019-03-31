#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 64
#define N 1024

__global__ void doubleValues(int* numbers, int length) {
  numbers[BLOCK_SIZE*blockIdx.x + threadIdx.x] *= 2;  
}


int main() {
  int* cpu_arr = (int*)malloc(N * sizeof(int));
  if(!cpu_arr) {
    perror("malloc");
    exit(1);
  }  

  for(int i = 0; i < N; i++) {
    cpu_arr[i] = i;
  }

  int* gpu_arr;

  if(cudaMalloc(&gpu_arr, sizeof(int) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate array on GPU\n");
    exit(2);
  }

  if(cudaMemcpy(gpu_arr, cpu_arr, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy array to the GPU\n");
  }

  doubleValues<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(gpu_arr, N);
  doubleValues<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(gpu_arr, N);
  doubleValues<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(gpu_arr, N);
  cudaDeviceSynchronize();

  if(cudaMemcpy(cpu_arr, gpu_arr, sizeof(int) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy array to the CPU\n");
  }
  
  for(int i = 0; i < N; i++) {
    printf("%d\n", cpu_arr[i]);
  }

  return 0;

}
