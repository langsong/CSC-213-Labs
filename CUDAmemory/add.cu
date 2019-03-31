#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32 
#define N 10240

__global__ void sumValues(int* sum, int* numbers, int length) {
  int index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  atomicAdd()
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

  int *sum;
  if(cudaMalloc(&sum, sizeof(int)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate array on GPU\n");
    exit(2);
  }

  sumValues<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(sum, gpu_arr, N);
  cudaDeviceSynchronize();

  if(cudaMemcpy(cpu_arr, gpu_difference, sizeof(int) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy array to the CPU\n");
  }
  
  for(int i = 0; i < N; i++) {
    printf("%d\n", cpu_arr[i]);
  }

  free(cpu_arr);
  cudaFree(gpu_arr);
  cudaFree(gpu_difference);

  return 0;

}
