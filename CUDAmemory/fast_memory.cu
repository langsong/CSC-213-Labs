#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32 
#define N 10240

__global__ void doubleValues(int*difference, int* numbers, int length) {
  __shared__ int local_values[N];
  int index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  local_values[index] = numbers[index];
  __syncthreads();
  if(index != length - 1) {
    difference[index] = local_values[index + 1] - local_values[index]; 
  }
}


int main() {
  int* cpu_arr = (int*)malloc(N * sizeof(int));
  if(!cpu_arr) {
    perror("malloc");
    exit(1);
  }  

  for(int i = 0; i < N; i++) {
    cpu_arr[i] = i * i;
  }

  int* gpu_arr;

  if(cudaMalloc(&gpu_arr, sizeof(int) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate array on GPU\n");
    exit(2);
  }

  if(cudaMemcpy(gpu_arr, cpu_arr, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy array to the GPU\n");
  }

  int* gpu_difference;
  if(cudaMalloc(&gpu_difference, sizeof(int) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate array on GPU\n");
    exit(2);
  }

  doubleValues<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(gpu_difference, gpu_arr, N);
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
