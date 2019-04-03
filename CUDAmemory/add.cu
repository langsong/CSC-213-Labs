#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32
#define N 321

__global__ void sumValues(int *arr, int *sum) {
  int index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  __shared__ float temp[BLOCK_SIZE];
  if (index < N) {
    temp[threadIdx.x] = arr[index] * arr[index];
    __syncthreads();
    // The thread with index zero will sum up the values in temp
    if (threadIdx.x == 0) {
      int s = 0;
      for (int i = 0; i < BLOCK_SIZE; i++) {
        s += temp[i];
      }

      // Add the sum for this block to the
      atomicAdd(sum, s);
    }
  }
}

int main() {
  int *arr;
  int *sum;

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&arr, N * sizeof(int));
  cudaMallocManaged(&sum, sizeof(int));

  for (int i = 0; i < N; i++) {
    arr[i] = i;
  }

  int block_number =
      N / BLOCK_SIZE * BLOCK_SIZE == N ? N / BLOCK_SIZE : N / BLOCK_SIZE + 1;
  sumValues<<<block_number, BLOCK_SIZE>>>(arr, sum);
  cudaDeviceSynchronize();

  printf("sum = %d\n", *sum);

  return 0;
}
