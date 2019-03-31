#include <stdint.h>
#include <stdio.h>

#define N 48
#define THREADS_PER_BLOCK 32 

__global__ void saxpy(float a, float* x, float* y) {
  // Which index of the array should this thread use?
  size_t index = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
  if(index < blockDim.x * THREADS_PER_BLOCK) {
    y[index] = a * x[index] + y[index];
  }
}

int main() {
  // Allocate arrays for X and Y on the CPU
  float* cpu_x = (float*)malloc(sizeof(float) * N);
  float* cpu_y = (float*)malloc(sizeof(float) * N);

  // Initialize X and Y
  int i;
  for(i=0; i<N; i++) {
    cpu_x[i] = (float)i;
    cpu_y[i] = 0.0;
  }

  // Allocate space for X and Y on the GPU
  float* gpu_x;
  float* gpu_y;

  if(cudaMalloc(&gpu_x, sizeof(float) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate X array on GPU\n");
    exit(2);
  }

  if(cudaMalloc(&gpu_y, sizeof(float) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate Y array on GPU\n");
    exit(2);
  }

  // Copy the host X and Y arrays to the device X and Y arrays
  if(cudaMemcpy(gpu_x, cpu_x, sizeof(float) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy X to the GPU\n");
  }

  if(cudaMemcpy(gpu_y, cpu_y, sizeof(float) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy Y to the GPU\n");
  }

  // How many blocks should be run, rounding up to include all threads?
  size_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // Run the saxpy kernel
  saxpy<<<blocks, THREADS_PER_BLOCK>>>(0.5, gpu_x, gpu_y);

  // Wait for the kernel to finish
  cudaDeviceSynchronize();

  // Copy values from the GPU back to the CPU
  if(cudaMemcpy(cpu_y, gpu_y, sizeof(float) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy Y from the GPU\n");
  }

  for(i=0; i<N; i++) {
    printf("%d: %f\n", i, cpu_y[i]);
  }

  cudaFree(gpu_x);
  cudaFree(gpu_y);
  free(cpu_x);
  free(cpu_y);

  return 0;
}
