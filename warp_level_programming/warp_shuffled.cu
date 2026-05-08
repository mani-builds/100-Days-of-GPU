#include <stdio.h>

#define FULL_MASK 0xFFFFFFFF

__global__ void warp_shuffle(int *array, int *output, int n) {
  // Map data to warp
  int laneId = threadIdx.x % 32;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int result1=0;
  int result2=0;
  int result3=0;

  // smem or load data
  int data;
  if(i < n)
  data = array[i];

  // logic
  // result1 = __shfl_sync(FULL_MASK, data,0);
  // if (laneId < 4) {
  //   result2 = __shfl_sync(FULL_MASK, data, 1);
  // }
  result3 = __shfl_sync( FULL_MASK, data, 0, warpSize/2);

  // HBM
  if(i < n)
  output[i] = result3;
}

int main() {
  int n = 64;
  int *a, *output;

  a = (int*) malloc(sizeof(int)*n);
  output = (int *) malloc(sizeof(int)*n);
  for (int i = 0; i < n; i++) {
    a[i] = n-i;
    output[i] = -1;
  }
  printf("\nInput 1st value: a[0] : %d", a[0]);

  int *a_d, *out_d;
  cudaMalloc(&a_d, sizeof(int)*n);
  cudaMalloc(&out_d, sizeof(int)*n);

  cudaMemcpy(a_d, a, sizeof(int)*n, cudaMemcpyHostToDevice);
  cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
   }
   cudaDeviceSynchronize();
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     printf("CUDA sync error: %s\n", cudaGetErrorString(err));
   }

  int threads = 32;
  int blocks = (n + threads - 1) / threads;
  printf("\nthreads: %d, blocks: %d\n", threads, blocks);
  warp_shuffle<<<blocks, threads>>>(a_d, out_d, n);

  cudaMemcpy(output, out_d, sizeof(int)*n, cudaMemcpyDeviceToHost);

  printf("Output (first 32):\n");
  for (int i = 0; i < 32; i++) {
    printf("%d ", output[i]);
  }
  cudaFree(out_d);
  cudaFree(a_d);
  free(a);
  free(output);
  return 0;
}
