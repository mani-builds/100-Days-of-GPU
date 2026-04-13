#include <stdio.h>

#define BLOCK 32

__global__ void red(float *a, float *o, int M, int N) {

  // Map data to threads
  int col = 2*blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.x; // each block processes one row

  int row_start = row * N;
  // smem
  __shared__ float as[BLOCK];
   as[threadIdx.x] = a[row_start + col] + a[row_start + col + BLOCK];
  __syncthreads();

  // logic implement
  for (int stride = blockDim.x / 2; stride >= 1; stride  /= 2) {
    if(threadIdx.x < stride){
      as[threadIdx.x] = as[threadIdx.x] + as[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // HBM
  if (threadIdx.x == 0 && row < M)
    o[row] = as[0];
    // atomicAdd(o, as[0]);

}
