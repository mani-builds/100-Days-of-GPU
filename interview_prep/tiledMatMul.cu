#include <stdio.h>

#define SIZE 4096
#define TILE_WIDTH 32

__global__ void matMul(float *M, float *N, float *P, int width) {

  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // smem
 __shared__ float  M_s[TILE_WIDTH][TILE_WIDTH];
 __shared__ float N_s[TILE_WIDTH][TILE_WIDTH];

  float Pvalue = 0.0f;
  for (int ph = 0; ph < width / TILE_WIDTH; ph++) {
    M_s[ty][tx] = M[row * width + (ph * TILE_WIDTH + tx)];
    N_s[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      Pvalue += M_s[ty][k] * N_s[k][tx];
    }
    __syncthreads();
  }

   P[row*width + col] = Pvalue;

}
