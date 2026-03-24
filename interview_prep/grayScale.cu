#include <stdio.h>

#define N 4096
#define CHANNELS 3

__global__ void grayScalConvertion(float *A, float *B, int height, int width) {


  int row = blockDim.y + blockIdx.y + threadIdx.y;
  int col = blockDim.x + blockIdx.x + threadIdx.x;
  // __shared__ float a_s[1024];
  // __shared__ float b_s[1024];

  int grayOffset = row * width + col;
  int rgbOffset = grayOffset * CHANNELS;
  if (row < height && col < width) {
    int r = A[rgbOffset];
    int g = A[rgbOffset + 1];
    int b = A[rgbOffset + 2];

    B[grayOffset] = 0.21*r + 0.71 * g + 0.07 * b;
  }

}
