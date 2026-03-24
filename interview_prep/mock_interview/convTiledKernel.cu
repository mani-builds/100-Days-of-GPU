
// Filter radius r; 2*r + 1 is the size
// #include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_builtin_vars.h>
#define RADIUS 1
__constant__ float f[2*RADIUS+1][2*RADIUS+1];

#define IN_TILE 32
#define OU_TILE ((IN_TILE) - (2*RADIUS))

__global__ void convKernel(float *in, float *ou,  int height, int width) {

  // threads to data mapping
  // 1. Map threads to the INPUT coordinates (including the halo)
    int col = blockIdx.x * OU_TILE + threadIdx.x - RADIUS;
    int row = blockIdx.y * OU_TILE + threadIdx.y - RADIUS;

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  // smem
  __shared__ float in_s[IN_TILE][IN_TILE];

  // 2. Load into Shared Memory (with bounds checking)
    if (row >= 0 && row < height && col >= 0 && col < width) {
        in_s[threadIdx.y][threadIdx.x] = in[row * width + col];
    } else {
        in_s[threadIdx.y][threadIdx.x] = 0.0f; // Padding for out-of-bounds
    }
    __syncthreads();

  // Logic implementation
  // Disable the threads at the end of the OU_TILE
  // Only threads that aren't part of the "halo" edge perform the math
    if (threadIdx.y >= RADIUS && threadIdx.y < IN_TILE - RADIUS &&
        threadIdx.x >= RADIUS && threadIdx.x < IN_TILE - RADIUS) {
  float value = 0.0f;
  for (int rowId = -RADIUS; rowId <= RADIUS; rowId++) {
    for (int colId = -RADIUS; colId <= RADIUS; colId++) {
        value += f[rowId + RADIUS][colId + RADIUS] *
               in_s[(threadIdx.y + rowId)][(threadIdx.x + colId)];
    }
    }


  // HBM
  if (row >= 0 && row < height && col >= 0 && col < width) {
    ou[row * width + col] = value;
  }

 }
}
