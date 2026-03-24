
// Filter radius r; 2*r + 1 is the size
#include <__clang_cuda_builtin_vars.h>
#define RADIUS 1
__constant__ float f[2*RADIUS+1][2*RADIUS+1];

__global__ void convKernel(float *in, float *ou,  int height, int width) {

  // threads to data mapping
  int outCol = blockDim.x * blockIdx.x + threadIdx.x;
  int outRow = blockDim.y * blockIdx.y + threadIdx.y;

  // smem

  // Logic implementation

  float value = 0.0f;
  for (int rowId = -RADIUS; rowId <= RADIUS; rowId++) {
    for (int colId = -RADIUS; colId <= RADIUS; colId++) {
      if ((outRow + rowId) >= 0 && (outRow + rowId) < height &&
          (outCol + colId) >= 0
          && (outCol + colId) < width)
      value += f[rowId + RADIUS][colId + RADIUS] *
               in[(outRow + rowId) * width + (outCol + colId)];

    }
  }

  // HBM
  if (outRow < height && outCol < width)
  ou[outRow*width + outCol] = value;

}
