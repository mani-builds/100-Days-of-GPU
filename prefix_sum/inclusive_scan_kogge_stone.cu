#include <stdio.h>

#define SECTION_SIZE 32
__global__ void Kogge_Stone_scan_kernel(float *x, float *y, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // Load a segment into shared memory
  __shared__ float XY[SECTION_SIZE];
  if (i < N) {
    XY[threadIdx.x] = x[i];
  } else {
    XY[threadIdx.x] = 0.0f;
  }

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads(); // This barrier synchronization is used to sync the complete load of XY from HBM
    float temp;
    if (threadIdx.x >= stride) {
      temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
    }
    __syncthreads(); // This barrier synchronization and 'temp' vairable is used to prevent
                     // write-after-read race condition that occurs when we try to write
                     // at the same location as we read from.
    if (threadIdx.x >= stride) 
      XY[threadIdx.x] = temp;
  }
  if (i < N){
    y[i] = XY[threadIdx.x];
  }
}

int main() {

  float *x_h;
  float *y_h;
  int N;

  N = 32;
  x_h = (float *) malloc(N*sizeof(float));
  y_h = (float *) malloc(N*sizeof(float));

  for (int i=0;i <N; i++) x_h[i] = 1.0f;

  float *x;
  float *y;

  cudaMalloc(&x, N*sizeof(float));
  cudaMalloc(&y, N*sizeof(float));

  cudaMemcpy(x, x_h, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(y, y_h, N*sizeof(float), cudaMemcpyHostToDevice);

  int threads = SECTION_SIZE;
  int blocksPerGrid = (N + threads - 1) / threads;
  Kogge_Stone_scan_kernel<<<blocksPerGrid, threads>>>(x, y, N);

  cudaMemcpy(y_h, y, N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0;i <N; i++) printf("%f \t ",y_h[i]);
  printf("\n");
  return 0;
}
