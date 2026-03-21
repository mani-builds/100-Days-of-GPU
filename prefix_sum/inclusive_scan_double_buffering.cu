#include <stdio.h>

#define SECTION_SIZE 32

// In order using 2nd barrier synchronization might put too much contraints on resources.
// We can use 'double-buffering' optimization to prevent this by using two arrays in shared mem
// instead of one.

__global__ void Kogge_Stone_scan_kernel(float *x, float *y, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // Have an two arrays in shared memory
  __shared__ float XY[SECTION_SIZE];
  __shared__ float XY2[SECTION_SIZE];

  // Load the first array and 2nd with input data
  if (i < N) {
    XY[threadIdx.x] = x[i];
    XY2[threadIdx.x] = x[i];
  } else {
    XY[threadIdx.x] = 0.0f;
    XY2[threadIdx.x] = 0.0f;
  }

  int iteration_counter = 0;
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    iteration_counter += 1;
    __syncthreads(); // This barrier synchronization is used to sync the complete load of XY from HBM
    if (threadIdx.x >= stride) {
      // The trick of 'double-buffing' is to alternate between both buffers
      if (iteration_counter % 2 != 0)
      XY2[threadIdx.x] = XY[threadIdx.x] + XY[threadIdx.x - stride];
      else
       XY[threadIdx.x] = XY2[threadIdx.x] + XY2[threadIdx.x - stride];
    }
  }

  if (i < N){
    if (iteration_counter % 2 != 0)
    y[i] = XY2[threadIdx.x];
    else
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
