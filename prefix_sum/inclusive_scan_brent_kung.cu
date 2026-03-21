#include <stdio.h>

#define SECTION_SIZE 2048

// SECTION_SIZE can be doubled that of thread capacity per block, since each thread
// process two data elements

__global__ void Brent_Kung_scan_kernel(float *x, float *y, int N) {

  int i = 2*blockDim.x * blockIdx.x + threadIdx.x; // Each thread processes two elements

  // Load two segments into shared memory
  __shared__ float XY[SECTION_SIZE];
  if (i < N) XY[threadIdx.x] = x[i];
  if (i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = x[i+blockDim.x];

  // Reduction tree phase
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads(); // This barrier synchronization is used to sync the complete load of XY from HBM
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < SECTION_SIZE) {
      XY[index] = XY[index] + XY[index - stride];
    }
    // There is no need for 2nd barrier synchronization and 'temp' variable like in Kogge-Stone
    // because there is no dependecy i.e In Kogge-Stone many threads read from/write to
    // overlapping indices, that is not the case here.
   }

  // Reverse tree phase
  for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
     __syncthreads(); // This barrier synchronization is used to sync the complete load of XY from HBM
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index + stride < SECTION_SIZE) {
      XY[index + stride] = XY[index + stride] + XY[index];
    }
  }
  __syncthreads();
  if (i < N) y[i] = XY[threadIdx.x];
  if (i + blockDim.x < N) y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

int main() {

  float *x_h;
  float *y_h;
  int N;

  N = 2048; // Double the block capacity
  x_h = (float *) malloc(N*sizeof(float));
  y_h = (float *) malloc(N*sizeof(float));

  for (int i=0;i <N; i++) x_h[i] = 1.0f;

  float *x;
  float *y;

  cudaMalloc(&x, N*sizeof(float));
  cudaMalloc(&y, N*sizeof(float));

  cudaMemcpy(x, x_h, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(y, y_h, N*sizeof(float), cudaMemcpyHostToDevice);

  int threads = SECTION_SIZE/2;
  int blocksPerGrid = (N/2 + threads - 1) / threads;
  Brent_Kung_scan_kernel<<<blocksPerGrid, threads>>>(x, y, N);

  cudaMemcpy(y_h, y, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("\n 1st 10 :\n");
  for (int i=0;i <10; i++) printf("%f \t ",y_h[i]);
  printf("\n Last 10 :\n");
  for (int i=N-10;i <N; i++) printf("%f \t ",y_h[i]);
  printf("\n");

  return 0;
}
