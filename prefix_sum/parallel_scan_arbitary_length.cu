#include <stdio.h>

#define SECTION_SIZE 4

__global__ void scan_block_kernel(float *x, float *y, float *S, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // Load a segment into shared memory
  __shared__ float XY[SECTION_SIZE];
  if (i < N) {
    XY[threadIdx.x] = x[i];
  } else {
    XY[threadIdx.x] = 0.0f;
  }

  // Kogge-Stone algo for parallel scan
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads(); // This barrier synchronization is used to sync the complete load of XY from HBM
    float temp;
    if (threadIdx.x >= stride) {
      temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
    }
    __syncthreads(); //to prevent write-after-read race condition that occurs
                      //when we try to write at the same location as we read from.
    if (threadIdx.x >= stride)
      XY[threadIdx.x] = temp;
  }
  if (i < N){
    y[i] = XY[threadIdx.x];
  }
  if (threadIdx.x == blockDim.x - 1) {
    S[blockIdx.x] = XY[threadIdx.x];
  }
}

__global__ void scan_block_sum_kernel(float *S,  int n) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // Apply Kogge-Stone to parallel scan the Block Sum array on a single thread block
  __shared__ float XY[SECTION_SIZE];

  if (i < n) {
    XY[threadIdx.x] = S[i];
  } else {
    XY[threadIdx.x] = 0.0f;
  }

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    float temp;
    if (threadIdx.x >= stride) {
        temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
    }
      __syncthreads();
      if (threadIdx.x >= stride) {
        XY[threadIdx.x] = temp;
      }
  }
  // Write to global memory
    if (i < n) {
       S[i] = XY[threadIdx.x];
    }
}

__global__ void final_array_kernel(float *y, float *S, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (blockIdx.x > 0 && i < N)
  y[i] = y[i] + S[blockIdx.x - 1];
}

int main() {

  // float *x_h;
  float *y_h;
  int N;

  N = 16;
  // x_h = (float *) malloc(N*sizeof(float));
  y_h = (float *) malloc(N*sizeof(float));

  float x_h[] = {2,1,3,1,0,4,1,2,0,3,1,2,5,3,1,2};
  // for (int i=0;i <N; i++) x_h[i] = 1.0f;
  printf("\nInput array:\n");
  for (int i=0;i <N; i++) printf("%f \t ",x_h[i]);
  printf("\n");

  float *x;
  float *y;

  cudaMalloc(&x, N*sizeof(float));
  cudaMalloc(&y, N*sizeof(float));

  cudaMemcpy(x, x_h, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(y, y_h, N*sizeof(float), cudaMemcpyHostToDevice);

  // Auxilliary array to store Block Sum
  float *S_h;
  float *S;

  int threads = SECTION_SIZE;
  int blocksPerGrid = (N + threads - 1) / threads;
  printf("\nthreads: %d, blocksPerGrid: %d\n", threads, blocksPerGrid);
  int S_size = N/SECTION_SIZE;
  S_h = (float *) malloc(S_size * sizeof(float));

  cudaMalloc(&S, S_size * sizeof(float));

  scan_block_kernel<<<blocksPerGrid, threads>>>(x, y,S,N);

  cudaMemcpy(S_h, S, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
  printf("\nBlock Sum array:\n");
  for (int i=0;i <S_size; i++) printf("%f \t ",S_h[i]);
  printf("\n");

  int threads2 = S_size;
  int blocks2 = 1;

  float *S2;
  cudaMalloc(&S2, S_size * sizeof(float));

  scan_block_sum_kernel<<<blocks2, threads2>>>(S, S_size);

  cudaMemcpy(S_h, S, S_size*sizeof(float), cudaMemcpyDeviceToHost);
  printf("\nScaned Block Sum array:\n");
  for (int i=0;i <S_size; i++) printf("%f \t ",S_h[i]);
  printf("\n");

  int threads3 = SECTION_SIZE;
  int blocks3 = (N + threads3 - 1) / threads3;
  printf("\nthreads3: %d, blocks3: %d\n", threads3, blocks3);
  final_array_kernel<<<blocks3, threads3>>>(y, S, N);

  printf("\nOutput array:\n");
  cudaMemcpy(y_h, y, N*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i=0;i <N; i++) printf("%f \t ",y_h[i]);
  printf("\n");

  // free(x_h);
  free(y_h);
  free(S_h);
  cudaFree(x);
  cudaFree(y);
  cudaFree(S);
  cudaFree(S2);
  return 0;
}
