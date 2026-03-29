// #include <__clang_cuda_builtin_vars.h>
#include <stdio.h>

__global__ void dotKernel(float *a, float *b, float *o, int N) {
  // Map data to tID
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // Register
  float pValue = 0.0f;
  // Implement
  if (i < N) {
    pValue = a[i] * b[i];
  }
  __syncthreads();

  // HBM
  if (i < N) {
    atomicAdd(o, pValue);
  }
}

int main() {
  int N = 4096;

  float *a_h;
  float *b_h;

  float *o_h;

  a_h = (float *) malloc(N*sizeof(float));
  b_h = (float *) malloc(N*sizeof(float));
  o_h = (float *) malloc(sizeof(float));

  for (int i = 0; i < N; i++) {
    a_h[i] = 1.0f;
    b_h[i] = 1.0f;
  }

  *o_h = 0.0f;

  float *a, *b, *o;
  cudaMalloc(&a, N*sizeof(float));
  cudaMalloc(&b, N*sizeof(float));
  cudaMalloc(&o, sizeof(float));

  cudaMemcpy(a, a_h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b, b_h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(o, o_h, sizeof(float), cudaMemcpyHostToDevice);

  int threadsPerBlock = 1024;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  printf("blocks: %d, threads: %d\n", blocksPerGrid, threadsPerBlock);

  dotKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, o, N);

  cudaMemcpy(o_h, o, sizeof(float), cudaMemcpyDeviceToHost);
  printf("\nOutput: %f\n", *o_h);

  free(o_h);
  free(b_h);
  free(a_h);
  cudaFree(o);
  cudaFree(a);
  cudaFree(b);
}
