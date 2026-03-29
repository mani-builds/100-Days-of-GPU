#include <stdio.h>

__global__ void dotKernelShared(float *a, float *b, float *partial, int N) {
  // Map data to tID

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // smem
 extern __shared__ float cache[]; //size of this is blockDim

  float temp = 0.0f;
  // Grid level stride loop
  for (int idx = i; idx < N; idx += gridDim.x * blockDim.x) {
    temp += a[i] * b[i];
  }
  cache[threadIdx.x] = temp;
  __syncthreads();

  // Reduce in shared mem
  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride)
    cache[threadIdx.x] = cache[threadIdx.x] + cache[threadIdx.x + stride];
  __syncthreads();
  }

  // HBM
  if (threadIdx.x == 0)
    partial[blockIdx.x] = cache[0];

}

int main() {
  int N = 4096;

  float *a_h;
  float *b_h;

  float *p_h;

  a_h = (float *) malloc(N*sizeof(float));
  b_h = (float *) malloc(N*sizeof(float));

  for (int i = 0; i < N; i++) {
    a_h[i] = 1.0f;
    b_h[i] = 1.0f;
  }

  float *a, *b, *p;
  cudaMalloc(&a, N*sizeof(float));
  cudaMalloc(&b, N*sizeof(float));

  cudaMemcpy(a, a_h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b, b_h, N*sizeof(float), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid =  4; //(N + threadsPerBlock - 1) / threadsPerBlock;

  p_h = (float *) malloc(blocksPerGrid * sizeof(float));
  for (int i=0; i<blocksPerGrid; i++) p_h[i] = 0.0f;
  cudaMalloc(&p, blocksPerGrid * sizeof(float));
  cudaMemcpy(p, p_h, blocksPerGrid *sizeof(float), cudaMemcpyHostToDevice);

  size_t size = threadsPerBlock * sizeof(float);

  printf("blocks: %d, threads: %d\n", blocksPerGrid, threadsPerBlock);

  dotKernelShared<<<blocksPerGrid, threadsPerBlock, size>>>(a, b, p, N);

  cudaMemcpy(p_h, p, blocksPerGrid *sizeof(float), cudaMemcpyDeviceToHost);
  float dot = 0.0f;
  for (int i=0; i<blocksPerGrid; i++) dot += p_h[i];
  printf("\nDot: %f\n", dot);

  free(p_h);
  free(b_h);
  free(a_h);
  cudaFree(p);
  cudaFree(a);
  cudaFree(b);
}
