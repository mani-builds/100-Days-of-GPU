#include <stdio.h>

#define SIZE 4096
#define TILE 32

__global__ void tiledMatMul(float *a, float *b, int M, int K, int N, float *o) {

  // Map data to threads
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  // smem
  __shared__ float as[TILE][TILE+1]; // Avoid bank conflicts
  __shared__ float bs[TILE][TILE+1];

  // logic
  if (row < M && col < N){
  float pvalue = 0.0f;
  for (int ph = 0; ph <= K / TILE; ph++) {
   // Collaborative load elements into shared mem
   if (row < M && (ph *TILE + tx) < K && col < N && (ph * TILE + ty) < K){
   as[ty][tx] = a[row * K + (ph*TILE + tx)];
   bs[ty][tx] = b[(ph * TILE + ty) * N + col];
   } else {
    as[ty][tx] = 0.0f;
    bs[ty][tx] = 0.0f;
   }
   __syncthreads();

   for (int k = 0; k < TILE; k++) {
     pvalue += as[ty][k] * bs[k][tx];
   }
    __syncthreads();
  }
  // HBM
  o[row * N + col] = pvalue;
  }
}


void tiledMatMul_launcher() {
  int M = SIZE;
  int N = SIZE / 2;
  int K = SIZE / 2;

  float *a_h, *b_h, *c_h;
  a_h = (float *) malloc(M * K * sizeof(float));
  b_h = (float *) malloc(K * N * sizeof(float));
  c_h = (float *) malloc(M * N * sizeof(float));

  for (int i = 0; i < M * K; i++) {
    a_h[i] = 1.0;
  }
  for (int i = 0; i < K * N; i++) {
    b_h[i] = 1.0;
  }
  for (int i = 0; i < M * N; i++) {
    c_h[i] = 0.0;
  }

  float *a, *b, *c;
  cudaMalloc(&a, M * K * sizeof(float));
  cudaMalloc(&b, K * N * sizeof(float));
  cudaMalloc(&c, M * N * sizeof(float));

  cudaMemcpy(a, a_h, M * K * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, b_h, K * N * sizeof(float), cudaMemcpyDeviceToHost);
  dim3 threadsPerBlock (32, 32);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1)/threadsPerBlock.y);
  tiledMatMul<<<blocksPerGrid, threadsPerBlock>>>(a, b, M, N, K, c);
  cudaMemcpy(c_h, c, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  free(a);
  free(b);
  free(c);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

}
