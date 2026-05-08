#include <stdio.h>

#define TILE 32

__global__ void matmul(float *a, float *b, int M, int K, int N, float *o) {

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

int main() {
    int M = 1000, K = 2000, N = 1000;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size_a);
    float *h_b = (float*)malloc(size_b);
    float *h_c = (float*)malloc(size_c);

    // Initialize input matrices
    for (int i = 0; i < M * K; i++) h_a[i] = 1.0f; // or any pattern
    for (int i = 0; i < K * N; i++) h_b[i] = 1.0f;

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size_a);
    cudaMalloc((void**)&d_b, size_b);
    cudaMalloc((void**)&d_c, size_c);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    printf("block y: %d block x: %d\n", gridDim.y, gridDim.x);
    printf("thread y: %d thread x: %d\n", blockDim.y, blockDim.x);
    matmul<<<gridDim, blockDim>>>(d_a, d_b, M, K, N, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    // Print a small part of the result
    printf("Result matrix (first 5x5 block):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_c[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
