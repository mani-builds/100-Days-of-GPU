#include <stdio.h>


#define TILE 16

__global__ void relu_matmul(float *a, float *b, float *c, int M, int K, int N) {

  // map data to threads
  int col = blockIdx.x * TILE + threadIdx.x;
  int row = blockIdx.y * TILE + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // smem
  __shared__ float as[TILE][TILE];
  __shared__ float bs[TILE][TILE];
  // Implement
  float pvalue = 0.0f;
  if (row < M && col < N){
  for (int ph = 0; ph < K; ph++) {
    as[ty][tx] = a[row * K + (ph*TILE + tx)];
    bs[ty][tx] = b[(ph*TILE+ty)*N + col];
    __syncthreads();

    for (int k = 0; k < TILE; k++) {
      pvalue += as[ty][k] * bs[k][tx];
    }
    __syncthreads();
  }
  }
  // HBM
  if (row < M && col < N)
  c[row * M + col] = fmaxf(0, pvalue);
}

int main() {
    int M = 32, K = 32, N = 32;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    // Host allocations
    float *h_a = (float*)malloc(size_a);
    float *h_b = (float*)malloc(size_b);
    float *h_c = (float*)malloc(size_c);

    // Initialize input matrices with random values
    for (int i = 0; i < M * K; i++) h_a[i] = (float)(rand() % 10 - 5);
    for (int i = 0; i < K * N; i++) h_b[i] = (float)(rand() % 10 - 5);

    // Device allocations
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, size_c);

    // Kernel launch configuration
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Launch kernel
    relu_matmul<<<blocks, threads>>>(d_a, d_b, d_c, M, K, N);

    // Copy result back
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    // Print a portion of the output matrix
    printf("Output (ReLU(matmul)) sample:\n");
    for (int i = 0; i < (M < 8 ? M : 8); i++) {
        for (int j = 0; j < (N < 8 ? N : 8); j++) {
            printf("%6.1f ", h_c[i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
