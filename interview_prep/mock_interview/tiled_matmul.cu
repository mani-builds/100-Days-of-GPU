#include <stdio.h>

#define TILE 16

__global__ void matmul ( float *A, float *B, float *C, int M, int K, int N){
  // Map data to threads
  int col = TILE * blockIdx.x + threadIdx.x;
  int row = TILE * blockIdx.y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // smem
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  if (row < M && col < N){
  float pvalue = 0.0f;
  for (int ph=0; ph < K / TILE; ph++){
    // Each thread collaboratively loads an element
    As[ty][tx] = A[row*K + (ph * TILE + tx)];
    Bs[ty][tx] = B[(ph * TILE + ty) * N + col];
    __syncthreads();

    for (int i = 0; i < TILE; i++) {
      pvalue += As[ty][i] * Bs[i][tx];
    }
    __syncthreads();
  }

  // HBM
  C[row * N + col] = pvalue;
  }
}


int main() {
  int M = 1000;
  int K = 2000;
  int N = 1000;

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  // 1. Allocate host memory
  float *h_A = (float*)malloc(size_A);
  float *h_B = (float*)malloc(size_B);
  float *h_C = (float*)malloc(size_C);

  // 2. Initialize host matrices
  for (int i = 0; i < M*K; ++i) h_A[i] = 1.0f; // or any value
  for (int i = 0; i < K*N; ++i) h_B[i] = 1.0f;

  // 3. Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  // 4. Copy host data to device
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  // 5. Kernel launch configuration
  dim3 threadsPerBlock(TILE, TILE);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // 6. Launch kernel
  matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

  // 7. Copy result back to host
  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  // 8. (Optional) Print a few results
  printf("C[0]=%f, C[M*N-1]=%f\n", h_C[0], h_C[M*N-1]);

  // 9. Free memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
