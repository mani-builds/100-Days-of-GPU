#include <stdio.h>

#define TILE 32

__global__ void transpose(float *a, float *b, int M, int N) {
  // Threads to data
  int col = blockIdx.x * TILE + threadIdx.x;
  int row = blockIdx.y * TILE + threadIdx.y;

  // smem
  __shared__ float as[TILE][TILE + 1];
  if (row < M && col < N){
  as[threadIdx.y][threadIdx.x] = a[row * N + col];
  }
  __syncthreads();
  // Logic implement
  int t_col = TILE * blockIdx.y + threadIdx.x;
  int t_row = TILE * blockIdx.x + threadIdx.y;
  // HBM
  if (t_row < N && t_col < M)
  b[t_row * M + t_col] = as[threadIdx.x][threadIdx.y];
}

int main() {
    int M = 2000; // number of rows
    int N = 1000; // number of columns

    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * M * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);

    // Initialize input matrix
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            h_A[i*N + j] = i * N + j;

    printf("Original matrix (first 8x8 block):\n");
for (int i = 0; i < 8 && i < M; ++i) {
    for (int j = 0; j < 8 && j < N; ++j)
        printf("%5.1f ", h_A[i*N + j]);
    printf("\n");
}


    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    transpose<<<gridDim, blockDim>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost);

    // Print part of the result for verification
    printf("Transposed matrix (first 8x8 block):\n");
    for (int i = 0; i < 8 && i < N; ++i) {
        for (int j = 0; j < 8 && j < M; ++j)
            printf("%5.1f ", h_B[i*M + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
