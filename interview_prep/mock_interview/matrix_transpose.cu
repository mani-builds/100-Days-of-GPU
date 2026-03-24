
#include <stdio.h>

#define M 1024
#define N 1024

#define TILE_SIZE 32

__global__ void transpose(float *A, float *B) {

  // Threads to data
  int col = TILE_SIZE * blockIdx.x + threadIdx.x;
  int row = TILE_SIZE * blockIdx.y + threadIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // smem
  __shared__ float As[TILE_SIZE][TILE_SIZE + 1]; // Avoid bank conflicts

// implementation

  if (row < M && col < N) {
    As[ty][tx] = A[row * N + col];
  }
   __syncthreads();

  int transposed_col = TILE_SIZE * blockIdx.y + threadIdx.x;
  int transposed_row = TILE_SIZE * blockIdx.x + threadIdx.y;

  // HBM
  if (transposed_col < M && transposed_row < N)
    B[transposed_row * M + transposed_col] = As[tx][ty];

}

int main() {
    // int M = 32; // number of rows
    // int N = 48; // number of columns

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

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    transpose<<<gridDim, blockDim>>>(d_A, d_B);

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
