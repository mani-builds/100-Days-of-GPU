#include <stdio.h>

#define TILE_SIZE 32

__global__ void trans_kernel(float *in, float *out, int M, int N) {

  // Map data to threads
  int col = blockIdx.x * blockDim.x  + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // smem
  __shared__ float in_s[TILE_SIZE][TILE_SIZE+1];
  if (row < M && col < N){
  in_s[ty][tx] = in[row * N + col];
  }
  __syncthreads();
  // // Logic
  int out_col = blockIdx.y * blockDim.x + threadIdx.x;
  int out_row = blockIdx.x * blockDim.y + threadIdx.y;

  // // HBM
  if (out_row < N && out_col < M){
  out[out_row * M + out_col] = in_s[tx][ty];
  }
}

int main() {
    int M = 1000; // number of rows
    int N = 2000; // number of columns

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

    trans_kernel<<<gridDim, blockDim>>>(d_A, d_B, M, N);

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
