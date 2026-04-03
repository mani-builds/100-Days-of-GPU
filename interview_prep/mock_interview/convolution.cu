#include <stdio.h>

#define FILTER_RADIUS 1
#define IN_TILE  16
#define OUT_TILE ((IN_TILE) - 2*(FILTER_RADIUS))

__constant__ float K[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];

__global__ void conv(float *a, int M, int N, float *b) {

    // 1. Calculate global coordinates based on Input Tile
    int row = blockIdx.y * OUT_TILE + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE + threadIdx.x - FILTER_RADIUS;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // smem
  __shared__ float as[IN_TILE][IN_TILE];

  // Boundary check for the input matrix
    if (row >= 0 && row < M && col >= 0 && col < N) {
        as[ty][tx] = a[row * N + col];
    } else {
        as[ty][tx] = 0.0f; // Padding
    }

    __syncthreads();

  // logic
  if (ty >= FILTER_RADIUS && ty < IN_TILE - FILTER_RADIUS &&
        tx >= FILTER_RADIUS && tx < IN_TILE - FILTER_RADIUS) {
  float pvalue = 0.0f;
    if (row >= 0 && row < M && col >=0 && col <N){
    for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
        pvalue += K[i + FILTER_RADIUS][j + FILTER_RADIUS] * as[ty + i][tx + j];
        }
    }
    // HBM
    b[row * N + col] = pvalue;
  }
  }
}


int main() {
    int M = 1000, N = 2000;
    size_t size_a = M * N * sizeof(float);
    size_t size_b = M * N * sizeof(float);

    // Host allocations
    float *h_a = (float*)malloc(size_a);
    float *h_b = (float*)malloc(size_b);

    // Initialize input matrix with values
    for (int i = 0; i < M * N; i++) {
        h_a[i] = (float)(i % 10);
    }
    printf("Input (sample 8x8):\n");
    for (int i = 0; i < (M < 8 ? M : 8); i++) {
        for (int j = 0; j < (N < 8 ? N : 8); j++) {
            printf("%6.1f ", h_a[i * N + j]);
        }
        printf("\n");
    }


    // Define a simple 3x3 kernel (e.g., all ones)
    float h_K[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1] = {
        {1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f}
    };

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(K, h_K, sizeof(h_K));

    // Device allocations
    float *d_a, *d_b;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, size_b);

    // Kernel launch configuration
    dim3 threads(IN_TILE, IN_TILE);
    dim3 blocks((N + OUT_TILE - 1) / OUT_TILE, (M + OUT_TILE - 1) / OUT_TILE);

    // Launch kernel
    conv<<<blocks, threads>>>(d_a, M, N, d_b);

    // Copy result back
    cudaMemcpy(h_b, d_b, size_b, cudaMemcpyDeviceToHost);

    // Print a portion of the output matrix
    printf("Convolution output (sample 8x8):\n");
    for (int i = 0; i < (M < 8 ? M : 8); i++) {
        for (int j = 0; j < (N < 8 ? N : 8); j++) {
            printf("%6.1f ", h_b[i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    return 0;
}
