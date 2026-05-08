#include <stdio.h>

// Note : Inplace transpose can be done with 2 shared mem variable but
// only fo squared matrices, for rectangular matrices, we have to use PPMC
// algorithm


#define TILE_WIDTH 32

__global__ void trans_inplace(float *in, int N) {
  if (blockIdx.y > blockIdx.x) return;// We only iterate over the
                                      // upper triangle of the BLOCKS

  //smem
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH+1]; // Avoid bank conflict
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH+1];

  // Coordinates for upper tile
  int x1 = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int y1 = blockIdx.y * TILE_WIDTH + threadIdx.y;

  // Coordinates for lower tile
  int x2 = blockIdx.y * TILE_WIDTH + threadIdx.x;
  int y2 = blockIdx.x * TILE_WIDTH + threadIdx.y;

  // if (blockIdx.y == blockIdx.x) {
  //   // --- CASE 1: DIAGONAL TILE ---
  //   // Swap inplace with just one tile
  //   if (x1 < N && y1 < N){
  //   tileA[threadIdx.y][threadIdx.x] = in[y1 * N + x1];
  //   }
  //   __syncthreads(); // Wait till all the threads are finished

  //   // Write back transposed to the same location
  //   if (x1 < N && y1 < N){
  //     in[y1 * N + x1] = tileA[threadIdx.x][threadIdx.y];
  //   }
  // } else
    {
    // --- CASE 2: NON-DIAGONAL TILE --
    // Load both upper and lower tile
    if (x1 < N && y1 < N){
    tileA[threadIdx.y][threadIdx.x] = in[y1 * N + x1];
    }
    if (x2 < N && y2 < N){
    tileB[threadIdx.y][threadIdx.x] = in[y2 * N + x2];
    }
    __syncthreads(); // Wait till all the threads are finished

    // Write back transposed to each other tile
    if (x2 < N && y2 < N){
      in[y2 * N + x2] = tileA[threadIdx.x][threadIdx.y];
    }
    if (x1 < N && y1 < N){
      in[y1 * N + x1] = tileB[threadIdx.x][threadIdx.y];
    }
  }
}



int main() {
    int M = 2000; // number of rows
    int N = M; // number of columns

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

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    trans_inplace<<<gridDim, blockDim>>>(d_A, N);

    cudaMemcpy(h_A, d_A, bytes_A, cudaMemcpyDeviceToHost);

    // Print part of the result for verification
    printf("Original matrix transposed (first 8x8 block):\n");
for (int i = 0; i < 8 && i < M; ++i) {
    for (int j = 0; j < 8 && j < N; ++j)
        printf("%5.1f ", h_A[i*N + j]);
    printf("\n");
}

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
