#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 1024

__global__ void softmax_kernel(float *arr, int N, float *accu, float *out) {
  // Map threads to data
  int i = blockIdx.x * TILE_WIDTH + threadIdx.x;

  // smem
  __shared__ float arr_s[TILE_WIDTH];
  if(i < N) arr_s[threadIdx.x] = arr[i];
  __syncthreads();

  // Logic
  float sum = 0;
  for (int k = 0; k < TILE_WIDTH; k++) sum += expf(arr[k]);
  atomicAdd(accu,sum);
  // HBM
  if (i < N) out[i] = expf(arr_s[i]) / *accu;
}



int main() {
    int N = TILE_WIDTH;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_arr = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // Initialize input
    for (int i = 0; i < N; i++) h_arr[i] = (float)i / (float)N;
    // Print first 10 results
    printf("Input:\n");
    for (int i = N-1; i > N - 10; i--) {
        printf("arr[%d] = %f\n", i, h_arr[i]);
    }
    printf("\n");

    // Allocate device memory
    float *d_arr, *d_out, *d_accu;
    cudaMalloc((void**)&d_arr, size);
    cudaMalloc((void**)&d_out, size);
    cudaMalloc((void**)&d_accu, sizeof(float));

    // Copy input to device
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    printf("blocks: %d threads: %d\n", numBlocks, TILE_WIDTH);
    softmax_kernel<<<numBlocks, TILE_WIDTH>>>(d_arr, N, d_accu, d_out);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    for (int i = N-1; i > N-10; i--) {
        printf("softmax[%d] = %f\n", i, h_out[i]);
    }

    // Free memory
    cudaFree(d_arr);
    cudaFree(d_out);
    free(h_arr);
    free(h_out);

    return 0;
}
