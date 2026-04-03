#include <stdio.h>

// Histogram of factors of {2, 4, 6, 8,10}

__constant__ int FACTORS[5];

__global__ void hist(int *in, int *hist, int N) {
  // Map threads to data
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // smem
  int local_value;
  // Implement
  if (i < N) {
    local_value = in[i];
    for (int k = 0; k < 5; k++) {
      if (local_value % FACTORS[k] == 0) {
        atomicAdd(&hist[k],1);
      }
    }

  }
  // HBM
}

int main() {
  int N = 2000;
  int h_factors[5] = {2, 4, 6, 8, 10};
  int *h_in = (int*)malloc(N * sizeof(int));
  int *h_hist = (int*)calloc(5, sizeof(int));

  // Fill input with numbers 1..N
  for (int i = 0; i < N; i++) h_in[i] = i + 1;

  // Copy factors to constant memory
  cudaMemcpyToSymbol(FACTORS, h_factors, 5 * sizeof(int));

  // Allocate device memory
  int *d_in, *d_hist;
  cudaMalloc(&d_in, N * sizeof(int));
  cudaMalloc(&d_hist, 5 * sizeof(int));
  cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_hist, 0, 5 * sizeof(int));

  // Launch kernel
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  hist<<<blocks, threads>>>(d_in, d_hist, N);

  // Copy result back
  cudaMemcpy(h_hist, d_hist, 5 * sizeof(int), cudaMemcpyDeviceToHost);

  // Print histogram
  printf("Histogram of factors:\n");
  for (int k = 0; k < 5; k++) {
    printf("Divisible by %d: %d\n", h_factors[k], h_hist[k]);
  }

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_hist);
  free(h_in);
  free(h_hist);

  return 0;
}
