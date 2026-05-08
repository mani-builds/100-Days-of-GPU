#include <cmath>
#include <stdio.h>

// Naive row-wise softmax with global memory loading of a Matrix

__global__ void navie_softmax_kernel(float *in, int M, int N, float *ou) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread in a block process an entire row of input matrix

  if (row < M){
  float max_value = -INFINITY;
  float norm = 0.0f;
  // Logic: 3 passes (max_value, norm, final softmax)
  for (int col = 0; col < N; col++) {
    // calculate max_value
    max_value = fmax(max_value, in[row * N + col]);
  }
  for (int col = 0; col < N; col++) {
    // norm
    norm += expf(in[row * N + col] - max_value);
  }
  // HBM
  for (int col = 0; col < N; col++) {
    ou[row*N + col] = expf(in[row * N + col] - max_value) / norm;
  }
  }
}



int main() {
    int M = 3; // number of rows
    int N = 3; //number of columns
    size_t size = M * N * sizeof(float);

    // Allocate and initialize host memory
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    for (int i = 0; i < M * N; i++) {
        h_in[i] = (float) i / (float) N; // simple initialization
    }
    // Print first 10 results
    printf("Input:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", h_in[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // Copy input to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 128;
  // each thread in a block process one row
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
    printf("blocks: %d threads: %d\n", blocksPerGrid, threadsPerBlock);
    navie_softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, M, N, d_out);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Softmax output:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", h_out[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
